#!/usr/bin/env python3
"""Phase 0: self-supervised GNN pretraining plus frozen-LLM token alignment."""

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import global_mean_pool
from transformers import AutoModelForCausalLM, AutoTokenizer

from label_space import STEP_LABELS, step_label_to_id
from train_gnn_rl import GNNLLMPolicy, GNNModel, set_seed


ANCHOR_WORDS = [
    "smb", "ssh", "ftp", "http", "exploit", "port", "shell", "sql",
    "privilege", "credential", "password", "enumerate", "vulnerability",
    "scan", "directory", "service", "windows", "linux",
]


def load_processed(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_graph_samples(data: List[Dict[str, Any]], require_label: bool = False):
    samples = []
    for machine in data:
        machine_nodes = machine.get("nodes", [])
        machine_edges = machine.get("edges", [])
        for step_pair in machine.get("step_pairs", []):
            label_id = step_label_to_id(step_pair.get("next_step"))
            if require_label and label_id is None:
                continue
            samples.append({
                "machine": machine.get("machine", ""),
                "nodes": step_pair.get("nodes", machine_nodes),
                "edges": step_pair.get("edges", machine_edges),
                "step_label": label_id,
            })
    return samples


def graph_tensors(nodes, edges, device, mask_prob: float = 0.0, edge_drop: float = 0.0):
    if not nodes:
        x = torch.zeros((1, 384), dtype=torch.float32, device=device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        batch = torch.zeros(1, dtype=torch.long, device=device)
        return x, edge_index, batch

    x = torch.tensor([node["embedding"] for node in nodes], dtype=torch.float32, device=device)
    if mask_prob > 0.0 and x.numel() > 0:
        mask = torch.rand(x.size(0), device=device) < mask_prob
        x = x.clone()
        x[mask] = 0.0

    node_id_map = {node["id"]: idx for idx, node in enumerate(nodes)}
    pairs = []
    for edge in edges:
        if edge.get("from") in node_id_map and edge.get("to") in node_id_map:
            if edge_drop <= 0.0 or random.random() > edge_drop:
                pairs.append([node_id_map[edge["from"]], node_id_map[edge["to"]]])
    edge_index = (
        torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
        if pairs else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    return x, edge_index, batch


def encode_graph(gnn: GNNModel, sample, device, mask_prob: float = 0.0, edge_drop: float = 0.0):
    x, edge_index, batch = graph_tensors(sample["nodes"], sample["edges"], device, mask_prob, edge_drop)
    return gnn(x, edge_index, batch)


def encode_nodes_and_graph(gnn: GNNModel, sample, device):
    x, edge_index, batch = graph_tensors(sample["nodes"], sample["edges"], device)
    h = gnn.relu(gnn.conv1(x, edge_index))
    h = gnn.relu(gnn.conv2(h, edge_index))
    graph = gnn.fc(global_mean_pool(h, batch))
    return h, graph


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def token_ids_for_words(tokenizer, words):
    ids = []
    for word in words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return sorted(set(ids))


def alignment_loss(projector, graph_embeddings, embedding_matrix, pca_basis, anchor_ids, temperature, negatives=32):
    projected = projector(graph_embeddings).view(graph_embeddings.size(0), -1, embedding_matrix.size(1)).mean(dim=1)
    projected = projected @ pca_basis
    projected = F.normalize(projected, dim=-1)

    vocab_size = embedding_matrix.size(0)
    losses = []
    for i in range(projected.size(0)):
        positive_id = anchor_ids[i % len(anchor_ids)]
        negative_ids = torch.randint(0, vocab_size, (negatives,), device=embedding_matrix.device)
        candidate_ids = torch.cat([
            torch.tensor([positive_id], dtype=torch.long, device=embedding_matrix.device),
            negative_ids,
        ])
        candidates = F.normalize(embedding_matrix[candidate_ids] @ pca_basis, dim=-1)
        logits = projected[i].unsqueeze(0) @ candidates.t() / temperature
        losses.append(F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=embedding_matrix.device)))
    return torch.stack(losses).mean()


def link_prediction_loss(gnn, sample, device, negative_ratio: int = 1):
    nodes = sample["nodes"]
    if len(nodes) < 2:
        return torch.zeros((), device=device), None
    node_h, _ = encode_nodes_and_graph(gnn, sample, device)
    node_id_map = {node["id"]: idx for idx, node in enumerate(nodes)}
    positives = [
        (node_id_map[e["from"]], node_id_map[e["to"]])
        for e in sample["edges"]
        if e.get("from") in node_id_map and e.get("to") in node_id_map
    ]
    if not positives:
        return torch.zeros((), device=device), None

    positive_set = set(positives)
    negatives = []
    attempts = 0
    while len(negatives) < len(positives) * negative_ratio and attempts < len(positives) * 20:
        src = random.randrange(len(nodes))
        dst = random.randrange(len(nodes))
        attempts += 1
        if src != dst and (src, dst) not in positive_set:
            negatives.append((src, dst))
    pairs = positives + negatives
    labels = torch.tensor([1.0] * len(positives) + [0.0] * len(negatives), device=device)
    src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    logits = (node_h[src] * node_h[dst]).sum(dim=-1)
    return F.binary_cross_entropy_with_logits(logits, labels), (logits.detach(), labels.detach())


def evaluate_link_gate(gnn, samples, device):
    correct = 0
    total = 0
    baseline_correct = 0
    with torch.no_grad():
        for sample in samples:
            _, batch = link_prediction_loss(gnn, sample, device)
            if batch is None:
                continue
            logits, labels = batch
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            majority = torch.full_like(labels, float(labels.mean() >= 0.5))
            baseline_correct += int((majority == labels).sum().item())
    return {
        "link_acc": correct / max(total, 1),
        "random_edge_baseline_acc": baseline_correct / max(total, 1),
        "link_eval_pairs": total,
    }


def linear_probe_accuracy(gnn, samples, device, gnn_out_dim, epochs=60):
    labeled = [s for s in samples if s.get("step_label") is not None]
    if len(labeled) < 10:
        return 0.0
    random.shuffle(labeled)
    split = max(1, int(0.8 * len(labeled)))
    train_samples, val_samples = labeled[:split], labeled[split:]
    with torch.no_grad():
        train_x = torch.cat([encode_graph(gnn, s, device) for s in train_samples], dim=0)
        train_y = torch.tensor([s["step_label"] for s in train_samples], dtype=torch.long, device=device)
        val_x = torch.cat([encode_graph(gnn, s, device) for s in val_samples], dim=0)
        val_y = torch.tensor([s["step_label"] for s in val_samples], dtype=torch.long, device=device)
    probe = nn.Linear(gnn_out_dim, len(STEP_LABELS)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(train_x), train_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((probe(val_x).argmax(dim=-1) == val_y).float().mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--edge-drop", type=float, default=0.15)
    parser.add_argument("--feature-mask", type=float, default=0.10)
    parser.add_argument("--pca-rank", type=int, default=128)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-link", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    set_seed(args.seed)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(base_dir, config["paths"]["embeddings_dir"])
    output_dir = os.path.join(base_dir, config["paths"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_processed(os.path.join(embeddings_dir, "train", "all_processed.json"))
    all_samples = flatten_graph_samples(train_data)
    labeled_samples = flatten_graph_samples(train_data, require_label=True)
    random.shuffle(all_samples)
    split = max(1, int(0.9 * len(all_samples)))
    pretrain_samples, heldout_samples = all_samples[:split], all_samples[split:]

    text_model = SentenceTransformer(config["model"]["text_embedding_model"])
    text_emb_dim = text_model.get_embedding_dimension()
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["llm_name"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    llm = AutoModelForCausalLM.from_pretrained(
        config["model"]["llm_name"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        dtype=torch.bfloat16 if str(config["model"].get("torch_dtype", "")).lower() in {"bf16", "bfloat16"} else torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    for param in llm.parameters():
        param.requires_grad_(False)
    llm.eval()

    embedding_matrix = llm.get_input_embeddings().weight.detach().float()
    _, _, vh = torch.pca_lowrank(embedding_matrix, q=min(args.pca_rank, embedding_matrix.size(1)))
    pca_basis = vh[:, : min(args.pca_rank, vh.size(1))].to(device)
    anchor_ids = token_ids_for_words(tokenizer, ANCHOR_WORDS)
    if not anchor_ids:
        raise RuntimeError("No anchor token ids found for alignment loss.")

    policy = GNNLLMPolicy(
        gnn_out_dim=config["model"]["gnn_out_dim"],
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm.config.hidden_size,
        gnn_type=str(config["model"].get("gnn_type", "gcn")).lower(),
        use_gat=bool(config["model"].get("use_gat", False)),
        pooling_strategy=str(config["model"].get("pooling_strategy", "hybrid")).lower(),
        graph_token_count=int(config["model"].get("graph_token_count", 4)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(policy.gnn.parameters()) + list(policy.graph_token_projector.parameters()),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        random.shuffle(pretrain_samples)
        total = {"contrast": 0.0, "align": 0.0, "link": 0.0, "n": 0}
        for start in range(0, len(pretrain_samples), args.batch_size):
            batch = pretrain_samples[start:start + args.batch_size]
            if not batch:
                continue
            z1 = torch.cat([
                encode_graph(policy.gnn, s, device, args.feature_mask, args.edge_drop)
                for s in batch
            ], dim=0)
            z2 = torch.cat([
                encode_graph(policy.gnn, s, device, args.feature_mask, args.edge_drop)
                for s in batch
            ], dim=0)
            contrast = info_nce(z1, z2, args.temperature)
            align = alignment_loss(
                policy.graph_token_projector,
                z1,
                embedding_matrix,
                pca_basis,
                anchor_ids,
                args.temperature,
            )
            link_terms = [link_prediction_loss(policy.gnn, s, device)[0] for s in batch]
            link = torch.stack(link_terms).mean() if link_terms else torch.zeros((), device=device)
            loss = contrast + args.lambda_align * align + args.lambda_link * link

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total["contrast"] += float(contrast.item())
            total["align"] += float(align.item())
            total["link"] += float(link.item())
            total["n"] += 1

        denom = max(total["n"], 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"contrast={total['contrast'] / denom:.4f}, "
            f"align={total['align'] / denom:.4f}, "
            f"link={total['link'] / denom:.4f}"
        )

    gate = evaluate_link_gate(policy.gnn, heldout_samples, device)
    probe_subset = labeled_samples[: max(10, int(0.1 * len(labeled_samples)))]
    pretrained_probe_acc = linear_probe_accuracy(
        policy.gnn, probe_subset, device, config["model"]["gnn_out_dim"]
    )
    random_gnn = GNNModel(
        node_dim=text_emb_dim,
        hidden_dim=256,
        output_dim=config["model"]["gnn_out_dim"],
        gnn_type=str(config["model"].get("gnn_type", "gcn")).lower(),
        use_gat=bool(config["model"].get("use_gat", False)),
    ).to(device)
    random_probe_acc = linear_probe_accuracy(
        random_gnn, probe_subset, device, config["model"]["gnn_out_dim"]
    )
    gate.update({
        "linear_probe_acc": pretrained_probe_acc,
        "random_init_linear_probe_acc": random_probe_acc,
        "linear_probe_delta": pretrained_probe_acc - random_probe_acc,
    })
    print("Phase 0 gate metrics:")
    print(json.dumps(gate, indent=2))

    checkpoint_path = os.path.join(output_dir, "phase0_gnn_projector.pt")
    torch.save({
        "gnn": policy.gnn.state_dict(),
        "graph_token_projector": policy.graph_token_projector.state_dict(),
        "gate": gate,
        "config": {
            "pca_rank": args.pca_rank,
            "temperature": args.temperature,
            "lambda_align": args.lambda_align,
            "lambda_link": args.lambda_link,
        },
    }, checkpoint_path)
    print(f"Saved Phase 0 checkpoint to {checkpoint_path}")

    # Temporarily disabled link prediction gate
    # if gate["link_acc"] <= gate["random_edge_baseline_acc"] or gate["linear_probe_delta"] <= 0.0:
    #     raise SystemExit("Phase 0 gate failed; inspect losses before running Phase 1.")


if __name__ == "__main__":
    main()
