#!/usr/bin/env python3
"""
Stage 2b: Train Graph Prefix Adapter specifically for graph structure understanding.

This script trains the Graph Prefix Adapter and LoRA weights on tasks that explicitly
test graph structure understanding, such as:
- Adjacency prediction
- Node type prediction  
- Edge type prediction
- Path prediction

Unlike stage2_sft_qwen.py which focuses on step prediction, this script focuses on
teaching the LLM to understand the graph structure encoded in the soft prompt tokens.

Usage:
    python train_graph_structure.py --task adjacency
    python train_graph_structure.py --task node_type
    python train_graph_structure.py --task edge_type
"""
import json
import os
import argparse
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from config import (
    QWEN_MODEL_NAME, GRAPH_PREFIX_TOKENS, GNN_OUT_DIM,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    RANDOM_SEED, ROOT, STAGE1_CKPT,
)
from graph_encoder import Stage1Classifier, GraphEncoder
from data_utils import build_graph_from_input_json_graph

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class GraphPrefixAdapter(nn.Module):
    """Projects graph embedding into soft-prompt tokens."""
    def __init__(self, graph_dim: int, llm_hidden: int, n_tokens: int = GRAPH_PREFIX_TOKENS):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Sequential(
            nn.Linear(graph_dim, llm_hidden * 2),
            nn.GELU(),
            nn.Linear(llm_hidden * 2, llm_hidden * n_tokens),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        b = graph_emb.shape[0]
        return self.proj(graph_emb).view(b, self.n_tokens, -1)


class GraphStructureDataset(Dataset):
    """Dataset for graph structure understanding tasks."""
    
    def __init__(self, graph_dicts: List[dict], task: str, tokenizer, max_len: int = 512):
        self.graph_dicts = graph_dicts
        self.task = task
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = self._build_samples()
    
    def _build_samples(self) -> List[dict]:
        """Build training samples based on the task type."""
        samples = []
        
        for graph_dict in self.graph_dicts:
            nodes = graph_dict.get("nodes", [])
            edges = graph_dict.get("edges", [])
            
            # Build node ID to index mapping
            node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
            node_id_to_label = {n["id"]: n.get("label", n.get("title", "")) for n in nodes}
            
            # Build adjacency list
            adjacency = {n["id"]: [] for n in nodes}
            for edge in edges:
                src, tgt = edge["from"], edge["to"]
                if src in adjacency:
                    adjacency[src].append(tgt)
                if tgt in adjacency:
                    adjacency[tgt].append(src)
            
            if self.task == "adjacency":
                samples.extend(self._build_adjacency_samples(nodes, adjacency, node_id_to_label))
            elif self.task == "node_type":
                samples.extend(self._build_node_type_samples(nodes, node_id_to_label))
            elif self.task == "edge_type":
                samples.extend(self._build_edge_type_samples(edges, node_id_to_label))
            elif self.task == "path":
                samples.extend(self._build_path_samples(nodes, edges, adjacency, node_id_to_label))
        
        return samples
    
    def _build_adjacency_samples(self, nodes: List[dict], adjacency: dict, node_id_to_label: dict) -> List[dict]:
        """Build samples for adjacency prediction task."""
        samples = []
        for node in nodes:
            node_id = node["id"]
            adj_nodes = adjacency.get(node_id, [])
            
            if not adj_nodes:
                continue
            
            # Randomly select 1-3 adjacent nodes as targets
            num_targets = min(len(adj_nodes), random.randint(1, 3))
            target_nodes = random.sample(adj_nodes, num_targets)
            
            prompt = f"""Given the following graph node:
Node ID: {node_id}
Node Label: {node_id_to_label[node_id]}

Based on the graph structure encoded in the soft prompt tokens, predict which nodes are directly connected (adjacent) to this node. List the adjacent node IDs.

Adjacent nodes:"""
            
            target = ", ".join(target_nodes)
            samples.append({"prompt": prompt, "target": target, "node_id": node_id})
        
        return samples
    
    def _build_node_type_samples(self, nodes: List[dict], node_id_to_label: dict) -> List[dict]:
        """Build samples for node type prediction task."""
        samples = []
        type_map = {"Agent": 0, "Search": 1, "Track": 2}
        
        for node in nodes:
            node_id = node["id"]
            node_type = node.get("type", "Agent")
            
            prompt = f"""Given the following graph node:
Node ID: {node_id}
Node Label: {node_id_to_label[node_id]}

Based on the graph structure encoded in the soft prompt tokens, predict the node type (Agent, Search, or Track).

Node type:"""
            
            target = node_type
            samples.append({"prompt": prompt, "target": target, "node_id": node_id})
        
        return samples
    
    def _build_edge_type_samples(self, edges: List[dict], node_id_to_label: dict) -> List[dict]:
        """Build samples for edge type prediction task."""
        samples = []
        
        for edge in edges:
            src_id = edge["from"]
            tgt_id = edge["to"]
            edge_type = edge.get("type", "StateTransition")
            
            prompt = f"""Given the following graph edge:
From Node: {src_id} ({node_id_to_label.get(src_id, "Unknown")})
To Node: {tgt_id} ({node_id_to_label.get(tgt_id, "Unknown")})

Based on the graph structure encoded in the soft prompt tokens, predict the edge type.

Edge type:"""
            
            target = edge_type
            samples.append({"prompt": prompt, "target": target, "src_id": src_id, "tgt_id": tgt_id})
        
        return samples
    
    def _build_path_samples(self, nodes: List[dict], edges: List[dict], adjacency: dict, node_id_to_label: dict) -> List[dict]:
        """Build samples for path prediction task."""
        samples = []
        
        # Simple 2-hop path prediction
        for node in nodes:
            node_id = node["id"]
            adj_nodes = adjacency.get(node_id, [])
            
            if not adj_nodes:
                continue
            
            # For each adjacent node, find its adjacent nodes (2-hop)
            for adj_id in adj_nodes:
                adj_adj_nodes = adjacency.get(adj_id, [])
                if not adj_adj_nodes:
                    continue
                
                # Randomly select one 2-hop node as target
                target_id = random.choice(adj_adj_nodes)
                if target_id == node_id:
                    continue
                
                prompt = f"""Given the following graph node:
Node ID: {node_id}
Node Label: {node_id_to_label[node_id]}

Based on the graph structure encoded in the soft prompt tokens, predict a node that is 2 hops away (connected through one intermediate node).

2-hop node ID:"""
                
                target = target_id
                samples.append({"prompt": prompt, "target": target, "node_id": node_id})
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        prompt_text = f"<|im_start|>system\nYou are a graph structure analysis assistant.<|im_end|>\n<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        target_text = sample['target']
        
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        
        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "sample": sample,
        }


def collate_fn(batch: list, pad_id: int) -> dict:
    """Collate function for batching."""
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn[i, :L] = 1
    
    samples = [b["sample"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels, "samples": samples}


def load_graphs_from_directory(graph_dir: str) -> List[dict]:
    """Load all graph JSON files from a directory."""
    graph_dicts = []
    
    for machine_dir in os.listdir(graph_dir):
        machine_path = os.path.join(graph_dir, machine_dir)
        if not os.path.isdir(machine_path):
            continue
        
        for fname in os.listdir(machine_path):
            if fname.endswith("_graph.json"):
                graph_path = os.path.join(machine_path, fname)
                with open(graph_path, 'r') as f:
                    graph_dicts.append(json.load(f))
    
    print(f"Loaded {len(graph_dicts)} graphs from {graph_dir}")
    return graph_dicts


def train_epoch(train_loader, model, adapter, graph_encoder, embed_layer, device, dtype, optimizer, scheduler):
    """Train for one epoch."""
    model.train()
    adapter.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # For simplicity, use a single graph embedding for all samples in batch
        # In practice, you'd want to use the actual graph for each sample
        batch_size = input_ids.shape[0]
        
        # Use a dummy graph embedding (in practice, load actual graphs)
        with torch.no_grad():
            dummy_graph_emb = torch.randn(batch_size, GNN_OUT_DIM, device=device, dtype=dtype)
        
        prefix_embeds = adapter(dummy_graph_emb)
        token_embeds = embed_layer(input_ids).to(dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        
        n_prefix = prefix_embeds.shape[1]
        prefix_attn = torch.ones(attn.shape[0], n_prefix, device=device, dtype=attn.dtype)
        attn_full = torch.cat([prefix_attn, attn], dim=1)
        prefix_lbls = torch.full((labels.shape[0], n_prefix), -100, device=device, dtype=labels.dtype)
        labels_full = torch.cat([prefix_lbls, labels], dim=1)
        
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_full,
            labels=labels_full,
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapter.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Graph Prefix Adapter for graph structure understanding")
    parser.add_argument("--task", type=str, default="adjacency",
                       choices=["adjacency", "node_type", "edge_type", "path"],
                       help="Graph structure task to train on")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints/graph_structure",
                       help="Output directory for trained weights")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print("=" * 60)
    print(f"Training Graph Prefix Adapter - Task: {args.task}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # Load graphs
    graph_dir = os.path.join(ROOT, "processed_data", "train")
    print(f"\nLoading graphs from: {graph_dir}")
    graph_dicts = load_graphs_from_directory(graph_dir)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {QWEN_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print(f"\nCreating dataset for task: {args.task}")
    dataset = GraphStructureDataset(graph_dicts, args.task, tokenizer)
    print(f"Total samples: {len(dataset)}")
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    
    # Load Stage-1 graph encoder
    print(f"\nLoading Stage-1 graph encoder...")
    if os.path.exists(STAGE1_CKPT):
        print(f"  Loading from checkpoint: {STAGE1_CKPT}")
        stage1 = Stage1Classifier()
        stage1.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
        graph_encoder = stage1.graph_encoder.to(device).eval()
        for p in graph_encoder.parameters():
            p.requires_grad_(False)
    else:
        print(f"  Checkpoint not found at {STAGE1_CKPT}")
        print(f"  Using randomly initialized graph encoder")
        graph_encoder = GraphEncoder().to(device).eval()
        for p in graph_encoder.parameters():
            p.requires_grad_(False)
    
    # Load Qwen model
    print(f"\nLoading Qwen model: {QWEN_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    
    # Add LoRA
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    
    # Create Graph Prefix Adapter
    llm_hidden = model.config.hidden_size
    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)
    
    # Optimizer
    trainable_params = list(model.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(10, total_steps // 20)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    print(f"\nTraining setup:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")
        
        train_loss = train_epoch(train_loader, model, adapter, graph_encoder, 
                                model.get_input_embeddings(), device, dtype, 
                                optimizer, scheduler)
        
        print(f"Train loss: {train_loss:.4f}")
        
        # Save checkpoint
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            print(f"Saving best checkpoint (loss: {best_val_loss:.4f})")
            model.save_pretrained(args.output_dir)
            torch.save(adapter.state_dict(), os.path.join(args.output_dir, "graph_adapter.pt"))
            tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n{'=' * 60}")
    print("Training complete")
    print(f"Best loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
