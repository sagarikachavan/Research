"""
Stage 2: Supervised fine-tuning of Qwen on the FULL context (same textual
fields the paper feeds its one-shot prompt, but now with worked examples
instead of a single hand-written one) plus a graph-conditioning adapter
that turns the frozen Stage-1 graph embedding into a handful of soft-prompt
token embeddings prepended to the Qwen input -- analogous to how the paper
grounds the LLM with one worked example, except grounding is now a learned
vector instead of static text.

The model is trained to autoregressively produce a structured block:

    New step: <one of STEP_LABELS, verbatim>
    Step explanation: <free text>
    MCP_tasks: <json dict subset of MCP_LABELS -> short action string>

Only the target block contributes to the loss (prompt tokens are masked).

Run:
    python stage2_sft_qwen.py
"""
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from config import (
    TRAIN_CSV, QWEN_MODEL_NAME, GRAPH_PREFIX_TOKENS, GNN_OUT_DIM,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, STAGE2_LR, STAGE2_EPOCHS,
    STAGE2_BATCH_SIZE, STAGE2_GRAD_ACCUM, STAGE1_CKPT, STAGE2_ADAPTER_DIR,
    RANDOM_SEED,
)
from data_utils import load_and_clean, load_graph, CONTEXT_COLUMNS, _embed_texts
from graph_encoder import Stage1Classifier

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


SYSTEM_PROMPT = (
    "You are an autonomous penetration-testing planning assistant operating "
    "strictly within an authorized lab environment. Given the current "
    "reconnaissance graph state and the previous/new strategy context, "
    "choose exactly one next-step type from the fixed taxonomy, exactly one "
    "or more tool(s) from the fixed MCP taxonomy, and explain your reasoning."
)


def build_prompt(ex):
    ctx = ex["context"]
    lines = [
        f"Machine: {ex['machine']}",
        f"Previous strategy: {ctx['Previous strategy']}",
        f"Previous step: {ctx['Previous step']}",
        f"Previous step result: {ctx['Previous step result']}",
        f"New strategy: {ctx['New strategy']}",
        f"Strategy explanation: {ctx['Strategy explanation']}",
    ]
    return "\n".join(lines)


def build_target(ex):
    mcp_dict = {}
    # best-effort: reuse whatever action text existed for tools we detected
    for label in ex["mcp_labels"]:
        mcp_dict[label] = f"Use {label} as part of: {ex['step_label']}"
    target = {
        "New step": ex["step_label"],
        "Step explanation": ex["gold_step_explanation"],
        "MCP_tasks": mcp_dict,
    }
    return json.dumps(target, ensure_ascii=False)


class GraphPrefixAdapter(nn.Module):
    """Projects a single graph embedding vector into GRAPH_PREFIX_TOKENS
    soft-prompt embeddings living in the LLM's hidden space."""

    def __init__(self, graph_dim, llm_hidden, n_tokens=GRAPH_PREFIX_TOKENS):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Sequential(
            nn.Linear(graph_dim, llm_hidden * 2),
            nn.GELU(),
            nn.Linear(llm_hidden * 2, llm_hidden * n_tokens),
        )

    def forward(self, graph_emb):  # (B, graph_dim) -> (B, n_tokens, llm_hidden)
        b = graph_emb.shape[0]
        out = self.proj(graph_emb)
        return out.view(b, self.n_tokens, -1)


class SFTDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=2048):
        self.examples = load_and_clean(csv_path, "train")
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = build_prompt(ex)
        target = build_target(ex)

        prompt_ids = self.tok(
            f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n",
            add_special_tokens=False,
        )["input_ids"]
        target_ids = self.tok(target, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]

        graph = load_graph(ex["machine"], ex["row_id"], ex["ptt"], "train")
        field_embs = _embed_texts([ex["context"][c] or "empty" for c in CONTEXT_COLUMNS])

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "graph": graph,
            "field_embs": torch.tensor(field_embs, dtype=torch.float32),
        }


def collate(batch, pad_id):
    from torch_geometric.data import Batch as PyGBatch
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn[i, :L] = 1
    graphs = PyGBatch.from_data_list([b["graph"] for b in batch])
    field_embs = torch.stack([b["field_embs"] for b in batch])
    return input_ids, attn, labels, graphs, field_embs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Load the frozen Stage-1 graph encoder to produce graph embeddings.
    stage1 = Stage1Classifier()
    stage1.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
    stage1.graph_encoder.to(device).eval()
    for p in stage1.graph_encoder.parameters():
        p.requires_grad_(False)

    llm_hidden = model.config.hidden_size
    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(torch.bfloat16)

    ds = SFTDataset(TRAIN_CSV, tokenizer)
    loader = DataLoader(
        ds, batch_size=STAGE2_BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    trainable_params = list(model.parameters()) + list(adapter.parameters())
    opt = torch.optim.AdamW([p for p in trainable_params if p.requires_grad], lr=STAGE2_LR)
    total_steps = (len(loader) // STAGE2_GRAD_ACCUM) * STAGE2_EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(10, total_steps // 20),
                                             num_training_steps=total_steps)

    embed_layer = model.get_input_embeddings()
    step = 0
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        for i, (input_ids, attn, labels, graphs, field_embs) in enumerate(loader):
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            graphs = graphs.to(device)

            with torch.no_grad():
                graph_emb = stage1.graph_encoder(graphs.x, graphs.edge_index, graphs.batch)
            prefix_embeds = adapter(graph_emb.to(torch.bfloat16))          # (B, n_tokens, H)
            token_embeds = embed_layer(input_ids)                          # (B, T, H)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            prefix_attn = torch.ones(attn.shape[0], prefix_embeds.shape[1], device=device, dtype=attn.dtype)
            attn_full = torch.cat([prefix_attn, attn], dim=1)
            prefix_labels = torch.full((labels.shape[0], prefix_embeds.shape[1]), -100, device=device, dtype=labels.dtype)
            labels_full = torch.cat([prefix_labels, labels], dim=1)

            out = model(inputs_embeds=inputs_embeds, attention_mask=attn_full, labels=labels_full)
            loss = out.loss / STAGE2_GRAD_ACCUM
            loss.backward()

            if (i + 1) % STAGE2_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step} loss {out.loss.item():.4f}")

    model.save_pretrained(STAGE2_ADAPTER_DIR)
    torch.save(adapter.state_dict(), STAGE2_ADAPTER_DIR + "/graph_adapter.pt")
    tokenizer.save_pretrained(STAGE2_ADAPTER_DIR)
    print(f"Stage 2 SFT complete. Adapter saved to {STAGE2_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
