"""
train_structure_adapter.py
============================

Trains the GraphPrefixAdapter + LoRA on the graph-structure probe tasks
(adjacency / node_type / edge_type / two_hop / graph_aggregate) built by
build_structure_tasks.py.

This is the "structure-only" arm referenced throughout RESEARCH_PLAN.md:
starts from the Stage-2 checkpoint (so it's a fair comparison against the
already-trained step-prediction model, not a fresh random init) and
continues training ONLY on structure tasks. Compare its behavior against:
  - `training/stage2_sft_qwen.py`'s checkpoint (step-prediction only)
  - `train_multitask_adapter.py`'s checkpoint (both objectives jointly)

Unlike the original report's train_graph_structure.py, this script NEVER
uses dummy/random graph embeddings -- every training example computes its
embedding through the real, frozen Stage-1 GNN
(`stage1.encode_and_predict`), the same call path Stage 2/3 use. There is
no dummy-embedding code path in this file to accidentally regress into.

Checkpoint selection: every EVAL_EVERY steps, scores the current adapter on
structure_tasks/held_out.jsonl (graphs from machines NEVER seen during this
training run) and only promotes the best-scoring checkpoint -- the same
fix applied to training/stage3_grpo_rl.py, for the same reason (the last
step is not reliably the best one).

Usage:
    python graph_adapter_experiments/train_structure_adapter.py \
        --init_from checkpoints/stage2_qwen_lora \
        --out_dir   checkpoints/graph_structure \
        --steps 1500
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CKPT_DIR, TASKS_DIR, RANDOM_SEED, GRAPH_PREFIX_SRC_DIM,
    load_all_examples, index_examples_by_key, load_task_items,
    format_task_prompt, score_task_item, parse_json_answer,
)

import torch
import torch.nn as nn
import numpy as np


def build_target_text(item: dict) -> str:
    return json.dumps({"answer": item["gold"]})


def forward_item(item, ex_index, stage1, adapter, model, tokenizer, embed_layer, device, dtype):
    """Teacher-forced single-item forward pass. Returns loss (or None if the
    referenced example/graph can't be found -- skipped rather than crashing
    a whole training run over one bad row)."""
    from torch_geometric.data import Batch as PyGBatch
    from data_utils import _embed_texts, CONTEXT_COLUMNS

    key = (item["machine"], item["row_id"], item["split"])
    ex = ex_index.get(key)
    if ex is None:
        return None

    prompt_text = (
        f"<|system|>\nYou are a graph-structure reasoning assistant.\n"
        f"<|user|>\n{format_task_prompt(item)}\n"
        f"<|assistant|>\n"
    )
    target_text = build_target_text(item)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = torch.tensor([(prompt_ids + target_ids)[:900]], device=device)
    labels = torch.tensor([([-100] * len(prompt_ids) + target_ids)[:900]], device=device)
    attn = torch.ones_like(input_ids)

    pyg_batch = PyGBatch.from_data_list([ex["graph"]]).to(device)
    edge_attr = getattr(pyg_batch, "edge_attr", None)
    field_embs = torch.tensor(
        _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        combined_emb, _, _ = stage1.encode_and_predict(
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch, field_embs, edge_attr=edge_attr
        )

    prefix_embeds = adapter(combined_emb.to(dtype))
    token_embeds = embed_layer(input_ids).to(dtype)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    n_prefix = prefix_embeds.shape[1]
    prefix_attn = torch.ones(1, n_prefix, device=device, dtype=attn.dtype)
    attn_full = torch.cat([prefix_attn, attn], dim=1)
    prefix_lbls = torch.full((1, n_prefix), -100, device=device, dtype=labels.dtype)
    labels_full = torch.cat([prefix_lbls, labels], dim=1)

    out = model(inputs_embeds=inputs_embeds, attention_mask=attn_full, labels=labels_full)
    return out.loss


@torch.no_grad()
def evaluate_on_items(items, ex_index, stage1, adapter, model, tokenizer, embed_layer,
                       device, dtype, max_items=60, seed=RANDOM_SEED):
    from torch_geometric.data import Batch as PyGBatch
    from data_utils import _embed_texts, CONTEXT_COLUMNS

    rng = random.Random(seed)
    sample = items if len(items) <= max_items else rng.sample(items, max_items)
    scores, parsed_ok_count = [], 0

    was_training = model.training
    model.eval()
    for item in sample:
        key = (item["machine"], item["row_id"], item["split"])
        ex = ex_index.get(key)
        if ex is None:
            continue
        prompt_text = (
            f"<|system|>\nYou are a graph-structure reasoning assistant.\n"
            f"<|user|>\n{format_task_prompt(item)}\n"
            f"<|assistant|>\n"
        )
        ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        pyg_batch = PyGBatch.from_data_list([ex["graph"]]).to(device)
        edge_attr = getattr(pyg_batch, "edge_attr", None)
        field_embs = torch.tensor(
            _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        combined_emb, _, _ = stage1.encode_and_predict(
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch, field_embs, edge_attr=edge_attr
        )
        prefix_embeds = adapter(combined_emb.to(dtype))
        token_embeds = embed_layer(ids).to(dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        out = model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn, max_new_tokens=150,
            do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        obj = parse_json_answer(text)
        pred = obj.get("answer") if isinstance(obj, dict) else None
        score, ok = score_task_item(item, pred)
        scores.append(score)
        parsed_ok_count += int(ok)
    if was_training:
        model.train()
    return (float(np.mean(scores)) if scores else 0.0,
            parsed_ok_count / max(1, len(sample)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default=None, help="default: STAGE2_ADAPTER_DIR")
    ap.add_argument("--out_dir", default=os.path.join(CKPT_DIR, "graph_structure"))
    ap.add_argument("--tasks", default="adjacency,node_type,edge_type,two_hop,graph_aggregate")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import config
    from stage2_sft_qwen import GraphPrefixAdapter
    import common as C

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    init_from = args.init_from or config.STAGE2_ADAPTER_DIR
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    task_filter = set(args.tasks.split(","))
    train_items = [it for it in load_task_items(os.path.join(TASKS_DIR, "train.jsonl"))
                   if it["task"] in task_filter]
    held_out_items = [it for it in load_task_items(os.path.join(TASKS_DIR, "held_out.jsonl"))
                       if it["task"] in task_filter]
    print(f"[train_structure_adapter] {len(train_items)} train items / "
          f"{len(held_out_items)} held-out items across tasks {sorted(task_filter)}")
    if not train_items:
        raise SystemExit("No task items found -- run build_structure_tasks.py first.")

    examples = load_all_examples()
    ex_index = index_examples_by_key(examples)

    print(f"[train_structure_adapter] Loading base model from {init_from} ...")
    tokenizer = AutoTokenizer.from_pretrained(init_from, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config.QWEN_MODEL_NAME, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, init_from, is_trainable=True, local_files_only=True)
    model.gradient_checkpointing_enable()
    model.train()

    stage1 = C.load_stage1(device)
    embed_layer = model.get_input_embeddings()

    llm_hidden = model.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).to(dtype)
    init_adapter_ckpt = os.path.join(init_from, "graph_adapter.pt")
    if os.path.exists(init_adapter_ckpt):
        adapter.load_state_dict(torch.load(init_adapter_ckpt, map_location=device))
        print(f"[train_structure_adapter] Warm-started adapter from {init_adapter_ckpt}")
    adapter.train()

    params = [p for p in model.parameters() if p.requires_grad] + list(adapter.parameters())
    n_trainable = sum(p.numel() for p in params)
    print(f"[train_structure_adapter] Trainable params: {n_trainable:,}")
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.out_dir, exist_ok=True)
    best_dir = os.path.join(args.out_dir, "best")

    print("[train_structure_adapter] Scoring the Stage-2 starting point on held-out "
          "structure tasks (this is the bar structure training has to clear)...")
    baseline_score, baseline_parse_rate = evaluate_on_items(
        held_out_items, ex_index, stage1, adapter, model, tokenizer, embed_layer, device, dtype
    )
    print(f"[train_structure_adapter] Baseline held-out score: {baseline_score:.4f} "
          f"(parse rate {baseline_parse_rate:.2%})")
    best_score, best_step = baseline_score, 0

    rng = random.Random(args.seed)
    step, accum_loss = 0, 0.0
    optim.zero_grad()
    while step < args.steps:
        item = rng.choice(train_items)
        loss = forward_item(item, ex_index, stage1, adapter, model, tokenizer, embed_layer, device, dtype)
        if loss is None:
            continue
        (loss / args.grad_accum).backward()
        accum_loss += loss.item()
        step += 1

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            optim.zero_grad()

        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps} | loss (last 50) ≈ {accum_loss / 50:.4f}")
            accum_loss = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            score, parse_rate = evaluate_on_items(
                held_out_items, ex_index, stage1, adapter, model, tokenizer, embed_layer, device, dtype
            )
            flag = ""
            if score > best_score:
                best_score, best_step = score, step
                model.save_pretrained(best_dir)
                torch.save(adapter.state_dict(), os.path.join(best_dir, "graph_adapter.pt"))
                tokenizer.save_pretrained(best_dir)
                flag = "  <-- new best, saved"
            print(f"[train_structure_adapter] step {step:5d} | held-out score {score:.4f} "
                  f"(parse rate {parse_rate:.2%}, baseline {baseline_score:.4f}, "
                  f"best {best_score:.4f} @ step {best_step}){flag}")

    print("\n" + "=" * 70)
    print(f"[train_structure_adapter] DONE. Baseline (Stage 2, no structure training): "
          f"{baseline_score:.4f}")
    print(f"[train_structure_adapter] Best structure-trained checkpoint: "
          f"{best_score:.4f} @ step {best_step} -> {best_dir}")
    if best_step == 0:
        print("[train_structure_adapter] ⚠ Structure training never beat the Stage-2 "
              "starting point on held-out graphs -- the 'best' checkpoint above is "
              "just the untrained starting point. Try more steps / a higher lr / "
              "fewer simultaneous tasks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
