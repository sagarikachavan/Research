"""
train_adapter.py
=================
Trains a StructureGNN + GraphPrefixAdapter (both fresh, random-init) so
that a base LLM's soft-prompt tokens carry graph STRUCTURE, and — via the
`graph_consistency` task — so the LLM learns to notice when the tokens it
was given don't match the claim it's being asked about (the "wrong graph"
behavior you described).

Nothing here loads any main-pipeline checkpoint. The LLM is a fresh base
model (optionally with a freshly-initialized LoRA adapter — never the
main pipeline's stage2/stage3 LoRA). The only thing borrowed from your
existing setup is the raw graph JSON data itself.

Usage:
    python build_probe_tasks.py                 # once, first
    python train_adapter.py --steps 1500 --use_lora
    python eval_right_vs_wrong_graph.py --checkpoint standalone_checkpoints/best
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch

from standalone_config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, TASKS_DIR, CKPT_DIR, RANDOM_SEED,
    LLM_MODEL_NAME, GNN_OUT_DIM, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    TRAIN_LR, TRAIN_GRAD_ACCUM, TRAIN_STEPS, TRAIN_EVAL_EVERY,
    TRAIN_WARMUP_STEPS, TRAIN_GRAD_CLIP, MAX_NEW_TOKENS,
)
from graph_json import load_records, index_records, to_pyg_data, graph_signature
from structure_gnn import StructureGNN
from prefix_adapter import GraphPrefixAdapter
from probe_prompts import build_question, target_json, parse_answer, score

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_llm(model_name: str, device: str, dtype, use_lora: bool):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    if use_lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, cfg)
        llm.print_trainable_parameters()
    else:
        for p in llm.parameters():
            p.requires_grad_(False)
    return tok, llm



def sample_decoy(rng, own_key, own_sig, records_by_key, sig_by_key, keys):
    """Legacy generic decoy sampler for non-paired uses."""
    candidates = [
        k for k in keys
        if k != own_key and sig_by_key[k] != own_sig
    ]
    if candidates:
        return records_by_key[rng.choice(candidates)]
    candidates = [k for k in keys if k != own_key]
    return records_by_key[rng.choice(candidates)] if candidates else records_by_key[own_key]


def prepare_example(item, records_by_key, sig_by_key, keys, rng, real_frac: float):
    """
    For graph_consistency, use the EXPLICIT counterfactual graph selected by
    build_probe_tasks.py. This keeps training aligned with the actual
    cross-machine evaluation:

        anchor graph + same question -> TRUE
        explicit decoy graph + same question -> FALSE

    For non-consistency tasks, always use the item's own graph.
    """
    key = (item["machine"], item["row_id"])
    own_rec = records_by_key[key]

    if item["task"] != "graph_consistency" or rng.random() < real_frac:
        return own_rec["graph"], item, True

    decoy_key_obj = item.get("decoy_graph")
    if decoy_key_obj:
        decoy_key = (decoy_key_obj["machine"], decoy_key_obj["row_id"])
        if decoy_key in records_by_key:
            flipped = dict(item)
            flipped["gold"] = bool(item.get("counterfactual_gold", False))
            return records_by_key[decoy_key]["graph"], flipped, False

    # This should never happen for newly built tasks. Keep a safe fallback
    # for old task files rather than crashing.
    own_sig = sig_by_key[key]
    decoy_rec = sample_decoy(rng, key, own_sig, records_by_key, sig_by_key, keys)
    flipped = dict(item)
    flipped["gold"] = False
    return decoy_rec["graph"], flipped, False

def prepare_consistency_pair(item, records_by_key):
    """Return the explicitly verified (right_graph, TRUE) and
    (wrong_graph, FALSE) versions of the exact same claim.

    Training both sides together is the key intervention: the model sees
    identical question text with different graph-prefix tokens and must
    produce different answers.
    """
    key = (item["machine"], item["row_id"])
    own = records_by_key[key]["graph"]
    decoy_obj = item.get("decoy_graph")
    if not decoy_obj:
        raise ValueError("graph_consistency item has no explicit decoy_graph")
    decoy_key = (decoy_obj["machine"], decoy_obj["row_id"])
    if decoy_key not in records_by_key:
        raise KeyError(f"Explicit decoy not found: {decoy_key}")
    right_item = dict(item)
    right_item["gold"] = bool(item["gold"])
    wrong_item = dict(item)
    wrong_item["gold"] = bool(item.get("counterfactual_gold", False))
    return own, right_item, records_by_key[decoy_key]["graph"], wrong_item


def _load_graph_batch(graph_dict, device, dtype):
    """to_pyg_data() always returns float32 tensors (independent of whatever
    dtype the LLM/GNN happen to be running in). Cast x/edge_attr to the
    model's dtype here so the GNN's Linear layers (which were moved to
    `dtype` via `.to(dtype)` on the module) don't hit a float32-vs-bfloat16
    mismatch. edge_index stays int64 — that's an index tensor, not a
    floating-point one, and must NOT be cast."""
    data = to_pyg_data(graph_dict)
    n_nodes = data.x.shape[0]
    batch = PyGBatch.from_data_list([data]).to(device)
    batch.x = batch.x.to(dtype)
    if getattr(batch, "edge_attr", None) is not None:
        batch.edge_attr = batch.edge_attr.to(dtype)
    return batch, n_nodes


def forward_loss(gnn, adapter, tok, llm, embed_layer, device, dtype,
                  graph_dict, item, n_nodes_hint=None):
    batch, n_nodes = _load_graph_batch(graph_dict, device, dtype)

    graph_emb = gnn(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None))
    prefix_embeds = adapter(graph_emb).to(dtype)  # (1, K, H)

    question = build_question(item, n_nodes)
    target = target_json(item["gold"]) + tok.eos_token

    q_ids = tok(question, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=600).input_ids.to(device)
    t_ids = tok(target, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=100).input_ids.to(device)

    q_embeds = embed_layer(q_ids).to(dtype)
    t_embeds = embed_layer(t_ids).to(dtype)
    inputs_embeds = torch.cat([prefix_embeds, q_embeds, t_embeds], dim=1)

    n_prefix, n_q, n_t = prefix_embeds.shape[1], q_ids.shape[1], t_ids.shape[1]
    labels = torch.cat([
        torch.full((1, n_prefix + n_q), -100, dtype=torch.long, device=device),
        t_ids,
    ], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    out = llm(inputs_embeds=inputs_embeds, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                            shift_labels.reshape(-1), ignore_index=-100)
    return loss


@torch.no_grad()
def generate_answer(gnn, adapter, tok, llm, embed_layer, device, dtype, graph_dict, item):
    batch, n_nodes = _load_graph_batch(graph_dict, device, dtype)
    graph_emb = gnn(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None))
    prefix_embeds = adapter(graph_emb).to(dtype)

    question = build_question(item, n_nodes)
    q_ids = tok(question, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=600).input_ids.to(device)
    q_embeds = embed_layer(q_ids).to(dtype)
    inputs_embeds = torch.cat([prefix_embeds, q_embeds], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    out = llm.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return parse_answer(text)


def evaluate(gnn, adapter, tok, llm, embed_layer, device, dtype,
             held_items, records_by_key, max_items: int = 80):
    gnn.eval(); adapter.eval(); llm.eval()
    subset = held_items[:max_items]
    scores = []
    for item in subset:
        key = (item["machine"], item["row_id"])
        graph_dict = records_by_key[key]["graph"]
        pred = generate_answer(gnn, adapter, tok, llm, embed_layer, device, dtype, graph_dict, item)
        scores.append(score(item, pred))
    gnn.train(); adapter.train()
    return float(np.mean(scores)) if scores else 0.0


def save_checkpoint(out_dir, gnn, adapter, llm, use_lora: bool, model_name: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(gnn.state_dict(), os.path.join(out_dir, "structure_gnn.pt"))
    torch.save(adapter.state_dict(), os.path.join(out_dir, "graph_adapter.pt"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"llm_model_name": model_name, "use_lora": use_lora}, f, indent=2)
    if use_lora:
        llm.save_pretrained(os.path.join(out_dir, "lora"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=LLM_MODEL_NAME)
    ap.add_argument("--out_dir", default=os.path.join(CKPT_DIR, "run1"))
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--grad_accum", type=int, default=TRAIN_GRAD_ACCUM)
    ap.add_argument("--lr", type=float, default=TRAIN_LR)
    ap.add_argument("--eval_every", type=int, default=TRAIN_EVAL_EVERY)
    ap.add_argument("--use_lora", action="store_true",
                     help="Also train a fresh LoRA on the base LLM (still never the "
                          "main pipeline's LoRA). Default: LLM fully frozen, so any "
                          "capability must come from the GNN + adapter alone.")
    ap.add_argument("--real_frac", type=float, default=0.5,
                     help="Legacy option. Ignored when paired consistency training is enabled.")
    ap.add_argument("--consistency_frac", type=float, default=0.5,
                     help="Fraction of training steps devoted to explicit paired right/wrong "
                          "graph consistency supervision. Each such step computes BOTH losses.")
    ap.add_argument("--paired_consistency", action="store_true", default=True,
                     help="Train graph_consistency as an explicit right(TRUE)+wrong(FALSE) pair. "
                          "Enabled by default.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    train_items = load_jsonl(os.path.join(TASKS_DIR, "train.jsonl"))
    held_items = load_jsonl(os.path.join(TASKS_DIR, "held_out.jsonl"))
    if not train_items:
        raise SystemExit("No train items found — run build_probe_tasks.py first.")

    records = index_records(load_records(INPUT_TRAIN_JSON) +
                             (load_records(INPUT_TEST_JSON) if os.path.exists(INPUT_TEST_JSON) else []))
    keys = list(records.keys())
    sig_by_key = {k: graph_signature(v["graph"]) for k, v in records.items()}

    tok, llm = build_llm(args.model_name, device, dtype, args.use_lora)
    embed_layer = llm.get_input_embeddings()
    llm_hidden = llm.config.hidden_size

    gnn = StructureGNN(out_dim=GNN_OUT_DIM).to(device).to(dtype)
    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)

    trainable = list(gnn.parameters()) + list(adapter.parameters())
    if args.use_lora:
        trainable += [p for p in llm.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    def lr_lambda(step):
        if step < TRAIN_WARMUP_STEPS:
            return step / max(1, TRAIN_WARMUP_STEPS)
        prog = (step - TRAIN_WARMUP_STEPS) / max(1, args.steps - TRAIN_WARMUP_STEPS)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    rng = random.Random(RANDOM_SEED)
    best_score = -1.0
    running_loss = 0.0
    optim.zero_grad()

    consistency_items = [x for x in train_items if x["task"] == "graph_consistency"]
    structural_items = [x for x in train_items if x["task"] != "graph_consistency"]
    if not consistency_items:
        raise SystemExit("No graph_consistency items found — rebuild tasks first.")

    for step in range(1, args.steps + 1):
        use_pair = args.paired_consistency and rng.random() < args.consistency_frac
        if use_pair:
            item_src = rng.choice(consistency_items)
            right_graph, right_item, wrong_graph, wrong_item = prepare_consistency_pair(item_src, records)
            loss_right = forward_loss(gnn, adapter, tok, llm, embed_layer, device, dtype, right_graph, right_item)
            loss_wrong = forward_loss(gnn, adapter, tok, llm, embed_layer, device, dtype, wrong_graph, wrong_item)
            loss = 0.5 * (loss_right + loss_wrong)
        else:
            item_src = rng.choice(structural_items)
            key = (item_src["machine"], item_src["row_id"])
            loss = forward_loss(gnn, adapter, tok, llm, embed_layer, device, dtype, records[key]["graph"], item_src)

        (loss / args.grad_accum).backward()
        running_loss += loss.item()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, TRAIN_GRAD_CLIP)
            optim.step()
            sched.step()
            optim.zero_grad()

        if step % 50 == 0:
            print(f"[step {step}/{args.steps}] loss={running_loss / 50:.4f} lr={sched.get_last_lr()[0]:.2e}")
            running_loss = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            held_score = evaluate(gnn, adapter, tok, llm, embed_layer, device, dtype, held_items, records)
            print(f"[step {step}] held-out structure score = {held_score:.4f}")
            if held_score > best_score:
                best_score = held_score
                save_checkpoint(os.path.join(args.out_dir, "best"), gnn, adapter, llm,
                                 args.use_lora, args.model_name)
                print(f"  -> new best ({best_score:.4f}), saved to {args.out_dir}/best")

    save_checkpoint(os.path.join(args.out_dir, "last"), gnn, adapter, llm, args.use_lora, args.model_name)
    if best_score < 0:
        print("WARNING: held-out score never improved over its initial value — "
              "only 'last' was saved, no 'best' checkpoint exists. This itself is "
              "informative (see README_STANDALONE.md, 'Interpreting a null result').")
    print(f"Done. Best held-out structure score: {best_score:.4f}")


if __name__ == "__main__":
    main()
