"""
eval_right_vs_wrong_graph.py
=============================
Reliability evaluation for the standalone Graph Prefix Adapter.

This version fixes the original graph_consistency reporting bug.  The old
script only counted TRUE claims on the right graph, and counted every wrong-
graph example as a negative.  That allowed a degenerate "always false"
model to report 0.000 / 1.000 and look superficially good.

For every held-out item, we evaluate:
  * REAL graph: original graph + original gold label.
  * WRONG graph: a structurally different decoy graph + expected label FALSE
    for graph_consistency items.

For graph_consistency we report both conditional accuracies and an overall
paired accuracy, plus the confusion matrix.  We also prefer decoys with the
same number of nodes so the prompt's node-id legend does not itself reveal
that the graph was swapped.

Usage:
    python eval_right_vs_wrong_graph_fixed.py \
        --checkpoint standalone_checkpoints/run1/best

The file can replace eval_right_vs_wrong_graph.py directly if desired.
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from standalone_config import INPUT_TRAIN_JSON, INPUT_TEST_JSON, TASKS_DIR, GNN_OUT_DIM, RANDOM_SEED
from graph_json import load_records, index_records, graph_signature, parse_graph_dict
from structure_gnn import StructureGNN
from prefix_adapter import GraphPrefixAdapter
from probe_prompts import score
from train_adapter import build_llm, generate_answer, load_jsonl, sample_decoy

TRUE_STRINGS = {"true", "yes", "1"}
FALSE_STRINGS = {"false", "no", "0"}


def as_bool(pred):
    """Return True/False for boolean-like model outputs, else None."""
    if isinstance(pred, bool):
        return pred
    s = str(pred).strip().lower()
    if s in TRUE_STRINGS:
        return True
    if s in FALSE_STRINGS:
        return False
    return None


def load_checkpoint(ckpt_dir, device, dtype, override_model_name=None):
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)
    model_name = override_model_name or meta["llm_model_name"]
    tok, llm = build_llm(model_name, device, dtype, use_lora=False)
    if meta.get("use_lora"):
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, os.path.join(ckpt_dir, "lora")).eval()
    llm_hidden = llm.config.hidden_size

    gnn = StructureGNN(out_dim=GNN_OUT_DIM).to(device).to(dtype)
    gnn.load_state_dict(torch.load(os.path.join(ckpt_dir, "structure_gnn.pt"), map_location=device))
    gnn.eval()

    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)
    adapter.load_state_dict(torch.load(os.path.join(ckpt_dir, "graph_adapter.pt"), map_location=device))
    adapter.eval()

    return tok, llm, gnn, adapter, llm.get_input_embeddings()


def node_count(graph):
    return len(parse_graph_dict(graph)["node_ids"])


def sample_decoy_same_size(rng, own_key, own_sig, own_graph, records_by_key, sig_by_key, keys):
    """Prefer a structurally different decoy with the same node count.

    Keeping the node count equal means build_question() produces the same
    anonymized node-id legend, so the consistency comparison is not confounded
    by a different number of visible node ids.
    """
    own_n = node_count(own_graph)
    candidates = [
        k for k in keys
        if k != own_key
        and sig_by_key[k] != own_sig
        and node_count(records_by_key[k]["graph"]) == own_n
    ]
    if candidates:
        return records_by_key[rng.choice(candidates)], True

    # If the dataset has no same-size structural alternative, retain the
    # original behavior but explicitly report that this was a fallback.
    candidates = [k for k in keys if k != own_key and sig_by_key[k] != own_sig]
    if candidates:
        return records_by_key[rng.choice(candidates)], False

    candidates = [k for k in keys if k != own_key]
    if candidates:
        return records_by_key[rng.choice(candidates)], False

    return records_by_key[own_key], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", default=None, help="override the LLM recorded in meta.json")
    ap.add_argument("--split", default="held_out", choices=["train", "held_out"])
    ap.add_argument("--max_items", type=int, default=150)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="seed for deterministic decoy selection")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok, llm, gnn, adapter, embed_layer = load_checkpoint(
        args.checkpoint, device, dtype, args.model_name
    )

    items = load_jsonl(os.path.join(TASKS_DIR, f"{args.split}.jsonl"))[: args.max_items]
    records = index_records(
        load_records(INPUT_TRAIN_JSON)
        + (load_records(INPUT_TEST_JSON) if os.path.exists(INPUT_TEST_JSON) else [])
    )
    keys = list(records.keys())
    sig_by_key = {k: graph_signature(v["graph"]) for k, v in records.items()}
    rng = random.Random(args.seed)

    per_task_real = defaultdict(list)
    per_task_wrong_changed = defaultdict(list)

    # Each entry is a dict containing the original label and both predictions.
    consistency_rows = []
    same_size_decoys = 0
    fallback_decoys = 0

    for item in items:
        key = (item["machine"], item["row_id"])
        real_graph = records[key]["graph"]
        real_pred = generate_answer(
            gnn, adapter, tok, llm, embed_layer, device, dtype, real_graph, item
        )
        real_score = score(item, real_pred)
        per_task_real[item["task"]].append(real_score)

        decoy_rec, same_size = sample_decoy_same_size(
            rng, key, sig_by_key[key], real_graph, records, sig_by_key, keys
        )
        same_size_decoys += int(same_size)
        fallback_decoys += int(not same_size)

        wrong_pred = generate_answer(
            gnn, adapter, tok, llm, embed_layer, device, dtype,
            decoy_rec["graph"], item
        )

        if item["task"] == "graph_consistency":
            real_bool = as_bool(real_pred)
            wrong_bool = as_bool(wrong_pred)
            original_gold = bool(item["gold"])

            # On the swapped graph the anchored claim is false by construction.
            wrong_gold = False
            consistency_rows.append({
                "gold_real": original_gold,
                "real_pred": real_bool,
                "wrong_pred": wrong_bool,
                "real_correct": real_bool is not None and real_bool == original_gold,
                "wrong_correct": wrong_bool is not None and wrong_bool == wrong_gold,
                "changed": real_bool is not None and wrong_bool is not None and real_bool != wrong_bool,
            })
        else:
            # For structural QA, changing the graph should generally change the
            # answer. We intentionally do NOT score the decoy against the old gold.
            per_task_wrong_changed[item["task"]].append(
                0.0 if wrong_pred == real_pred else 1.0
            )

    print(f"\n=== Reliability report ({args.split}, n={len(items)}) ===\n")
    print("Structural QA accuracy on the REAL graph (higher = better structure reading):")
    for task, scores in per_task_real.items():
        if task == "graph_consistency":
            continue
        print(f"  {task:16s} n={len(scores):3d}  score={np.mean(scores):.3f}")

    print("\nFraction of answers that CHANGE when the graph is swapped for a decoy,")
    print("same claim/query text (higher is generally better for graph-sensitive tasks):")
    for task, changed in per_task_wrong_changed.items():
        print(f"  {task:16s} n={len(changed):3d}  changed={np.mean(changed):.3f}")

    if consistency_rows:
        total = len(consistency_rows)
        real_acc = np.mean([r["real_correct"] for r in consistency_rows])
        wrong_acc = np.mean([r["wrong_correct"] for r in consistency_rows])
        paired_acc = np.mean([
            r["real_correct"] and r["wrong_correct"] for r in consistency_rows
        ])
        flip_rate = np.mean([r["changed"] for r in consistency_rows])

        # Report the two original claim classes separately. This makes class
        # imbalance visible and prevents a trivial majority-class answer from
        # looking like good graph grounding.
        true_rows = [r for r in consistency_rows if r["gold_real"] is True]
        false_rows = [r for r in consistency_rows if r["gold_real"] is False]

        def acc(rows, field):
            return float(np.mean([r[field] for r in rows])) if rows else float("nan")

        print("\ngraph_consistency task (paired right-graph vs wrong-graph test):")
        print(f"  right graph accuracy                 = {real_acc:.3f}  (n={total})")
        print(f"  wrong graph accuracy (expected false)= {wrong_acc:.3f}  (n={total})")
        print(f"  paired accuracy (both correct)       = {paired_acc:.3f}  (n={total})")
        print(f"  prediction changed right -> wrong   = {flip_rate:.3f}  (n={total})")

        if true_rows:
            print(f"  original TRUE claims: right={acc(true_rows, 'real_correct'):.3f}, "
                  f"wrong={acc(true_rows, 'wrong_correct'):.3f} (n={len(true_rows)})")
        if false_rows:
            print(f"  original FALSE claims: right={acc(false_rows, 'real_correct'):.3f}, "
                  f"wrong={acc(false_rows, 'wrong_correct'):.3f} (n={len(false_rows)})")

        # Explicit prediction counts expose degenerate always-true/always-false behavior.
        for name, field in (("RIGHT", "real_pred"), ("WRONG", "wrong_pred")):
            vals = [r[field] for r in consistency_rows]
            n_true = sum(v is True for v in vals)
            n_false = sum(v is False for v in vals)
            n_invalid = sum(v is None for v in vals)
            print(f"  {name} predictions: true={n_true}, false={n_false}, invalid={n_invalid}")

        if true_rows and false_rows:
            balanced_real = 0.5 * (
                acc(true_rows, "real_correct") + acc(false_rows, "real_correct")
            )
            print(f"  balanced accuracy on RIGHT graph   = {balanced_real:.3f}")

        print("\n  Interpretation:")
        print("  - Do not interpret RIGHT=0% / WRONG=100% as graph grounding: "
              "that can be produced by always answering FALSE.")
        print("  - Strong evidence requires high right-graph accuracy, high wrong-graph "
              "accuracy, and substantial paired correctness/appropriate prediction flips.")
        print(f"\n  Decoy quality: same-node-count={same_size_decoys}, "
              f"fallback-size={fallback_decoys}")


if __name__ == "__main__":
    main()
