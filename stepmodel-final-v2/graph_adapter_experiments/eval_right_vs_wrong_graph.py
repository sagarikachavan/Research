"""
eval_right_vs_wrong_graph.py
=============================
The actual test you described: same query, right graph vs same query,
wrong graph.

For every held-out item:
  - "real":        item's own graph  -> score against its own gold answer.
  - "wrong_graph": a structurally different decoy graph, SAME question text
                    -> for graph_consistency items, the correct answer is
                    now "false" (the claim was anchored to a different
                    graph); for the other structural tasks, we report
                    whether the answer CHANGES from the "real" answer
                    (it should — a model reading the graph tokens has no
                    reason to give the same adjacency/type/etc. answer for
                    a different graph) rather than scoring against the
                    original gold (which is no longer the right target).

Usage:
    python eval_right_vs_wrong_graph.py --checkpoint standalone_checkpoints/run1/best
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from standalone_config import INPUT_TRAIN_JSON, INPUT_TEST_JSON, TASKS_DIR, GNN_OUT_DIM, RANDOM_SEED
from graph_json import load_records, index_records, graph_signature
from structure_gnn import StructureGNN
from prefix_adapter import GraphPrefixAdapter
from probe_prompts import score
from train_adapter import build_llm, generate_answer, load_jsonl, sample_decoy


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", default=None, help="override the LLM recorded in meta.json")
    ap.add_argument("--split", default="held_out", choices=["train", "held_out"])
    ap.add_argument("--max_items", type=int, default=150)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok, llm, gnn, adapter, embed_layer = load_checkpoint(args.checkpoint, device, dtype, args.model_name)

    items = load_jsonl(os.path.join(TASKS_DIR, f"{args.split}.jsonl"))[: args.max_items]
    records = index_records(load_records(INPUT_TRAIN_JSON) +
                             (load_records(INPUT_TEST_JSON) if os.path.exists(INPUT_TEST_JSON) else []))
    keys = list(records.keys())
    sig_by_key = {k: graph_signature(v["graph"]) for k, v in records.items()}
    rng = random.Random(RANDOM_SEED)

    per_task_real = defaultdict(list)
    per_task_wrong_changed = defaultdict(list)   # non-consistency tasks: did the answer change?
    consistency_true_pos, consistency_true_neg = [], []

    for item in items:
        key = (item["machine"], item["row_id"])
        real_graph = records[key]["graph"]
        real_pred = generate_answer(gnn, adapter, tok, llm, embed_layer, device, dtype, real_graph, item)
        real_score = score(item, real_pred)
        per_task_real[item["task"]].append(real_score)

        decoy_rec = sample_decoy(rng, key, sig_by_key[key], records, sig_by_key, keys)
        wrong_pred = generate_answer(gnn, adapter, tok, llm, embed_layer, device, dtype,
                                      decoy_rec["graph"], item)

        if item["task"] == "graph_consistency":
            if item["gold"] is True:
                consistency_true_pos.append(1.0 if str(real_pred).strip().lower() in ("true", "yes", "1") else 0.0)
            wrong_is_false = str(wrong_pred).strip().lower() in ("false", "no", "0")
            consistency_true_neg.append(1.0 if wrong_is_false else 0.0)
        else:
            per_task_wrong_changed[item["task"]].append(0.0 if wrong_pred == real_pred else 1.0)

    print(f"\n=== Reliability report ({args.split}, n={len(items)}) ===\n")
    print("Structural QA accuracy on the REAL graph (higher = better structure reading):")
    for task, scores in per_task_real.items():
        if task == "graph_consistency":
            continue
        print(f"  {task:16s} n={len(scores):3d}  score={np.mean(scores):.3f}")

    print("\nFraction of answers that CHANGE when the graph is swapped for a decoy,")
    print("same question text (higher = better — a model truly reading the graph")
    print("tokens has no reason to repeat the same answer for a different graph):")
    for task, changed in per_task_wrong_changed.items():
        print(f"  {task:16s} n={len(changed):3d}  changed={np.mean(changed):.3f}")

    if consistency_true_pos or consistency_true_neg:
        print("\ngraph_consistency task (this is the direct 'right graph vs wrong graph' test):")
        if consistency_true_pos:
            print(f"  same query + RIGHT graph  -> answers 'true' correctly: "
                  f"{np.mean(consistency_true_pos):.3f}  (n={len(consistency_true_pos)})")
        if consistency_true_neg:
            print(f"  same query + WRONG graph  -> answers 'false' correctly: "
                  f"{np.mean(consistency_true_neg):.3f}  (n={len(consistency_true_neg)})")
        print("  (Both numbers well above chance and comparable to each other is the")
        print("   signature of the model actually reading the graph tokens rather than")
        print("   pattern-matching the claim text or defaulting to one answer.)")


if __name__ == "__main__":
    main()
