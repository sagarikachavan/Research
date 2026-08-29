"""
run_reliability_suite.py
==========================

The core experiment. For every held-out structure-probe question, this
generates an answer under each of several PREFIX CONDITIONS -- swapping out
what the Graph Prefix Adapter actually receives -- while keeping the
question text and everything else identical. This is a causal intervention
(patching) design, not just "measure accuracy once":

  real            correct graph + correct context      (normal operation)
  wrong_graph     a DIFFERENT real graph + correct context
  wrong_context   correct graph + a DIFFERENT real context
  wrong_both      different graph AND different context
  shuffled_nodes  correct graph structure, node features permuted among nodes
  zero            fused embedding replaced with the zero vector (no info)
  noise           fused embedding replaced with matched-scale Gaussian noise
  mean_prototype  fused embedding replaced with the average over many graphs

If the model "understands the graph", `real` should clearly beat every
control on tasks that require reading edges (adjacency/edge_type/two_hop),
and `wrong_graph`/`shuffled_nodes` should be close to the `zero`/`noise`
floor (if the model were ignoring the graph and just pattern-matching on
question text, `wrong_graph` and `real` would score about the same instead).
`wrong_context` isolates whether any observed "graph understanding" is
actually coming from the fused-in context text rather than the graph GNN
output at all -- see common.py's module docstring for why this matters here
specifically (the fusion architecture).

Output: results/raw_results_<checkpoint_name>.jsonl -- one row per
(item, condition), with the parsed prediction and score. Feed this into
analyze_results.py for the statistical writeup.

Usage:
    python graph_adapter_experiments/run_reliability_suite.py \
        --checkpoint checkpoints/graph_structure/best \
        --split held_out --max_items 150
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    RESULTS_DIR, TASKS_DIR, RANDOM_SEED, CONDITIONS,
    load_all_examples, index_examples_by_key, load_task_items,
    load_model_for_eval, make_condition_embedding, format_task_prompt,
    generate_with_prefix, parse_json_answer, score_task_item,
)

import torch


def precompute_mean_prototype(model, ex_index, keys, n_samples=40, seed=RANDOM_SEED):
    from common import compute_fused_embedding
    rng = random.Random(seed)
    sample_keys = keys if len(keys) <= n_samples else rng.sample(keys, n_samples)
    embs = []
    with torch.no_grad():
        for k in sample_keys:
            ex = ex_index.get(k)
            if ex is None:
                continue
            emb, _, _ = compute_fused_embedding(model, ex["graph"], ex["context"])
            embs.append(emb)
    if not embs:
        raise RuntimeError("Could not compute any embeddings for mean_prototype baseline.")
    return torch.stack(embs, dim=0).mean(dim=0)


def pick_decoy(rng, items, current_key, ex_index):
    """A different example than `current_key`, resampled until it actually
    resolves to a loadable graph (guards against a handful of missing rows
    in ex_index without derailing the whole run)."""
    for _ in range(20):
        cand = rng.choice(items)
        k = (cand["machine"], cand["row_id"], cand["split"])
        if k != current_key and k in ex_index:
            return ex_index[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                     help="Directory with adapter files + graph_adapter.pt "
                          "(e.g. checkpoints/stage2_qwen_lora, checkpoints/graph_structure/best)")
    ap.add_argument("--split", choices=["train", "held_out"], default="held_out")
    ap.add_argument("--tasks", default="adjacency,node_type,edge_type,two_hop,graph_aggregate")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--max_items", type=int, default=150)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    task_filter = set(args.tasks.split(","))
    condition_list = args.conditions.split(",")
    for c in condition_list:
        assert c in CONDITIONS, f"Unknown condition {c!r}, choose from {CONDITIONS}"

    items = [it for it in load_task_items(os.path.join(TASKS_DIR, f"{args.split}.jsonl"))
             if it["task"] in task_filter]
    rng = random.Random(args.seed)
    if len(items) > args.max_items:
        items = rng.sample(items, args.max_items)
    print(f"[run_reliability_suite] {len(items)} items x {len(condition_list)} conditions "
          f"= {len(items) * len(condition_list)} generations")

    examples = load_all_examples()
    ex_index = index_examples_by_key(examples)

    print(f"[run_reliability_suite] Loading checkpoint: {args.checkpoint}")
    model = load_model_for_eval(args.checkpoint)

    prototype = None
    if "mean_prototype" in condition_list:
        print("[run_reliability_suite] Precomputing mean-graph prototype embedding...")
        all_keys = list(ex_index.keys())
        prototype = precompute_mean_prototype(model, ex_index, all_keys, seed=args.seed)

    out_path = os.path.join(RESULTS_DIR, f"raw_results_{model.name}.jsonl")
    n_written = 0
    with open(out_path, "w") as out_f:
        for i, item in enumerate(items):
            key = (item["machine"], item["row_id"], item["split"])
            ex = ex_index.get(key)
            if ex is None:
                continue

            decoy = pick_decoy(rng, items, key, ex_index)
            decoy_graph = decoy["graph"] if decoy else None
            decoy_context = decoy["context"] if decoy else None

            prompt_text = f"<|user|>\n{format_task_prompt(item)}\n<|assistant|>\n"

            for condition in condition_list:
                if condition in ("wrong_graph", "wrong_both") and decoy_graph is None:
                    continue
                if condition in ("wrong_context", "wrong_both") and decoy_context is None:
                    continue
                try:
                    emb = make_condition_embedding(
                        model, condition, ex["graph"], ex["context"],
                        decoy_graph_data=decoy_graph, decoy_context=decoy_context,
                        prototype_embedding=prototype, rng=rng,
                    )
                    prefix_embeds = model.graph_adapter(emb.to(model.dtype))
                    gen_text = generate_with_prefix(model, prefix_embeds, prompt_text)
                    obj = parse_json_answer(gen_text)
                    pred = obj.get("answer") if isinstance(obj, dict) else None
                    score, parsed_ok = score_task_item(item, pred)
                except Exception as e:
                    gen_text, pred, score, parsed_ok = f"<ERROR: {e}>", None, 0.0, False

                row = dict(
                    task=item["task"], machine=item["machine"], row_id=item["row_id"],
                    condition=condition, gold=item["gold"], pred=pred, score=score,
                    parsed_ok=parsed_ok, raw_generation=gen_text[:500],
                )
                out_f.write(json.dumps(row) + "\n")
                n_written += 1

            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1}/{len(items)} items done ({n_written} rows written)")

    print(f"[run_reliability_suite] Wrote {n_written} rows -> {out_path}")
    print("[run_reliability_suite] Next: python analyze_results.py "
          f"--results {out_path}")


if __name__ == "__main__":
    main()
