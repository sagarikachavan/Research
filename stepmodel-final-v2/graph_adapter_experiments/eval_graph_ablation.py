"""
eval_graph_ablation.py
======================
Ablation/diagnostic suite for the standalone Graph Prefix Adapter.

This script is designed to answer four questions:
  1. Does the LLM answer the questions without graph information?
  2. Does adding the correct graph change/improve the answer?
  3. Does swapping the graph to another machine change the answer?
  4. Does a random graph produce a different answer from the correct graph?

IMPORTANT:
- "no_graph" is implemented as a NULL graph-prefix baseline: the same number
  of prefix slots are supplied, but their values are forced to zero. This is
  preferable to changing the LLM input shape and is a clean control for the
  information carried by the learned graph prefix.
- For graph_consistency, the primary paired test uses the exact same question
  with the source graph (expected original gold) and a verified cross-machine
  decoy graph (expected FALSE).
- Random-graph results are a sensitivity diagnostic, NOT an accuracy metric,
  because a random graph may accidentally satisfy a particular query.

Run from graph_adapter_experiments/:
  python eval_graph_ablation.py \
      --checkpoint standalone_checkpoints/run1/best \
      --max_items 150 \
      --max_consistency_pairs 50 \
      --max_random_pairs 50 \
      --show_examples 10
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
from train_adapter import build_llm, generate_answer, load_jsonl

TRUE_STRINGS = {"true", "yes", "1"}
FALSE_STRINGS = {"false", "no", "0"}


class NullPrefixAdapter(torch.nn.Module):
    """Drop-in adapter returning a zero graph prefix with the correct shape."""
    def __init__(self, real_adapter):
        super().__init__()
        self.real_adapter = real_adapter

    def forward(self, graph_emb):
        # real_adapter(graph_emb) is only used to obtain the expected prefix shape.
        with torch.no_grad():
            shape = self.real_adapter(graph_emb).shape
        return torch.zeros(shape, device=graph_emb.device, dtype=graph_emb.dtype)


def as_bool(pred):
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
    gnn.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "structure_gnn.pt"), map_location=device)
    )
    gnn.eval()

    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)
    adapter.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "graph_adapter.pt"), map_location=device)
    )
    adapter.eval()

    return tok, llm, gnn, adapter, llm.get_input_embeddings()


def node_count(graph):
    return len(parse_graph_dict(graph)["node_ids"])


def sample_same_size_decoy(rng, own_key, own_sig, own_graph, records, sig_by_key, keys):
    own_n = node_count(own_graph)
    candidates = [
        k for k in keys
        if k != own_key
        and sig_by_key[k] != own_sig
        and node_count(records[k]["graph"]) == own_n
    ]
    if candidates:
        return records[rng.choice(candidates)], True

    candidates = [k for k in keys if k != own_key and sig_by_key[k] != own_sig]
    if candidates:
        return records[rng.choice(candidates)], False
    return records[own_key], False


def sample_random_graph(rng, own_key, records, keys):
    candidates = [k for k in keys if k != own_key]
    if not candidates:
        return records[own_key]
    return records[rng.choice(candidates)]


def evaluate_one(graph, item, tok, llm, gnn, adapter, embed_layer, device, dtype):
    return generate_answer(
        gnn, adapter, tok, llm, embed_layer, device, dtype, graph, item
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--split", default="held_out", choices=["train", "held_out"])
    ap.add_argument("--max_items", type=int, default=150)
    ap.add_argument("--max_consistency_pairs", type=int, default=50)
    ap.add_argument("--max_random_pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--show_examples", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok, llm, gnn, adapter, embed_layer = load_checkpoint(
        args.checkpoint, device, dtype, args.model_name
    )
    null_adapter = NullPrefixAdapter(adapter).to(device)
    null_adapter.eval()

    items = load_jsonl(os.path.join(TASKS_DIR, f"{args.split}.jsonl"))
    items = items[: args.max_items]

    records = index_records(
        load_records(INPUT_TRAIN_JSON)
        + (load_records(INPUT_TEST_JSON) if os.path.exists(INPUT_TEST_JSON) else [])
    )
    keys = list(records.keys())
    sig_by_key = {k: graph_signature(v["graph"]) for k, v in records.items()}
    rng = random.Random(args.seed)

    print("\n=== GRAPH ADAPTER ABLATION ===")
    print(f"split={args.split} items={len(items)}")
    print(f"checkpoint={args.checkpoint}")

    # ------------------------------------------------------------
    # 1) No-graph/null-prefix baseline vs correct graph
    # ------------------------------------------------------------
    structural = [x for x in items if x["task"] != "graph_consistency"]
    if structural:
        null_scores = []
        graph_scores = []
        changed = []
        by_task = defaultdict(lambda: [[], [], []])

        for item in structural:
            key = (item["machine"], item["row_id"])
            graph = records[key]["graph"]

            pred_null = evaluate_one(
                graph, item, tok, llm, gnn, null_adapter, embed_layer, device, dtype
            )
            pred_graph = evaluate_one(
                graph, item, tok, llm, gnn, adapter, embed_layer, device, dtype
            )

            s_null = score(item, pred_null)
            s_graph = score(item, pred_graph)
            null_scores.append(s_null)
            graph_scores.append(s_graph)
            changed.append(float(str(pred_null).strip() != str(pred_graph).strip()))

            by_task[item["task"]][0].append(s_null)
            by_task[item["task"]][1].append(s_graph)
            by_task[item["task"]][2].append(changed[-1])

        print("\n--- 1. NULL-PREFIX (NO-GRAPH) ABLATION ---")
        print("Null-prefix = zero graph-prefix embeddings; LLM architecture is unchanged.")
        print(f"overall null-prefix accuracy = {np.mean(null_scores):.3f} (n={len(null_scores)})")
        print(f"overall graph-prefix accuracy = {np.mean(graph_scores):.3f} (n={len(graph_scores)})")
        print(f"graph-prefix improvement      = {np.mean(graph_scores)-np.mean(null_scores):+.3f}")
        print(f"answer changed null -> graph   = {np.mean(changed):.3f}")
        for task, (a, b, c) in by_task.items():
            print(
                f"  {task:16s} n={len(a):3d} "
                f"null={np.mean(a):.3f} graph={np.mean(b):.3f} changed={np.mean(c):.3f}"
            )

    # ------------------------------------------------------------
    # 2) Correct vs cross-machine verified-wrong graph
    # ------------------------------------------------------------
    consistency_items = [x for x in items if x["task"] == "graph_consistency"]
    consistency_items = consistency_items[: args.max_consistency_pairs]
    consistency_rows = []
    examples = []

    for item in consistency_items:
        key = (item["machine"], item["row_id"])
        source_graph = records[key]["graph"]
        source_sig = sig_by_key[key]
        decoy_rec, same_size = sample_same_size_decoy(
            rng, key, source_sig, source_graph, records, sig_by_key, keys
        )

        p_source = evaluate_one(
            source_graph, item, tok, llm, gnn, adapter, embed_layer, device, dtype
        )
        p_decoy = evaluate_one(
            decoy_rec["graph"], item, tok, llm, gnn, adapter, embed_layer, device, dtype
        )

        b_source = as_bool(p_source)
        b_decoy = as_bool(p_decoy)
        gold_source = bool(item["gold"])
        gold_decoy = False

        row = {
            "source": key,
            "decoy": (decoy_rec["machine"], decoy_rec["row_id"]),
            "source_pred": b_source,
            "decoy_pred": b_decoy,
            "source_correct": b_source is not None and b_source == gold_source,
            "decoy_correct": b_decoy is not None and b_decoy == gold_decoy,
            "paired_correct": b_source is not None and b_decoy is not None and b_source == gold_source and b_decoy == gold_decoy,
            "changed": b_source is not None and b_decoy is not None and b_source != b_decoy,
            "correct_flip": b_source is True and b_decoy is False,
            "question": item.get("question", item.get("prompt", "")),
            "same_size": same_size,
        }
        consistency_rows.append(row)
        if len(examples) < args.show_examples:
            examples.append(row)

    if consistency_rows:
        print("\n--- 2. CROSS-MACHINE GRAPH ABLATION ---")
        n = len(consistency_rows)
        src_acc = np.mean([r["source_correct"] for r in consistency_rows])
        dec_acc = np.mean([r["decoy_correct"] for r in consistency_rows])
        pair_acc = np.mean([r["paired_correct"] for r in consistency_rows])
        flip = np.mean([r["changed"] for r in consistency_rows])
        good_flip = np.mean([r["correct_flip"] for r in consistency_rows])
        print(f"pairs={n}")
        print(f"source/right graph accuracy          = {src_acc:.3f}")
        print(f"cross-machine wrong graph accuracy   = {dec_acc:.3f}")
        print(f"paired accuracy (source + decoy)     = {pair_acc:.3f}")
        print(f"prediction changed source -> decoy   = {flip:.3f}")
        print(f"correct TRUE -> FALSE directional flip = {good_flip:.3f}")
        print("prediction distribution:")
        for label, field in (("SOURCE", "source_pred"), ("DECOY", "decoy_pred")):
            vals = [r[field] for r in consistency_rows]
            print(
                f"  {label:6s}: true={sum(v is True for v in vals)}, "
                f"false={sum(v is False for v in vals)}, invalid={sum(v is None for v in vals)}"
            )

    # ------------------------------------------------------------
    # 3) Random-graph sensitivity test
    # ------------------------------------------------------------
    random_items = structural[: args.max_random_pairs]
    random_rows = []
    for item in random_items:
        key = (item["machine"], item["row_id"])
        source_graph = records[key]["graph"]
        random_rec = sample_random_graph(rng, key, records, keys)

        p_source = evaluate_one(
            source_graph, item, tok, llm, gnn, adapter, embed_layer, device, dtype
        )
        p_random = evaluate_one(
            random_rec["graph"], item, tok, llm, gnn, adapter, embed_layer, device, dtype
        )
        random_rows.append({
            "source": key,
            "random": (random_rec["machine"], random_rec["row_id"]),
            "source_pred": str(p_source).strip(),
            "random_pred": str(p_random).strip(),
            "changed": str(p_source).strip() != str(p_random).strip(),
            "task": item["task"],
        })

    if random_rows:
        print("\n--- 3. RANDOM-GRAPH SENSITIVITY TEST ---")
        print("Diagnostic only: a random graph may accidentally satisfy the question.")
        print(f"pairs={len(random_rows)}")
        print(f"prediction changed source -> random = {np.mean([r['changed'] for r in random_rows]):.3f}")
        by_task = defaultdict(list)
        for r in random_rows:
            by_task[r["task"]].append(float(r["changed"]))
        for task, vals in by_task.items():
            print(f"  {task:16s} n={len(vals):3d} changed={np.mean(vals):.3f}")

    if examples:
        print("\n--- VERIFIED CROSS-MACHINE EXAMPLES ---")
        for i, r in enumerate(examples, 1):
            print(f"Pair {i}: source={r['source']} decoy={r['decoy']} same_size={r['same_size']}")
            print(f"  question: {r['question']}")
            print(f"  source={r['source_pred']} decoy={r['decoy_pred']} changed={r['changed']}")

    print("\n=== INTERPRETATION GUIDE ===")
    print("1. Null ≈ graph: the prefix may be ignored or not carrying useful information.")
    print("2. Graph > null but no source→decoy flip: graph helps some QA, but not robust counterfactual grounding.")
    print("3. High source + high decoy + high paired + high TRUE→FALSE flip: strong evidence of graph grounding.")
    print("4. Random-graph changes support graph sensitivity, but are not sufficient for grounding because random graphs can accidentally answer a query.")


if __name__ == "__main__":
    main()
