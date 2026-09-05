"""
build_probe_tasks.py
=====================
Builds the question/answer tasks used to test (and train) whether the
Graph Prefix Adapter's soft-prompt tokens actually carry graph STRUCTURE.

Two families of task, both derived purely from the raw graph JSON (never
from node titles, never from the "New strategy"/"Strategy explanation"
text — see graph_json.py for why):

  1. Structural QA (adjacency / node_type / edge_type / two_hop /
     graph_aggregate) — "can the LLM read a structural fact off the
     tokens for THIS graph". Node ids are anonymized to "N0", "N1", ...
     per graph (stable order = order in the JSON) so the answer cannot be
     shortcut-guessed from a revealing id string or a title.

  2. graph_consistency — "same claim text, does the LLM notice when the
     graph tokens underneath it are swapped for a different graph". Each
     item is a claim like "Node N2 is directly connected to node N5."
     with a gold true/false label for its OWN (anchor) graph. At train/
     eval time (see train_adapter.py / eval_right_vs_wrong_graph.py) the
     SAME claim is also served with a different, decoy graph's tokens
     swapped in — in which case the correct answer is "false" regardless
     of the anchor graph's original label, since the claim no longer
     describes the graph actually being shown. This is the direct
     operationalization of "same query + right graph -> answer; same
     query + wrong graph -> should recognize it's wrong".

Usage:
    python build_probe_tasks.py
Writes standalone_tasks/{train,held_out}.jsonl + split_manifest.json.
"""
import argparse
import json
import random

from standalone_config import INPUT_TRAIN_JSON, INPUT_TEST_JSON, TASKS_DIR, RANDOM_SEED, VAL_FRAC
from graph_json import load_records, machine_level_split, parse_graph_dict

TASKS = ["adjacency", "node_type", "edge_type", "two_hop", "graph_aggregate", "graph_consistency"]


def _anon_ids(parsed):
    return [f"N{i}" for i in range(len(parsed["node_ids"]))]


def _neighbors(parsed):
    """idx -> set of directly-connected idx (undirected, dedups multi-edges)."""
    nbrs = [set() for _ in parsed["node_ids"]]
    for s, t in parsed["edge_index"]:
        nbrs[s].add(t)
        nbrs[t].add(s)
    return nbrs


def _two_hop(parsed, nbrs):
    """idx -> set of idx reachable in EXACTLY 2 hops (excludes self and 1-hop)."""
    out = []
    for i in range(len(parsed["node_ids"])):
        one_hop = nbrs[i] | {i}
        two = set()
        for j in nbrs[i]:
            two |= nbrs[j]
        two -= one_hop
        out.append(two)
    return out


NODE_TYPE_NAMES = ["State", "Action", "Finding", "Unknown"]


def _bucket(value, edges):
    for b, e in enumerate(edges):
        if value <= e:
            return f"bucket_{b}"
    return f"bucket_{len(edges)}"


def build_items_for_record(rec: dict, rng: random.Random, all_ids: list):
    """all_ids: list of (machine, row_id) for OTHER records, used to sample
    a decoy anchor for graph_consistency negative-by-construction items."""
    graph = rec["graph"]
    parsed = parse_graph_dict(graph)
    n = len(parsed["node_ids"])
    anon = _anon_ids(parsed)
    nbrs = _neighbors(parsed)
    two_hop = _two_hop(parsed, nbrs)
    key = {"machine": rec["machine"], "row_id": rec["row_id"]}

    items = []

    # -- adjacency --
    for i in rng.sample(range(n), k=min(2, n)):
        items.append({**key, "task": "adjacency", "query_node": anon[i],
                      "gold": sorted(anon[j] for j in nbrs[i])})

    # -- node_type --
    for i in rng.sample(range(n), k=min(2, n)):
        items.append({**key, "task": "node_type", "query_node": anon[i],
                      "gold": NODE_TYPE_NAMES[parsed["node_type"][i]]})

    # -- edge_type --
    if parsed["edge_index"] and parsed["edge_type"][0] != -1:
        idxs = [j for j, et in enumerate(parsed["edge_type"]) if et != -1]
        for j in rng.sample(idxs, k=min(2, len(idxs))):
            s, t = parsed["edge_index"][j]
            from standalone_config import EDGE_TYPES
            items.append({**key, "task": "edge_type", "query_edge": [anon[s], anon[t]],
                          "gold": EDGE_TYPES[parsed["edge_type"][j]]})

    # -- two_hop --
    for i in rng.sample(range(n), k=min(2, n)):
        items.append({**key, "task": "two_hop", "query_node": anon[i],
                      "gold": sorted(anon[j] for j in two_hop[i])})

    # -- graph_aggregate (buckets filled in globally, see finalize_aggregate_buckets) --
    n_edges = sum(1 for et in parsed["edge_type"])
    density = (2 * n_edges) / (n * (n - 1)) if n > 1 else 0.0
    type_counts = [0, 0, 0, 0]
    for t in parsed["node_type"]:
        type_counts[t] += 1
    dominant = NODE_TYPE_NAMES[type_counts.index(max(type_counts))]
    items.append({**key, "task": "graph_aggregate",
                  "_raw_n_nodes": n, "_raw_n_edges": n_edges, "_raw_density": density,
                  "gold": {"dominant_node_type": dominant}})

    # -- graph_consistency: 1 true claim + 1 false-by-construction claim,
    # anchored to THIS graph (used as the "real"/"wrong_graph" pair by the
    # training/eval scripts).
    if n >= 2:
        i = rng.randrange(n)
        connected = list(nbrs[i])
        if connected:
            j = rng.choice(connected)
            items.append({**key, "task": "graph_consistency",
                          "claim": f"Node {anon[i]} is directly connected to node {anon[j]}.",
                          "gold": True})
        non_neighbors = [k for k in range(n) if k != i and k not in nbrs[i]]
        if non_neighbors:
            j = rng.choice(non_neighbors)
            items.append({**key, "task": "graph_consistency",
                          "claim": f"Node {anon[i]} is directly connected to node {anon[j]}.",
                          "gold": False})

    return items


def finalize_aggregate_buckets(items: list):
    """graph_aggregate bucket edges are computed from the actual pooled
    distribution (quintiles) rather than hardcoded, so buckets stay
    meaningful regardless of dataset size/shape."""
    agg = [it for it in items if it["task"] == "graph_aggregate"]
    if not agg:
        return
    import statistics
    for field, raw_key in [("node_count_bucket", "_raw_n_nodes"),
                            ("edge_count_bucket", "_raw_n_edges"),
                            ("density_bucket", "_raw_density")]:
        vals = sorted(it[raw_key] for it in agg)
        edges = [vals[int(q * (len(vals) - 1))] for q in (0.2, 0.4, 0.6, 0.8)]
        for it in agg:
            it["gold"][field] = _bucket(it[raw_key], edges)
    for it in agg:
        for k in ("_raw_n_nodes", "_raw_n_edges", "_raw_density"):
            it.pop(k, None)


def build(split_name: str, records: list, seed: int):
    rng = random.Random(seed)
    all_ids = [(r["machine"], r["row_id"]) for r in records]
    items = []
    for rec in records:
        items.extend(build_items_for_record(rec, rng, all_ids))
    finalize_aggregate_buckets(items)
    rng.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_frac", type=float, default=VAL_FRAC)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    records = load_records(INPUT_TRAIN_JSON)
    try:
        records += load_records(INPUT_TEST_JSON)
    except FileNotFoundError:
        pass
    if not records:
        raise SystemExit(
            f"No graph records found. Set STANDALONE_INPUT_TRAIN_JSON / "
            f"STANDALONE_INPUT_TEST_JSON to your main pipeline's input/train.json "
            f"and input/test.json (only the 'graph' field of each record is read)."
        )

    train_recs, held_out_recs = machine_level_split(records, args.val_frac, args.seed)
    train_machines = {r["machine"] for r in train_recs}
    held_machines = {r["machine"] for r in held_out_recs}
    assert not (train_machines & held_machines), "LEAKAGE: a machine is in both splits"

    train_items = build("train", train_recs, args.seed)
    held_items = build("held_out", held_out_recs, args.seed + 1)

    import os
    with open(os.path.join(TASKS_DIR, "train.jsonl"), "w") as f:
        for it in train_items:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(TASKS_DIR, "held_out.jsonl"), "w") as f:
        for it in held_items:
            f.write(json.dumps(it) + "\n")
    manifest = {
        "n_train_machines": len(train_machines), "n_held_out_machines": len(held_machines),
        "n_train_items": len(train_items), "n_held_out_items": len(held_items),
        "train_machines": sorted(train_machines), "held_out_machines": sorted(held_machines),
    }
    with open(os.path.join(TASKS_DIR, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"train: {len(train_recs)} graphs / {len(train_machines)} machines / {len(train_items)} task items")
    print(f"held_out: {len(held_out_recs)} graphs / {len(held_machines)} machines / {len(held_items)} task items")
    print("LEAKAGE check passed: no machine appears in both splits.")


if __name__ == "__main__":
    main()
