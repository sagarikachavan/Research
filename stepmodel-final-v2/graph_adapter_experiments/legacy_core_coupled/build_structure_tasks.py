"""
build_structure_tasks.py
=========================

Turns the existing input/train.json + input/test.json graphs into 5 families
of graph-structure probe tasks, and writes a REPRODUCIBLE machine-level
train/held-out split to disk so every later script (training, evaluation,
analysis) uses the identical split.

Why a fresh held-out split instead of just using input/test.json:
Your original report trained the structure adapter on "175 graphs" and then
tested on "5 representative nodes from the graph" without saying whether
that test graph was one of the 175 training graphs. If it was, a 90% recall
number tells you almost nothing about generalization -- it could be pure
memorization of that one graph's embedding-to-answer mapping. Every task
here is explicitly split by MACHINE (matching how Stage 1/2/3 already avoid
leakage elsewhere in this repo), and every evaluation script downstream only
ever reports numbers on the held-out split.

Tasks produced (each item references a machine + row_index, not raw text
answers -- graphs are re-loaded from input/*.json at train/eval time so this
file stays small and never gets out of sync with the graph JSON):

  1. adjacency        -- given an anonymized node id, list its anonymized neighbors
  2. node_type         -- given an anonymized node id, predict State / Action / Finding
  3. edge_type         -- given two anonymized connected node ids, predict the edge type
  4. two_hop           -- given an anonymized node id, list nodes exactly 2 hops away
  5. graph_aggregate   -- given the whole graph, predict node_count bucket / edge_count
                           bucket / density bucket / dominant node type

IMPORTANT: node ids are ANONYMIZED (n0, n1, n2, ...) for tasks 1-4. The real
ids look like "state:bashed:r0:1.1", which trivially leaks the node's type
("state:") and would let a model "solve" node_type prediction by string
matching instead of by reading the graph representation. Anonymized ids are
assigned in a fixed, graph-local, non-alphabetic order (BFS from a random
root, reseeded per graph) so they carry no residual naming signal.
"""
import os
import sys
import json
import re
import random
import argparse
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    TASKS_DIR, load_all_examples, machine_level_split, RANDOM_SEED,
)

# Words that would let a model answer node_type "for free" by reading the
# title text instead of the graph representation -- stripped defensively.
_TYPE_LEAK_WORDS = re.compile(
    r"\b(state|action|finding|agent|search|track)\b", re.IGNORECASE
)


def _clean_title(title: str) -> str:
    t = _TYPE_LEAK_WORDS.sub("[item]", title or "")
    return re.sub(r"\s+", " ", t).strip() or "[unnamed item]"


def _anonymize_graph(graph_dict: dict, rng: random.Random):
    """Returns (anon_id_map, adjacency, edge_types, node_types, node_titles).
    node_titles gives every node a SHORT, TYPE-WORD-STRIPPED description so
    later prompts can ground "n3" in something concrete without leaking the
    answer to the node_type task."""
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    real_ids = [n["id"] for n in nodes]
    order = real_ids[:]
    rng.shuffle(order)
    anon_id_map = {rid: f"n{k}" for k, rid in enumerate(order)}

    node_types = {}
    node_titles = {}
    for n in nodes:
        aid = anon_id_map[n["id"]]
        node_types[aid] = n.get("type", "Unknown")
        raw_title = n.get("title") or n.get("label") or n.get("id", "")
        node_titles[aid] = _clean_title(raw_title)

    adjacency = defaultdict(set)
    edge_types = {}
    for e in edges:
        src = e.get("from", e.get("source"))
        tgt = e.get("to", e.get("target"))
        if src in anon_id_map and tgt in anon_id_map:
            au, av = anon_id_map[src], anon_id_map[tgt]
            adjacency[au].add(av)
            adjacency[av].add(au)  # treat as undirected for adjacency/2-hop tasks
            edge_types[(au, av)] = e.get("type", "Unknown")

    return anon_id_map, adjacency, edge_types, node_types, node_titles


def _two_hop(adjacency: dict, node: str) -> set:
    visited = {node}
    frontier = {node}
    for _ in range(2):
        nxt = set()
        for u in frontier:
            for v in adjacency.get(u, ()):
                if v not in visited:
                    nxt.add(v)
        visited |= nxt
        frontier = nxt
    visited.discard(node)
    # exclude direct neighbors -- we want EXACTLY 2 hops away
    return visited - adjacency.get(node, set())


def build_tasks_for_example(ex: dict, rng: random.Random, max_nodes_per_graph: int = 6,
                             legend_cap: int = 12):
    graph_dict = ex["graph_dict"] if "graph_dict" in ex else None
    if graph_dict is None:
        return []  # caller must attach raw graph_dict; see main()

    anon_id_map, adjacency, edge_types, node_types, node_titles = _anonymize_graph(graph_dict, rng)
    all_anon_nodes = list(anon_id_map.values())
    if not all_anon_nodes:
        return []

    # The text "legend" grounds each anonymized id in a short, type-word-
    # stripped description -- WITHOUT this, a question like "who is n3
    # connected to?" is unanswerable in principle (nothing in the prompt
    # would tell the model what n3 even refers to), which would make any
    # experiment result meaningless. With it, the legend supplies node
    # IDENTITY (fair -- a real system would always have this) while the
    # actual EDGES are only ever available through the graph prefix tokens
    # (not fair to also put in text), so success genuinely requires reading
    # structure out of the soft-prompt tokens.
    legend_nodes = all_anon_nodes if len(all_anon_nodes) <= legend_cap else \
        rng.sample(all_anon_nodes, legend_cap)
    legend = {n: node_titles[n] for n in legend_nodes}

    sample_nodes = [n for n in rng.sample(all_anon_nodes, min(max_nodes_per_graph, len(all_anon_nodes)))
                    if n in legend]
    if not sample_nodes:
        sample_nodes = legend_nodes[:max_nodes_per_graph]

    items = []
    key = dict(machine=ex["machine"], row_id=ex["row_id"], split=ex["_orig_split"])

    # 1. adjacency (restrict gold to nodes that are actually in the legend,
    # so an answer can't be judged "wrong" purely because that neighbor
    # wasn't nameable in the prompt)
    for node in sample_nodes:
        gold = sorted(n for n in adjacency.get(node, set()) if n in legend)
        items.append(dict(task="adjacency", **key, legend=legend, query_node=node, gold=gold))

    # 2. node_type
    for node in sample_nodes:
        items.append(dict(task="node_type", **key, legend=legend, query_node=node,
                           gold=node_types.get(node, "Unknown")))

    # 3. edge_type -- sample up to max_nodes_per_graph real edges, both endpoints in legend
    edge_items = [((au, av), t) for (au, av), t in edge_types.items() if au in legend and av in legend]
    rng.shuffle(edge_items)
    for (au, av), etype in edge_items[:max_nodes_per_graph]:
        items.append(dict(task="edge_type", **key, legend=legend, query_edge=[au, av], gold=etype))

    # 4. two_hop
    for node in sample_nodes:
        gold = sorted(n for n in _two_hop(adjacency, node) if n in legend)
        items.append(dict(task="two_hop", **key, legend=legend, query_node=node, gold=gold))

    # 5. graph_aggregate (one per graph, not per node) -- deliberately does
    # NOT need a legend at all, since it asks about the whole graph, not a
    # specific node. This is the fairest test for a POOLED graph embedding
    # (see RESEARCH_PLAN.md) since it doesn't require per-node addressing.
    n_nodes = len(all_anon_nodes)
    n_edges = len(edge_types)
    density = n_edges / max(1, n_nodes * (n_nodes - 1) / 2)
    type_counts = defaultdict(int)
    for t in node_types.values():
        type_counts[t] += 1
    dominant_type = max(type_counts, key=type_counts.get) if type_counts else "Unknown"
    items.append(dict(
        task="graph_aggregate", **key,
        gold=dict(
            node_count_bucket=_bucket(n_nodes, [5, 10, 20, 40]),
            edge_count_bucket=_bucket(n_edges, [5, 10, 20, 40]),
            density_bucket=_bucket(density, [0.05, 0.1, 0.2, 0.4]),
            dominant_node_type=dominant_type,
        ),
    ))

    return items


def _bucket(v, edges):
    for i, e in enumerate(edges):
        if v <= e:
            return f"bucket_{i}"
    return f"bucket_{len(edges)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held_out_frac", type=float, default=0.2)
    ap.add_argument("--max_nodes_per_graph", type=int, default=6)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    examples = load_all_examples()
    print(f"[build_structure_tasks] Loaded {len(examples)} graphs total "
          f"(train+test pooled -- we do our OWN held-out split below).")

    # Need raw graph_dict (nodes/edges with real ids), not the pyg Data
    # object -- re-derive it by re-reading input/*.json directly. Examples
    # from load_from_input_json() are enumerated in the SAME order as the
    # raw JSON list and use that enumerate() position as "row_id" (see
    # data_utils.py: `for row_id, record in enumerate(data)`), so we index
    # back into the raw list by position, not by any field inside the record.
    import config
    for split, path in (("train", config.INPUT_TRAIN_JSON), ("test", config.INPUT_TEST_JSON)):
        with open(path) as f:
            raw = json.load(f)
        for ex in examples:
            if ex["_orig_split"] == split and 0 <= ex["row_id"] < len(raw):
                ex["graph_dict"] = raw[ex["row_id"]].get("graph", raw[ex["row_id"]].get("Graph", {}))

    examples = [e for e in examples if e.get("graph_dict")]
    train_examples, held_out_examples = machine_level_split(
        examples, val_frac=args.held_out_frac, seed=args.seed
    )
    print(f"[build_structure_tasks] Machine-level split: "
          f"{len(train_examples)} train graphs / {len(held_out_examples)} held-out graphs "
          f"({len(set(e['machine'] for e in train_examples))} / "
          f"{len(set(e['machine'] for e in held_out_examples))} machines)")
    overlap = (set(e["machine"] for e in train_examples) &
               set(e["machine"] for e in held_out_examples))
    assert not overlap, f"LEAKAGE: machines in both splits: {overlap}"

    for split_name, split_examples in (("train", train_examples), ("held_out", held_out_examples)):
        out_path = os.path.join(TASKS_DIR, f"{split_name}.jsonl")
        n_items = 0
        with open(out_path, "w") as f:
            for ex in split_examples:
                for item in build_tasks_for_example(ex, rng, args.max_nodes_per_graph):
                    f.write(json.dumps(item) + "\n")
                    n_items += 1
        print(f"[build_structure_tasks] Wrote {n_items} task items -> {out_path}")

    manifest = dict(
        held_out_frac=args.held_out_frac, seed=args.seed,
        n_train_graphs=len(train_examples), n_held_out_graphs=len(held_out_examples),
        train_machines=sorted(set(e["machine"] for e in train_examples)),
        held_out_machines=sorted(set(e["machine"] for e in held_out_examples)),
    )
    with open(os.path.join(TASKS_DIR, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_structure_tasks] Split manifest written -> "
          f"{os.path.join(TASKS_DIR, 'split_manifest.json')}")


if __name__ == "__main__":
    main()
