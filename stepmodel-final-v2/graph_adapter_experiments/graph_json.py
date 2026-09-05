"""
graph_json.py
=============
Turns the raw "graph" JSON (nodes/edges, produced by the main pipeline's
data_prep/graph_builder.py — vis-network-style: nodes have id/type/status,
edges have from/to/type) into a torch_geometric.data.Data object using
STRUCTURE-ONLY features.

This module is fully independent: it does not import anything from
`core/`, `data_prep/`, or `training/`. It only assumes the JSON shape
below, which is a stable, documented data contract, not code:

    graph = {
      "nodes": [{"id": str, "type": "State"|"Action"|"Finding", "status": str, ...}, ...],
      "edges": [{"from": str, "to": str, "type": "StateTransition"|"SearchUpdate"|
                                                  "TrackUpdate"|"Prediction", ...}, ...]
    }

Deliberate exclusion: node "title"/"label" text is never read here. The
main pipeline's graph_encoder.py embeds node titles with a text encoder
and mixes that into the node features; doing that here would let the LLM
solve structure questions by reading semantics out of the title strings
instead of the graph topology, which defeats the point of this experiment.
The only signal a node carries into the GNN is: its type, its status, and
where it sits in the graph (degree, BFS depth from the row's start node).
"""
import json
import random
from collections import deque, Counter

import numpy as np

from standalone_config import NODE_TYPES, NODE_STATUSES, EDGE_TYPES, RANDOM_SEED

NODE_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
NODE_STATUS_IDX = {s: i for i, s in enumerate(NODE_STATUSES)}
EDGE_TYPE_IDX = {t: i for i, t in enumerate(EDGE_TYPES)}


def _norm_status(raw):
    s = str(raw or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    return s if s in NODE_STATUS_IDX else "unknown"


def _norm_type(raw):
    return raw if raw in NODE_TYPE_IDX else "Unknown"


def _find_start_node(node_ids):
    """The main pipeline always names the synthetic start node
    'state:<machine>:r<row>:START'. Fall back to node 0 if that convention
    isn't present (e.g. hand-built test graphs)."""
    for i, nid in enumerate(node_ids):
        if str(nid).upper().endswith(":START") or str(nid).upper() == "START":
            return i
    return 0


def parse_graph_dict(graph: dict):
    """
    Returns a plain-Python structural representation (no torch dependency),
    so this function alone can be unit-tested without torch installed:

        {
          "node_ids": [...],                  # original string ids, in order
          "node_type": [int, ...],            # index into NODE_TYPES
          "node_status": [int, ...],          # index into NODE_STATUSES
          "in_deg": [...], "out_deg": [...], "total_deg": [...],
          "bfs_depth": [...],                 # hops from start node (undirected)
          "is_start": [...],                  # 0/1
          "edge_index": [[src, tgt], ...],    # 0-indexed into node_ids
          "edge_type": [int, ...],            # index into EDGE_TYPES, or -1 if unknown
        }
    """
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not nodes:
        nodes = [{"id": "__empty__", "type": "Unknown", "status": "unknown"}]
        edges = []

    node_ids = [n.get("id") for n in nodes]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    node_type = [NODE_TYPE_IDX[_norm_type(nd.get("type"))] for nd in nodes]
    node_status = [NODE_STATUS_IDX[_norm_status(nd.get("status"))] for nd in nodes]

    edge_index, edge_type = [], []
    for e in edges:
        src, tgt = e.get("from"), e.get("to")
        if src in id2idx and tgt in id2idx:
            edge_index.append([id2idx[src], id2idx[tgt]])
            etype = e.get("type")
            edge_type.append(EDGE_TYPE_IDX.get(etype, -1))
    if not edge_index:
        edge_index = [[0, 0]]
        edge_type = [-1]

    in_deg = [0] * n
    out_deg = [0] * n
    adj = [[] for _ in range(n)]  # undirected adjacency, for BFS depth only
    for s, t in edge_index:
        out_deg[s] += 1
        in_deg[t] += 1
        adj[s].append(t)
        adj[t].append(s)
    total_deg = [in_deg[i] + out_deg[i] for i in range(n)]

    start = _find_start_node(node_ids)
    bfs_depth = [None] * n
    bfs_depth[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if bfs_depth[v] is None:
                bfs_depth[v] = bfs_depth[u] + 1
                q.append(v)
    max_depth = max((d for d in bfs_depth if d is not None), default=0) or 1
    # Unreachable nodes (shouldn't normally happen) get a depth just beyond
    # the max observed depth rather than an arbitrary sentinel.
    bfs_depth = [(d if d is not None else max_depth + 1) for d in bfs_depth]

    is_start = [1 if i == start else 0 for i in range(n)]

    return {
        "node_ids": node_ids,
        "node_type": node_type,
        "node_status": node_status,
        "in_deg": in_deg,
        "out_deg": out_deg,
        "total_deg": total_deg,
        "bfs_depth": bfs_depth,
        "max_depth": max_depth,
        "is_start": is_start,
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


def to_pyg_data(graph: dict):
    """Structural-only torch_geometric.data.Data. Requires torch/torch_geometric
    (imported lazily so parse_graph_dict() above can be used/tested without them)."""
    import torch
    from torch_geometric.data import Data

    parsed = parse_graph_dict(graph)
    n = len(parsed["node_ids"])
    max_deg = max(parsed["total_deg"]) or 1
    max_depth = parsed["max_depth"] or 1

    x = np.zeros((n, len(NODE_TYPES) + len(NODE_STATUSES) + 5), dtype=np.float32)
    for i in range(n):
        x[i, parsed["node_type"][i]] = 1.0
        x[i, len(NODE_TYPES) + parsed["node_status"][i]] = 1.0
        base = len(NODE_TYPES) + len(NODE_STATUSES)
        x[i, base + 0] = parsed["in_deg"][i] / max_deg
        x[i, base + 1] = parsed["out_deg"][i] / max_deg
        x[i, base + 2] = parsed["total_deg"][i] / max_deg
        x[i, base + 3] = parsed["bfs_depth"][i] / max_depth
        x[i, base + 4] = float(parsed["is_start"][i])

    edge_index = np.array(parsed["edge_index"], dtype=np.int64).T  # (2, E)
    e = edge_index.shape[1]
    edge_attr = np.zeros((e, len(EDGE_TYPES) + 1), dtype=np.float32)
    for j, et in enumerate(parsed["edge_type"]):
        if et == -1:
            edge_attr[j, len(EDGE_TYPES)] = 1.0  # self-loop / unknown-type indicator
        else:
            edge_attr[j, et] = 1.0

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    data.node_ids = parsed["node_ids"]
    return data


def load_records(json_path: str):
    """
    Reads one of the main pipeline's input/{train,test}.json files and
    returns ONLY what this experiment needs from each record:
    machine, a stable row id, and the raw graph dict. Every other field
    in the record ("new_strategy", "strategy_explanation", "gold_*") is
    dropped here — never read by anything downstream in this folder.
    """
    with open(json_path) as f:
        raw = json.load(f)
    records = []
    for i, rec in enumerate(raw):
        graph = rec.get("graph")
        if not graph or not graph.get("nodes"):
            continue
        row_id = rec.get("row_index", rec.get("row_id", i))
        records.append({
            "machine": rec.get("machine", f"unknown_machine_{i}"),
            "row_id": row_id,
            "graph": graph,
        })
    return records


def index_records(records: list) -> dict:
    """(machine, row_id) -> record, for O(1) lookup from a probe-task item
    back to its raw graph dict."""
    return {(r["machine"], r["row_id"]): r for r in records}


def machine_level_split(records: list, val_frac: float, seed: int = RANDOM_SEED):
    """Split by MACHINE (never by row), so no held-out graph's machine also
    appears in training — mirrors the leakage guard used throughout the
    main pipeline, reimplemented here with zero import from it."""
    machines = sorted({r["machine"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(machines)
    n_val = max(1, int(len(machines) * val_frac))
    val_machines = set(machines[:n_val])
    train = [r for r in records if r["machine"] not in val_machines]
    held_out = [r for r in records if r["machine"] in val_machines]
    assert not (val_machines & {r["machine"] for r in train}), "machine leakage between splits"
    return train, held_out


def graph_signature(graph: dict) -> str:
    """A short structural fingerprint used to make sure a 'decoy' graph
    sampled as the wrong-graph condition is actually structurally
    different from the real one (not just a different object with an
    identical shape by coincidence)."""
    parsed = parse_graph_dict(graph)
    type_counts = Counter(parsed["node_type"])
    edge_counts = Counter(t for t in parsed["edge_type"] if t != -1)
    return json.dumps({
        "n_nodes": len(parsed["node_ids"]),
        "n_edges": len(parsed["edge_index"]),
        "type_counts": dict(sorted(type_counts.items())),
        "edge_counts": dict(sorted(edge_counts.items())),
    }, sort_keys=True)
