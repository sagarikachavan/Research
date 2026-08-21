"""
Data utilities:
  1. Robust normalization of the noisy free-text "New step" column onto the
     fixed STEP_LABELS taxonomy (embedding similarity + regex fallback).
  2. Robust extraction of tool names from the "MCP_tasks" column (which is
     sometimes a real python-dict string, sometimes free text) onto the
     fixed MCP_LABELS taxonomy -> multi-hot vector.
  3. Primary loader: load_from_input_json() reads input/train.json or
     input/test.json (produced by build_input_json.py) and builds
     torch_geometric Data objects directly from the embedded Graph JSON.
     Node features: 387-dim = 384-dim bge-small-en-v1.5 title embedding +
     3-dim one-hot type (Agent=0, Search=1, Track=2).
  4. Fallback loader for pre-built per-row graphs from processed_data/,
     with a PTT-text graph builder when no pre-built file exists.
  5. A GNN-stage PyTorch Dataset that returns (graph, context_text_fields,
     step_label, mcp_multihot).
"""
import ast
import os
import re
import glob
import json
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

from config import (
    STEP_LABELS, STEP2IDX, MCP_LABELS, MCP2IDX,
    GRAPH_DIR_TRAIN, GRAPH_DIR_TEST,
    INPUT_TRAIN_JSON, INPUT_TEST_JSON,
)

CONTEXT_COLUMNS = [
    "New strategy",
    "Strategy explanation",
]

# ----------------------------------------------------------------------------
# 1. Step label normalization
# ----------------------------------------------------------------------------
_STEP_REGEX = [
    (re.compile(r"google search", re.I), STEP_LABELS[0]),
    (re.compile(r"enumerate further on the .*(service|http|ftp|smb|ssh)", re.I), STEP_LABELS[1]),
    (re.compile(r"explore.*(suspicious|files|commands).*summary", re.I), STEP_LABELS[2]),
    (re.compile(r"further enumerate the website", re.I), STEP_LABELS[3]),
    (re.compile(r"enumerate the domain", re.I), STEP_LABELS[4]),
    (re.compile(r"exploit the selected exploitation", re.I), STEP_LABELS[5]),
    (re.compile(r"analy[sz]e the outcomes.*attack path", re.I), STEP_LABELS[6]),
    (re.compile(r"ask for human", re.I), STEP_LABELS[7]),
    (re.compile(r"explore the source code", re.I), STEP_LABELS[8]),
    (re.compile(r"end task.*(permission|report)", re.I), STEP_LABELS[9]),
]


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower().rstrip(".")


class StepLabelNormalizer:
    """
    Maps noisy free-text "New step" strings onto the fixed STEP_LABELS set.

    Strategy (cheap -> expensive, first match wins):
      (a) exact match after whitespace/punctuation normalization
      (b) regex/keyword rules (_STEP_REGEX) - handles almost all observed
          variants (missing trailing period, "commands and" fragments, etc.)
      (c) embedding cosine-similarity fallback against the canonical labels,
          using the same sentence encoder used for context text. This is
          lazily loaded so data_utils can be imported without a GPU/network.
    """

    def __init__(self, sim_threshold: float = 0.55):
        self.sim_threshold = sim_threshold
        self._canon_norm = {_normalize_whitespace(l): l for l in STEP_LABELS}
        self._encoder = None
        self._canon_emb = None

    def _lazy_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            from config import TEXT_ENCODER_NAME
            self._encoder = SentenceTransformer(TEXT_ENCODER_NAME)
            self._canon_emb = self._encoder.encode(STEP_LABELS, normalize_embeddings=True)
        return self._encoder

    def normalize(self, raw: str) -> str:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        norm = _normalize_whitespace(raw)
        if norm in self._canon_norm:
            return self._canon_norm[norm]

        for pattern, label in _STEP_REGEX:
            if pattern.search(raw):
                return label

        # embedding fallback for anything the regexes didn't catch
        enc = self._lazy_encoder()
        emb = enc.encode([raw], normalize_embeddings=True)
        sims = self._canon_emb @ emb[0]
        best = int(np.argmax(sims))
        if sims[best] >= self.sim_threshold:
            return STEP_LABELS[best]
        return None  # unresolvable -> drop row / route to "Ask for human assistant" at caller's discretion


# ----------------------------------------------------------------------------
# 2. MCP multi-label extraction
# ----------------------------------------------------------------------------
_MCP_PATTERNS = {
    "Nmap": re.compile(r"\bnmap\b", re.I),
    "Metasploit": re.compile(r"\bmetasploit|msfconsole|msfvenom\b", re.I),
    "Netcat": re.compile(r"\bnetcat|\bnc\b", re.I),
    "Dirbuster": re.compile(r"\bdirbuster|gobuster|dirb\b", re.I),
    "SQLmap": re.compile(r"\bsqlmap\b", re.I),
    "Smb client": re.compile(r"smb\s*client|smbclient|\bsmb\b", re.I),
    "hydra": re.compile(r"\bhydra\b", re.I),
    "John-the-ripper": re.compile(r"john[\s\-]?the[\s\-]?ripper|\bjohn\b", re.I),
    "Google search": re.compile(r"google\s*search|\bgoogle\b", re.I),
    "Interactive CLI": re.compile(r"interactive\s*cli|\bssh\b|\bbash\b|\bshell\b", re.I),
    "Web page interaction": re.compile(r"web\s*page\s*interaction|\bbrowser\b|\bcurl\b", re.I),
}


def extract_mcp_labels(raw) -> list:
    """
    Returns a list of canonical MCP_LABELS found in the raw MCP_tasks cell.

    FIX: when MCP_tasks parses as a real python dict (the common case —
    `{'Smb client': 'Enumerate SMB service...'}`), the dict KEYS are an
    exact, high-precision gold signal and are used on their own. The
    previous version also regex-matched against the dict VALUES (the free
    text description), which occasionally pattern-matched an unrelated tool
    mentioned only in passing inside the description — e.g. a value like
    "...explore the file system, run commands..." would spuriously add
    "Interactive CLI" even when the actual dict only had {"SQLmap": ...}.
    That silently injected extra positive labels into the gold multi-hot
    vector, which both trains the model on slightly wrong targets and
    depresses eval metrics (subset accuracy in particular, since it
    requires an exact set match). Verified against test_data.csv: ~3% of
    rows had at least one such spurious extra label before this fix.

    The free-text regex fallback (_MCP_PATTERNS over the whole cell) is now
    used ONLY when the cell isn't a parseable dict, i.e. genuinely free text.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []

    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        parsed = None

    if isinstance(parsed, dict) and parsed:
        # High-precision path: keys ARE the tool names. Still run each key
        # through the canonical-label matcher (in case of near-miss casing/
        # spelling like "Smb Client" vs "Smb client"), but never look at
        # the free-text values.
        found = []
        for k in parsed.keys():
            k_str = str(k)
            # exact / case-insensitive match against canonical labels first
            exact = next((l for l in MCP_LABELS if l.lower() == k_str.strip().lower()), None)
            if exact:
                found.append(exact)
                continue
            # fall back to pattern match on the key text only (not the value)
            for label, pat in _MCP_PATTERNS.items():
                if pat.search(k_str):
                    found.append(label)
                    break
        # de-duplicate, preserve order
        seen = set()
        return [l for l in found if not (l in seen or seen.add(l))]

    # Free-text fallback (cell wasn't a parseable non-empty dict)
    text = str(parsed) if parsed is not None else str(raw)
    return [label for label, pat in _MCP_PATTERNS.items() if pat.search(text)]


def mcp_multihot(labels: list) -> np.ndarray:
    vec = np.zeros(len(MCP_LABELS), dtype=np.float32)
    for l in labels:
        vec[MCP2IDX[l]] = 1.0
    return vec


# ----------------------------------------------------------------------------
# 3. NEW: Primary input loader — input/train.json, input/test.json
# ----------------------------------------------------------------------------

def build_graph_from_input_json_graph(graph_dict: dict):
    """
    Builds a torch_geometric.data.Data object from the Graph JSON (as exported
    by build_input_json.py, which comes from generate_graphs.py).

    Node features (388-dim):
      - 384-dim: BAAI/bge-small-en-v1.5 embedding of node title  (cached)
      - 3-dim: one-hot type encoding (Agent=0, Search=1, Track=2)
      - 1-dim: normalized node degree feature

    Edge index: constructed from the 'from' → 'to' field of each edge.

    Node title embeddings are served from _TITLE_EMB_CACHE — each unique
    title string is encoded at most once per process, regardless of how many
    records share the same graph nodes.
    """
    import torch
    from torch_geometric.data import Data

    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])

    if not nodes:
        # empty graph fallback — single dummy node
        nodes = [{"id": "dummy", "title": "empty graph", "type": "Agent"}]
        edges = []

    # Node ID → index mapping
    node_ids = [n["id"] for n in nodes]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    # Embed node titles — uses process-level cache, no repeat encoder calls
    titles = [n.get("title", n.get("label", "")).strip() or "unknown node"
              for n in nodes]
    title_embs = _embed_titles_cached(titles)  # (N, TEXT_EMB_DIM)
    # stepmodelv2: Agent=0, Search=1, Track=2
    # stepmodelv3: State=0, Action=1, Finding=2
    type_map_v2 = {"Agent": 0, "Search": 1, "Track": 2}
    type_map_v3 = {"State": 0, "Action": 1, "Finding": 2}
    type_onehot = np.zeros((len(nodes), 3), dtype=np.float32)
    for i, n in enumerate(nodes):
        ntype = n.get("type", "Agent")
        # Try stepmodelv3 mapping first, fallback to stepmodelv2
        if ntype in type_map_v3:
            type_onehot[i, type_map_v3[ntype]] = 1.0
        else:
            type_onehot[i, type_map_v2.get(ntype, 0)] = 1.0

    # Build edge_list first (support both stepmodelv2 "from"/"to" and stepmodelv3 "source"/"target")
    edge_list = []
    for e in edges:
        src_id = e.get("from", e.get("source", ""))
        tgt_id = e.get("to", e.get("target", ""))
        if src_id in id2idx and tgt_id in id2idx:
            edge_list.append([id2idx[src_id], id2idx[tgt_id]])

    if not edge_list:
        # single self-loop to avoid empty edge_index
        edge_list = [[0, 0]]

    edge_index = np.array(edge_list, dtype=np.int64).T  # (2, E)

    # Calculate node degrees for additional features
    from collections import Counter
    degree_counts = Counter()
    for e in edge_list:
        degree_counts[e[0]] += 1  # out-degree
        degree_counts[e[1]] += 1  # in-degree (undirected)

    # Normalize degrees and add as feature
    max_degree = max(degree_counts.values()) if degree_counts else 1
    degree_features = np.zeros((len(nodes), 1), dtype=np.float32)
    for i in range(len(nodes)):
        degree_features[i, 0] = degree_counts.get(i, 0) / max_degree

    # Combine: (N, TEXT_EMB_DIM + 3 + 1) = (N, TEXT_EMB_DIM + 4)
    x = np.concatenate([title_embs, type_onehot, degree_features], axis=1)

    # Build edge_attr: one-hot [type: [parent_child, sequential, self_loop] = 3-dim
    edge_attr = np.zeros((edge_index.shape[1], 3), dtype=np.float32)
    for e_idx in range(edge_index.shape[1]):
        u, v = edge_index[0, e_idx], edge_index[1, e_idx]
        if u == v:
            edge_attr[e_idx, 2] = 1.0
        else:
            # Detect sequential/sibling (sequential nodes)
            edge_attr[e_idx, 1] = 0.5  # default weight — real edge
            # Parent-child / hierarchical gets boosted based on structure (remain 0 in second dim, 1 in first
            edge_attr[e_idx, 0] = 0.5

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    return data


def load_from_input_json(json_path: str, split: str = "train"):
    """
    Primary loader: reads input/train.json or input/test.json (produced by
    build_input_json_finalv.py from stepmodelv3) and returns a list of examples,
    where each example has:
      {
        machine, row_id, graph (torch_geometric Data), context: {...},
        step_label, step_idx, mcp_labels, mcp_vec,
        gold_step_explanation, gold_mcp_raw
      }

    Records whose "gold_new_step" cannot be normalized to STEP_LABELS are
    dropped.

    PERFORMANCE: All unique node titles across the entire file are collected
    first and embedded in a single batched encoder call before any graphs are
    built, populating _TITLE_EMB_CACHE. Subsequent calls to
    build_graph_from_input_json_graph() are then pure cache hits — no further
    encoder calls needed for titles seen in this file.
    
    DATA LEAKAGE PREVENTION: Cache is cleared when loading test data to ensure
    no information from training set influences test set embeddings.
    """
    # Clear cache for test data to prevent data leakage
    if split == "test":
        global _TITLE_EMB_CACHE
        _TITLE_EMB_CACHE.clear()
        print(f"[{split}] Cleared title embedding cache to prevent data leakage")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Pre-warm the title cache with one batched encoder call ────────────────
    # Collect every unique node title across the whole file
    all_titles: set[str] = set()
    for record in data:
        graph_dict = record.get("graph", record.get("Graph", {}))
        for node in graph_dict.get("nodes", []):
            t = node.get("title", node.get("label", "")).strip() or "unknown node"
            all_titles.add(t)

    # Filter to only titles not already in cache
    unseen = [t for t in all_titles if t not in _TITLE_EMB_CACHE]
    if unseen:
        print(f"[{split}] Pre-embedding {len(unseen)} unique node titles "
              f"({len(all_titles) - len(unseen)} already cached) ...")
        embs = _embed_texts(unseen)          # single batched call
        for title, emb in zip(unseen, embs):
            _TITLE_EMB_CACHE[title] = emb
        print(f"[{split}] Title cache now holds {len(_TITLE_EMB_CACHE)} entries.")
    else:
        print(f"[{split}] All {len(all_titles)} node titles already cached — skipping encoder.")

    # ── Build examples (all graph builds are now pure cache hits) ────────────
    normalizer = StepLabelNormalizer()
    examples = []
    dropped = 0

    for row_id, record in enumerate(data):
        machine = record.get("machine", record.get("Machine", "unknown"))
        graph_dict = record.get("graph", record.get("Graph", {}))

        # Normalize step label (stepmodelv3 format uses "gold_new_step")
        raw_step = record.get("gold_new_step", record.get("Gold New step", ""))
        step_label = normalizer.normalize(raw_step)
        if step_label is None:
            dropped += 1
            continue

        # Extract MCP labels (stepmodelv3 format uses "gold_mcp_tasks")
        mcp_raw = record.get("gold_mcp_tasks", record.get("Gold MCP_tasks", ""))
        mcp_labels = extract_mcp_labels(mcp_raw)

        # Build context dict (only actual fields from JSON)
        new_strategy = record.get("new_strategy", record.get("New strategy", ""))
        strategy_explanation = record.get("strategy_explanation", record.get("Strategy explanation", ""))
        gold_new_step = record.get("gold_new_step", record.get("Gold New step", ""))
        gold_step_explanation = record.get("gold_step_explanation", record.get("Gold Step explanation", ""))
        
        context = {
            "New strategy":         new_strategy,
            "Strategy explanation": strategy_explanation,
        }

        # Build graph — every _embed_titles_cached call here is a cache hit
        graph = build_graph_from_input_json_graph(graph_dict)

        examples.append({
            "machine":               machine,
            "row_id":                row_id,
            "graph":                 graph,
            "context":               context,
            "step_label":            step_label,
            "step_idx":              STEP2IDX[step_label],
            "mcp_labels":            mcp_labels,
            "mcp_vec":               mcp_multihot(mcp_labels),
            "gold_step_explanation": gold_step_explanation,
            "gold_mcp_raw":          mcp_raw,
            "ptt":                   "",  # not needed when using input JSON
        })

    print(f"[{split}] loaded {len(examples)} usable rows from {json_path}, "
          f"dropped {dropped} unmappable 'gold_new_step' rows "
          f"({dropped / (len(data) or 1):.1%})")
    return examples


# ----------------------------------------------------------------------------
# 4. FALLBACK: Graph loading (pre-built) with PTT-text fallback builder
# ----------------------------------------------------------------------------
def _find_prebuilt_graph(machine: str, row_id: int, split: str):
    graph_dir = GRAPH_DIR_TRAIN if split == "train" else GRAPH_DIR_TEST
    candidates = [
        os.path.join(graph_dir, f"{machine}__{row_id}.pt"),
        os.path.join(graph_dir, f"{machine}_{row_id}.pt"),
    ]
    # also allow "one graph per machine" layout
    candidates += glob.glob(os.path.join(graph_dir, f"{machine}*.pt"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def parse_ptt_to_tree(ptt_text: str):
    """
    Parses the indentation-structured Penetration Testing Tree (PTT) text
    column into a list of (depth, node_text, status) tuples, in document
    order. Depth is inferred from leading whitespace (3 spaces / level, as
    seen in the data), matched against the numeric prefix as a sanity check.
    """
    nodes = []
    if not isinstance(ptt_text, str):
        return nodes
    for line in ptt_text.split("\n"):
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip(" "))
        depth = leading // 4  # empirically ~4 spaces per indent level in the dumps
        m = re.search(r"-\s*\[?\(?(completed|in progress|pending|failed)\)?\]?", line, re.I)
        status = m.group(1).lower() if m else "unknown"
        text = line.strip()
        nodes.append((depth, text, status))
    return nodes


def build_graph_from_ptt(ptt_text: str):
    """
    Fallback graph construction: builds a torch_geometric Data object
    directly from the PTT text when no pre-exported graph file is found.

    Node feature = frozen sentence-embedding of the node's text (+ a
    3-dim one-hot type, all set to Agent since PTT has no type info).
    This matches the 387-dim feature space expected by the GNN.
    Edges = parent -> child (tree structure) plus "next sibling" temporal
    edges, both directions (undirected GNN).

    Title embeddings are served from _TITLE_EMB_CACHE — same cache as the
    primary JSON loader, so PTT nodes that happen to share text with graph
    JSON nodes are also free.
    """
    import torch
    from torch_geometric.data import Data

    nodes = parse_ptt_to_tree(ptt_text)
    if not nodes:
        nodes = [(0, "root", "unknown")]

    texts = [n[1] for n in nodes]
    embs = _embed_titles_cached(texts)  # (N, 384) — uses cache

    # All PTT nodes treated as "Agent" type (index 0)
    type_onehot = np.zeros((len(nodes), 3), dtype=np.float32)
    type_onehot[:, 0] = 1.0  # Agent = index 0

    # Calculate node degrees for consistency with JSON loader (needed for 388-dim)
    from collections import Counter
    edges_for_degree = []
    stack = []  # (depth, index)
    for i, (depth, _, _) in enumerate(nodes):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            parent_idx = stack[-1][1]
            edges_for_degree.append((parent_idx, i))
            edges_for_degree.append((i, parent_idx))
        stack.append((depth, i))
    for i in range(len(nodes) - 1):
        edges_for_degree.append((i, i + 1))
        edges_for_degree.append((i + 1, i))

    degree_counts = Counter()
    for e in edges_for_degree:
        degree_counts[e[0]] += 1
        degree_counts[e[1]] += 1
    max_degree = max(degree_counts.values()) if degree_counts else 1
    degree_features = np.zeros((len(nodes), 1), dtype=np.float32)
    for i in range(len(nodes)):
        degree_features[i, 0] = degree_counts.get(i, 0) / max_degree

    # Now: 384 + 3 + 1 = 388 (matches JSON loader and graph_encoder NODE_FEAT_DIM)
    x = np.concatenate([embs, type_onehot, degree_features], axis=1)  # (N, 388)

    edges = []
    stack = []  # (depth, index)
    for i, (depth, _, _) in enumerate(nodes):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            parent_idx = stack[-1][1]
            edges.append((parent_idx, i))
            edges.append((i, parent_idx))
        stack.append((depth, i))
    for i in range(len(nodes) - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))

    if not edges:
        edges = [(0, 0)]

    edge_index = np.array(edges, dtype=np.int64).T

    # Simple edge features: [parent_child, sibling, self_loop] one-hot
    edge_attr = np.zeros((edge_index.shape[1], 3), dtype=np.float32)
    sibling_set = set()
    for i in range(len(nodes) - 1):
        sibling_set.add((i, i + 1))
        sibling_set.add((i + 1, i))
    for e_idx, (u, v) in enumerate(edge_index.T):
        if u == v:
            edge_attr[e_idx, 2] = 1.0  # self-loop
        elif (int(u), int(v)) in sibling_set:
            edge_attr[e_idx, 1] = 1.0  # sibling (sequential)
        else:
            edge_attr[e_idx, 0] = 1.0  # parent-child (hierarchical)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    return data


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    from config import TEXT_ENCODER_NAME
    return SentenceTransformer(TEXT_ENCODER_NAME)


def _embed_texts(texts):
    enc = _get_embedder()
    return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Title embedding cache — process-level, persists across all dataset loads.
#
# Without this, build_graph_from_input_json_graph calls _embed_texts() once
# per graph (~17 nodes × 1,894 records = ~32,000 encoder calls at Stage 1
# init alone, repeated identically at Stage 2 and Stage 3). Many node titles
# are shared across records of the same machine (e.g. the START node title
# appears in every row for that machine), so the same string gets re-embedded
# thousands of times.
#
# The cache maps title_string → np.ndarray(384, float32).
# build_graph_from_input_json_graph batches all cache-miss titles into a
# single encoder call, then stores results before building the Data object.
# ---------------------------------------------------------------------------
_TITLE_EMB_CACHE: dict[str, np.ndarray] = {}


def _embed_titles_cached(titles: list[str]) -> np.ndarray:
    """
    Return (N, 384) embeddings for `titles`, using and populating
    _TITLE_EMB_CACHE so each unique string is encoded at most once
    per process lifetime.

    Steps:
      1. Identify titles not yet in cache.
      2. Encode all misses in a single batched encoder call.
      3. Store results in cache.
      4. Build the output array from cache, preserving input order.
    """
    global _TITLE_EMB_CACHE

    # Collect unique unseen titles (preserve first-seen order)
    seen = set()
    misses = []
    for t in titles:
        if t not in _TITLE_EMB_CACHE and t not in seen:
            misses.append(t)
            seen.add(t)

    # One batched encoder call for all cache misses
    if misses:
        new_embs = _embed_texts(misses)          # (len(misses), 384)
        for title, emb in zip(misses, new_embs):
            _TITLE_EMB_CACHE[title] = emb

    # Build output in original order from cache
    return np.stack([_TITLE_EMB_CACHE[t] for t in titles])  # (N, 384)


def load_graph(machine: str, row_id: int, ptt_text: str, split: str = "train"):
    """
    Tries to load a pre-built graph exported by the graph-building stage
    (stepmodelv2/processed_data/{split}). Falls back to constructing the
    graph on the fly from the PTT column if no export is found.
    """
    path = _find_prebuilt_graph(machine, row_id, split)
    if path is not None:
        import torch
        return torch.load(path)
    return build_graph_from_ptt(ptt_text)


# ----------------------------------------------------------------------------
# 4. End-to-end CSV -> examples
# ----------------------------------------------------------------------------
def load_and_clean(csv_path: str, split: str = "train"):
    """
    Returns a list of dicts, one per usable row:
      {machine, row_id, context: {...}, ptt, step_label, step_idx,
       mcp_labels, mcp_vec, gold_step_text, gold_step_explanation}
    Rows whose "New step" cannot be normalized to any STEP_LABELS entry are
    dropped (logged) rather than silently mislabeled.
    """
    df = pd.read_csv(csv_path)
    normalizer = StepLabelNormalizer()
    examples = []
    dropped = 0
    for row_id, row in df.iterrows():
        step_label = normalizer.normalize(row.get("New step"))
        if step_label is None:
            dropped += 1
            continue
        mcp_labels = extract_mcp_labels(row.get("MCP_tasks"))
        context = {c: ("" if pd.isna(row.get(c)) else str(row.get(c))) for c in CONTEXT_COLUMNS}
        examples.append({
            "machine": row.get("Machine"),
            "row_id": row_id,
            "context": context,
            "ptt": row.get("PTT"),
            "step_label": step_label,
            "step_idx": STEP2IDX[step_label],
            "mcp_labels": mcp_labels,
            "mcp_vec": mcp_multihot(mcp_labels),
            "gold_step_explanation": ("" if pd.isna(row.get("Step explanation")) else str(row.get("Step explanation"))),
            "gold_mcp_raw": row.get("MCP_tasks"),
        })
    print(f"[{split}] loaded {len(examples)} usable rows, dropped {dropped} "
          f"unmappable 'New step' rows ({dropped/len(df):.1%})")
    return examples


class GNNStageDataset:
    """torch.utils.data.Dataset-like wrapper (kept framework-light here;
    see stage1_gnn_train.py for the actual torch Dataset subclass)."""

    def __init__(self, csv_path: str, split: str = "train"):
        self.split = split
        self.examples = load_and_clean(csv_path, split)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        graph = load_graph(ex["machine"], ex["row_id"], ex["ptt"], self.split)
        return graph, ex