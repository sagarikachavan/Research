"""
Data utilities:
  1. Robust normalization of the noisy free-text "New step" column onto the
     fixed STEP_LABELS taxonomy (embedding similarity + regex fallback).
  2. Robust extraction of tool names from the "MCP_tasks" column (which is
     sometimes a real python-dict string, sometimes free text) onto the
     fixed MCP_LABELS taxonomy -> multi-hot vector.
  3. A loader for pre-built per-row graphs from stepmodelv2/processed_data,
     with a fallback that parses the indentation-structured "PTT" text into
     a torch_geometric graph when no pre-built graph file exists.
  4. A GNN-stage PyTorch Dataset that returns (graph, context_text_fields,
     step_label, mcp_multihot).
"""
import ast
import os
import re
import glob
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd

from config import (
    STEP_LABELS, STEP2IDX, MCP_LABELS, MCP2IDX,
    GRAPH_DIR_TRAIN, GRAPH_DIR_TEST,
)

CONTEXT_COLUMNS = [
    "Previous strategy",
    "Previous step",
    "Previous step result",
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
    """Returns a list of canonical MCP_LABELS found in the raw MCP_tasks cell."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    text_parts = []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            # dict keys are (usually) the tool names -> highest-precision signal
            text_parts.extend(str(k) for k in parsed.keys())
            text_parts.extend(str(v) for v in parsed.values())
        else:
            text_parts.append(str(parsed))
    except Exception:
        text_parts.append(str(raw))

    joined = " | ".join(text_parts)
    found = [label for label, pat in _MCP_PATTERNS.items() if pat.search(joined)]
    return found


def mcp_multihot(labels: list) -> np.ndarray:
    vec = np.zeros(len(MCP_LABELS), dtype=np.float32)
    for l in labels:
        vec[MCP2IDX[l]] = 1.0
    return vec


# ----------------------------------------------------------------------------
# 3. Graph loading (pre-built) with PTT-text fallback builder
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
    one-hot status flag). Edges = parent -> child (tree structure) plus
    "next sibling" temporal edges, both directions (undirected GNN).
    """
    import torch
    from torch_geometric.data import Data

    nodes = parse_ptt_to_tree(ptt_text)
    if not nodes:
        # single empty node to avoid crashing on rows with no PTT context
        nodes = [(0, "root", "unknown")]

    texts = [n[1] for n in nodes]
    embs = _embed_texts(texts)  # (N, TEXT_EMB_DIM)

    status_map = {"completed": 0, "in progress": 1, "pending": 2, "failed": 3, "unknown": 4}
    status_onehot = np.zeros((len(nodes), 5), dtype=np.float32)
    for i, (_, _, status) in enumerate(nodes):
        status_onehot[i, status_map.get(status, 4)] = 1.0

    x = np.concatenate([embs, status_onehot], axis=1)

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
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
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
