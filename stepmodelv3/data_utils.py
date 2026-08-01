"""
data_utils.py — shared data loading / normalization utilities for stepmodelv3.

IMPORTANT — this replaces the assumptions baked into the original
stage1_grpo_rl.py / evaluate.py. Those scripts were written against a
schema (`Machine`, `Graph`, `step_label`, `mcp_labels`, fixed 10-way
STEP_LABELS, fixed nmap/ssh/... MCP_LABELS with clean JSON `gold_mcp_tasks`)
that does NOT match your actual input/train.json and input/test.json.

The real files have, per row:
    machine                 : str
    graph                   : dict (nodes/edges/legend/graph_statistics)
    new_strategy             : str  (a draft/candidate next-step, NOT the target)
    strategy_explanation     : str  (explanation for the draft candidate)
    gold_new_step            : str  (free-text gold next step -> TRAINING TARGET)
    gold_step_explanation    : str  (free-text gold explanation -> TRAINING TARGET)
    gold_mcp_tasks           : str  (semi-structured "Tool: desc; Tool2: desc2"
                                      OR python-dict-repr string -> TRAINING TARGET)

`gold_new_step` only has 46 raw unique strings across 1728 rows, and they
cluster tightly into 10 canonical categories (verified against your data).
`gold_mcp_tasks` tool names cluster into 18 canonical tools (verified,
>99% coverage). We use those canonical sets for classification-style
accuracy / F1 metrics, while still training the model to generate the
full free-text step / explanation / MCP dict (so generation quality and
the graph-grounded specifics — e.g. which service, which IP — aren't lost).
"""
from __future__ import annotations

import ast
import json
import re
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Canonical step categories (data-derived, validated against train.json:
# 1721/1728 rows classified, 6 blank gold_new_step, 1 typo)
# ---------------------------------------------------------------------------
STEP_LABELS = [
    "recon_scan",
    "enumerate_further",
    "enumerate_website",
    "enumerate_domain",
    "explore_files",
    "source_code_review",
    "google_search",
    "exploit",
    "analyze_outcomes",
    "end_task",
]

_STEP_KEYWORDS = [
    ("recon_scan", ["netdiscover", "nmap", "port scan", "discover the target ip",
                     "identify the target ip"]),
    ("enumerate_further", ["enumerate further"]),
    ("enumerate_website", ["enumerate the website", "website"]),
    ("enumerate_domain", ["enumerate the domain", "domain"]),
    ("explore_files", ["explore the suspicious", "explore the suspecious",
                        "xplore the suspicious", "suspicious files", "suspecious files"]),
    ("source_code_review", ["source code"]),
    ("google_search", ["google search"]),
    ("exploit", ["exploit"]),
    ("analyze_outcomes", ["analyze the outcomes", "analyze outcomes", "attack path"]),
    ("end_task", ["end task"]),
]


def classify_step(text: str) -> str:
    """Map a free-text step string to one of the 10 canonical STEP_LABELS."""
    if not text or not text.strip():
        return "unknown"
    t = text.lower()
    for canon, kws in _STEP_KEYWORDS:
        if any(kw in t for kw in kws):
            return canon
    return "unknown"


# ---------------------------------------------------------------------------
# Canonical MCP tool vocabulary (data-derived, validated: only 6/~1800 tool
# mentions in train.json fail to normalize)
# ---------------------------------------------------------------------------
MCP_LABELS = [
    "nmap", "metasploit", "dirbuster", "john_the_ripper", "smb_client",
    "sqlmap", "hydra", "hashcat", "netexec", "git_dumper", "burp_suite",
    "ftp_client", "responder", "autopsy", "netcat", "google_search",
    "web_page_interaction", "interactive_cli",
]

_MCP_KEYWORDS = [
    ("nmap", ["nmap", "netdiscover"]),
    ("metasploit", ["metasploit", "msf"]),
    ("dirbuster", ["dirbuster", "gobuster"]),
    ("john_the_ripper", ["john-the-ripper", "john the ripper", "johntheripper"]),
    ("smb_client", ["smb client", "smbclient"]),
    ("sqlmap", ["sqlmap"]),
    ("hydra", ["hydra"]),
    ("hashcat", ["hashcat"]),
    ("netexec", ["netexec"]),
    ("git_dumper", ["git-dumper", "git dumper", "gitdumper"]),
    ("burp_suite", ["burp"]),
    ("ftp_client", ["ftp client", "ftp"]),
    ("responder", ["responder"]),
    ("autopsy", ["autopsy"]),
    ("netcat", ["netcat", " nc "]),
    ("google_search", ["google search"]),
    ("web_page_interaction", ["web page", "browse http", "login", "access "]),
    ("interactive_cli", ["interactive cli", "ssh", "bash", "shell", "cli"]),
]


def normalize_tool(name: str) -> Optional[str]:
    """Map a raw MCP tool-name string to a canonical MCP_LABELS entry, or None."""
    if not name:
        return None
    n = name.lower().strip()
    for canon, kws in _MCP_KEYWORDS:
        if any(kw in n for kw in kws):
            return canon
    return None


def parse_mcp_tasks(raw: str) -> dict:
    """
    Parse gold_mcp_tasks / a model's predicted MCP_tasks string into a
    {tool_name: description} dict. Handles two formats seen in the data:
      1. Python-dict-repr string:  "{'Tool': 'desc', 'Tool2': 'desc2'}"
      2. Semicolon-joined string:  "Tool: desc; Tool2: desc2"
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k).strip(): str(v).strip() for k, v in raw.items()}
    s = str(raw).strip()
    if not s:
        return {}
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return {str(k).strip(): str(v).strip() for k, v in obj.items()}
    except Exception:
        pass
    result = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            tool, desc = part.split(":", 1)
            tool, desc = tool.strip(), desc.strip()
            if tool:
                result[tool] = desc
    return result


def mcp_labels_from_dict(mcp_dict: dict) -> list[str]:
    """Canonicalize the tool-name keys of a parsed MCP dict."""
    out = []
    for tool in mcp_dict.keys():
        c = normalize_tool(tool)
        if c and c not in out:
            out.append(c)
    return out


def mcp_multihot(labels: list[str]) -> np.ndarray:
    vec = np.zeros(len(MCP_LABELS), dtype=int)
    for label in labels:
        if label in MCP_LABELS:
            vec[MCP_LABELS.index(label)] = 1
    return vec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_data(json_path: str) -> list[dict]:
    """
    Load train/test rows from the REAL stepmodelv3 schema and attach
    canonicalized labels used for reward computation and metrics.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        gold_step_raw = (item.get("gold_new_step") or "").strip()
        gold_expl = (item.get("gold_step_explanation") or "").strip()
        gold_mcp_dict = parse_mcp_tasks(item.get("gold_mcp_tasks", ""))

        examples.append({
            "machine": item.get("machine", ""),
            "graph_json": item.get("graph", {}),
            "candidate_step": (item.get("new_strategy") or "").strip(),
            "candidate_step_explanation": (item.get("strategy_explanation") or "").strip(),
            "gold_step_text": gold_step_raw,
            "gold_step_category": classify_step(gold_step_raw),
            "gold_step_explanation": gold_expl,
            "gold_mcp_dict": gold_mcp_dict,
            "gold_mcp_labels": mcp_labels_from_dict(gold_mcp_dict),
        })

    print(f"[data] Loaded {len(examples)} examples from {json_path}")
    return examples


# ---------------------------------------------------------------------------
# Completion parsing
# ---------------------------------------------------------------------------

def parse_completion(text: str) -> Optional[dict]:
    """Extract the first {...} JSON block from generated text."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text similarity (deterministic, sentence-transformer based)
# ---------------------------------------------------------------------------

_sentence_encoder = None


def get_sentence_encoder():
    global _sentence_encoder
    if _sentence_encoder is None:
        from sentence_transformers import SentenceTransformer
        print("[embed] Loading sentence-transformers/all-MiniLM-L6-v2 ...")
        _sentence_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sentence_encoder


def cosine_sim(a: str, b: str, min_score: float = 0.0) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    encoder = get_sentence_encoder()
    embs = encoder.encode([a[:512], b[:512]])
    cos = float(np.dot(embs[0], embs[1]) /
                (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8))
    cos = max(0.0, cos)
    return 0.0 if cos < min_score else cos


# ---------------------------------------------------------------------------
# Reward function used by GRPO training
# ---------------------------------------------------------------------------

def compute_reward(
    completion: str,
    gold: dict,
    w_fmt: float = 0.10,
    w_step: float = 0.35,
    w_mcp: float = 0.25,
    w_exp: float = 0.30,
) -> float:
    """
    Composite reward.
      - format:       valid JSON with the 3 required keys
      - step:         0.6 * (predicted category == gold category)
                     + 0.4 * cosine_sim(pred step text, gold step text)
                       -> rewards both getting the right kind of step AND
                          reproducing the graph-specific details (service name,
                          IP, etc.) that the coarse category collapses away.
      - mcp:          set-F1 between canonicalized predicted / gold tool sets
      - explanation:  cosine sim between predicted and gold explanation
    """
    obj = parse_completion(completion)
    if obj is None or not all(k in obj for k in ("New step", "Step explanation", "MCP_tasks")):
        return 0.0

    fmt_r = 1.0

    pred_step = str(obj.get("New step", "")).strip()
    pred_cat = classify_step(pred_step)
    cat_match = 1.0 if (pred_cat != "unknown" and pred_cat == gold["gold_step_category"]) else 0.0
    text_sim = cosine_sim(pred_step, gold["gold_step_text"])
    step_r = 0.6 * cat_match + 0.4 * text_sim

    mcp_val = obj.get("MCP_tasks", {})
    if isinstance(mcp_val, str):
        mcp_val = parse_mcp_tasks(mcp_val)
    pred_mcp = set(mcp_labels_from_dict(mcp_val)) if isinstance(mcp_val, dict) else set()
    gold_mcp = set(gold["gold_mcp_labels"])
    if not pred_mcp and not gold_mcp:
        mcp_r = 1.0
    else:
        inter = len(pred_mcp & gold_mcp)
        prec = inter / len(pred_mcp) if pred_mcp else 0.0
        rec = inter / len(gold_mcp) if gold_mcp else 0.0
        mcp_r = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    pred_expl = str(obj.get("Step explanation", "")).strip()
    exp_r = cosine_sim(pred_expl, gold["gold_step_explanation"], min_score=0.30)

    return w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * exp_r
