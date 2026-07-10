import ast
import difflib
import re
from typing import Iterable, Optional, Set

import numpy as np


STEP_LABELS = [
    "Do a google search for more information",
    "Enumerate further on the X service to find software versions, hidden directories and file.",
    "Explore the suspicious files, commands and create a summary of the findings.",
    "Further Enumerate the website. - hidden directories, links and software",
    "Enumerate the domain",
    "Exploit the selected exploitations",
    "Analyze the outcomes of the previous step and find an attack path",
    "Ask for human assistant",
    "Explore the source code for vulnerabilities.",
    "End task and ask permission to generate the report",
]


MCP_LABELS = [
    "Nmap",
    "Metasploit",
    "Netcat",
    "Dirbuster",
    "SQLmap",
    "Smb client",
    "hydra",
    "John-the-ripper",
    "Google search",
    "Interactive CLI",
    "Web page interaction",
]


PREFIX_N = 16
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    value = " ".join(str(text or "").strip().lower().split())
    if value in {"", "nan", "none", "null", "n/a"}:
        return ""
    return value


STEP_LABEL2ID_NORM = {_norm(s): i for i, s in enumerate(STEP_LABELS)}
STEP_LABEL2ID_PREFIX = {}
for i, label in enumerate(STEP_LABELS):
    STEP_LABEL2ID_PREFIX.setdefault(_norm(label)[:PREFIX_N], i)
STEP_ID2LABEL = {i: s for i, s in enumerate(STEP_LABELS)}

MCP_LABEL2ID = {_norm(s): i for i, s in enumerate(MCP_LABELS)}
MCP_ID2LABEL = {i: s for i, s in enumerate(MCP_LABELS)}


def step_label_to_id(raw_label: str) -> Optional[int]:
    norm = _norm(raw_label)
    if not norm:
        return None
    if norm in STEP_LABEL2ID_NORM:
        return STEP_LABEL2ID_NORM[norm]
    prefix_match = STEP_LABEL2ID_PREFIX.get(norm[:PREFIX_N])
    if prefix_match is not None:
        return prefix_match
    candidates = difflib.get_close_matches(norm, list(STEP_LABEL2ID_NORM.keys()), n=1, cutoff=0.72)
    return STEP_LABEL2ID_NORM[candidates[0]] if candidates else None


def normalize_step_label(raw_label: str) -> str:
    step_id = step_label_to_id(raw_label)
    return STEP_ID2LABEL[step_id] if step_id is not None else ""


def _split_mcp_candidates(text: str):
    if text is None:
        return []
    value = str(text).strip()
    if not value:
        return []
    parts = re.split(r"[\n,;|]+", value)
    out = []
    for part in parts:
        item = part.strip()
        if not item:
            continue
        if item.startswith("-"):
            item = item[1:].strip()
        item = re.sub(r"^\d+\)?\.?\s*", "", item).strip()
        if item:
            out.append(item)
    return out


def normalize_mcp_label(raw_label: str) -> str:
    norm = _norm(raw_label)
    if not norm:
        return ""
    if norm in MCP_LABEL2ID:
        return MCP_ID2LABEL[MCP_LABEL2ID[norm]]
    matches = difflib.get_close_matches(norm, list(MCP_LABEL2ID.keys()), n=1, cutoff=0.72)
    return MCP_ID2LABEL[MCP_LABEL2ID[matches[0]]] if matches else ""


def parse_mcp_tools(raw_value) -> Set[str]:
    if raw_value is None:
        return set()
    value = raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        try:
            value = ast.literal_eval(stripped)
        except Exception:
            value = raw_value

    tools = set()
    if isinstance(value, dict):
        iterable: Iterable = value.keys()
    elif isinstance(value, (list, tuple, set)):
        iterable = value
    else:
        iterable = _split_mcp_candidates(str(value))

    for item in iterable:
        if isinstance(item, str) and ":" in item:
            item = item.split(":", 1)[0].strip()
        canonical = normalize_mcp_label(item)
        if canonical:
            tools.add(canonical)
    return tools


def mcp_tools_to_multihot(tools: Set[str]) -> np.ndarray:
    multihot = np.zeros(len(MCP_LABELS), dtype=np.float32)
    for tool in tools:
        idx = MCP_LABEL2ID.get(_norm(tool))
        if idx is not None:
            multihot[idx] = 1.0
    return multihot


def raw_mcp_to_multihot(raw_value) -> np.ndarray:
    return mcp_tools_to_multihot(parse_mcp_tools(raw_value))


def step_id_to_label(step_id: int) -> str:
    return STEP_ID2LABEL[int(step_id)]


def multihot_to_mcp_tools(multihot, threshold: float = 0.5) -> Set[str]:
    arr = np.asarray(multihot, dtype=np.float32)
    return {MCP_ID2LABEL[i] for i, value in enumerate(arr) if value >= threshold}


def set_f1(pred_set: Set[str], true_set: Set[str]) -> float:
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    inter = len(pred_set & true_set)
    precision = inter / max(len(pred_set), 1)
    recall = inter / max(len(true_set), 1)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2.0 * precision * recall / denom


def classification_reward(
    pred_step_id: int,
    true_step_id: int,
    pred_mcp_multihot,
    true_mcp_multihot,
    threshold: float = 0.5,
) -> float:
    step_score = 1.0 if int(pred_step_id) == int(true_step_id) else 0.0
    pred_tools = multihot_to_mcp_tools(pred_mcp_multihot, threshold=threshold)
    true_tools = multihot_to_mcp_tools(true_mcp_multihot, threshold=0.5)
    mcp_score = set_f1(pred_tools, true_tools)
    mcp_exact = 1.0 if pred_tools == true_tools else 0.0
    both_exact = step_score * mcp_exact
    joint_score = step_score * mcp_score

    if not pred_tools and not true_tools:
        mcp_recall = 1.0
    elif not true_tools:
        mcp_recall = 0.0
    else:
        mcp_recall = len(pred_tools & true_tools) / len(true_tools)

    # Denser task-aligned reward:
    # - exact Step correctness remains primary
    # - MCP F1 gives partial credit
    # - MCP recall rewards finding the needed tools
    # - Step+MCP joint term encourages tool quality only when Step is right
    # - both_exact rewards the hardest target without harsh negative penalties
    reward = (
        0.30 * step_score
        + 0.25 * mcp_score
        + 0.20 * mcp_recall
        + 0.15 * joint_score
        + 0.10 * both_exact
    )
    return float(reward)


def token_set(text: str):
    return set(_WORD_RE.findall(_norm(text)))
