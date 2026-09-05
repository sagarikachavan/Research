"""
graph_adapter_experiments/common.py
====================================

Shared building blocks for the Graph Prefix Adapter reliability suite.

This module intentionally re-uses the REAL model classes and REAL loading
code from your pipeline (`Stage1Classifier`, `GraphPrefixAdapter`,
`build_prompt`, `SYSTEM_PROMPT`, `build_graph_from_input_json_graph`) rather
than reimplementing anything, so every experiment here is testing the exact
architecture you actually train and ship -- not a simplified stand-in.

Key architectural fact this whole suite is built around (verified by reading
graph_encoder.py / stage2_sft_qwen.py directly, not assumed):

    The "graph prefix tokens" the LLM sees are NOT a pure function of graph
    structure. `Stage1Classifier.encode_and_predict()` fuses the GNN's graph
    embedding `g` with a TEXT embedding `c` of the "New strategy" /
    "Strategy explanation" columns via cross-attention + learned gating,
    and only THAT fused vector `h` (dim = FUSION_HIDDEN // 2) is what
    `GraphPrefixAdapter` turns into the 8 soft-prompt tokens.

    This means any experiment that only swaps the graph while leaving the
    context text unchanged cannot, by itself, prove the LLM is reading
    graph structure -- the fused vector still carries the (unchanged) text
    signal through `c`. Every intervention in this suite therefore varies
    graph and context INDEPENDENTLY (a factorial design), so we can tell
    apart "the LLM used the graph" from "the LLM used the leftover text
    signal that rides along in the same fused vector".
"""
import os
import sys
import json
import random
import copy
from dataclasses import dataclass, field
from typing import Optional

# ── Path bootstrap: works whether this folder sits inside the original flat
# repo layout, or inside the restructured core/ data_prep/ training/ eval/
# layout -- adds whichever of these exist to sys.path. ─────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _sub in ("", "core", "data_prep", "training", "eval"):
    _p = os.path.join(_ROOT, _sub) if _sub else _ROOT
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ROOT, STEP_LABELS, MCP_LABELS, STAGE1_CKPT, STAGE2_ADAPTER_DIR,
    STAGE3_ADAPTER_DIR, QWEN_MODEL_NAME, GRAPH_PREFIX_TOKENS,
    FUSION_HIDDEN, RANDOM_SEED, INPUT_TRAIN_JSON, INPUT_TEST_JSON,
)
from data_utils import (
    build_graph_from_input_json_graph, load_from_input_json,
    _embed_texts, CONTEXT_COLUMNS,
)
from graph_encoder import Stage1Classifier

EXPERIMENTS_DIR = _THIS_DIR
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
TASKS_DIR = os.path.join(EXPERIMENTS_DIR, "structure_tasks")
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT, "checkpoints"))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TASKS_DIR, exist_ok=True)

GRAPH_PREFIX_SRC_DIM = FUSION_HIDDEN // 2  # matches stage2_sft_qwen.py exactly

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────
# Model loading -- mirrors eval/evaluate.py's eval_llm() loading path
# exactly, so the models under test here are byte-for-byte what your
# pipeline actually deploys, not a re-implementation that could silently
# diverge.
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    tokenizer: object
    llm: object                # PeftModel (base + LoRA), or plain base if no adapter
    stage1: object              # frozen Stage1Classifier
    graph_adapter: object       # GraphPrefixAdapter
    embed_layer: object
    device: str
    dtype: object
    name: str = "model"


def load_stage1(device: str):
    from graph_encoder import Stage1Classifier as _S1
    stage1 = _S1()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
    else:
        stage1.load_state_dict(ckpt)
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad_(False)
    return stage1


def load_model_for_eval(adapter_dir: str, device: Optional[str] = None,
                         name: Optional[str] = None) -> LoadedModel:
    """
    Load a Qwen + LoRA + GraphPrefixAdapter checkpoint the SAME way
    eval/evaluate.py does for `--model llm`. `adapter_dir` can be
    STAGE2_ADAPTER_DIR, STAGE3_ADAPTER_DIR, or any checkpoint directory
    produced by this suite's own training scripts (they save in the same
    format: HF adapter files + graph_adapter.pt).
    """
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from graph_encoder import Stage1Classifier as _S1

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, torch_dtype=dtype).to(device)
    try:
        llm = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True).eval()
    except Exception as e:
        print(f"[common] No LoRA adapter found / failed to load ({e}); using base model unmodified.")
        llm = base.eval()

    stage1 = load_stage1(device)

    from stage2_sft_qwen import GraphPrefixAdapter
    llm_hidden = llm.config.hidden_size if hasattr(llm, "config") else base.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).to(dtype)
    adapter_ckpt = os.path.join(adapter_dir, "graph_adapter.pt")
    if os.path.exists(adapter_ckpt):
        adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    else:
        print(f"[common] WARNING: no graph_adapter.pt in {adapter_dir} -- "
              f"using a randomly initialised adapter.")
    adapter.eval()

    return LoadedModel(
        tokenizer=tokenizer, llm=llm, stage1=stage1, graph_adapter=adapter,
        embed_layer=llm.get_input_embeddings(), device=device, dtype=dtype,
        name=name or os.path.basename(os.path.normpath(adapter_dir)),
    )


# ─────────────────────────────────────────────────────────────────────────
# Fused-embedding computation + INTERVENTIONS
#
# `compute_fused_embedding()` reproduces stage1.encode_and_predict() exactly
# (graph + context -> fused h). Every function below it takes that same
# call and deliberately corrupts/substitutes one or both of its inputs,
# so we can causally test what each component contributes.
# ─────────────────────────────────────────────────────────────────────────

def _field_embs_for_context(context: dict, device: str) -> torch.Tensor:
    arr = _embed_texts([context.get(c, "") or "empty" for c in CONTEXT_COLUMNS])
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)


def compute_fused_embedding(model: LoadedModel, graph_data, context: dict):
    """The real, unmodified path: real graph + real context -> fused embedding."""
    from torch_geometric.data import Batch as PyGBatch
    pyg_batch = PyGBatch.from_data_list([graph_data]).to(model.device)
    edge_attr = getattr(pyg_batch, "edge_attr", None)
    field_embs = _field_embs_for_context(context, model.device)
    with torch.no_grad():
        combined_emb, step_logits, mcp_logits = model.stage1.encode_and_predict(
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch, field_embs, edge_attr=edge_attr
        )
    return combined_emb, step_logits, mcp_logits


CONDITIONS = [
    "real",            # real graph + real context (normal operating condition)
    "wrong_graph",     # a DIFFERENT real graph + real context
    "wrong_context",   # real graph + a DIFFERENT real context
    "wrong_both",      # different graph AND different context
    "shuffled_nodes",  # real graph structure, node id/title strings permuted
    "zero",            # fused embedding replaced with 0 vector (no info at all)
    "noise",           # fused embedding replaced with matched-scale Gaussian noise
    "mean_prototype",  # fused embedding replaced with the mean over many graphs
]


def make_condition_embedding(model: LoadedModel, condition: str,
                              real_graph_data, real_context: dict,
                              decoy_graph_data=None, decoy_context=None,
                              prototype_embedding: Optional[torch.Tensor] = None,
                              rng: Optional[random.Random] = None):
    """
    Returns the fused embedding to feed into GraphPrefixAdapter for the given
    condition. `decoy_graph_data`/`decoy_context` must come from a DIFFERENT
    example than `real_graph_data`/`real_context` (the caller is responsible
    for sampling a decoy that is not the current example).
    """
    rng = rng or random
    if condition == "real":
        emb, _, _ = compute_fused_embedding(model, real_graph_data, real_context)
        return emb

    if condition == "wrong_graph":
        assert decoy_graph_data is not None
        emb, _, _ = compute_fused_embedding(model, decoy_graph_data, real_context)
        return emb

    if condition == "wrong_context":
        assert decoy_context is not None
        emb, _, _ = compute_fused_embedding(model, real_graph_data, decoy_context)
        return emb

    if condition == "wrong_both":
        assert decoy_graph_data is not None and decoy_context is not None
        emb, _, _ = compute_fused_embedding(model, decoy_graph_data, decoy_context)
        return emb

    if condition == "shuffled_nodes":
        shuffled = shuffle_node_identity(real_graph_data, rng)
        emb, _, _ = compute_fused_embedding(model, shuffled, real_context)
        return emb

    if condition == "zero":
        real_emb, _, _ = compute_fused_embedding(model, real_graph_data, real_context)
        return torch.zeros_like(real_emb)

    if condition == "noise":
        real_emb, _, _ = compute_fused_embedding(model, real_graph_data, real_context)
        std = real_emb.std().item() or 1.0
        return torch.randn_like(real_emb) * std

    if condition == "mean_prototype":
        assert prototype_embedding is not None
        return prototype_embedding.to(model.device).to(real_graph_data.x.dtype
                                                          if hasattr(real_graph_data, "x") else torch.float32)

    raise ValueError(f"Unknown condition: {condition}")


def shuffle_node_identity(graph_data, rng: random.Random):
    """
    Returns a COPY of graph_data with node feature rows permuted among nodes
    (structure/edge_index untouched). This dissociates "what the text prompt
    calls each node" from "what the GNN actually encoded for that position",
    since the prompt's node-name strings come from the ORIGINAL graph JSON,
    not from this permuted tensor. Used to test whether the LLM's answers
    track true structure or just the node-name strings it can already read
    directly out of the text prompt.
    """
    import torch as _t
    g = copy.copy(graph_data)
    n = g.x.shape[0]
    perm = list(range(n))
    rng.shuffle(perm)
    idx = _t.tensor(perm, dtype=_t.long)
    g.x = g.x[idx].clone()
    return g


# ─────────────────────────────────────────────────────────────────────────
# Prompting + generation
# ─────────────────────────────────────────────────────────────────────────

def format_legend(legend: dict) -> str:
    lines = [f"  {nid}: {title}" for nid, title in legend.items()]
    return "Nodes in this graph (id: short description):\n" + "\n".join(lines)


def format_task_prompt(item: dict) -> str:
    """
    Builds the user-facing question text for one structure_tasks.jsonl item.
    The legend (node id -> description) is always given in TEXT so the model
    has something to ground ids in; the actual answer (edges/types) is only
    ever derivable from the graph prefix tokens, never from this text.
    """
    task = item["task"]
    legend_txt = format_legend(item["legend"]) if item.get("legend") else ""

    if task == "adjacency":
        q = (f"{legend_txt}\n\nUsing the graph structure encoded in the tokens above "
             f"(not the list of node descriptions, which does not state connections), "
             f"which node ids are DIRECTLY connected to {item['query_node']}? "
             f"Answer with a JSON list of node ids, e.g. [\"n1\", \"n4\"]. "
             f"Use an empty list [] if none.")
    elif task == "node_type":
        q = (f"{legend_txt}\n\nBased on the graph structure encoded in the tokens above, "
             f"what TYPE of node is {item['query_node']}? Answer with exactly one of: "
             f"\"State\", \"Action\", \"Finding\".")
    elif task == "edge_type":
        au, av = item["query_edge"]
        q = (f"{legend_txt}\n\nBased on the graph structure encoded in the tokens above, "
             f"there is an edge between {au} and {av}. What TYPE is that edge? Answer with "
             f"exactly one of: \"StateTransition\", \"SearchUpdate\", \"TrackUpdate\", \"Prediction\".")
    elif task == "two_hop":
        q = (f"{legend_txt}\n\nUsing the graph structure encoded in the tokens above, "
             f"which node ids are reachable from {item['query_node']} in EXACTLY 2 hops "
             f"(not 1 hop, not 3+ hops)? Answer with a JSON list of node ids. "
             f"Use an empty list [] if none.")
    elif task == "graph_aggregate":
        q = ("Based ONLY on the graph structure encoded in the tokens above (you are not "
             "given a node list for this question), estimate the following about the WHOLE "
             "graph and answer with a JSON object with exactly these keys: "
             "\"node_count_bucket\" (one of bucket_0..bucket_4, roughly smallest to largest), "
             "\"edge_count_bucket\" (one of bucket_0..bucket_4), "
             "\"density_bucket\" (one of bucket_0..bucket_4, sparsest to densest), "
             "\"dominant_node_type\" (one of \"State\", \"Action\", \"Finding\").")
    else:
        raise ValueError(f"Unknown task: {task}")

    return build_structure_prompt(q)


def build_structure_prompt(question: str) -> str:
    """
    A minimal, task-neutral instruction wrapper. `node_id_map`, if given, is
    an {anonymized_id: ...} hint block telling the LLM which anonymized
    labels are in play for this graph (used by node-type / edge-type probes
    to prevent the LLM from reading the answer off a revealing ID string --
    see build_structure_tasks.py for why raw IDs like "state:bashed:r0:1.1"
    are a shortcut that has nothing to do with the soft-prompt tokens).
    """
    lines = [
        "You are given ONLY a graph representation as soft prompt tokens "
        "(no text description of the graph). Answer the structural question "
        "below using ONLY the graph representation. If you cannot determine "
        "the answer from the graph representation, say \"unknown\" rather "
        "than guessing.",
        "",
        f"Question: {question}",
        "",
        "Respond with ONLY a JSON object: {\"answer\": ...}. No other text.",
    ]
    return "\n".join(lines)


def generate_with_prefix(model: LoadedModel, prefix_embeds: torch.Tensor,
                          prompt_text: str, max_new_tokens: int = 150) -> str:
    ids = model.tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=700,
    ).input_ids.to(model.device)
    token_embeds = model.embed_layer(ids).to(model.dtype)
    inputs_embeds = torch.cat([prefix_embeds.to(model.dtype), token_embeds], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    return model.tokenizer.decode(out[0], skip_special_tokens=True)


def parse_json_answer(text: str):
    """Best-effort JSON object extraction, mirroring evaluate.py's parser."""
    import re
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            return json.loads(match.group())
        except Exception:
            continue
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────
# Stats helpers -- bootstrap CI + paired permutation test, used by
# analyze_results.py so every "condition A beats condition B" claim in the
# final report is backed by a number, not a single point estimate.
# ─────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(values.mean()), float(lo), float(hi))


def paired_permutation_test(a, b, n_perm: int = 5000, seed: int = RANDOM_SEED) -> float:
    """
    Two-sided paired permutation test on mean(a) - mean(b) for equal-length
    paired arrays (same items scored under two conditions). Returns a
    p-value: probability of seeing a difference this large under the null
    that condition labels don't matter for these items.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b) and len(a) > 0
    rng = np.random.default_rng(seed)
    diffs = a - b
    observed = diffs.mean()
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([1, -1], size=len(diffs))
        perm_stat = (diffs * signs).mean()
        if abs(perm_stat) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def machine_level_split(examples: list, val_frac: float = 0.2, seed: int = RANDOM_SEED):
    """
    Same pattern used throughout the rest of the pipeline (stage1/2/3):
    split by MACHINE, not by row, so no held-out graph shares a machine with
    a training graph. Returns (train_examples, held_out_examples).
    """
    machines = sorted(set(e["machine"] for e in examples))
    rng = random.Random(seed)
    rng.shuffle(machines)
    n_val = max(1, int(len(machines) * val_frac))
    val_machines = set(machines[:n_val])
    train = [e for e in examples if e["machine"] not in val_machines]
    held_out = [e for e in examples if e["machine"] in val_machines]
    return train, held_out


def load_all_examples():
    """All train+test examples pooled, each still tagged with its machine and
    original split, so downstream scripts can re-split however they need."""
    train = load_from_input_json(INPUT_TRAIN_JSON, "train")
    test = load_from_input_json(INPUT_TEST_JSON, "test")
    for e in train:
        e["_orig_split"] = "train"
    for e in test:
        e["_orig_split"] = "test"
    return train + test


def index_examples_by_key(examples: list) -> dict:
    """{(machine, row_id, split): example} for O(1) lookup from a
    structure_tasks.jsonl item back to its real graph Data + context."""
    return {(e["machine"], e["row_id"], e["_orig_split"]): e for e in examples}


def load_task_items(path: str) -> list:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def score_task_item(item: dict, pred) -> tuple:
    """
    Returns (score in [0,1], parsed_ok: bool). `pred` is whatever
    parse_json_answer(...).get("answer") produced (or None if parsing failed).
    """
    task = item["task"]
    gold = item["gold"]

    if pred is None:
        return 0.0, False

    if task in ("adjacency", "two_hop"):
        if not isinstance(pred, list):
            return 0.0, False
        pred_set = set(str(p) for p in pred)
        gold_set = set(gold)
        if not gold_set and not pred_set:
            return 1.0, True
        if not gold_set or not pred_set:
            return 0.0, True
        inter = len(pred_set & gold_set)
        prec = inter / len(pred_set)
        rec = inter / len(gold_set)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        return f1, True

    if task in ("node_type", "edge_type"):
        pred_s = str(pred).strip().lower()
        gold_s = str(gold).strip().lower()
        return (1.0 if pred_s == gold_s else 0.0), True

    if task == "graph_aggregate":
        if not isinstance(pred, dict):
            return 0.0, False
        keys = ["node_count_bucket", "edge_count_bucket", "density_bucket", "dominant_node_type"]
        matched = sum(
            1 for k in keys
            if str(pred.get(k, "")).strip().lower() == str(gold.get(k, "")).strip().lower()
        )
        return matched / len(keys), True

    raise ValueError(f"Unknown task: {task}")
