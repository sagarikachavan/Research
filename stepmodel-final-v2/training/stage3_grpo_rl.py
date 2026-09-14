"""
Stage 3: Custom GRPO (Group Relative Policy Optimization) with full graph
conditioning — the same GraphPrefixAdapter soft-prompt tokens used in Stage 2
are injected during every RL rollout, keeping the input distribution identical
to how the policy was trained in Stage 2.

WHY A CUSTOM LOOP INSTEAD OF trl.GRPOTrainer
---------------------------------------------
trl.GRPOTrainer drives generation through plain text token IDs.  It has no
hook to prepend arbitrary embedding tensors before the token sequence, so
using it forces us to drop the graph soft-prompt during RL rollouts.  That
shifts the input distribution relative to Stage 2 — the KL penalty (β=0.02)
is far too small to compensate, meaning Stage 3 effectively fine-tunes a
different model than what Stage 2 produced.

The custom loop is not complicated:
  1. For each example, build the graph-prefix embeddings (frozen GNN +
     trainable GraphPrefixAdapter) and prepend them to the token embeddings.
  2. Call model.generate() with inputs_embeds instead of input_ids.
  3. Score each of the G completions with the reward function.
  4. Compute group-relative advantages  A_i = (r_i - mean) / (std + ε).
  5. Re-run a forward pass with inputs_embeds for the generated tokens,
     compute per-token log-probs, apply the clipped policy-gradient loss,
     add a KL penalty against a frozen reference copy of Stage 2.
  6. Gradient update on LoRA weights + GraphPrefixAdapter weights.

Reward composition (see compute_reward() below for the exact, current weights):
  r = 0.01 × format_ok          — valid JSON with all 3 required keys
    + 0.33 × step_r              — exact match on the normalized predicted step
    + 0.33 × mcp_r                — Jaccard set-F1 between predicted and gold tools
    + 0.33 × exp_r                — deterministic explanation-quality score:
                                     0.60 * BGE cosine similarity
                                   + 0.20 * lexical (difflib) ratio
                                   + 0.10 * step-keyword support
                                   + 0.10 * predicted/gold tool-set overlap
                                   (see _deterministic_explanation_score() below).

WHY A DETERMINISTIC SCORE FOR EXPLANATION, NOT THE TEST-TIME LLM JUDGE:
  - The project's actual test-time explanation metric is core/llm_judge.py's
    4-dimension rubric gate (a separate Qwen model, used only by eval/evaluate.py).
    An earlier version of this file called that judge in-loop during RL (a GPT-4o
    call was never implemented here; the removed code called a local Qwen judge).
  - Using the same noisy, slow evaluator as both the RL reward AND the reported
    test metric risks the policy learning to game the judge's specific quirks
    rather than the underlying explanation quality (reward hacking against your
    own eval). The current deterministic proxy is cheap (reuses the project's
    frozen BGE encoder, no extra model forward pass) and reference-aware, and
    is deliberately kept separate from the test-time judge -- see compute_reward
    below and its docstring.

-----------------------------------------------------------------------------
FIX (this revision): completion-slicing bug when generating with inputs_embeds
-----------------------------------------------------------------------------
When `model.generate()` is called with ONLY `inputs_embeds` (no `input_ids`),
HF's `generate()` has no token-ID representation of the prompt to prepend to
its output, so the returned tensor contains ONLY the newly generated tokens
— it is NOT `[prompt_tokens | generated_tokens]` the way generation with
`input_ids` would be. The previous version of this file assumed the latter
and sliced `gen_out[:, L_prefix_plus_prompt:]`, which — since gen_out is
already shorter than the prompt length — produced an empty tensor on nearly
every step, causing "No valid completions" warnings almost every step.

The fix: treat `gen_out` itself as the completion batch, and simply trim the
per-row trailing pad tokens (rows are padded to a common length because
`num_return_sequences=G` generates a batch of sequences together).
"""
import json
import os
import random
import csv
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch as PyGBatch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Path bootstrap (folder was restructured into core/ data_prep/ training/ eval/) ──
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "core"), _os.path.join(_ROOT, "data_prep"), _os.path.join(_ROOT, "training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from config import (
    INPUT_TRAIN_JSON,
    INPUT_TEST_JSON,
    QWEN_MODEL_NAME,
    STAGE1_CKPT,
    STAGE2_ADAPTER_DIR,
    STAGE3_ADAPTER_DIR,
    STAGE3_GROUP_SIZE,
    STAGE3_LR,
    STAGE3_STEPS,
    STAGE3_KL_COEF,
    STAGE3_PPO_CLIP,
    STAGE3_USE_CLIP_HIGHER,
    STAGE3_CLIP_HIGH,
    STAGE3_GRAD_ACCUM,
    STAGE3_GRAD_CLIP,
    STAGE3_DUAL_CLIP_COEF,
    STAGE3_KL_HARD_CAP,
    STAGE3_EARLY_STOP_PATIENCE,
    RANDOM_SEED,
    STEP_LABELS,
    MCP_LABELS,
    ROOT,
    GRAPH_PREFIX_TOKENS,
    GNN_OUT_DIM,
    STAGE2_VAL_SPLIT,
)
from data_utils import load_from_input_json, _embed_texts, StepLabelNormalizer, extract_mcp_labels
from graph_encoder import Stage1Classifier
from stage2_sft_qwen import GraphPrefixAdapter, build_prompt, SYSTEM_PROMPT, build_obj_parser, GRAPH_PREFIX_SRC_DIM

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Reward function
#
# CLEANUP (architecture re-audit): removed a dead `ValueHead` class (never
# instantiated anywhere -- advantages here are GRPO's group-relative z-score,
# not a learned value baseline) and a dead in-loop LLM-judge path
# (`LLM_JUDGE_SYSTEM_PROMPT`, `_get_cache_key`, `set_llm_judge_model`,
# `_explanation_llm_judge_cached`): `set_llm_judge_model` was never called
# from anywhere in the repo, so `_llm_judge_model`/`_llm_judge_tokenizer` were
# always None and every call would have silently fallen through to the
# length-bucket heuristic branch, not an actual judge call -- confirmed dead,
# not merely unused. The module docstring above previously described this
# reward component as "GPT-4o" LLM-judge scoring, which was never true of any
# code in this file; the actual, live explanation reward is
# `_deterministic_explanation_score()` below.
# ---------------------------------------------------------------------------

def _parse_completion(text: str) -> dict | None:
    """Extract the first {...} JSON block from generated text."""
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


# Shared normalizer instance so the RL reward's step-correctness check uses
# EXACTLY the same canonicalization evaluate.py uses to compute the
# "Step Exact Match" metric this reward is meant to optimize toward.
_step_normalizer = StepLabelNormalizer()


_MCP_WEIGHT_CACHE = None


def _mcp_label_weights() -> dict:
    """
    Inverse-sqrt-frequency weights per MCP_LABELS, computed once from
    INPUT_TRAIN_JSON's gold_mcp_tasks and cached for the process lifetime.

    sqrt (not plain inverse-frequency) so a label with 10x fewer examples
    gets ~3x the weight, not 10x -- plain inverse-frequency over-corrects
    on a dataset this small (Netcat=14 vs Interactive CLI=152 support would
    otherwise imply an ~11x weight, which can make the reward gradient
    dominated by a handful of rare-tool examples and destabilize GRPO's
    group-relative advantage estimate). Weights are normalized to mean 1.0
    so the *overall scale* of w_mcp in compute_reward is unaffected --
    only the relative balance across labels shifts.

    Falls back to uniform weights (all 1.0) if the training file can't be
    read, so this never hard-fails a training run.
    """
    global _MCP_WEIGHT_CACHE
    if _MCP_WEIGHT_CACHE is not None:
        return _MCP_WEIGHT_CACHE
    counts = {l: 0 for l in MCP_LABELS}
    try:
        rows = load_from_input_json(INPUT_TRAIN_JSON)
        for row in rows:
            for l in extract_mcp_labels(str(row.get("gold_mcp_tasks", ""))):
                if l in counts:
                    counts[l] += 1
    except Exception:
        pass
    if sum(counts.values()) == 0:
        _MCP_WEIGHT_CACHE = {l: 1.0 for l in MCP_LABELS}
        return _MCP_WEIGHT_CACHE
    raw = {l: 1.0 / np.sqrt(c + 1.0) for l, c in counts.items()}
    mean_w = sum(raw.values()) / len(raw)
    _MCP_WEIGHT_CACHE = {l: v / mean_w for l, v in raw.items()}
    return _MCP_WEIGHT_CACHE


def _deterministic_explanation_score(pred_expl: str, gold_expl: str,
                                      pred_step: str = "", gold_step: str = "",
                                      pred_mcp: set[str] | None = None,
                                      gold_mcp: set[str] | None = None) -> float:
    """Cheap, reference-aware explanation reward for RL.

    Uses semantic sentence embeddings when available plus bounded technical
    consistency checks. It is deliberately not the LLM judge used at test
    time, preventing a noisy evaluator from becoming the optimization target.
    """
    from difflib import SequenceMatcher
    pred = str(pred_expl or "").strip()
    gold = str(gold_expl or "").strip()
    if not pred or not gold:
        return 0.0
    # lexical fallback is always available
    lexical = SequenceMatcher(None, pred.lower(), gold.lower()).ratio()
    # Reuse the project's BGE encoder if available; this is frozen and cheap
    # relative to Qwen.
    semantic = lexical
    try:
        emb = _embed_texts([pred, gold])
        a, b = emb[0], emb[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 0:
            semantic = float(np.dot(a, b) / denom)
            semantic = max(0.0, min(1.0, (semantic + 1.0) / 2.0))
    except Exception:
        pass
    # Technical consistency gates: the explanation should support the
    # selected step/tools rather than merely resemble the reference.
    text = pred.lower()
    step_support = 1.0 if pred_step and any(tok in text for tok in pred_step.lower().split()[:4]) else 0.5
    pred_mcp = pred_mcp or set()
    gold_mcp = gold_mcp or set()
    tool_support = (len(pred_mcp & gold_mcp) / len(gold_mcp)) if gold_mcp else (1.0 if not pred_mcp else 0.5)
    # Semantic similarity is primary; technical support is a bounded modifier.
    return float(max(0.0, min(1.0, 0.60 * semantic + 0.20 * lexical + 0.10 * step_support + 0.10 * tool_support)))


def compute_reward(completion: str, gold: dict,
                   w_fmt: float = 0.01,
                   w_step: float = 0.33,
                   w_mcp: float = 0.33,
                   w_exp: float = 0.33,
                   return_components: bool = False):
    """Equal-objective reward: Step, MCP, and explanation dominate.

    Raw component scales are intentionally kept in [0,1]. Stage-3 group
    advantages are normalized per objective (GDPO-style decoupled
    normalization) before the policy update, so a numerically noisy objective
    cannot dominate the other two.
    """
    obj = _parse_completion(completion)
    if obj is None or not all(k in obj for k in ("New step", "Step explanation", "MCP_tasks")):
        partial = 0.0 if obj is None else sum(k in obj for k in ("New step","Step explanation","MCP_tasks")) / 3.0
        out = {"total": w_fmt * partial, "fmt": partial, "step": 0.0, "mcp": 0.0, "exp": 0.0}
        return out if return_components else out["total"]

    pred_step = str(obj.get("New step", "")).strip()
    gold_step = str(gold["step_label"]).strip()
    step_r = 1.0 if _step_normalizer.normalize(pred_step) == gold_step else 0.0

    mcp_val = obj.get("MCP_tasks", {})
    pred_mcp = set(extract_mcp_labels(str(mcp_val))) if isinstance(mcp_val, dict) else set()
    gold_mcp = set(gold["mcp_labels"])
    union = pred_mcp | gold_mcp
    inter = pred_mcp & gold_mcp
    mcp_r = (len(inter) / len(union)) if union else 1.0

    exp_r = _deterministic_explanation_score(
        str(obj.get("Step explanation", "")), gold.get("gold_step_explanation", ""),
        pred_step, gold_step, pred_mcp, gold_mcp
    )
    fmt_r = 1.0
    total = w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * exp_r
    out = {"total": total, "fmt": fmt_r, "step": step_r, "mcp": mcp_r, "exp": exp_r}
    return out if return_components else total


def compute_reward_curriculum(completion: str, gold: dict, step_num: int,
                              total_steps: int = 2000, return_components: bool = False):
    # No curriculum weighting: all three research objectives remain equally important.
    return compute_reward(completion, gold, return_components=return_components)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def build_prefix_embeds(ex, stage1, adapter, device, dtype):
    """Build exactly the Stage-2 graph-prefix input from the frozen GINE.

    Contract:
      PTT graph -> frozen Stage-1 GINE -> 512-d graph embedding
      -> frozen Stage-2 GraphPrefixAdapter -> 8 Qwen soft tokens.

    The Stage-1 checkpoint itself is never passed to the adapter, and no
    Stage-1 classifier/fusion logits are used as a shortcut.
    """
    from torch_geometric.data import Batch as PyGBatch
    graph = PyGBatch.from_data_list([ex["graph"]]).to(device)
    with torch.no_grad():
        edge_attr = getattr(graph, "edge_attr", None)
        graph_emb = stage1.graph_encoder(
            graph.x, graph.edge_index, graph.batch, edge_attr=edge_attr
        )
    if graph_emb.shape[-1] != GNN_OUT_DIM:
        raise RuntimeError(
            f"Stage-1 GINE must produce {GNN_OUT_DIM} dims, got {graph_emb.shape[-1]}"
        )
    expected_dim = adapter.proj[0].in_features
    if expected_dim != GNN_OUT_DIM:
        raise RuntimeError(
            f"GraphPrefixAdapter expects {expected_dim} dims; expected raw GINE {GNN_OUT_DIM}."
        )
    return adapter(graph_emb.float()).to(dtype)


def build_prompt_embeds(prompt_text: str, tokenizer, embed_layer, prefix_embeds, device, dtype):
    """
    Tokenise prompt_text, embed the token IDs, then prepend prefix_embeds.

    Returns:
        inputs_embeds : (1, n_prefix + n_prompt, H)
        prompt_len    : total length (prefix + prompt tokens) — used to slice
                        out the generated portion later
    """
    ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,  # Don't truncate - let model handle full prompt
    ).input_ids.to(device)

    token_embeds = embed_layer(ids).to(dtype)             # (1, T_prompt, H)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # (1, n_prefix+T_prompt, H)
    return inputs_embeds, inputs_embeds.shape[1]


def trim_generated_row(row: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    """
    Trim a single generated row down to the "real" generated tokens.

    IMPORTANT: when `generate()` is called with ONLY `inputs_embeds` (no
    `input_ids`), the returned tensor contains ONLY the newly generated
    tokens — there is no prompt prefix to slice off. `num_return_sequences=G`
    does, however, pad all G rows in the batch to a common max length, so we
    still need to trim trailing pad tokens per row.

    Keeps tokens up to and including the first EOS token if present;
    otherwise keeps everything up to the last non-pad token.
    """
    ids = row.tolist()
    if eos_id in ids:
        idx = ids.index(eos_id)
        return row[: idx + 1]
    nonpad_positions = (row != pad_id).nonzero(as_tuple=True)[0]
    if len(nonpad_positions) == 0:
        return row[:0]
    last = nonpad_positions[-1].item()
    return row[: last + 1]


# ---------------------------------------------------------------------------
# Per-token log-prob of a completion given inputs_embeds prefix
# ---------------------------------------------------------------------------

def completion_logprobs(
    model,
    inputs_embeds: torch.Tensor,   # (1, L_prefix, H)  — includes BOTH graph prefix AND prompt tokens
    completion_ids: torch.Tensor,  # (1, L_gen)  — NEW tokens generated
    embed_layer,
    dtype,
    device,
) -> torch.Tensor:
    """
    Compute the sum of per-token log-probs for `completion_ids` given the
    prefix represented as `inputs_embeds`.

    IMPORTANT INDEXING:
    - inputs_embeds contains: [graph_prefix | prompt_tokens], total length = L_prefix
    - When we concatenate [inputs_embeds | completion_embeds], we get a sequence of
      length L_full = L_prefix + L_gen
    - For causal LM, position i in the sequence PREDICTS token at position i+1
    - The first completion token (completion_ids[:, 0]) is predicted FROM position L_prefix-1
      in the full sequence (the LAST token of the prompt/prefix)
    - logits[:, L_prefix-1] predicts completion_ids[:, 0]
    - logits[:, L_prefix + k - 1] predicts completion_ids[:, k] for k=0..L_gen-1
    - So we want logits[:, L_prefix-1 : L_prefix+L_gen-1]  (total L_gen positions)

    Returns scalar tensor (grad-enabled).
    """
    comp_embeds = embed_layer(completion_ids).to(dtype)          # (1, L_gen, H)
    full_embeds = torch.cat([inputs_embeds, comp_embeds], dim=1) # (1, L_prefix+L_gen, H)

    L_prefix = inputs_embeds.shape[1]
    L_gen = completion_ids.shape[1]

    attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)
    out  = model(inputs_embeds=full_embeds, attention_mask=attn)  # no labels → no loss
    logits = out.logits  # (1, L_prefix+L_gen, V)

    # CRITICAL: slice [L_prefix-1 : L_prefix+L_gen-1] to get exactly the positions
    # that predict the L_gen completion tokens (one per position):
    comp_logits = logits[:, L_prefix - 1 : L_prefix + L_gen - 1, :]  # (1, L_gen, V)
    log_probs   = F.log_softmax(comp_logits, dim=-1)                 # (1, L_gen, V)
    token_lp    = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)  # (1, L_gen)

    return token_lp.sum()  # scalar



# ---------------------------------------------------------------------------
# Gold-target SFT anchor
# ---------------------------------------------------------------------------

def completion_nll(model, prompt_embeds: torch.Tensor, target_ids: torch.Tensor,
                    embed_layer, dtype, device) -> torch.Tensor:
    """Mean teacher-forced NLL on the gold target only.

    This is a small Stage-2 anchor used alongside GRPO.  It prevents sparse
    group-relative rewards from pushing a strong SFT policy away from the
    learned answer distribution.  Prompt tokens are never included in the
    loss; only gold completion tokens contribute.
    """
    if target_ids.numel() == 0:
        return torch.zeros((), device=device)
    target_ids = target_ids.view(1, -1).to(device)
    target_embeds = embed_layer(target_ids).to(dtype)
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)
    Lp = prompt_embeds.shape[1]
    Lt = target_ids.shape[1]
    attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)
    out = model(inputs_embeds=full_embeds, attention_mask=attn)
    logits = out.logits[:, Lp - 1:Lp + Lt - 1, :]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))


def gold_target_text(ex: dict) -> str:
    """Canonical gold JSON target matching Stage-2's output contract."""
    return json.dumps({
        "New step": ex["step_label"],
        "Step explanation": ex.get("gold_step_explanation", ""),
        "MCP_tasks": {k: True for k in ex.get("mcp_labels", [])},
    }, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def evaluate_policy_on_val(policy, adapter, stage1, embed_layer, tokenizer,
                            val_examples, device, dtype, max_examples: int = 96) -> dict:
    """
    Greedy-decode the current policy on a capped sample of the held-out
    (machine-level) validation set and score it with the *fixed* (non-curriculum)
    `compute_reward` weights, so the number is comparable across the whole run
    and against the Stage-2 starting point.

    THIS IS THE MODEL-SELECTION SIGNAL THAT WAS MISSING BEFORE: previously
    Stage 3 only logged reward on the single training example sampled that
    step and saved whatever the LAST step happened to produce, with nothing
    checking whether that was actually better than where RL started. Because
    the reward is noisy (LLM-judge component, small per-step group size,
    non-stationary curriculum weights), the last step is not reliably the
    best step -- which is consistent with Stage 3 finishing statistically
    indistinguishable from (or slightly worse than) Stage 2 in practice.

    Returns a dict of mean component scores {"total","fmt","step","mcp","exp"}
    instead of a single scalar. The scalar `total` blends step-label
    correctness (25-35%) with an LLM-judge explanation score (35-45%), so a
    single-number gate can promote a checkpoint that improved MCP/explanation
    quality while quietly trading away step accuracy. The "step" component is
    tracked separately so the caller can require step performance not to
    regress before promoting a checkpoint (see the model-selection block in
    `main()`).

    ALSO returns the detailed Step/MCP metric suite requested for periodic
    Stage-3 validation (not just the end-of-run test report): step_micro_f1
    (mathematically identical to step exact-match accuracy for single-label
    classification -- sklearn's average="micro" F1 over a single-label task
    reduces to accuracy, since every example contributes exactly one
    predicted and one gold label from the same space), mcp_exact_match,
    mcp_micro_f1/precision/recall (pooled TP/FP/FN across the whole sample,
    matching eval/evaluate.py's definition), and avg_missing/extra_mcp_tools.
    All computed from the same greedy-decoded completions already generated
    for the reward above -- no extra generation calls.
    """
    was_training = policy.training
    policy.eval()
    rng = random.Random(RANDOM_SEED)
    sample = val_examples if len(val_examples) <= max_examples else rng.sample(val_examples, max_examples)

    components = {"total": [], "fmt": [], "step": [], "mcp": [], "exp": []}
    step_exact = []
    mcp_jaccard = []
    mcp_pass = []
    mcp_exact = []
    missing_counts = []
    extra_counts = []
    mcp_tp = mcp_fp = mcp_fn = 0
    with torch.no_grad():
        for ex in sample:
            gold = {
                "step_label": ex["step_label"],
                "mcp_labels": ex["mcp_labels"],
                "gold_step_explanation": ex["gold_step_explanation"],
            }
            prefix_embeds = build_prefix_embeds(
                ex, stage1, adapter, device, dtype
            )
            prompt_text = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{build_prompt(ex)}\n"
                f"<|assistant|>\n"
            )
            prompt_embeds, L_prefix_plus_prompt = build_prompt_embeds(
                prompt_text, tokenizer, embed_layer, prefix_embeds, device, dtype
            )
            attn_prompt = torch.ones(1, L_prefix_plus_prompt, dtype=torch.long, device=device)

            gen_out = policy.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_prompt,
                max_new_tokens=300,
                do_sample=False,       # greedy -- deterministic model-selection signal
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = trim_generated_row(gen_out[0], tokenizer.eos_token_id, tokenizer.pad_token_id)
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            # Fixed weights (not the curriculum schedule) so scores are
            # comparable at step 0, step 1000, and step 3000 alike.
            comp = compute_reward(completion_text, gold, return_components=True)
            for k in components:
                components[k].append(comp[k])
            step_exact.append(comp["step"])
            mcp_jaccard.append(comp["mcp"])
            mcp_pass.append(1.0 if comp["mcp"] >= 0.5 else 0.0)

            # Detailed MCP tool-set breakdown, parsed the same way
            # compute_reward's own MCP scoring does (extract_mcp_labels on
            # the parsed "MCP_tasks" object) so this never drifts from the
            # reward's own notion of "predicted tools".
            obj = _parse_completion(completion_text)
            mcp_val = obj.get("MCP_tasks", {}) if obj else {}
            pred_mcp_set = set(extract_mcp_labels(str(mcp_val))) if isinstance(mcp_val, dict) else set()
            gold_mcp_set = set(gold["mcp_labels"])
            missing = gold_mcp_set - pred_mcp_set
            extra = pred_mcp_set - gold_mcp_set
            missing_counts.append(len(missing))
            extra_counts.append(len(extra))
            mcp_exact.append(1.0 if pred_mcp_set == gold_mcp_set else 0.0)
            mcp_tp += len(pred_mcp_set & gold_mcp_set)
            mcp_fp += len(extra)
            mcp_fn += len(missing)

    if was_training:
        policy.train()
    mcp_micro_precision = mcp_tp / max(1, mcp_tp + mcp_fp)
    mcp_micro_recall = mcp_tp / max(1, mcp_tp + mcp_fn)
    mcp_micro_f1 = (
        2 * mcp_micro_precision * mcp_micro_recall / max(1e-9, mcp_micro_precision + mcp_micro_recall)
        if (mcp_micro_precision + mcp_micro_recall) > 0 else 0.0
    )
    return {**{k: (float(np.mean(v)) if v else 0.0) for k, v in components.items()},
            "step_exact": float(np.mean(step_exact)) if step_exact else 0.0,
            "step_micro_f1": float(np.mean(step_exact)) if step_exact else 0.0,
            "mcp_jaccard": float(np.mean(mcp_jaccard)) if mcp_jaccard else 0.0,
            "mcp_pass": float(np.mean(mcp_pass)) if mcp_pass else 0.0,
            "mcp_exact_match": float(np.mean(mcp_exact)) if mcp_exact else 0.0,
            "mcp_micro_f1": float(mcp_micro_f1),
            "mcp_micro_precision": float(mcp_micro_precision),
            "mcp_micro_recall": float(mcp_micro_recall),
            "avg_missing_mcp_tools": float(np.mean(missing_counts)) if missing_counts else 0.0,
            "avg_extra_mcp_tools": float(np.mean(extra_counts)) if extra_counts else 0.0}


def _save_policy_snapshot(policy, adapter, value_head, tokenizer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    policy.save_pretrained(out_dir)
    torch.save(adapter.state_dict(), os.path.join(out_dir, "graph_adapter.pt"))
    torch.save(value_head.state_dict(), os.path.join(out_dir, "value_head.pt"))
    tokenizer.save_pretrained(out_dir)


def main():
    """Conservative GRPO fine-tuning anchored to the Stage-2 policy.

    The previous implementation had three important problems:
      1. PPO ratios were computed against the frozen reference model rather
         than the rollout (old) policy.  That is not the GRPO/PPO objective.
      2. The RL reward could be dominated by explanation heuristics and the
         validation score was measured on only 96 examples, so a noisy subset
         could promote a checkpoint that later lost badly on the 268-example
         test set.
      3. Stage-3 was allowed to modify the very large graph-prefix adapter.
         Stage 2 already learned this mapping; changing it during RL makes it
         easy to destroy graph conditioning while the language model reward
         still looks good.

    This version therefore:
      * starts exactly from Stage 2;
      * consumes the Stage-1 raw 512-d GINE representation through the frozen
        graph-prefix adapter; the private Stage-1 fusion/classifiers never enter RL;
      * freezes the graph-prefix adapter by default;
      * uses the actual GRPO group-relative advantage;
      * uses rollout-policy log-probabilities for the PPO ratio;
      * uses an equal-objective reward: 33% Step + 33% MCP Jaccard + 33%
        explanation + 1% format; the optimization-time explanation reward is
        deterministic/reference-aware rather than the test-time LLM judge;
      * adds a small supervised Stage-2 target anchor to prevent reward drift;
      * rejects near-zero-variance groups instead of learning from noise;
      * evaluates the FULL 239-example machine-held-out validation set;
      * only promotes a checkpoint if both Step and MCP improve over Stage 2;
      * otherwise copies Stage 2 forward unchanged.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # These CAN be overridden per-run via env vars without editing config.py,
    # but the fallback default (when no env var is set) now comes from
    # config.py's corresponding STAGE3_* constant instead of an independently
    # hardcoded literal -- architecture re-audit found all ten STAGE3_*/GRPO
    # constants in config.py were imported above but silently never read
    # again, so tuning config.py had zero effect on a real run. This restores
    # config.py as the actual source of truth with NO change to today's
    # values (every default below matches what was already hardcoded here,
    # except SAFE_GRAD_CLIP/SAFE_DUAL_CLIP/SAFE_KL_HARD_CAP -- see below).
    G = int(os.environ.get("STAGE3_SAFE_GROUP_SIZE", str(STAGE3_GROUP_SIZE)))
    SAFE_LR = float(os.environ.get("STAGE3_SAFE_LR", str(STAGE3_LR)))
    SAFE_STEPS = int(os.environ.get("STAGE3_SAFE_STEPS", str(STAGE3_STEPS)))
    SAFE_KL = float(os.environ.get("STAGE3_SAFE_KL", str(STAGE3_KL_COEF)))
    SAFE_CLIP = float(os.environ.get("STAGE3_SAFE_CLIP", str(STAGE3_PPO_CLIP)))
    # DAPO Clip-Higher (see config.py's STAGE3_USE_CLIP_HIGHER comment): an
    # asymmetric upper PPO clip bound, wider than SAFE_CLIP, to avoid
    # prematurely capping the update for a completion whose probability
    # should increase a lot. Falls back to SAFE_CLIP (symmetric clipping)
    # when disabled.
    SAFE_USE_CLIP_HIGHER = os.environ.get(
        "STAGE3_SAFE_USE_CLIP_HIGHER", str(STAGE3_USE_CLIP_HIGHER)
    ).lower() in ("1", "true", "yes")
    SAFE_CLIP_HIGH = float(os.environ.get("STAGE3_SAFE_CLIP_HIGH", str(STAGE3_CLIP_HIGH)))
    SAFE_ACCUM = int(os.environ.get("STAGE3_SAFE_GRAD_ACCUM", str(STAGE3_GRAD_ACCUM)))
    SAFE_PATIENCE = int(os.environ.get("STAGE3_SAFE_PATIENCE", str(STAGE3_EARLY_STOP_PATIENCE)))
    # Grad-norm clip: config.py had drifted to 1.0 while the training loop
    # below hardcoded 0.5 -- config.py's value is now corrected to 0.5 (the
    # value actually exercised by every real run so far), so this env-var
    # override is truly optional rather than silently ignored either way.
    SAFE_GRAD_CLIP = float(os.environ.get("STAGE3_SAFE_GRAD_CLIP_NORM", str(STAGE3_GRAD_CLIP)))
    # Dual-clip PPO coefficient and per-micro-batch KL hard cap: config.py
    # documents both as the fix for a real observed pg_loss/KL explosion (see
    # the STAGE3_DUAL_CLIP_COEF/STAGE3_KL_HARD_CAP comments there) but neither
    # was actually wired into the loss below until this pass -- see the PPO
    # loss and KL-cap sections further down.
    SAFE_DUAL_CLIP = float(os.environ.get("STAGE3_SAFE_DUAL_CLIP_COEF", str(STAGE3_DUAL_CLIP_COEF)))
    SAFE_KL_HARD_CAP = float(os.environ.get("STAGE3_SAFE_KL_HARD_CAP", str(STAGE3_KL_HARD_CAP)))
    EVAL_EVERY = int(os.environ.get("STAGE3_SAFE_EVAL_EVERY", "200"))
    VAL_MAX = int(os.environ.get("STAGE3_SAFE_VAL_MAX", "239"))
    MAX_NEW_TOKENS = int(os.environ.get("STAGE3_SAFE_MAX_NEW_TOKENS", "260"))
    TRAIN_ADAPTER = False  # Research contract: Stage-2 GraphPrefixAdapter is frozen throughout Stage 3.
    SFT_ANCHOR = float(os.environ.get("STAGE3_SFT_ANCHOR", "0.20"))

    print(f"[Stage 3] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 3] Device         : {device}")
    print(f"[Stage 3] Total steps    : {SAFE_STEPS}")
    print(f"[Stage 3] Group size (G) : {G}")
    print(f"[Stage 3] KL coef        : {SAFE_KL}")
    print(f"[Stage 3] PPO clip eps   : {SAFE_CLIP}"
          + (f" (clip-higher: upper={SAFE_CLIP_HIGH})" if SAFE_USE_CLIP_HIGHER else " (symmetric)"))
    print(f"[Stage 3] LR             : {SAFE_LR:.2e}")
    print(f"[Stage 3] Grad accum     : {SAFE_ACCUM}")
    print(f"[Stage 3] Grad clip norm : {SAFE_GRAD_CLIP}")
    print(f"[Stage 3] Dual-clip coef : {SAFE_DUAL_CLIP}")
    print(f"[Stage 3] KL hard cap    : {SAFE_KL_HARD_CAP}")
    print(f"[Stage 3] Train graph adapter during RL: {TRAIN_ADAPTER}")
    print(f"[Stage 3] Stage-2 supervised anchor weight: {SFT_ANCHOR:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------- Stage-2 policy --------------------------
    print(f"\n[Stage 3] Loading policy base model: {QWEN_MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    policy = PeftModel.from_pretrained(base, STAGE2_ADAPTER_DIR, is_trainable=True)
    policy.train()

    # ---------------------- Frozen Stage-2 reference --------------------
    print("[Stage 3] Loading frozen Stage-2 reference model")
    ref_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    ref_base.config.use_cache = False
    ref_model = PeftModel.from_pretrained(ref_base, STAGE2_ADAPTER_DIR, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ------------------------- Frozen Stage-1 ----------------------------
    print(f"[Stage 3] Loading Stage-1 GNN checkpoint: {STAGE1_CKPT}")
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
        be = ckpt.get("best_epoch", "?")
        bs = ckpt.get("best_score", "?")
        print(f"[Stage 3]   loaded (epoch={be}, score={bs:.4f})" if isinstance(bs, (float, int)) else
              f"[Stage 3]   loaded (epoch={be}, score={bs})")
    else:
        stage1.load_state_dict(ckpt)
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    # ------------------------- Prefix adapter ---------------------------
    llm_hidden = policy.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).float()
    adapter_ckpt = os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt")
    if not os.path.isfile(adapter_ckpt):
        raise FileNotFoundError(f"Stage-2 graph adapter not found: {adapter_ckpt}")
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device, weights_only=False))
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    print(f"[Stage 3] GraphPrefixAdapter source dim: {GRAPH_PREFIX_SRC_DIM}")
    print("[Stage 3] ✓ Loaded Stage-2 GraphPrefixAdapter")

    embed_layer = policy.get_input_embeddings()
    ref_embed_layer = ref_model.get_input_embeddings()

    # ---------------------------- Data split ----------------------------
    all_examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
    print(f"[Stage 3] Total labeled examples loaded: {len(all_examples)}")
    machine_order = sorted(set(e["machine"] for e in all_examples))
    rng_split = np.random.default_rng(RANDOM_SEED + 1)
    perm = rng_split.permutation(len(machine_order))
    n_val_machines = max(1, int(len(machine_order) * STAGE2_VAL_SPLIT))
    val_machines = {machine_order[i] for i in perm[:n_val_machines]}
    train_machines = set(machine_order) - val_machines
    train_examples = [e for e in all_examples if e["machine"] in train_machines]
    val_examples = [e for e in all_examples if e["machine"] in val_machines]
    print(f"[Stage 3] RL training on {len(train_examples)} examples, {len(train_machines)} machines")
    print(f"[Stage 3] Held-out val set: {len(val_examples)} examples, {len(val_machines)} machines")

    test_pre = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = {e["machine"] for e in test_pre}
    if train_machines & test_machines or val_machines & test_machines:
        raise RuntimeError("Stage 3 machine split overlaps the test set; refusing to train/evaluate.")
    print("[Stage 3] ✓ No machine overlap between train/val and test")
    del test_pre

    # Gentle inverse-sqrt class balancing.  Do not let rare classes dominate.
    step_counts = np.bincount([e["step_idx"] for e in train_examples], minlength=len(STEP_LABELS)).astype(float)
    safe_counts = np.maximum(step_counts, 1.0)
    class_w = 1.0 / np.sqrt(safe_counts)
    class_w = np.clip(class_w, class_w.max() / 4.0, class_w.max())
    sample_weights = np.asarray([class_w[e["step_idx"]] for e in train_examples], dtype=np.float64)

    # ------------------------- Output management ------------------------
    os.makedirs(STAGE3_ADAPTER_DIR, exist_ok=True)
    BEST_DIR = os.path.join(STAGE3_ADAPTER_DIR, "best")
    if os.path.isdir(BEST_DIR):
        shutil.rmtree(BEST_DIR)
    for name in os.listdir(STAGE3_ADAPTER_DIR):
        if name.startswith("step_"):
            path = os.path.join(STAGE3_ADAPTER_DIR, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    # ------------------------ Validation baseline ----------------------
    print(f"\n[Stage 3] Scoring Stage-2 starting checkpoint on FULL held-out val set ({min(VAL_MAX, len(val_examples))} examples)")
    baseline = evaluate_policy_on_val(
        policy, adapter, stage1, embed_layer, tokenizer,
        val_examples, device, dtype, max_examples=VAL_MAX
    )
    baseline_step = float(baseline["step_exact"])
    baseline_mcp = float(baseline["mcp_jaccard"])
    baseline_exp = float(baseline["exp"])
    baseline_score = (baseline_step + baseline_mcp + baseline_exp) / 3.0
    print(f"[Stage 3] Stage-2 baseline: task={baseline_score:.4f} | "
          f"step={baseline_step:.4f} | mcpJ={baseline_mcp:.4f} | exp={baseline_exp:.4f}")
    print(f"[Stage 3] Stage-2 baseline (detailed): step_microF1={baseline['step_micro_f1']:.4f} | "
          f"mcp_exact={baseline['mcp_exact_match']:.4f} | mcp_microF1={baseline['mcp_micro_f1']:.4f} | "
          f"mcp_P={baseline['mcp_micro_precision']:.4f} | mcp_R={baseline['mcp_micro_recall']:.4f} | "
          f"avg_missing={baseline['avg_missing_mcp_tools']:.3f} | avg_extra={baseline['avg_extra_mcp_tools']:.3f}")

    best_score = baseline_score
    best_step_metric = baseline_step
    best_mcp_metric = baseline_mcp
    best_exp_metric = baseline_exp
    best_step = 0
    no_improve = 0

    # ----------------------------- Optimizer ----------------------------
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if TRAIN_ADAPTER:
        trainable += [p for p in adapter.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No Stage-3 trainable parameters found.")
    print(f"[Stage 3] Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    optimizer = AdamW(
        trainable, lr=SAFE_LR, weight_decay=0.01,
        betas=(0.9, 0.95), eps=1e-8, foreach=False
    )
    total_updates = max(1, SAFE_STEPS // SAFE_ACCUM)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_updates)
    optimizer.zero_grad(set_to_none=True)

    # ---------------------------- GRPO loop -----------------------------
    kl_skipped = 0
    applied = 0
    consecutive_zero_var = 0

    for step in range(1, SAFE_STEPS + 1):
        ex = random.choices(train_examples, weights=sample_weights.tolist(), k=1)[0]
        gold = {
            "step_label": ex["step_label"],
            "mcp_labels": ex["mcp_labels"],
            "gold_step_explanation": ex.get("gold_step_explanation", ""),
        }

        prefix_embeds = build_prefix_embeds(ex, stage1, adapter, device, dtype)
        prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{build_prompt(ex)}\n"
            f"<|assistant|>\n"
        )
        prompt_embeds, prompt_len = build_prompt_embeds(
            prompt_text, tokenizer, embed_layer, prefix_embeds, device, dtype
        )
        target_ids = tokenizer(
            gold_target_text(ex), return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=MAX_NEW_TOKENS
        ).input_ids.to(device)
        attn_prompt = torch.ones(1, prompt_len, dtype=torch.long, device=device)

        # Dynamic sampling: if all candidates have effectively identical task
        # reward, retry at a slightly higher temperature.  If diversity is
        # still absent, skip the update rather than injecting a fake gradient.
        chosen_ids = None
        chosen_text = None
        rewards_np = None
        for attempt, temp in enumerate((0.70, 0.82, 0.95)):
            policy.eval()
            with torch.no_grad():
                gen_out = policy.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=attn_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    num_return_sequences=G,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            ids = [trim_generated_row(gen_out[i], tokenizer.eos_token_id, tokenizer.pad_token_id)
                   for i in range(gen_out.shape[0])]
            texts = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in ids]
            rs = np.asarray([
                float(compute_reward(t, gold))
                for t in texts
            ], dtype=np.float32)
            chosen_ids, chosen_text, rewards_np = ids, texts, rs
            if float(rs.std()) >= 0.03:
                break

        if rewards_np is None or float(rewards_np.std()) < 0.03:
            consecutive_zero_var += 1
            if step % 50 == 0:
                print(f"[Stage 3] step {step:4d}: reward std={0.0 if rewards_np is None else rewards_np.std():.4f}; skipping low-information group")
            if consecutive_zero_var >= 100:
                print("[Stage 3] Too many consecutive zero-variance groups; stopping safely.")
                break
            continue
        consecutive_zero_var = 0

        # GDPO-style decoupled normalization: normalize Step, MCP and
        # explanation rewards independently inside each rollout group, then
        # average their standardized advantages with equal importance.
        component_rows = [compute_reward(t, gold, return_components=True) for t in chosen_text]
        comp_adv = []
        for key in ("step", "mcp", "exp"):
            vals = torch.tensor([float(r[key]) for r in component_rows],
                                dtype=torch.float32, device=device)
            mu = vals.mean()
            sd = vals.std(unbiased=False)
            z = torch.zeros_like(vals) if float(sd) < 1e-6 else (vals - mu) / (sd + 1e-8)
            comp_adv.append(z)
        advantages = (comp_adv[0] + comp_adv[1] + comp_adv[2]) / 3.0
        advantages = advantages.clamp(-3.0, 3.0)
        rewards = torch.tensor([float(r["total"]) for r in component_rows],
                               dtype=torch.float32, device=device)

        # Rollout-policy log-probability is the PPO/GRPO denominator.  This is
        # computed BEFORE the backward pass and detached from the graph.
        old_lps = []
        with torch.no_grad():
            for ids in chosen_ids:
                if ids.numel() == 0:
                    old_lps.append(torch.tensor(0.0, device=device))
                    continue
                lp = completion_logprobs(
                    policy, prompt_embeds.detach(), ids.unsqueeze(0).to(device),
                    embed_layer, dtype, device
                )
                old_lps.append(lp / max(1, int(ids.numel())))

        policy.train()

        loss_sum = torch.zeros((), device=device)
        kl_sum = 0.0
        valid = 0

        for i, ids in enumerate(chosen_ids):
            if ids.numel() == 0:
                continue
            ids = ids.unsqueeze(0).to(device)
            valid += 1
            adv = advantages[i]

            new_lp = completion_logprobs(
                policy, prompt_embeds, ids, embed_layer, dtype, device
            ) / max(1, int(ids.shape[1]))
            old_lp = old_lps[i]

            # Correct PPO ratio: current policy / rollout (old) policy.
            log_ratio = torch.clamp(new_lp - old_lp, -4.0, 4.0)
            ratio = torch.exp(log_ratio)
            # DAPO Clip-Higher (see config.py's STAGE3_USE_CLIP_HIGHER
            # comment): widen only the upper clip bound so a completion
            # whose probability should increase a lot isn't prematurely
            # capped -- the lower bound (downweighting a bad completion)
            # is unaffected. Falls back to symmetric SAFE_CLIP when off.
            clip_hi = SAFE_CLIP_HIGH if SAFE_USE_CLIP_HIGHER else SAFE_CLIP
            clipped_ratio = torch.clamp(ratio, 1.0 - SAFE_CLIP, 1.0 + clip_hi)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            clip_obj = torch.minimum(surr1, surr2)
            # Dual-clip PPO (Ye et al. 2020; see config.py's
            # STAGE3_DUAL_CLIP_COEF comment for the real pg_loss-explosion
            # incident -- pg_loss 127->8255, val reward 0.454->0.29 -- this
            # fixes). For a NEGATIVE-advantage sample whose ratio has drifted
            # far above 1, single-clip's min(surr1, surr2) does not bound the
            # objective from below -- only the "good news" direction is
            # capped, so a single exploding-ratio, negative-advantage
            # completion can dominate the whole batch loss. Floor the
            # objective at SAFE_DUAL_CLIP * adv (adv < 0 here, SAFE_DUAL_CLIP
            # > 1, so this floor is always looser than the raw unclipped
            # surr1 could otherwise fall to as ratio -> exp(4) ~ 55).
            dual_clip_obj = torch.maximum(clip_obj, SAFE_DUAL_CLIP * adv)
            obj = torch.where(adv < 0, dual_clip_obj, clip_obj)
            pg = -obj

            # KL anchor to the actual Stage-2 policy.  This is separate from
            # the PPO denominator; conflating the two was a major bug before.
            with torch.no_grad():
                ref_lp = completion_logprobs(
                    ref_model, prompt_embeds.detach(), ids,
                    ref_embed_layer, dtype, device
                ) / max(1, int(ids.shape[1]))
            delta_ref = torch.clamp(new_lp - ref_lp, -4.0, 4.0)
            # Non-negative sampled KL approximation: exp(delta)-delta-1.
            kl = torch.clamp(torch.exp(delta_ref) - delta_ref - 1.0, min=0.0, max=4.0)
            kl_sum += float(kl.detach().item())

            loss_sum = loss_sum + (pg + SAFE_KL * kl) / max(1, G)

        if valid == 0:
            continue

        mean_kl = kl_sum / valid
        if mean_kl > SAFE_KL_HARD_CAP:
            # Per-micro-batch KL safety rail (config.py's STAGE3_KL_HARD_CAP:
            # this used to be hardcoded to 1.0 here regardless of config, which
            # config.py's own comment documents as too tight -- it discarded
            # ~28% of micro-batches, including good ones, for no real safety
            # benefit once dual-clip PPO above already keeps pg_loss bounded
            # even when an individual micro-batch's KL spikes to 5-6. Now
            # reads the documented 4.0 default from config.py.
            kl_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"[Stage 3] step {step:4d}: KL {mean_kl:.3f} > {SAFE_KL_HARD_CAP:.2f}; skipping update")
            continue

        # Keep a small supervised anchor to the exact Stage-2 target contract.
        # This is deliberately applied only after a useful GRPO group exists.
        if SFT_ANCHOR > 0.0:
            anchor_nll = completion_nll(policy, prompt_embeds, target_ids, embed_layer, dtype, device)
            loss_sum = loss_sum + SFT_ANCHOR * anchor_nll
        else:
            anchor_nll = torch.zeros((), device=device)

        (loss_sum / SAFE_ACCUM).backward()
        applied += 1

        if step % SAFE_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, SAFE_GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if step <= 3 or step % 50 == 0:
            fmt = sum(_parse_completion(t) is not None for t in chosen_text)
            step_hit = np.mean([
                1.0 if compute_reward(t, gold, w_fmt=0.0, w_step=1.0, w_mcp=0.0, w_exp=0.0) else 0.0
                for t in chosen_text
            ])
            mcp_vals = []
            for t in chosen_text:
                c = compute_reward(t, gold, w_fmt=0.0, w_step=0.0, w_mcp=1.0, w_exp=0.0, return_components=True)
                mcp_vals.append(c["mcp"])
            print(
                f"step {step:4d}/{SAFE_STEPS} | lr {scheduler.get_last_lr()[0]:.2e} | "
                f"avg_r {rewards.mean().item():.3f} | step {step_hit:.2f} mcp {np.mean(mcp_vals):.2f} | "
                f"fmt {fmt}/{G} | reward_std {rewards.std(unbiased=False).item():.3f} | kl {mean_kl:.4f} | anchor {anchor_nll.item():.3f}"
            )

        # ---------------- checkpoint + full validation ----------------
        if step % EVAL_EVERY == 0:
            ckpt_path = os.path.join(STAGE3_ADAPTER_DIR, f"step_{step}")
            _save_policy_snapshot(policy, adapter, nn.Identity(), tokenizer, ckpt_path)
            val = evaluate_policy_on_val(
                policy, adapter, stage1, embed_layer, tokenizer,
                val_examples, device, dtype, max_examples=VAL_MAX
            )
            val_step = float(val["step_exact"])
            val_mcp = float(val["mcp_jaccard"])
            val_exp = float(val["exp"])
            val_score = (val_step + val_mcp + val_exp) / 3.0

            # Equal-objective promotion: explanation, Step and MCP all matter.
            # RL may only replace Stage 2 when the aggregate improves and no
            # objective suffers a material regression versus the Stage-2 anchor.
            step_ok = val_step >= baseline_step - 0.01
            mcp_ok = val_mcp >= baseline_mcp - 0.01
            exp_ok = val_exp >= baseline_exp - 0.01
            better = val_score >= best_score + 0.002
            flag = ""
            if step_ok and mcp_ok and exp_ok and better:
                best_score = val_score
                best_step_metric = val_step
                best_mcp_metric = val_mcp
                best_exp_metric = val_exp
                best_step = step
                _save_policy_snapshot(policy, adapter, nn.Identity(), tokenizer, BEST_DIR)
                no_improve = 0
                flag = " <-- NEW BEST"
            else:
                no_improve += 1

            print(
                f"[Stage 3] step {step:4d} | val task={val_score:.4f} "
                f"(step={val_step:.4f}, mcpJ={val_mcp:.4f}, exp={val_exp:.4f}) | "
                f"baseline={baseline_score:.4f} (step={baseline_step:.4f}, mcpJ={baseline_mcp:.4f}, exp={baseline_exp:.4f}) | "
                f"best={best_score:.4f} @ {best_step}{flag}"
            )
            print(
                f"[Stage 3] step {step:4d} | val detailed: step_microF1={val['step_micro_f1']:.4f} | "
                f"mcp_exact={val['mcp_exact_match']:.4f} | mcp_microF1={val['mcp_micro_f1']:.4f} | "
                f"mcp_P={val['mcp_micro_precision']:.4f} | mcp_R={val['mcp_micro_recall']:.4f} | "
                f"avg_missing={val['avg_missing_mcp_tools']:.3f} | avg_extra={val['avg_extra_mcp_tools']:.3f}"
            )

            if no_improve >= SAFE_PATIENCE:
                print(f"[Stage 3] Early stop: {no_improve} consecutive validation checks without a strict improvement.")
                break

    # Flush any partial accumulation only if useful gradients are present.
    # We intentionally do not force an extra optimizer step after an early stop
    # because it has not been validated.
    optimizer.zero_grad(set_to_none=True)

    print("\n" + "=" * 72)
    print("[Stage 3] FINAL MODEL SELECTION")
    print(f"  Stage-2 full-val task : {baseline_score:.4f} (step={baseline_step:.4f}, mcpJ={baseline_mcp:.4f}, exp={baseline_exp:.4f})")
    print(f"  Best RL full-val task : {best_score:.4f} (step={best_step_metric:.4f}, mcpJ={best_mcp_metric:.4f}, step={best_step})")
    print(f"  RL optimizer updates  : {applied}")
    print(f"  KL-skipped updates    : {kl_skipped}")
    print("=" * 72)

    # Always produce a canonical Stage-3 directory.  If RL did not beat the
    # complete Stage-2 validation baseline on BOTH objectives, Stage 3 is a
    # no-op by design and Stage 2 is copied forward.
    canonical_is_stage2 = False
    if best_step > 0 and best_score > baseline_score and best_step_metric >= baseline_step + 0.002 and best_mcp_metric >= baseline_mcp - 0.002:
        for name in os.listdir(STAGE3_ADAPTER_DIR):
            if name == "best" or name.startswith("step_") or name == "last_step_raw":
                continue
            path = os.path.join(STAGE3_ADAPTER_DIR, name)
            if os.path.isfile(path):
                os.remove(path)
        for name in os.listdir(BEST_DIR):
            src = os.path.join(BEST_DIR, name)
            dst = os.path.join(STAGE3_ADAPTER_DIR, name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"[Stage 3] ✓ Promoted RL checkpoint step {best_step}.")
    else:
        canonical_is_stage2 = True
        print("[Stage 3] ⚠ RL did not clear the full Stage-2 validation baseline on both objectives.")
        print("[Stage 3] ✓ Falling back to Stage 2; no regression will be shipped.")
        for name in os.listdir(STAGE2_ADAPTER_DIR):
            src = os.path.join(STAGE2_ADAPTER_DIR, name)
            dst = os.path.join(STAGE3_ADAPTER_DIR, name)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # Keep the final raw RL state only when it differs from the canonical copy.
    last_dir = os.path.join(STAGE3_ADAPTER_DIR, "last_step_raw")
    if os.path.isdir(last_dir):
        shutil.rmtree(last_dir)
    _save_policy_snapshot(policy, adapter, nn.Identity(), tokenizer, last_dir)
    print(f"[Stage 3] Canonical policy: {STAGE3_ADAPTER_DIR}")
    print(f"[Stage 3] Raw final RL state: {last_dir}")

    # If Stage 2 won model selection, evaluate the actual canonical Stage-2
    # policy rather than the rejected last RL state.  This prevents a misleading
    # Stage-3 test score after a safe fallback.
    if canonical_is_stage2:
        print("[Stage 3] Reloading Stage-2 canonical policy for final test evaluation")
        del policy
        torch.cuda.empty_cache() if device == "cuda" else None
        base_eval = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
        ).to(device)
        base_eval.config.use_cache = False
        policy = PeftModel.from_pretrained(base_eval, STAGE2_ADAPTER_DIR, is_trainable=False)
        policy.eval()
        embed_layer = policy.get_input_embeddings()

    # -------------------------- Final test eval --------------------------
    # This uses the same deterministic parser/metrics as the project and is
    # kept here so `run.py` can report Stage-3 results immediately.
    test_examples = load_from_input_json(INPUT_TEST_JSON, "test")
    policy.eval()
    adapter.eval()
    normalizer = StepLabelNormalizer()
    parser = build_obj_parser()
    rows = []

    with torch.no_grad():
        for ex in test_examples:
            prefix = build_prefix_embeds(ex, stage1, adapter, device, dtype)
            user_prompt = build_prompt(ex)
            full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
            p_emb, p_len = build_prompt_embeds(full_prompt, tokenizer, embed_layer, prefix, device, dtype)
            attn = torch.ones(1, p_len, dtype=torch.long, device=device)
            out = policy.generate(
                inputs_embeds=p_emb, attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                num_return_sequences=1, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            ids = trim_generated_row(out[0], tokenizer.eos_token_id, tokenizer.pad_token_id)
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            obj = parser(text, normalizer)

            raw_step = str(obj.get("New step", "")).strip()
            pred_step = normalizer.normalize(raw_step) if raw_step else ""
            if pred_step not in STEP_LABELS and raw_step in STEP_LABELS:
                pred_step = raw_step
            gold_step = STEP_LABELS[ex["step_idx"]]

            mcp_obj = obj.get("MCP_tasks", {})
            pred_mcp = extract_mcp_labels(str(list(mcp_obj.keys()))) if isinstance(mcp_obj, dict) else []
            ps, gs = set(pred_mcp), set(ex["mcp_labels"])
            union = ps | gs
            mcp_j = 1.0 if not union else len(ps & gs) / len(union)

            rows.append({
                "machine": ex.get("machine", ""),
                "new_strategy": ex["context"].get("New strategy", ""),
                "strategy_explanation": ex["context"].get("Strategy explanation", ""),
                "step_prediction": pred_step,
                "gold_new_step": gold_step,
                "mcp_tool_prediction": "|".join(pred_mcp),
                "mcp_tool_gold": "|".join(ex["mcp_labels"]),
                "step_explanation_predicted": str(obj.get("Step explanation", "")),
                "step_explanation_gold": ex.get("gold_step_explanation", ""),
                "step_jaccard": 1.0 if pred_step == gold_step else 0.0,
                "mcp_jaccard": mcp_j,
            })

    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "stage3.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        step_acc = float(np.mean([r["step_jaccard"] for r in rows]))
        mcp_j = float(np.mean([r["mcp_jaccard"] for r in rows]))
        mcp_pass = float(np.mean([r["mcp_jaccard"] >= 0.5 for r in rows]))
        print("\n[Stage 3] ═══════════ TEST SET RESULTS ═══════════")
        print(f"  Step Exact Match      : {int(step_acc*len(rows))}/{len(rows)} ({step_acc*100:.2f}%)")
        print(f"  MCP Jaccard ≥0.5      : {int(mcp_pass*len(rows))}/{len(rows)} ({mcp_pass*100:.2f}%)")
        print(f"  Mean Step Jaccard     : {step_acc:.4f}")
        print(f"  Mean MCP Jaccard      : {mcp_j:.4f}")
        print(f"  Combined (Step+MCP)/2 : {(step_acc+mcp_j)/2:.4f}")
        print(f"[Stage 3] Evaluation CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
