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

Reward composition:
  r = 0.10 × format_ok          — valid JSON with all 3 required keys
    + 0.30 × step_similarity    — embedding similarity between predicted and gold step
    + 0.30 × mcp_set_F1         — set F1 between predicted and gold tools
    + 0.30 × explanation_score  — LLM judge correctness score (0.0-1.0)
                                   using GPT-4o to evaluate if explanation
                                   conveys the same meaning as gold explanation.
                                   Uses caching to avoid repeated API calls.

WHY LLM JUDGE FOR EXPLANATION:
  - Teacher-style evaluation focusing on semantic correctness
  - Captures whether the explanation conveys the same meaning, not just lexical overlap
  - More robust to paraphrasing than BERTScore/BLEU/ROUGE
  - Caching mechanism makes it feasible for training

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
import hashlib
import csv
import shutil
from functools import lru_cache

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
from stage2_sft_qwen import GraphPrefixAdapter, build_prompt, SYSTEM_PROMPT, build_obj_parser

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Value function for baseline reduction
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """
    Value function head for computing state value estimates.
    Used in GRPO to reduce variance by subtracting a learned baseline.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, seq_len, hidden_size) or (B, hidden_size)
        Returns:
            value: (B, 1) scalar value estimate
        """
        if hidden_states.dim() == 3:
            # Pool over sequence dimension (mean pooling)
            hidden_states = hidden_states.mean(dim=1)
        return self.value_net(hidden_states)

# ---------------------------------------------------------------------------
# Explanation quality: LLM judge with caching
# ---------------------------------------------------------------------------

# LLM judge system prompt for reward computation
LLM_JUDGE_SYSTEM_PROMPT = """You are an expert penetration-testing instructor evaluating student answers in a pentesting planning system.

You will be given:
1. A predicted step explanation (what the model/student generated)
2. A ground truth step explanation (what a human expert wrote)

Your task is to evaluate whether the predicted explanation conveys the SAME MEANING as the ground truth explanation, like a teacher grading a student's answer.

Evaluation Criteria:
- Does the predicted explanation convey the same core reasoning and justification as the ground truth?
- Are the technical concepts and logic equivalent, even if worded differently?
- Would this explanation be acceptable as a correct answer in a classroom setting?

Scoring:
- Return a correctness score between 0.0 and 1.0
- 1.0 = Perfect match - conveys exactly the same meaning and reasoning
- 0.8-0.9 = Very good - minor differences in wording but same core meaning
- 0.6-0.7 = Good - mostly correct with some minor omissions or slight inaccuracies
- 0.4-0.5 = Partial - captures some key points but misses important aspects
- 0.2-0.3 = Poor - misses the main point or has significant errors
- 0.0-0.1 = Very poor - completely wrong or irrelevant

Respond in JSON format:
{
    "correctness_score": <float 0.0-1.0>,
    "justification": "<brief explanation of why this score was given>",
    "is_correct": <boolean - true if score >= 0.6, false otherwise>
}"""


def _get_cache_key(pred_expl: str, gold_expl: str) -> str:
    """Generate a cache key from the explanation pair."""
    combined = f"{pred_expl}|||{gold_expl}"
    return hashlib.md5(combined.encode()).hexdigest()


# Global reference to the loaded LLM judge model (separate from training model)
_llm_judge_model = None
_llm_judge_tokenizer = None
_llm_judge_device = None

def set_llm_judge_model(model, tokenizer, device):
    """Set the LLM judge model reference (separate from training model)."""
    global _llm_judge_model, _llm_judge_tokenizer, _llm_judge_device
    _llm_judge_model = model
    _llm_judge_tokenizer = tokenizer
    _llm_judge_device = device

@lru_cache(maxsize=1000)
def _explanation_llm_judge_cached(pred_expl: str, gold_expl: str) -> float:
    """
    LLM judge evaluation using separate QWEN model for explanation quality assessment.

    Returns correctness score (0.0-1.0) using cached results when available.
    Uses a separate model from the one being fine-tuned to avoid bias.
    """
    if not pred_expl.strip() or not gold_expl.strip():
        return 0.0

    # If LLM judge model is not available, use heuristic fallback
    if _llm_judge_model is None or _llm_judge_tokenizer is None:
        expl_len = len(pred_expl)
        if expl_len < 20:
            return 0.3
        elif expl_len < 50:
            return 0.5
        elif expl_len < 100:
            return 0.7
        else:
            return 0.8

    try:
        judge_prompt = f"""Evaluate whether the predicted explanation conveys the same meaning as the ground truth explanation.

PREDICTED EXPLANATION: {pred_expl}

GROUND TRUTH EXPLANATION: {gold_expl}

Rate the similarity on a scale of 0.0 to 1.0 where:
- 0.0: Completely different meaning
- 0.5: Partially similar
- 1.0: Identical or very similar meaning

Respond with just the number (e.g., 0.7)."""

        inputs = _llm_judge_tokenizer(
            judge_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(_llm_judge_device)

        with torch.no_grad():
            outputs = _llm_judge_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=_llm_judge_tokenizer.pad_token_id
            )

        response = _llm_judge_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the score from the response
        import re
        score_match = re.search(r'(\d+\.?\d*)', response)
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        else:
            return 0.5  # Fallback if parsing fails

    except Exception as e:
        print(f"[LLM Judge Error] LLM judge evaluation failed: {e}, using heuristic fallback")
        expl_len = len(pred_expl)
        if expl_len < 20:
            return 0.3
        elif expl_len < 50:
            return 0.5
        elif expl_len < 100:
            return 0.7
        else:
            return 0.8


# ---------------------------------------------------------------------------
# Reward function
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


def compute_reward_curriculum(completion: str, gold: dict, step_num: int,
                              total_steps: int = 2000) -> float:
    """
    Enhanced curriculum learning reward function based on research from
    "Curriculum Reinforcement Learning for Complex Reward Functions" and
    "Decoupling Task and Behavior: A Two-Stage Reward Curriculum".

    Three-stage curriculum with smooth transitions:
    1. Foundation (0-25%): Format + basic step classification
    2. Integration (25-50%): Add MCP tools with increasing complexity
    3. Refinement (50-100%): Full reward with explanation quality emphasis

    Args:
        completion: Generated completion text
        gold: Gold standard dict with step_label, mcp_labels, gold_step_explanation
        step_num: Current training step number
        total_steps: Total training steps (default 2000)
    """
    # Enhanced curriculum with smooth transitions
    progress = step_num / max(1, total_steps)
    
    if progress < 0.25:
        # Foundation stage: master format and step classification
        w_fmt, w_step, w_mcp, w_exp = 0.25, 0.55, 0.15, 0.05
    elif progress < 0.50:
        # Integration stage: gradually introduce MCP tools
        # Linear interpolation between foundation and integration weights
        t = (progress - 0.25) / 0.25  # 0 to 1
        w_fmt = 0.25 * (1 - t) + 0.15 * t
        w_step = 0.55 * (1 - t) + 0.35 * t
        w_mcp = 0.15 * (1 - t) + 0.30 * t
        w_exp = 0.05 * (1 - t) + 0.20 * t
    else:
        # Refinement stage: full reward with explanation emphasis
        # Continue gradual shift toward explanation quality
        t = min(1.0, (progress - 0.50) / 0.50)  # 0 to 1
        w_fmt = 0.15 * (1 - t) + 0.10 * t
        w_step = 0.35 * (1 - t) + 0.25 * t
        w_mcp = 0.30 * (1 - t) + 0.25 * t
        w_exp = 0.20 * (1 - t) + 0.40 * t

    return compute_reward(completion, gold, w_fmt=w_fmt, w_step=w_step, w_mcp=w_mcp, w_exp=w_exp)


def _deterministic_explanation_score(pred_expl: str, pred_step: str, pred_mcp: set[str]) -> float:
    """Low-noise explanation reward used during RL; LLM judge is eval-only."""
    text = str(pred_expl or "").strip().lower()
    if not text:
        return 0.0
    score = 0.35
    n = len(text)
    if 40 <= n <= 350:
        score += 0.20
    elif n >= 20:
        score += 0.10
    if pred_step and pred_step.lower() in text:
        score += 0.20
    evidence_terms = ("port", "service", "version", "directory", "file", "vulnerability",
                      "credential", "authentication", "shell", "exploit", "enumerat", "scan")
    hits = sum(1 for t in evidence_terms if t in text)
    score += min(0.15, 0.03 * hits)
    if pred_mcp and any(tool.lower() in text for tool in pred_mcp):
        score += 0.10
    return float(max(0.0, min(1.0, score)))


def compute_reward(completion: str, gold: dict,
                   w_fmt: float = 0.05,
                   w_step: float = 0.65,
                   w_mcp: float = 0.30,
                   w_exp: float = 0.0,
                   return_components: bool = False):
    """Task-aligned dense reward for GRPO.

    Primary signals intentionally match the final research metrics:
      - exact canonical Step match
      - unweighted MCP set Jaccard
      - lightweight explanation quality
    The LLM judge is reserved for evaluation/model reporting, not the RL
    objective, so the policy is not incentivized to optimize a noisy judge.
    """
    obj = _parse_completion(completion)
    if obj is None or not all(k in obj for k in ("New step", "Step explanation", "MCP_tasks")):
        partial = 0.0
        if obj is not None:
            partial = sum(1 for k in ("New step", "Step explanation", "MCP_tasks") if k in obj) / 3.0
        total = w_fmt * partial
        out = {"total": total, "fmt": partial, "step": 0.0, "mcp": 0.0, "exp": 0.0}
        return out if return_components else total

    pred_step = str(obj.get("New step", "")).strip()
    gold_step = str(gold["step_label"]).strip()
    step_r = 1.0 if _step_normalizer.normalize(pred_step) == gold_step else 0.0

    mcp_val = obj.get("MCP_tasks", {})
    pred_mcp = set(extract_mcp_labels(str(mcp_val))) if isinstance(mcp_val, dict) and mcp_val else set()
    gold_mcp = set(gold["mcp_labels"])
    union = pred_mcp | gold_mcp
    mcp_r = (len(pred_mcp & gold_mcp) / len(union)) if union else 1.0

    exp_r = _deterministic_explanation_score(str(obj.get("Step explanation", "")), pred_step, pred_mcp)
    fmt_r = 1.0
    total = w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * exp_r
    out = {"total": total, "fmt": fmt_r, "step": step_r, "mcp": mcp_r, "exp": exp_r}
    return out if return_components else total


def compute_reward_curriculum(completion: str, gold: dict, step_num: int,
                              total_steps: int = 2000, return_components: bool = False):
    """Stable task-aligned reward; no changing weights during RL."""
    return compute_reward(completion, gold, return_components=return_components)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def build_prefix_embeds(ex, stage1, adapter, device, dtype):
    """
    Build the exact fused Stage-1 representation used to train the Stage-2
    GraphPrefixAdapter, then project it into graph-prefix soft tokens.

    Stage 2 was trained with GRAPH_PREFIX_SRC_DIM = FUSION_HIDDEN // 2
    (384-dim for the current model). The old Stage-3 implementation incorrectly
    fed the raw 512-dim graph embedding into that adapter.

    The current Stage-1 classifier's encode_and_predict() is intentionally used
    here so Stage 3 sees the same graph + strategy-conditioned representation
    that Stage 2 saw. When semantic token tensors are unavailable, Stage-1's
    built-in compatibility fallback derives a short semantic sequence from the
    two BGE field embeddings.
    """
    graph = ex["graph"]
    batch = PyGBatch.from_data_list([graph]).to(device)

    texts = [ex["context"].get(c, "") or "empty" for c in ("New strategy", "Strategy explanation")]
    field_embs = torch.tensor(_embed_texts(texts), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        edge_attr = getattr(batch, 'edge_attr', None)
        fused_h, _, _ = stage1.encode_and_predict(
            batch.x, batch.edge_index, batch.batch, field_embs, edge_attr=edge_attr
        )  # (1, 384) with current FUSION_HIDDEN=768

    src_dim = fused_h.shape[-1]
    expected_dim = adapter.proj[0].in_features
    if src_dim != expected_dim:
        raise RuntimeError(
            f"Stage-3 graph-prefix dimension mismatch: Stage-1 produced {src_dim} dims, "
            f"but the Stage-2 GraphPrefixAdapter expects {expected_dim}. "
            f"The Stage-1 checkpoint, Stage-2 adapter, and Stage-3 code must come from the same interface version."
        )

    # Stage-2 GraphPrefixAdapter is stored/trained in FP32, while the Qwen
    # policy and Stage-1 checkpoint may run in BF16.  Feed the adapter FP32
    # input, then cast its soft-prefix output back to the policy dtype.
    # This avoids mat1/mat2 dtype mismatches without changing the learned
    # Stage-2 adapter weights.
    prefix = adapter(fused_h.float())  # (1, n_tokens, H), adapter runs in FP32
    prefix = prefix.to(dtype=dtype)     # Qwen consumes BF16/FP16 embeddings
    return prefix  # kept on device


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
    """
    was_training = policy.training
    policy.eval()
    rng = random.Random(RANDOM_SEED)
    sample = val_examples if len(val_examples) <= max_examples else rng.sample(val_examples, max_examples)

    components = {"total": [], "fmt": [], "step": [], "mcp": [], "exp": []}
    step_exact = []
    mcp_jaccard = []
    mcp_pass = []
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

    if was_training:
        policy.train()
    return {**{k: (float(np.mean(v)) if v else 0.0) for k, v in components.items()},
            "step_exact": float(np.mean(step_exact)) if step_exact else 0.0,
            "mcp_jaccard": float(np.mean(mcp_jaccard)) if mcp_jaccard else 0.0,
            "mcp_pass": float(np.mean(mcp_pass)) if mcp_pass else 0.0}


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
      * keeps the Stage-1 fused-384 -> prefix interface unchanged;
      * freezes the graph-prefix adapter by default;
      * uses the actual GRPO group-relative advantage;
      * uses rollout-policy log-probabilities for the PPO ratio;
      * uses a task-aligned reward: 75% exact Step + 20% MCP Jaccard + 5%
        format, with NO LLM-judge/explanation reward during RL;
      * adds a small supervised Stage-2 target anchor to prevent reward drift;
      * rejects near-zero-variance groups instead of learning from noise;
      * evaluates the FULL 239-example machine-held-out validation set;
      * only promotes a checkpoint if both Step and MCP improve over Stage 2;
      * otherwise copies Stage 2 forward unchanged.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # These are deliberately conservative and can be overridden without
    # editing config.py.  They are safer for the already-strong Stage-2 model
    # than the previous 8e-7 / 1600-step configuration.
    G = int(os.environ.get("STAGE3_SAFE_GROUP_SIZE", "8"))
    SAFE_LR = float(os.environ.get("STAGE3_SAFE_LR", "1.0e-7"))
    SAFE_STEPS = int(os.environ.get("STAGE3_SAFE_STEPS", "600"))
    SAFE_KL = float(os.environ.get("STAGE3_SAFE_KL", "0.08"))
    SAFE_CLIP = float(os.environ.get("STAGE3_SAFE_CLIP", "0.20"))
    SAFE_ACCUM = int(os.environ.get("STAGE3_SAFE_GRAD_ACCUM", "4"))
    SAFE_PATIENCE = int(os.environ.get("STAGE3_SAFE_PATIENCE", "2"))
    EVAL_EVERY = int(os.environ.get("STAGE3_SAFE_EVAL_EVERY", "200"))
    VAL_MAX = int(os.environ.get("STAGE3_SAFE_VAL_MAX", "239"))
    MAX_NEW_TOKENS = int(os.environ.get("STAGE3_SAFE_MAX_NEW_TOKENS", "260"))
    TRAIN_ADAPTER = os.environ.get("STAGE3_TRAIN_ADAPTER", "0") == "1"
    SFT_ANCHOR = float(os.environ.get("STAGE3_SFT_ANCHOR", "0.20"))

    print(f"[Stage 3] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 3] Device         : {device}")
    print(f"[Stage 3] Total steps    : {SAFE_STEPS}")
    print(f"[Stage 3] Group size (G) : {G}")
    print(f"[Stage 3] KL coef        : {SAFE_KL}")
    print(f"[Stage 3] PPO clip eps   : {SAFE_CLIP}")
    print(f"[Stage 3] LR             : {SAFE_LR:.2e}")
    print(f"[Stage 3] Grad accum     : {SAFE_ACCUM}")
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
    from stage2_sft_qwen import GRAPH_PREFIX_SRC_DIM
    llm_hidden = policy.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).float()
    adapter_ckpt = os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt")
    if not os.path.isfile(adapter_ckpt):
        raise FileNotFoundError(f"Stage-2 graph adapter not found: {adapter_ckpt}")
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device, weights_only=False))
    adapter.eval() if not TRAIN_ADAPTER else adapter.train()
    if not TRAIN_ADAPTER:
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
    baseline_score = 0.65 * baseline_step + 0.35 * baseline_mcp
    print(f"[Stage 3] Stage-2 baseline: task={baseline_score:.4f} | step={baseline_step:.4f} | mcpJ={baseline_mcp:.4f}")

    best_score = baseline_score
    best_step_metric = baseline_step
    best_mcp_metric = baseline_mcp
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
                float(compute_reward(t, gold, w_fmt=0.05, w_step=0.75, w_mcp=0.20, w_exp=0.0))
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

        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False)
        advantages = (rewards - mean_r) / std_r.clamp_min(1e-6)
        advantages = advantages.clamp(-3.0, 3.0)

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
        if TRAIN_ADAPTER:
            adapter.train()

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
            clipped_ratio = torch.clamp(ratio, 1.0 - SAFE_CLIP, 1.0 + SAFE_CLIP)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            pg = -torch.minimum(surr1, surr2)

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
        if mean_kl > 1.0:
            # Safety rail is intentionally much tighter than the old cap of 4.
            kl_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"[Stage 3] step {step:4d}: KL {mean_kl:.3f} > 1.0; skipping update")
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
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
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
            val_score = 0.65 * val_step + 0.35 * val_mcp

            # Promotion is deliberately strict.  Stage 3 is not allowed to
            # trade away MCP for Step or vice versa.  Require a real margin,
            # not a one-example/noise improvement.
            step_ok = val_step >= baseline_step + 0.002
            mcp_ok = val_mcp >= baseline_mcp - 0.002
            better = val_score >= best_score + 0.002
            flag = ""
            if step_ok and mcp_ok and better:
                best_score = val_score
                best_step_metric = val_step
                best_mcp_metric = val_mcp
                best_step = step
                _save_policy_snapshot(policy, adapter, nn.Identity(), tokenizer, BEST_DIR)
                no_improve = 0
                flag = " <-- NEW BEST"
            else:
                no_improve += 1

            print(
                f"[Stage 3] step {step:4d} | val task={val_score:.4f} "
                f"(step={val_step:.4f}, mcpJ={val_mcp:.4f}) | "
                f"baseline={baseline_score:.4f} (step={baseline_step:.4f}, mcpJ={baseline_mcp:.4f}) | "
                f"best={best_score:.4f} @ {best_step}{flag}"
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
    print(f"  Stage-2 full-val task : {baseline_score:.4f} (step={baseline_step:.4f}, mcpJ={baseline_mcp:.4f})")
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
