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
import openai

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
    RANDOM_SEED,
    STEP_LABELS,
    MCP_LABELS,
    ROOT,
    GRAPH_PREFIX_TOKENS,
    GNN_OUT_DIM,
    STAGE2_VAL_SPLIT,
)
from data_utils import load_from_input_json, _embed_texts, CONTEXT_COLUMNS, StepLabelNormalizer, extract_mcp_labels
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


def compute_reward(completion: str, gold: dict,
                   w_fmt:  float = 0.10,  # Format weight
                   w_step: float = 0.25,  # Step similarity weight
                   w_mcp:  float = 0.20,  # MCP F1 weight
                   w_exp:  float = 0.45) -> float:  # LLM judge weight
    """
    Enhanced composite reward function based on research from:
    - "Dense Reward for Free in RLHF" - reward shaping
    - "ClaHF" - preference-based classification rewards
    - "TOLE" - token-level reward considerations

    Components:
      fmt_r   — 1.0 if output is valid JSON with all 3 required keys
      step_r  — exact match + embedding similarity for step classification
      mcp_r   — rarity-weighted F1 for MCP tool prediction
      exp_r   — LLM judge correctness + semantic bonuses for explanation
    """
    obj = _parse_completion(completion)
    if obj is None or not all(k in obj for k in ("New step", "Step explanation", "MCP_tasks")):
        # Enhanced partial credit for progressive learning
        partial_fmt_score = 0.0
        if obj is not None:
            required_keys = ["New step", "Step explanation", "MCP_tasks"]
            present_keys = sum(1 for k in required_keys if k in obj)
            partial_fmt_score = present_keys / len(required_keys) * 0.5
        return w_fmt * partial_fmt_score

    # ── Format ────────────────────────────────────────────────────────────────
    fmt_r = 1.0

    # ── Enhanced Step correctness ───────────────────────────────────────────────
    pred_step = obj["New step"].strip()
    gold_step = gold["step_label"]
    pred_step_norm = _step_normalizer.normalize(pred_step)
    
    if pred_step_norm == gold_step:
        step_r = 1.0
    else:
        # Enhanced partial credit with semantic similarity
        step_embs = _embed_texts([pred_step, gold_step])
        step_sim = max(0.0, float(np.dot(step_embs[0], step_embs[1])))
        
        # Additional bonus for partial semantic match
        step_r = 0.2 * step_sim
        
        # Bonus for correct step category (e.g., both enumeration steps)
        if any(kw in pred_step.lower() and kw in gold_step.lower() 
               for kw in ["enumerate", "exploit", "explore", "search", "analyze"]):
            step_r += 0.1

    # ── Enhanced MCP set F1 with macro-F1 optimization ─────────────────────────
    mcp_val = obj.get("MCP_tasks", {})
    if isinstance(mcp_val, dict) and mcp_val:
        pred_mcp = set(extract_mcp_labels(str(mcp_val)))
    else:
        pred_mcp = set()
    gold_mcp = set(gold["mcp_labels"])
    w = _mcp_label_weights()
    
    if not pred_mcp and not gold_mcp:
        mcp_r = 1.0
    else:
        tp = pred_mcp & gold_mcp
        fp = pred_mcp - gold_mcp
        fn = gold_mcp - pred_mcp
        w_tp = sum(w[l] for l in tp)
        w_fp = sum(w[l] for l in fp)
        w_fn = sum(w[l] for l in fn)
        
        prec = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0.0
        rec  = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0.0
        mcp_r = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        # Bonus for correct tool category (e.g., network tools)
        network_tools = {"Nmap", "Netcat", "Smb client"}
        if (pred_mcp & network_tools) and (gold_mcp & network_tools):
            mcp_r += 0.05

    # ── Enhanced Explanation with multi-dimensional evaluation ─────────────────
    pred_expl = str(obj.get("Step explanation", "")).strip()
    gold_expl = gold.get("gold_step_explanation", "")
    exp_r = _explanation_llm_judge_cached(pred_expl, gold_expl)
    
    # Multi-dimensional explanation bonuses
    exp_bonus = 0.0
    
    # Length and structure
    expl_len = len(pred_expl)
    if expl_len < 20:
        exp_bonus -= 0.15  # Stronger penalty for too short
    elif 40 <= expl_len <= 250:
        exp_bonus += 0.08  # Optimal length range
    
    # Step consistency
    if pred_step.lower() in pred_expl.lower():
        exp_bonus += 0.05
    
    # Technical depth
    tech_terms = ["vulnerability", "exploit", "enumerate", "scan", "privilege", "escalation",
                  "credential", "authentication", "service", "port", "attack", "defense",
                  "payload", "shell", "reverse", "bind", "lateral", "movement"]
    term_count = sum(1 for term in tech_terms if term.lower() in pred_expl.lower())
    if term_count >= 3:
        exp_bonus += 0.05
    elif term_count >= 1:
        exp_bonus += 0.02
    
    # Logical structure (has reasoning indicators)
    reasoning_indicators = ["because", "since", "due to", "therefore", "thus", "as", "to"]
    if any(ind in pred_expl.lower() for ind in reasoning_indicators):
        exp_bonus += 0.03
    
    # MCP consistency
    if pred_mcp:
        mcp_mentioned = any(tool.lower() in pred_expl.lower() for tool in pred_mcp)
        if mcp_mentioned:
            exp_bonus += 0.03
    
    # Clamp bonus
    exp_bonus = max(-0.15, min(0.15, exp_bonus))

    return w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * (exp_r + exp_bonus)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def build_prefix_embeds(graph, field_embs, stage1, adapter, embed_layer, device, dtype):
    """
    Given a single torch_geometric Data object + its context field
    embeddings, produce the (1, n_tokens, H) soft-prompt prefix that gets
    prepended to every prompt/completion.

    FIX: this previously ran graph_encoder ALONE (no context/strategy-text
    fusion at all -- not even the crude blend Stage 2 used), so the
    GraphPrefixAdapter loaded from the Stage-2 checkpoint (which WAS
    trained on a graph+context representation) was fed an out-of-
    distribution input throughout RL. Now calls
    stage1.encode_and_predict(...) -- the same fused representation Stage
    2 now trains against (see stage2_sft_qwen.py / graph_encoder.py) -- so
    GRPO fine-tunes the adapter starting from the distribution it was
    actually initialized on.

    stage1 and adapter must already be on device.
    """
    batch = PyGBatch.from_data_list([graph]).to(device)
    field_embs = field_embs.to(device)
    with torch.no_grad():
        edge_attr = getattr(batch, 'edge_attr', None)
        combined_emb, _, _ = stage1.encode_and_predict(
            batch.x, batch.edge_index, batch.batch, field_embs, edge_attr=edge_attr
        )  # (1, FUSION_HIDDEN // 2)
    prefix = adapter(combined_emb.to(dtype))  # (1, n_tokens, H)
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
# Main training loop
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[Stage 3] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 3] Device         : {device}")
    print(f"[Stage 3] Total steps    : {STAGE3_STEPS}")
    print(f"[Stage 3] Group size (G) : {STAGE3_GROUP_SIZE}")
    print(f"[Stage 3] KL coef        : {STAGE3_KL_COEF}")
    print(f"[Stage 3] PPO clip eps   : {STAGE3_PPO_CLIP}")
    print(f"[Stage 3] LR             : {STAGE3_LR}")
    print(f"[Stage 3] Grad accum     : {STAGE3_GRAD_ACCUM}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Policy model (Stage-2 LoRA, trainable) ───────────────────────────────
    print(f"\n[Stage 3] Loading policy base model: {QWEN_MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    base.gradient_checkpointing_enable()
    policy = PeftModel.from_pretrained(base, STAGE2_ADAPTER_DIR, is_trainable=True)
    policy.train()

    # ── Reference model (Stage-2 LoRA, frozen) ───────────────────────────────
    print(f"[Stage 3] Loading reference model (frozen copy)")
    ref_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    ref_base.gradient_checkpointing_enable()
    ref_model = PeftModel.from_pretrained(ref_base, STAGE2_ADAPTER_DIR, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Frozen Stage-1 graph encoder ─────────────────────────────────────────
    print(f"[Stage 3] Loading Stage-1 GNN checkpoint: {STAGE1_CKPT}")
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
        print(f"[Stage 3]   loaded (epoch={ckpt.get('best_epoch','?')}, "
              f"score={ckpt.get('best_score','?'):.4f})")
    else:
        stage1.load_state_dict(ckpt)
    # FIX: keep the whole frozen Stage-1 classifier (encoders + gates +
    # fusion), not just graph_encoder -- see build_prefix_embeds and
    # graph_encoder.Stage1Classifier.encode_and_predict for why.
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    # ── GraphPrefixAdapter (trainable — loaded from Stage-2 checkpoint) ──────
    # Dim must match GRAPH_PREFIX_SRC_DIM = FUSION_HIDDEN // 2 used by
    # stage2_sft_qwen.py, since we're loading ITS checkpoint here.
    from stage2_sft_qwen import GRAPH_PREFIX_SRC_DIM
    llm_hidden = policy.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).to(dtype)
    adapter_ckpt = os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt")
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device, weights_only=False))
    adapter.train()
    print(f"[Stage 3] Loaded GraphPrefixAdapter from Stage-2")

    # ── Embedding layers (shared, read-only during generation) ────────────────
    embed_layer = policy.get_input_embeddings()
    ref_embed_layer = ref_model.get_input_embeddings()

    # ── Load SEPARATE LLM judge model (CRITICAL: different from training model) ───
    from config import LLM_JUDGE_MODEL_NAME
    print(f"\n[Stage 3] Loading SEPARATE LLM judge model: {LLM_JUDGE_MODEL_NAME}")
    print(f"[Stage 3] ⚠  This is DIFFERENT from training model ({QWEN_MODEL_NAME})")
    judge_tokenizer = AutoTokenizer.from_pretrained(LLM_JUDGE_MODEL_NAME)
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_model = AutoModelForCausalLM.from_pretrained(
        LLM_JUDGE_MODEL_NAME,
        torch_dtype=dtype,
        device_map=None
    ).to(device)
    judge_model.eval()
    for p in judge_model.parameters():
        p.requires_grad_(False)
    set_llm_judge_model(judge_model, judge_tokenizer, device)
    print(f"[Stage 3] ✓ Separate LLM judge model loaded ({LLM_JUDGE_MODEL_NAME} != {QWEN_MODEL_NAME})")

    # ── Value function for baseline reduction ────────────────────────────────
    value_head = ValueHead(llm_hidden).to(device).to(dtype)
    value_optimizer = AdamW(value_head.parameters(), lr=STAGE3_LR * 2, weight_decay=0.01)

    # ── Optimizer — LoRA params + adapter, NOT base weights ──────────────────
    trainable = [p for p in policy.parameters() if p.requires_grad] + \
                list(adapter.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"\n[Stage 3] Trainable params: ~{n_trainable/1e6:.1f}M")

    optimizer = AdamW(trainable, lr=STAGE3_LR, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8)
    total_updates = STAGE3_STEPS // STAGE3_GRAD_ACCUM
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_updates))

    # ── Dataset — load training data, machine-based split for no leakage ─────
    all_train_examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
    print(f"\n[Stage 3] Total labeled examples loaded: {len(all_train_examples)}")

    # DATA LEAKAGE PREVENTION: apply same machine-based split as Stage 2
    # (so RL doesn't see Stage 2 validation examples either)
    machine_order = sorted(set(e["machine"] for e in all_train_examples))
    rng_split = np.random.default_rng(RANDOM_SEED + 1)
    perm_machines = rng_split.permutation(len(machine_order))
    n_val_machines = max(1, int(len(machine_order) * STAGE2_VAL_SPLIT))
    val_machine_set = set(machine_order[i] for i in perm_machines[:n_val_machines])
    examples = [e for e in all_train_examples if e["machine"] not in val_machine_set]

    train_machines = set(e["machine"] for e in examples)
    print(f"[Stage 3] RL training on {len(examples)} examples, "
          f"{len(train_machines)} machines (excluded {len(val_machine_set)} val machines)")

    # ── REMOVED: Stage-1 classifier hints ─────────────────────────────────────
    # Critical fix: Force Stage 3 RL to learn from graph prefix tokens instead of
    # copying Stage 1 predictions. This is essential for Stage 3 to actually
    # improve over Stage 1 performance.
    # Precompute field embeddings for efficiency
    for ex in examples:
        ex["_field_embs"] = torch.tensor(
            _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
            dtype=torch.float32,
        ).unsqueeze(0)

    # Class-balanced example sampling — same rationale as Stage 2's
    # WeightedRandomSampler (see stage2_sft_qwen.py): step_label support is
    # heavily skewed, and uniform random.choice() over `examples` means the
    # policy sees the majority class ("Exploit the selected exploitations")
    # far more often than rare ones, which biases what GRPO has gradient
    # signal to improve. Precompute inverse-frequency sampling weights once.
    # Same gentler sqrt + capped-ratio reweighting as Stage 2 (see
    # stage2_sft_qwen.py) -- raw 1/count overshot and flipped the imbalance
    # rather than correcting it.
    example_step_idxs = [e["step_idx"] for e in examples]
    example_step_counts = np.bincount(example_step_idxs, minlength=len(STEP_LABELS)).astype(np.float64)
    example_step_counts[example_step_counts == 0] = 1.0
    example_inv_freq = 1.0 / np.sqrt(example_step_counts)
    example_inv_freq = np.clip(example_inv_freq, example_inv_freq.max() / 4.0, example_inv_freq.max())
    example_sample_weights = [example_inv_freq[i] for i in example_step_idxs]

    # ── Load test data just for data leakage pre-check ──────────────────────
    test_examples_precheck = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = set(e["machine"] for e in test_examples_precheck)
    train_test_overlap = train_machines & test_machines
    if train_test_overlap:
        print(f"[Stage 3] ⚠  WARNING: train/test machine overlap: {sorted(train_test_overlap)}")
    else:
        print(f"[Stage 3] ✓ No machine overlap between RL train and test sets")
    del test_examples_precheck  # free memory

    # ── Output dir ────────────────────────────────────────────────────────────
    os.makedirs(STAGE3_ADAPTER_DIR, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    G          = STAGE3_GROUP_SIZE
    beta       = STAGE3_KL_COEF
    grad_accum = STAGE3_GRAD_ACCUM
    clip_eps   = STAGE3_PPO_CLIP

    optimizer.zero_grad()
    value_optimizer.zero_grad()
    global_step = 0

    # Running reward stats for adaptive advantage normalization
    reward_running_mean = 0.0
    reward_running_std = 1.0
    ema_alpha = 0.95

    for step in range(1, STAGE3_STEPS + 1):

        # ── 1. Sample one training example (class-balanced) ───────────────────
        ex = random.choices(examples, weights=example_sample_weights, k=1)[0]
        gold = {
            "step_label":             ex["step_label"],
            "mcp_labels":             ex["mcp_labels"],
            "gold_step_explanation":  ex["gold_step_explanation"],
        }

        # ── 2. Build graph prefix + prompt embeddings ─────────────────────────
        prefix_embeds = build_prefix_embeds(
            ex["graph"], ex["_field_embs"], stage1, adapter, embed_layer, device, dtype
        )  # (1, n_tokens, H)

        prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{build_prompt(ex, mask_hint=True)}\n"
            f"<|assistant|>\n"
        )
        prompt_embeds, L_prefix_plus_prompt = build_prompt_embeds(
            prompt_text, tokenizer, embed_layer, prefix_embeds, device, dtype
        )
        attn_prompt = torch.ones(
            1, L_prefix_plus_prompt, dtype=torch.long, device=device
        )

        # ── 3. Generate G completions ─────────────────────────────────────────
        policy.eval()   # disable dropout during generation

        with torch.no_grad():
            gen_out = policy.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_prompt,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=G,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # ── FIX: gen_out already contains ONLY the newly generated tokens ─────
        # (generate() cannot prepend the prompt when called with inputs_embeds
        # only, since the prompt has no token-id representation). We just trim
        # per-row trailing pad tokens; no slicing by L_prefix_plus_prompt.
        completion_ids_list = [
            trim_generated_row(gen_out[i], tokenizer.eos_token_id, tokenizer.pad_token_id)
            for i in range(gen_out.shape[0])
        ]

        if step <= 3:
            lens = [t.shape[0] for t in completion_ids_list]
            print(f"[Stage 3] Debug step {step}: gen_out shape = {gen_out.shape}, "
                  f"trimmed completion lengths = {lens}")

        # ── 4. Decode and score completions ───────────────────────────────────
        completions = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in completion_ids_list
        ]

        if step <= 3:
            print(f"[Stage 3] Debug step {step}: First 2 completions:")
            for i, comp in enumerate(completions[:2]):
                print(f"  Completion {i}: {comp[:200]}...")

        rewards_np = np.array([
            compute_reward_curriculum(c, gold, step, total_steps=STAGE3_STEPS)
            for c in completions
        ], dtype=np.float32)

        # Update EMA reward stats for adaptive normalization
        batch_mean = rewards_np.mean()
        batch_std = rewards_np.std() + 1e-8
        reward_running_mean = ema_alpha * reward_running_mean + (1 - ema_alpha) * batch_mean
        reward_running_std = ema_alpha * reward_running_std + (1 - ema_alpha) * batch_std

        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)

        # ── 5. Compute advantages (group-relative, GRPO-style) ────────────────
        # BUG FIX: this used to compute the standard GRPO group-relative
        # advantage (rewards - mean) / std -- which is *already* a properly
        # normalized, roughly unit-variance quantity by construction (that's
        # the whole point of GRPO: the group itself is the baseline, no
        # critic needed) -- and then subtract a SECOND baseline term
        # ((value_head output) - mean) / std on top of it. Stacking two
        # baselines like that double-counts the normalization and routinely
        # pushed `advantages` several std devs past the +-4.0 clamp, which
        # is visible directly in the training log: pg_loss lands on almost
        # exactly the same value (-4.719 = -4.0 * (1+clip_eps)) at steps 50,
        # 200, 600, 800, 1650, 1900, 2000, 2200... regardless of how avg_r or
        # kl vary at those steps. That constant value is what
        # `-adv_clamped * clamped_ratio` evaluates to once `adv` is pinned at
        # the clamp boundary -- i.e. most updates were being driven by the
        # clamp constant, not by the actual per-sample reward signal, which
        # is consistent with Stage 3 ending up statistically indistinguishable
        # from (or slightly worse than) the Stage 2 SFT checkpoint it started
        # from. Fix: use the plain GRPO group-relative advantage; the value
        # head is still trained below (as a monitoring/critic signal you can
        # inspect) but no longer feeds into the policy advantage.
        mean_r = rewards.mean()
        std_r = rewards.std()
        if std_r < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - mean_r) / std_r
        advantages = torch.clamp(advantages, -4.0, 4.0)

        with torch.no_grad():
            v_in = {"inputs_embeds": prompt_embeds.detach(),
                    "attention_mask": attn_prompt.detach(),
                    "output_hidden_states": True}
            v_out = policy(**v_in)
            baseline_v = value_head(v_out.hidden_states[-1]).squeeze().detach()

        # ── 6. PPO-clipped GRPO policy loss + KL penalty ─────────────────────
        policy.train()
        value_head.train()

        loss_accum = torch.zeros(1, device=device, requires_grad=True)
        value_loss_accum = torch.zeros(1, device=device, requires_grad=True)
        total_kl_accum = 0.0

        valid_completions = 0
        for g_idx in range(G):
            comp_ids = completion_ids_list[g_idx].unsqueeze(0).to(device)
            if comp_ids.shape[1] == 0:
                continue  # skip genuinely empty completions
            valid_completions += 1
            adv = advantages[g_idx]
            reward = rewards[g_idx].item()

            # Policy log-prob (grad-enabled)
            lp_policy = completion_logprobs(
                policy, prompt_embeds, comp_ids, embed_layer, dtype, device
            )

            # Reference log-prob (frozen, no grad)
            with torch.no_grad():
                lp_ref = completion_logprobs(
                    ref_model, prompt_embeds.detach(), comp_ids,
                    ref_embed_layer, dtype, device
                )

            # ── Standard PPO clipped importance-ratio loss ────────────────────
            # ratio = π_θ(a|s) / π_ref(a|s) = exp(lp_policy - lp_ref)
            # Clipping keeps ratio in [1-ε, 1+ε] for pessimistic bound.
            log_ratio = lp_policy - lp_ref.detach()
            # Clip log_ratio before exp to prevent ratio explosion
            log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
            ratio = torch.exp(log_ratio)
            clamped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

            # Standard PPO surrogate: pessimistic bound (minimum for positive adv, maximum for negative)
            # Since we MINIMIZE loss = -objective:
            #   loss_unclipped = -adv * ratio
            #   loss_clipped   = -adv * clamp(ratio, 1-ε, 1+ε)
            #   pg_loss = max(loss_unclipped, loss_clipped)   [pessimistic]
            pg_unclipped = -adv * ratio
            pg_clipped   = -adv * clamped_ratio
            pg_loss = torch.max(pg_unclipped, pg_clipped)

            # Clip-fraction monitoring: % of ratios that were clipped
            clip_frac = float(((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean().item())

            # KL penalty: keep policy close to Stage 2 SFT init (reversed KL, policy vs ref)
            L = max(1.0, float(comp_ids.shape[1]))
            # Clip log_ratio to prevent KL explosion
            log_ratio_clipped = torch.clamp(log_ratio, min=-10.0, max=10.0)
            kl = (torch.exp(log_ratio_clipped) - log_ratio_clipped - 1.0) / L  # non-negative KL-like term
            # Clip KL term to prevent explosion
            kl = torch.clamp(kl, max=10.0)
            total_kl_accum += kl.item()

            loss_accum = loss_accum + (pg_loss + beta * kl) / G

            # Value loss: learn baseline
            value_loss = 0.5 * ((baseline_v - reward) ** 2)
            value_loss_accum = value_loss_accum + value_loss / G

        # Scale by gradient accumulation
        if valid_completions > 0:
            loss_for_backward = loss_accum / grad_accum
            value_loss_for_backward = value_loss_accum / grad_accum
            total_loss = loss_for_backward + 0.5 * value_loss_for_backward
            total_loss.backward()
        else:
            print(f"[Stage 3] Warning: No valid completions in step {step}, skipping backward pass")

        # ── 7. Optimizer step every grad_accum steps ──────────────────────────
        if step % grad_accum == 0:
            if valid_completions > 0:
                torch.nn.utils.clip_grad_norm_(trainable, STAGE3_GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(value_head.parameters(), STAGE3_GRAD_CLIP)
                optimizer.step()
                value_optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                value_optimizer.zero_grad()
                global_step += 1
            else:
                print(f"[Stage 3] Warning: Skipping optimizer step at step {step} due to no valid completions")

        # ── 8. Logging ────────────────────────────────────────────────────────
        if step % 50 == 0:
            avg_reward = rewards.mean().item()
            fmt_ok = sum(1 for c in completions if _parse_completion(c) is not None)

            comp_scores = {"step": [], "mcp": [], "exp": []}
            for c in completions:
                obj = _parse_completion(c)
                if obj is None:
                    continue
                pred_step_dbg = obj.get("New step", "").strip()
                comp_scores["step"].append(
                    1.0 if _step_normalizer.normalize(pred_step_dbg) == gold["step_label"] else 0.0
                )
                mcp_val = obj.get("MCP_tasks", {})
                if isinstance(mcp_val, dict) and mcp_val:
                    pred_mcp = set(extract_mcp_labels(str(mcp_val)))
                else:
                    pred_mcp = set()
                gold_mcp = set(gold["mcp_labels"])
                if not pred_mcp and not gold_mcp:
                    comp_scores["mcp"].append(1.0)
                else:
                    inter = len(pred_mcp & gold_mcp)
                    prec  = inter / len(pred_mcp) if pred_mcp else 0.0
                    rec   = inter / len(gold_mcp) if gold_mcp else 0.0
                    comp_scores["mcp"].append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0)
                pred_expl = str(obj.get("Step explanation","")).strip()
                comp_scores["exp"].append(
                    _explanation_llm_judge_cached(pred_expl, gold["gold_step_explanation"])
                )

            avg_step = float(np.mean(comp_scores["step"])) if comp_scores["step"] else 0.0
            avg_mcp  = float(np.mean(comp_scores["mcp"]))  if comp_scores["mcp"]  else 0.0
            avg_exp  = float(np.mean(comp_scores["exp"]))  if comp_scores["exp"]  else 0.0
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"step {step:4d}/{STAGE3_STEPS} | "
                f"lr {current_lr:.2e} | "
                f"avg_r {avg_reward:.3f} | "
                f"step {avg_step:.2f} mcp {avg_mcp:.2f} exp {avg_exp:.2f} | "
                f"fmt {fmt_ok}/{G} | "
                f"kl {total_kl_accum/G:.3f} | "
                f"pg_loss {loss_accum.item():.3f} vloss {value_loss_accum.item():.3f}"
            )

        # ── 9. Checkpoint every 200 steps ────────────────────────────────────
        if step % 200 == 0:
            ckpt_path = os.path.join(STAGE3_ADAPTER_DIR, f"step_{step}")
            policy.save_pretrained(ckpt_path)
            torch.save(adapter.state_dict(), os.path.join(ckpt_path, "graph_adapter.pt"))
            torch.save(value_head.state_dict(), os.path.join(ckpt_path, "value_head.pt"))
            tokenizer.save_pretrained(ckpt_path)
            print(f"  -> checkpoint saved to {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    policy.save_pretrained(STAGE3_ADAPTER_DIR)
    torch.save(adapter.state_dict(), os.path.join(STAGE3_ADAPTER_DIR, "graph_adapter.pt"))
    torch.save(value_head.state_dict(), os.path.join(STAGE3_ADAPTER_DIR, "value_head.pt"))
    tokenizer.save_pretrained(STAGE3_ADAPTER_DIR)
    print(f"\n[Stage 3] GRPO complete. Policy saved to {STAGE3_ADAPTER_DIR}")

    # ── Evaluate on test set and save CSV ─────────────────────────────────────
    print("\n[Stage 3] Evaluating on test set...")
    test_examples = load_from_input_json(INPUT_TEST_JSON, "test")
    precompute_stage1_hints(test_examples, stage1, device, dtype)
    for ex in test_examples:
        ex["_field_embs"] = torch.tensor(
            _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
            dtype=torch.float32,
        ).unsqueeze(0)

    # Final data leakage check
    test_machines_final = set(e["machine"] for e in test_examples)
    overlap = train_machines & test_machines_final
    if not overlap:
        print("[Stage 3] ✓ Confirmed: NO machine overlap between train & test")

    normalizer = StepLabelNormalizer()
    csv_rows = []
    _obj_parser = build_obj_parser()

    policy.eval()
    adapter.eval()

    with torch.no_grad():
        for ex in test_examples:
            prefix_embeds = build_prefix_embeds(
                ex["graph"], ex["_field_embs"], stage1, adapter, embed_layer, device, dtype
            )
            user_prompt = build_prompt(ex, mask_hint=True)
            full_prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{user_prompt}\n"
                f"<|assistant|>\n"
            )
            prompt_embeds, L_prefix_plus_prompt = build_prompt_embeds(
                full_prompt, tokenizer, embed_layer, prefix_embeds, device, dtype
            )
            attn_prompt = torch.ones(1, L_prefix_plus_prompt, dtype=torch.long, device=device)

            gen_out = policy.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_prompt,
                max_new_tokens=2000,  # Increased to match training generation
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # FIX: gen_out already contains only the newly generated tokens —
            # do NOT slice by L_prefix_plus_prompt (see note above main()).
            completion_ids = trim_generated_row(
                gen_out[0], tokenizer.eos_token_id, tokenizer.pad_token_id
            )
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            obj = _obj_parser(completion_text, normalizer)

            pred_step_raw = obj.get("New step", "")
            pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
            pred_step_label = "UNPARSEABLE"
            if pred_step_norm and pred_step_norm in STEP_LABELS:
                pred_step_label = pred_step_norm
            elif pred_step_raw:
                if pred_step_raw in STEP_LABELS:
                    pred_step_label = pred_step_raw
                else:
                    import difflib
                    closest = difflib.get_close_matches(pred_step_raw, STEP_LABELS, n=1, cutoff=0.6)
                    if closest:
                        pred_step_label = closest[0]

            gold_step_label = STEP_LABELS[ex["step_idx"]]

            pred_mcp_keys = list(obj.get("MCP_tasks", {}).keys()) if isinstance(obj.get("MCP_tasks"), dict) else []
            pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
            pred_mcp_tools = "|".join(pred_mcp_labels)
            gold_mcp_tools = "|".join(ex["mcp_labels"])

            pred_expl = str(obj.get("Step explanation", "")).strip()
            gold_expl = ex.get("gold_step_explanation", "")

            # Jaccard for consistency with project metrics
            step_jaccard = 1.0 if pred_step_label == gold_step_label else 0.0
            pred_set, gold_set = set(pred_mcp_labels), set(ex["mcp_labels"])
            if not pred_set and not gold_set:
                mcp_jaccard = 1.0
            else:
                union = pred_set | gold_set
                mcp_jaccard = len(pred_set & gold_set) / len(union) if union else 0.0

            csv_rows.append({
                "machine": ex.get("machine", ""),
                "new_strategy": ex["context"].get("New strategy", ""),
                "strategy_explanation": ex["context"].get("Strategy explanation", ""),
                "step_prediction": pred_step_label,
                "gold_new_step": gold_step_label,
                "mcp_tool_prediction": pred_mcp_tools,
                "mcp_tool_gold": gold_mcp_tools,
                "step_explanation_predicted": pred_expl,
                "step_explanation_gold": gold_expl,
                "step_jaccard": step_jaccard,
                "mcp_jaccard": mcp_jaccard,
            })

    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "stage3.csv")

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Stage 3] Evaluation CSV saved to: {csv_path}")
        print(f"[Stage 3] Total test samples evaluated: {len(csv_rows)}")

        step_acc = np.mean([r["step_jaccard"] for r in csv_rows])
        mcp_jac = np.mean([r["mcp_jaccard"] for r in csv_rows])
        combined = (step_acc + mcp_jac) / 2.0
        step_pass = sum(1 for r in csv_rows if r["step_jaccard"] == 1.0)
        mcp_pass = sum(1 for r in csv_rows if r["mcp_jaccard"] >= 0.5)

        print(f"\n[Stage 3] ═══════════ TEST SET RESULTS ═══════════")
        print(f"  Step Exact Match     : {step_pass}/{len(csv_rows)}  ({step_acc*100:.2f}%)")
        print(f"  MCP Jaccard ≥0.5      : {mcp_pass}/{len(csv_rows)}  ({mcp_pass/len(csv_rows)*100:.2f}%)")
        print(f"  Mean Step Jaccard    : {step_acc:.4f}")
        print(f"  Mean MCP Jaccard     : {mcp_jac:.4f}")
        print(f"  Combined (Step+MCP)/2 : {combined:.4f}")
        print(f"[Stage 3] ═══════════════════════════════════════")
    else:
        print("[Stage 3] Warning: No CSV rows generated")


if __name__ == "__main__":
    main()