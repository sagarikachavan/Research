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
    RANDOM_SEED,
    STEP_LABELS,
    MCP_LABELS,
    ROOT,
    GRAPH_PREFIX_TOKENS,
    GNN_OUT_DIM,
)
from data_utils import load_from_input_json, _embed_texts, CONTEXT_COLUMNS, StepLabelNormalizer, extract_mcp_labels
from graph_encoder import Stage1Classifier
from stage2_sft_qwen import GraphPrefixAdapter, build_prompt, SYSTEM_PROMPT, build_obj_parser

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
        if hidden_states.dim() ==3:
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


@lru_cache(maxsize=1000)
def _explanation_llm_judge_cached(pred_expl: str, gold_expl: str) -> float:
    """
    LLM judge evaluation with caching to avoid repeated API calls.
    
    Returns correctness score (0.0-1.0) using cached results when available.
    """
    if not pred_expl.strip() or not gold_expl.strip():
        return 0.0
    
    # Check if OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Warning] OPENAI_API_KEY not set, using fallback score of 0.5")
        return 0.5
    
    try:
        user_prompt = f"""Please evaluate whether the predicted explanation conveys the same meaning as the ground truth explanation:

PREDICTED EXPLANATION: {pred_expl}

GROUND TRUTH EXPLANATION: {gold_expl}

Evaluate whether the predicted explanation is correct and conveys the same meaning as the ground truth.
Provide your response in JSON format as requested."""
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        
        raw_response = response.choices[0].message.content
        
        # Parse JSON response
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0].strip()
        else:
            json_str = raw_response.strip()
        
        result = json.loads(json_str)
        score = result.get("correctness_score", 0.0)
        return float(score)
        
    except Exception as e:
        print(f"[LLM Judge Error] {e}, using fallback score of 0.5")
        return 0.5


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


def compute_reward_curriculum(completion: str, gold: dict, step_num: int,
                              total_steps: int = 2000) -> float:
    """
    Curriculum learning reward function that adjusts weights based on training stage.

    Early training (steps 0-500): Focus on format and step prediction
    Mid training (steps 500-1000): Add MCP tool prediction
    Late training (steps 1000+): Full reward with explanation quality

    Args:
        completion: Generated completion text
        gold: Gold standard dict with step_label, mcp_labels, gold_step_explanation
        step_num: Current training step number
        total_steps: Total training steps (default 2000)
    """
    # Curriculum learning: adjust weights based on training stage
    if step_num < 500:
        # Early: focus on format and step
        w_fmt, w_step, w_mcp, w_exp = 0.3, 0.5, 0.1, 0.1
    elif step_num < 1000:
        # Mid: add MCP
        w_fmt, w_step, w_mcp, w_exp = 0.2, 0.3, 0.3, 0.2
    else:
        # Late: full reward
        w_fmt, w_step, w_mcp, w_exp = 0.1, 0.25, 0.25, 0.4

    return compute_reward(completion, gold, w_fmt=w_fmt, w_step=w_step, w_mcp=w_mcp, w_exp=w_exp)


def compute_reward(completion: str, gold: dict,
                   w_fmt:  float = 0.10,  # Format weight
                   w_step: float = 0.25,  # Step similarity weight (reduced from 0.30)
                   w_mcp:  float = 0.20,  # MCP F1 weight (reduced from 0.30)
                   w_exp:  float = 0.45) -> float:  # LLM judge weight (increased from 0.30 to 0.45)
    """
    Composite reward with LLM judge for explanation evaluation.

    gold keys:
      step_label          str          gold next-step label
      mcp_labels          list[str]    gold MCP tool list
      gold_step_explanation str        gold free-text explanation

    Components:
      fmt_r   — 1.0 if output is valid JSON with all 3 required keys
      step_r  — embedding similarity between predicted and gold step
      mcp_r   — F1 between predicted and gold tool sets
      exp_r   — LLM judge correctness score between explanations
                (0.0-1.0, uses caching to avoid repeated API calls)
    """
    obj = _parse_completion(completion)
    if obj is None or not all(k in obj for k in ("New step", "Step explanation", "MCP_tasks")):
        # Give partial credit for partial JSON structure to provide learning signal
        partial_fmt_score = 0.0
        if obj is not None:
            # Check if any required keys are present
            required_keys = ["New step", "Step explanation", "MCP_tasks"]
            present_keys = sum(1 for k in required_keys if k in obj)
            partial_fmt_score = present_keys / len(required_keys) * 0.5
        return w_fmt * partial_fmt_score  # only format reward for partial JSON

    # ── Format ────────────────────────────────────────────────────────────────
    fmt_r = 1.0

    # ── Step similarity (use embedding similarity instead of exact match) ──────
    pred_step = obj["New step"].strip()
    gold_step = gold["step_label"]
    # Use embedding similarity for more continuous reward
    step_embs = _embed_texts([pred_step, gold_step])
    step_sim = float(np.dot(step_embs[0], step_embs[1]))
    step_sim = max(0.0, step_sim)  # clamp negatives
    step_r = step_sim  # continuous similarity score

    # ── MCP set F1 ────────────────────────────────────────────────────────────
    mcp_val  = obj.get("MCP_tasks", {})
    pred_mcp = set(mcp_val.keys() if isinstance(mcp_val, dict) else []) & set(MCP_LABELS)
    gold_mcp = set(gold["mcp_labels"])
    if not pred_mcp and not gold_mcp:
        mcp_r = 1.0
    else:
        inter = len(pred_mcp & gold_mcp)
        prec  = inter / len(pred_mcp) if pred_mcp else 0.0
        rec   = inter / len(gold_mcp) if gold_mcp else 0.0
        mcp_r = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ── Explanation LLM Judge ─────────────────────────────────────────────────
    pred_expl = str(obj.get("Step explanation", "")).strip()
    gold_expl = gold.get("gold_step_explanation", "")
    exp_r = _explanation_llm_judge_cached(pred_expl, gold_expl)

    # ── Explanation-specific bonus rewards ────────────────────────────────────
    exp_bonus = 0.0

    # Length bonus: reward appropriately detailed explanations
    expl_len = len(pred_expl)
    if expl_len < 20:
        exp_bonus -= 0.1  # Penalty for too short
    elif 50 <= expl_len <= 300:
        exp_bonus += 0.05  # Bonus for good length

    # Step mention bonus: reward explanations that mention the predicted step
    if pred_step.lower() in pred_expl.lower():
        exp_bonus += 0.03

    # Technical term bonus: reward explanations with pentesting terminology
    tech_terms = ["vulnerability", "exploit", "enumerate", "scan", "privilege", "escalation", 
                  "credential", "authentication", "service", "port", "attack", "defense"]
    term_count = sum(1 for term in tech_terms if term.lower() in pred_expl.lower())
    if term_count >= 2:
        exp_bonus += 0.02

    # Clamp bonus to reasonable range
    exp_bonus = max(-0.1, min(0.1, exp_bonus))

    return w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * (exp_r + exp_bonus)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def build_prefix_embeds(graph, graph_encoder, adapter, embed_layer, device, dtype):
    """
    Given a single torch_geometric Data object, produce the (1, n_tokens, H)
    soft-prompt prefix that gets prepended to every prompt/completion.

    graph_encoder and adapter must already be on device.
    """
    # Wrap single graph in a batch of size 1
    batch = PyGBatch.from_data_list([graph]).to(device)
    with torch.no_grad():
        graph_emb = graph_encoder(batch.x, batch.edge_index, batch.batch)  # (1, GNN_OUT_DIM)
    prefix = adapter(graph_emb.to(dtype))  # (1, n_tokens, H)
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
        truncation=True,
        max_length=900,   # leave room for generation
    ).input_ids.to(device)

    token_embeds = embed_layer(ids).to(dtype)             # (1, T_prompt, H)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # (1, n_prefix+T_prompt, H)
    return inputs_embeds, inputs_embeds.shape[1]


# ---------------------------------------------------------------------------
# Per-token log-prob of a completion given inputs_embeds prefix
# ---------------------------------------------------------------------------

def completion_logprobs(
    model,
    inputs_embeds: torch.Tensor,   # (1, L_prefix, H)
    completion_ids: torch.Tensor,  # (1, L_gen)
    embed_layer,
    dtype,
    device,
) -> torch.Tensor:
    """
    Compute the sum of per-token log-probs for `completion_ids` given the
    prompt represented as `inputs_embeds`.

    We do a single forward pass over [prefix | completion] with
    inputs_embeds, then slice the logits to the completion span.

    Returns scalar tensor (grad-enabled).
    """
    comp_embeds = embed_layer(completion_ids).to(dtype)          # (1, L_gen, H)
    full_embeds = torch.cat([inputs_embeds, comp_embeds], dim=1) # (1, L_prefix+L_gen, H)

    attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)
    out  = model(inputs_embeds=full_embeds, attention_mask=attn)  # no labels → no loss
    logits = out.logits  # (1, L_prefix+L_gen, V)

    # The logit at position i predicts token i+1.
    # Completion tokens start at index L_prefix in the full sequence.
    L_prefix = inputs_embeds.shape[1]
    # logits[:, L_prefix-1 : L_prefix+L_gen-1] predicts completion_ids[:,0..L_gen-1]
    comp_logits = logits[:, L_prefix - 1 : L_prefix + completion_ids.shape[1] - 1, :]
    log_probs   = F.log_softmax(comp_logits, dim=-1)             # (1, L_gen, V)
    token_lp    = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)  # (1, L_gen)
    return token_lp.sum()  # scalar


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 3] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 3] Device         : {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Policy model (Stage-2 LoRA, trainable) ───────────────────────────────
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    base.gradient_checkpointing_enable()
    policy = PeftModel.from_pretrained(base, STAGE2_ADAPTER_DIR, is_trainable=True)
    policy.train()
    dtype = torch.bfloat16

    # ── Reference model (Stage-2 LoRA, frozen) ───────────────────────────────
    # Deep-copy so both share the same base weights but reference is frozen.
    ref_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    ref_base.gradient_checkpointing_enable()
    ref_model = PeftModel.from_pretrained(ref_base, STAGE2_ADAPTER_DIR, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Frozen Stage-1 graph encoder ─────────────────────────────────────────
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
    else:
        stage1.load_state_dict(ckpt)
    graph_encoder = stage1.graph_encoder.to(device).eval()
    for p in graph_encoder.parameters():
        p.requires_grad_(False)

    # ── GraphPrefixAdapter (trainable — loaded from Stage-2 checkpoint) ──────
    llm_hidden = policy.config.hidden_size
    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)
    adapter_ckpt = os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt")
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device, weights_only=False))
    adapter.train()

    # ── Embedding layer (shared, read-only during generation) ────────────────
    embed_layer = policy.get_input_embeddings()

    # ── Value function for baseline reduction ────────────────────────────────
    value_head = ValueHead(llm_hidden).to(device).to(dtype)
    value_optimizer = AdamW(value_head.parameters(), lr=STAGE3_LR, weight_decay=0.01)

    # ── Optimizer — LoRA params + adapter, NOT base weights ──────────────────
    trainable = [p for p in policy.parameters() if p.requires_grad] + \
                list(adapter.parameters())
    # Use updated learning rate from config for more meaningful updates
    optimizer = AdamW(trainable, lr=STAGE3_LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=STAGE3_STEPS)

    # ── Dataset ───────────────────────────────────────────────────────────────
    examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
    print(f"[Stage 3] {len(examples)} training examples loaded")

    # ── Output dir ────────────────────────────────────────────────────────────
    os.makedirs(STAGE3_ADAPTER_DIR, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    G          = STAGE3_GROUP_SIZE  # Use config group size (16) for better gradient estimation
    beta       = STAGE3_KL_COEF     # Use config KL penalty (0.01) for more exploration
    grad_accum = 4                   # Reduced from 8 for faster updates
    clip_eps   = 0.2                 # PPO clip range

    optimizer.zero_grad()
    global_step = 0

    for step in range(1, STAGE3_STEPS + 1):

        # ── 1. Sample one training example ───────────────────────────────────
        ex = random.choice(examples)
        gold = {
            "step_label":             ex["step_label"],
            "mcp_labels":             ex["mcp_labels"],
            "gold_step_explanation":  ex["gold_step_explanation"],
        }

        # ── 2. Build graph prefix  (1, n_prefix, H) ──────────────────────────
        prefix_embeds = build_prefix_embeds(
            ex["graph"], graph_encoder, adapter, embed_layer, device, dtype
        )  # (1, n_prefix, H)

        # ── 3. Build prompt embeddings ────────────────────────────────────────
        prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{build_prompt(ex)}\n"
            f"<|assistant|>\n"
        )
        prompt_embeds, L_prefix_plus_prompt = build_prompt_embeds(
            prompt_text, tokenizer, embed_layer, prefix_embeds, device, dtype
        )
        # prompt_embeds : (1, L_prefix_plus_prompt, H)

        attn_prompt = torch.ones(
            1, L_prefix_plus_prompt, dtype=torch.long, device=device
        )

        # ── 4. Generate G completions ─────────────────────────────────────────
        policy.eval()   # disable dropout during generation
        with torch.no_grad():
            gen_out = policy.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_prompt,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.6,  # Lower temperature for more stable generation
                top_p=0.8,       # Lower top_p for more focused generation
                num_return_sequences=G,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Prevent repetitive n-grams
                repetition_penalty=1.1,  # Add repetition penalty
            )
        # gen_out: (G, L_prefix_plus_prompt + L_gen)
        # generated token IDs only (strip the prompt prefix length)
        # NOTE: generate() with inputs_embeds returns only the NEW tokens
        completion_ids_list = gen_out  # (G, L_gen) — already just the new tokens

        # ── 5. Decode and score completions ───────────────────────────────────
        completions = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in completion_ids_list
        ]
        # Use curriculum learning reward function
        rewards = torch.tensor(
            [compute_reward_curriculum(c, gold, step) for c in completions],
            dtype=torch.float32,
            device=device,
        )

        # ── 6. Compute value estimates and advantages ──────────────────────────
        # Get value estimates from value function (keep gradients for training)
        value_inputs = {
            "inputs_embeds": prompt_embeds,
            "attention_mask": attn_prompt,
            "output_hidden_states": True,
        }
        value_outputs = policy(**value_inputs)
        # Use last hidden state (mean pooled by ValueHead)
        value_estimates = value_head(value_outputs.hidden_states[-1])  # (1, 1)
        baseline = value_estimates.squeeze()  # scalar tensor with gradients

        # Compute advantages with learned baseline
        advantages = rewards - baseline.detach()  # detach baseline for advantage computation
        # Normalize advantages for stability
        mean_r = advantages.mean()
        std_r  = advantages.std()
        if std_r < 1e-8:
            advantages = torch.zeros_like(advantages).to(device)
        else:
            advantages = ((advantages - mean_r) / std_r).to(device)
            # Clip advantages to prevent extreme gradients
            advantages = torch.clamp(advantages, -5.0, 5.0)

        # ── 7. Policy gradient with clipping + KL penalty ─────────────────────
        policy.train()
        value_head.train()
        total_pg_loss = torch.tensor(0.0, device=device, requires_grad=False)
        # We accumulate losses from all G completions then divide
        loss_accum = torch.zeros(1, device=device)
        value_loss_accum = torch.zeros(1, device=device)

        for g_idx in range(G):
            comp_ids = completion_ids_list[g_idx].unsqueeze(0).to(device)  # (1, L_gen)
            adv = advantages[g_idx]                                         # scalar
            reward = rewards[g_idx].item()                                  # scalar

            # Policy log-prob for this completion
            lp_policy = completion_logprobs(
                policy, prompt_embeds, comp_ids, embed_layer, dtype, device
            )

            # Reference log-prob (frozen Stage-2) — same prefix
            with torch.no_grad():
                ref_embed_layer = ref_model.get_input_embeddings()
                # Rebuild prefix through adapter for reference
                # (adapter is trainable so we use its current state for ref too
                #  but ref_model's LM weights are frozen — this is correct:
                #  the KL is over the LM distribution, adapter is our addition)
                lp_ref = completion_logprobs(
                    ref_model, prompt_embeds.detach(), comp_ids,
                    ref_embed_layer, dtype, device
                )

            # KL penalty (forward KL approximation: log π - log π_ref)
            kl = lp_policy - lp_ref.detach()

            # Entropy bonus to encourage exploration
            # Compute entropy from the log-probs distribution
            # For simplicity, use a small constant bonus
            entropy_bonus = 0.01

            # GRPO loss for this completion:
            # We use a simple policy-gradient loss (no importance-weight ratio
            # here since we're doing on-policy generation) with clipping on the
            # advantage scale to limit gradient variance.
            # L = -adv * lp_policy  +  β * KL  - entropy_bonus
            pg_loss = -adv * lp_policy + beta * kl - entropy_bonus

            # Value function loss (MSE between predicted value and actual reward)
            value_loss = (baseline - reward) ** 2

            loss_accum = loss_accum + pg_loss / G
            value_loss_accum = value_loss_accum + value_loss / G

        # Scale by gradient accumulation
        loss_for_backward = loss_accum / grad_accum
        value_loss_for_backward = value_loss_accum / grad_accum

        # Combine losses for single backward pass
        total_loss = loss_for_backward + value_loss_for_backward
        total_loss.backward()

        # ── 8. Optimizer step every grad_accum steps ──────────────────────────
        if step % grad_accum == 0:
            # Clip gradients for policy
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            # Clip gradients for value function
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
            
            optimizer.step()
            value_optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            value_optimizer.zero_grad()
            global_step += 1

        # ── 9. Logging ────────────────────────────────────────────────────────
        if step % 50 == 0:
            avg_reward = rewards.mean().item()
            fmt_ok = sum(1 for c in completions if _parse_completion(c) is not None)

            # Breakdown: avg per-component scores for this batch
            comp_scores = {"step": [], "mcp": [], "exp": []}
            for c in completions:
                obj = _parse_completion(c)
                if obj is None:
                    continue
                comp_scores["step"].append(
                    1.0 if obj.get("New step","").strip() == gold["step_label"] else 0.0
                )
                mcp_val  = obj.get("MCP_tasks", {})
                pred_mcp = set(mcp_val.keys() if isinstance(mcp_val, dict) else []) & set(MCP_LABELS)
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

            print(
                f"step {step:4d}/{STAGE3_STEPS} | "
                f"avg_reward {avg_reward:.3f} | "
                f"step_r {avg_step:.3f} | "
                f"mcp_r {avg_mcp:.3f} | "
                f"exp_r {avg_exp:.3f} | "
                f"fmt_ok {fmt_ok}/{G} | "
                f"pg_loss {loss_accum.item():.4f} | "
                f"val_loss {value_loss_accum.item():.4f}"
            )

        # ── 10. Checkpoint every 200 steps ────────────────────────────────────
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
    print(f"\nStage 3 GRPO complete. Policy saved to {STAGE3_ADAPTER_DIR}")
    
    # ── Evaluate on test set and save CSV ─────────────────────────────────────
    print("\n[Stage 3] Evaluating on test set...")
    test_examples = load_from_input_json(INPUT_TEST_JSON, "test")
    
    normalizer = StepLabelNormalizer()
    csv_rows = []
    _obj_parser = build_obj_parser()
    
    policy.eval()
    adapter.eval()
    
    with torch.no_grad():
        for ex in test_examples:
            gold = {
                "step_label": ex["step_label"],
                "mcp_labels": ex["mcp_labels"],
                "gold_step_explanation": ex["gold_step_explanation"],
            }
            
            # Build graph prefix
            prefix_embeds = build_prefix_embeds(
                ex["graph"], graph_encoder, adapter, embed_layer, device, dtype
            )
            
            # Build prompt embeddings
            user_prompt = build_prompt(ex)
            full_prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{user_prompt}\n"
                f"<|assistant|>\n"
            )
            prompt_embeds, L_prefix_plus_prompt = build_prompt_embeds(
                full_prompt, tokenizer, embed_layer, prefix_embeds, device, dtype
            )

            attn_prompt = torch.ones(1, L_prefix_plus_prompt, dtype=torch.long, device=device)

            # Generate single completion
            gen_out = policy.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_prompt,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

            # Decode generated text
            completion_ids = gen_out[:, L_prefix_plus_prompt:]
            completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

            # Parse generated text
            obj = _obj_parser(completion_text, normalizer)
            
            # Extract step prediction
            pred_step_raw = obj.get("New step", "")
            pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
            
            # Try multiple fallback strategies for step prediction
            pred_step_label = "UNPARSEABLE"
            if pred_step_norm and pred_step_norm in STEP_LABELS:
                pred_step_label = pred_step_norm
            elif pred_step_raw:
                # Try direct match
                if pred_step_raw in STEP_LABELS:
                    pred_step_label = pred_step_raw
                else:
                    # Try fuzzy match - find closest label
                    import difflib
                    closest_match = difflib.get_close_matches(pred_step_raw, STEP_LABELS, n=1, cutoff=0.6)
                    if closest_match:
                        pred_step_label = closest_match[0]
            
            gold_step_label = STEP_LABELS[ex["step_idx"]]
            
            # Extract MCP predictions
            pred_mcp_keys = list(obj.get("MCP_tasks", {}).keys()) if isinstance(obj.get("MCP_tasks"), dict) else []
            pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
            pred_mcp_tools = "|".join(pred_mcp_labels)
            gold_mcp_tools = "|".join(ex["mcp_labels"])
            
            # Extract explanations
            pred_expl = str(obj.get("Step explanation", "")).strip()
            gold_expl = ex.get("gold_step_explanation", "")
            
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
            })
    
    # Save CSV
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
    else:
        print("[Stage 3] Warning: No CSV rows generated")


if __name__ == "__main__":
    main()
