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
    + 0.30 × step_exact_match   — exact match against gold step label
    + 0.30 × mcp_set_F1         — set F1 between predicted and gold tools
    + 0.30 × explanation_score  — deterministic BERTScore-style cosine sim
                                   between generated and gold explanation
                                   using the frozen bge-small-en-v1.5 encoder
                                   already loaded for graph node embeddings.
                                   No LLM judge — fully deterministic.

WHY BERTSCORE FOR EXPLANATION (not BLEU/ROUGE):
  - BLEU/ROUGE require exact n-gram overlap → punishes valid paraphrases
  - Cosine similarity over bge-small embeddings captures semantic equivalence
  - It's the same encoder already in memory → zero extra cost
  - It's deterministic → no reward variance from a stochastic judge
  - Threshold: scores below 0.3 are clamped to 0 to avoid rewarding
    off-topic text that happens to share a few semantic dimensions
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch as PyGBatch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import (
    INPUT_TRAIN_JSON,
    QWEN_MODEL_NAME,
    STAGE1_CKPT,
    STAGE2_ADAPTER_DIR,
    STAGE3_ADAPTER_DIR,
    STAGE3_GROUP_SIZE,
    STAGE3_LR,
    STAGE3_STEPS,
    STAGE3_KL_COEF,
    GNN_OUT_DIM,
    GRAPH_PREFIX_TOKENS,
    MCP_LABELS,
    RANDOM_SEED,
)
from data_utils import load_from_input_json, _embed_texts, CONTEXT_COLUMNS
from graph_encoder import Stage1Classifier
from stage2_sft_qwen import GraphPrefixAdapter, build_prompt, SYSTEM_PROMPT

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Explanation quality: deterministic BERTScore via frozen bge-small encoder
# ---------------------------------------------------------------------------

def _explanation_bertscore(pred_expl: str, gold_expl: str,
                            min_score: float = 0.30) -> float:
    """
    Cosine similarity between bge-small-en-v1.5 embeddings of the generated
    and gold explanation.

    - Uses the same frozen sentence encoder already loaded for graph nodes
      (via data_utils._embed_texts) — zero extra cost.
    - Fully deterministic — no stochastic judge, no variance.
    - Scores below min_score clamped to 0.0 to avoid rewarding off-topic text.

    Returns float in [0.0, 1.0].
    """
    if not pred_expl.strip() or not gold_expl.strip():
        return 0.0
    # _embed_texts returns L2-normalised vectors, so dot = cosine similarity
    embs = _embed_texts([pred_expl[:512], gold_expl[:512]])  # (2, 384)
    cos  = float(np.dot(embs[0], embs[1]))                   # scalar in [-1,1]
    cos  = max(0.0, cos)                                      # clamp negatives
    return 0.0 if cos < min_score else cos


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


def compute_reward(completion: str, gold: dict,
                   w_fmt:  float = 0.20,  # Increased weight for format
                   w_step: float = 0.30,
                   w_mcp:  float = 0.30,
                   w_exp:  float = 0.20) -> float:  # Reduced weight for explanation
    """
    Composite reward — all components are deterministic.

    gold keys:
      step_label          str          gold next-step label
      mcp_labels          list[str]    gold MCP tool list
      gold_step_explanation str        gold free-text explanation

    Components:
      fmt_r   — 1.0 if output is valid JSON with all 3 required keys
      step_r  — embedding similarity between predicted and gold step
      mcp_r   — F1 between predicted and gold tool sets
      exp_r   — BERTScore cosine similarity between explanations
                (0.0 if below 0.30 threshold, to avoid rewarding off-topic text)
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

    # ── Explanation BERTScore ─────────────────────────────────────────────────
    pred_expl = str(obj.get("Step explanation", "")).strip()
    gold_expl = gold.get("gold_step_explanation", "")
    exp_r = _explanation_bertscore(pred_expl, gold_expl)

    return w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * exp_r


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
    policy = PeftModel.from_pretrained(base, STAGE2_ADAPTER_DIR, is_trainable=True)
    policy.train()
    dtype = torch.bfloat16

    # ── Reference model (Stage-2 LoRA, frozen) ───────────────────────────────
    # Deep-copy so both share the same base weights but reference is frozen.
    ref_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    ref_model = PeftModel.from_pretrained(ref_base, STAGE2_ADAPTER_DIR, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Frozen Stage-1 graph encoder ─────────────────────────────────────────
    stage1 = Stage1Classifier()
    stage1.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
    graph_encoder = stage1.graph_encoder.to(device).eval()
    for p in graph_encoder.parameters():
        p.requires_grad_(False)

    # ── GraphPrefixAdapter (trainable — loaded from Stage-2 checkpoint) ──────
    llm_hidden = policy.config.hidden_size
    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)
    adapter_ckpt = os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt")
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    adapter.train()

    # ── Embedding layer (shared, read-only during generation) ────────────────
    embed_layer = policy.get_input_embeddings()

    # ── Optimizer — LoRA params + adapter, NOT base weights ──────────────────
    trainable = [p for p in policy.parameters() if p.requires_grad] + \
                list(adapter.parameters())
    # Use very conservative learning rate for stability
    optimizer = AdamW(trainable, lr=5e-7, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=STAGE3_STEPS)

    # ── Dataset ───────────────────────────────────────────────────────────────
    examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
    print(f"[Stage 3] {len(examples)} training examples loaded")

    # ── Output dir ────────────────────────────────────────────────────────────
    os.makedirs(STAGE3_ADAPTER_DIR, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    G          = 8                   # Increased group size for better gradient estimation
    beta       = 0.05                # Increased KL penalty for more stable training
    grad_accum = 8                   # accumulate before optimizer step
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
                max_new_tokens=256,
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
        rewards = torch.tensor(
            [compute_reward(c, gold) for c in completions],
            dtype=torch.float32,
        )

        # ── 6. Group-relative advantage  A_i = (r_i - mean) / (std + ε) ──────
        mean_r = rewards.mean()
        std_r  = rewards.std()
        # If all rewards are the same (std=0), use zero advantages to prevent NaN
        if std_r < 1e-8:
            advantages = torch.zeros_like(rewards).to(device)
        else:
            advantages = ((rewards - mean_r) / std_r).to(device)
            # Clip advantages to prevent extreme gradients
            advantages = torch.clamp(advantages, -5.0, 5.0)

        # ── 7. Policy gradient with clipping + KL penalty ─────────────────────
        policy.train()
        total_pg_loss = torch.tensor(0.0, device=device, requires_grad=False)
        # We accumulate losses from all G completions then divide
        loss_accum = torch.zeros(1, device=device)

        for g_idx in range(G):
            comp_ids = completion_ids_list[g_idx].unsqueeze(0).to(device)  # (1, L_gen)
            adv = advantages[g_idx]                                         # scalar

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

            loss_accum = loss_accum + pg_loss / G

        # Scale by gradient accumulation
        loss_for_backward = loss_accum / grad_accum
        loss_for_backward.backward()

        # ── 8. Optimizer step every grad_accum steps ──────────────────────────
        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
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
                    _explanation_bertscore(pred_expl, gold["gold_step_explanation"])
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
                f"loss {loss_accum.item():.4f}"
            )

        # ── 10. Checkpoint every 200 steps ────────────────────────────────────
        if step % 200 == 0:
            ckpt_path = os.path.join(STAGE3_ADAPTER_DIR, f"step_{step}")
            policy.save_pretrained(ckpt_path)
            torch.save(adapter.state_dict(), os.path.join(ckpt_path, "graph_adapter.pt"))
            tokenizer.save_pretrained(ckpt_path)
            print(f"  -> checkpoint saved to {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    policy.save_pretrained(STAGE3_ADAPTER_DIR)
    torch.save(adapter.state_dict(), os.path.join(STAGE3_ADAPTER_DIR, "graph_adapter.pt"))
    tokenizer.save_pretrained(STAGE3_ADAPTER_DIR)
    print(f"\nStage 3 GRPO complete. Policy saved to {STAGE3_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
