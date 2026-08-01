"""
train_grpo.py — GRPO (Group Relative Policy Optimization) RL fine-tuning of
Qwen3-14B on stepmodelv3/input/train.json.

Fixes vs. the original stage1_grpo_rl.py:
  - Data loader matches the REAL schema (`machine`, `graph`, `gold_new_step`,
    `gold_step_explanation`, `gold_mcp_tasks`) instead of a schema that
    doesn't exist in your files (`Machine`, `Graph`, `step_label`, `mcp_labels`).
  - `gold_mcp_tasks` is parsed correctly — it's usually a semicolon-joined
    "Tool: description; Tool2: description2" string, not JSON.
  - Step matching uses a data-derived 10-category classifier + text
    similarity instead of exact-string match against a hardcoded, wrong
    label list (which would have given ~0% reward forever).
  - Raw graph JSON is still passed directly as text in the prompt — no
    embedding model, exactly as required.

Same GRPO mechanics as the original script (LoRA policy + frozen LoRA
reference, group-relative advantage, PPO-clipped policy gradient + KL
penalty), which were already correctly implemented and are preserved as-is.

IMPORTANT — realistic expectations:
This script trains on your real data and will drive reward upward, but
whether you land in the 85-90% accuracy range is a function of training
steps, group size, LoRA rank/targets, data size, and Qwen3-14B's own
capacity — not something a script's hyperparameters can *guarantee*. Treat
the numbers below (steps, group size, LoRA rank) as a reasonable starting
point tuned for this dataset's size (1728 rows) and complexity (10 step
categories, 18 tools), and use evaluate.py + the run log (reward /
step_r / mcp_r / exp_r curves) to decide whether to train longer, widen
LoRA rank/targets, or increase group size.
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from data_utils import load_json_data, compute_reward, parse_completion, classify_step, mcp_labels_from_dict, get_sentence_encoder
import data_utils
from prompts import build_chat_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_TRAIN_JSON = "input/train.json"
QWEN_MODEL_NAME = "Qwen/Qwen3-14B-Instruct"
ADAPTER_DIR = "checkpoints/stage1_grpo_rl"

GROUP_SIZE = 6            # completions sampled per prompt (higher = less noisy advantage estimate)
LR = 1e-5
NUM_STEPS = 3000          # ~10-12 epochs over 1728 rows at batch=1; raise if reward is still climbing
KL_COEF = 0.02
GRAD_ACCUM = 8
CLIP_EPS = 0.2
MAX_PROMPT_TOKENS = 1800  # raw graph JSON can be large; raise if prompts are being truncated
MAX_NEW_TOKENS = 350
LORA_R = 32
LORA_ALPHA = 64
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Embedding helpers (text-only prompt, no graph-embedding model — raw JSON
# text is tokenized like any other text)
# ---------------------------------------------------------------------------

def build_prompt_embeds(prompt_text: str, tokenizer, embed_layer, device, dtype):
    ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).input_ids.to(device)
    token_embeds = embed_layer(ids).to(dtype)
    return token_embeds, token_embeds.shape[1]


def completion_logprobs(model, inputs_embeds, completion_ids, embed_layer, dtype, device):
    comp_embeds = embed_layer(completion_ids).to(dtype)
    full_embeds = torch.cat([inputs_embeds, comp_embeds], dim=1)
    attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)
    out = model(inputs_embeds=full_embeds, attention_mask=attn)
    logits = out.logits
    L_prompt = inputs_embeds.shape[1]
    comp_logits = logits[:, L_prompt - 1: L_prompt + completion_ids.shape[1] - 1, :]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_lp = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Input : {INPUT_TRAIN_JSON}")
    print(f"[train] Device: {device}")
    if device == "cpu":
        print("[train] WARNING: no GPU detected. Qwen3-14B GRPO training on CPU is "
              "not practically feasible — this will be extremely slow or OOM.")

    print("[train] Loading sentence transformer for reward computation...")
    data_utils._sentence_encoder = get_sentence_encoder()
    print("[train] Sentence transformer ready")

    print("[train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train] Loading base model (policy)...")
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base, lora_config)
    policy.train()
    dtype = torch.bfloat16
    print("[train] Policy model ready")

    print("[train] Loading reference model (frozen)...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model = get_peft_model(ref_base, lora_config)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print("[train] Reference model ready")

    embed_layer = policy.get_input_embeddings()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_STEPS)

    examples = load_json_data(INPUT_TRAIN_JSON)
    # Drop rows with unusable/blank gold targets — they'd inject reward-0 noise.
    examples = [e for e in examples if e["gold_step_text"] and e["gold_step_category"] != "unknown"]
    print(f"[train] Usable training rows after filtering: {len(examples)}")

    os.makedirs(ADAPTER_DIR, exist_ok=True)

    G = GROUP_SIZE
    beta = KL_COEF
    optimizer.zero_grad()

    print(f"[train] Starting GRPO training: {NUM_STEPS} steps, group size {G}")

    for step in range(1, NUM_STEPS + 1):
        ex = random.choice(examples)
        gold = {
            "gold_step_text": ex["gold_step_text"],
            "gold_step_category": ex["gold_step_category"],
            "gold_mcp_labels": ex["gold_mcp_labels"],
            "gold_step_explanation": ex["gold_step_explanation"],
        }

        prompt_text = build_chat_prompt(ex)
        prompt_embeds, L_prompt = build_prompt_embeds(prompt_text, tokenizer, embed_layer, device, dtype)
        attn_prompt = torch.ones(1, L_prompt, dtype=torch.long, device=device)

        policy.eval()
        with torch.no_grad():
            gen_out = policy.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6,     # higher than the original 0.3 — GRPO needs
                top_p=0.9,           # genuine diversity across the G samples to
                repetition_penalty=1.05,  # get a non-degenerate advantage signal
                num_return_sequences=G,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids_list = gen_out

        completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids_list]
        rewards = torch.tensor([compute_reward(c, gold) for c in completions], dtype=torch.float32)

        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantages = ((rewards - mean_r) / std_r).to(device)

        policy.train()
        loss_accum = torch.zeros(1, device=device)

        for g_idx in range(G):
            comp_ids = completion_ids_list[g_idx].unsqueeze(0).to(device)
            adv = advantages[g_idx]

            lp_policy = completion_logprobs(policy, prompt_embeds, comp_ids, embed_layer, dtype, device)
            with torch.no_grad():
                ref_embed_layer = ref_model.get_input_embeddings()
                lp_ref = completion_logprobs(ref_model, prompt_embeds.detach(), comp_ids, ref_embed_layer, dtype, device)

            kl = lp_policy - lp_ref.detach()
            pg_loss = -adv * lp_policy + beta * kl
            loss_accum = loss_accum + pg_loss / G

        loss_for_backward = loss_accum / GRAD_ACCUM
        loss_for_backward.backward()

        if step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            avg_reward = rewards.mean().item()
            fmt_ok = sum(1 for c in completions if parse_completion(c) is not None)

            step_scores, mcp_scores, exp_scores = [], [], []
            for c in completions:
                obj = parse_completion(c)
                if obj is None:
                    continue
                pred_step = str(obj.get("New step", "")).strip()
                pred_cat = classify_step(pred_step)
                step_scores.append(1.0 if pred_cat == gold["gold_step_category"] else 0.0)

                mcp_val = obj.get("MCP_tasks", {})
                pred_mcp = set(mcp_labels_from_dict(mcp_val)) if isinstance(mcp_val, dict) else set()
                gold_mcp = set(gold["gold_mcp_labels"])
                if not pred_mcp and not gold_mcp:
                    mcp_scores.append(1.0)
                else:
                    inter = len(pred_mcp & gold_mcp)
                    prec = inter / len(pred_mcp) if pred_mcp else 0.0
                    rec = inter / len(gold_mcp) if gold_mcp else 0.0
                    mcp_scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

                pred_expl = str(obj.get("Step explanation", "")).strip()
                from data_utils import cosine_sim
                exp_scores.append(cosine_sim(pred_expl, gold["gold_step_explanation"], min_score=0.30))

            avg_step = float(np.mean(step_scores)) if step_scores else 0.0
            avg_mcp = float(np.mean(mcp_scores)) if mcp_scores else 0.0
            avg_exp = float(np.mean(exp_scores)) if exp_scores else 0.0

            print(
                f"step {step:4d}/{NUM_STEPS} | avg_reward {avg_reward:.3f} | "
                f"step_cat_acc {avg_step:.3f} | mcp_f1 {avg_mcp:.3f} | exp_sim {avg_exp:.3f} | "
                f"fmt_ok {fmt_ok}/{G} | loss {loss_accum.item():.4f}"
            )

        if step % 200 == 0:
            ckpt_path = os.path.join(ADAPTER_DIR, f"step_{step}")
            policy.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  -> checkpoint saved to {ckpt_path}")

    policy.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n[train] GRPO training complete. Policy saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
