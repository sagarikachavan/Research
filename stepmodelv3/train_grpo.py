"""
train_grpo.py — GRPO (Group Relative Policy Optimization) RL fine-tuning of
Qwen2.5-14B on stepmodelv3/input/train.json.

Stage 2: Loads supervised checkpoint and refines explanation quality using GRPO.
The supervised stage already teaches the model:
- Step classification from fixed STEP_LABELS
- MCP tool classification from fixed MCP_LABELS  
- Basic explanation generation

GRPO then refines the explanation quality using the reward function.
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
from peft import LoraConfig, get_peft_model, PeftModel

from data_utils import load_json_data, compute_reward, parse_completion, classify_step, mcp_labels_from_dict, get_sentence_encoder
import data_utils
from prompts import build_chat_prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_TRAIN_JSON = "input/train.json"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
SUPERVISED_ADAPTER_DIR = "/tmp/stage1_supervised"  # Load from supervised checkpoint (using /tmp for disk space)
ADAPTER_DIR = "/tmp/stage2_grpo_rl"  # Save GRPO-refined model here (using /tmp for disk space)

GROUP_SIZE = 4            # increased from 2 for more stable advantage estimation
LR = 5e-6                 # reduced from 1e-5 for more stable updates
NUM_STEPS = 3000          # ~10-12 epochs over 1728 rows at batch=1; raise if reward is still climbing
KL_COEF = 0.02
GRAD_ACCUM = 16           # increased from 8 for smoother gradient updates
CLIP_EPS = 0.2
MAX_PROMPT_TOKENS = 1200  # reduced from 1800 for memory
MAX_NEW_TOKENS = 350      # increased from 250 to allow complete JSON responses
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


def completion_logprobs(model, input_ids, completion_ids, dtype, device):
    """Compute log probabilities of completion tokens given input_ids."""
    full_ids = torch.cat([input_ids, completion_ids], dim=1)
    attn = torch.ones(full_ids.shape[:2], dtype=torch.long, device=device)
    out = model(input_ids=full_ids, attention_mask=attn)
    logits = out.logits
    L_prompt = input_ids.shape[1]
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
    # Load supervised checkpoint directly if available, otherwise load base model
    if os.path.exists(SUPERVISED_ADAPTER_DIR):
        print(f"[train] Loading supervised checkpoint from {SUPERVISED_ADAPTER_DIR}")
        base = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        base.gradient_checkpointing_enable()
        policy = PeftModel.from_pretrained(base, SUPERVISED_ADAPTER_DIR)
        # Enable training for LoRA parameters
        policy.enable_adapter_layers()
        for param in policy.parameters():
            if param.requires_grad:
                param.requires_grad_(True)
    else:
        print(f"[train] No supervised checkpoint found at {SUPERVISED_ADAPTER_DIR}, training from scratch")
        base = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        base.gradient_checkpointing_enable()
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base, lora_config)
    
    policy.train()
    dtype = torch.bfloat16
    print("[train] Policy model ready")

    print("[train] Loading reference model (frozen)...")
    # Load supervised checkpoint for reference model too
    if os.path.exists(SUPERVISED_ADAPTER_DIR):
        print(f"[train] Loading supervised checkpoint for reference model")
        ref_base = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        ref_base.gradient_checkpointing_enable()
        ref_model = PeftModel.from_pretrained(ref_base, SUPERVISED_ADAPTER_DIR)
    else:
        print(f"[train] No supervised checkpoint found for reference model, using base model")
        ref_base = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
        ref_base.gradient_checkpointing_enable()
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        ref_model = get_peft_model(ref_base, lora_config)
    
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print("[train] Reference model ready")

    embed_layer = policy.get_input_embeddings()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    
    # Debug: print trainable parameter count
    print(f"[train] Trainable parameters: {len(trainable)}")
    if len(trainable) == 0:
        print("[train] WARNING: No trainable parameters found!")
        print("[train] Forcing all LoRA parameters to be trainable...")
        # Force enable all parameters
        for name, param in policy.named_parameters():
            if "lora" in name.lower():
                param.requires_grad_(True)
                print(f"[train] Enabled: {name}")
        trainable = [p for p in policy.parameters() if p.requires_grad]
        print(f"[train] Trainable parameters after forcing: {len(trainable)}")
    
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
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        ).input_ids.to(device)
        L_prompt = prompt_ids.shape[1]
        attn_prompt = torch.ones(1, L_prompt, dtype=torch.long, device=device)

        policy.eval()
        with torch.no_grad():
            gen_out = policy.generate(
                input_ids=prompt_ids,
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

        # Slice to get only the completion tokens (after the prompt)
        # gen_out shape: [G, seq_len] when num_return_sequences=G
        completion_only = gen_out[:, L_prompt:]
        completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_only]

        # Brief check for first 10 steps to see if MCP_tasks are generated
        if step <= 10:
            from data_utils import parse_completion
            parsed = parse_completion(completions[0])
            if parsed:
                print(f"[step {step}] Full JSON response:")
                print(json.dumps(parsed, indent=2))
            else:
                print(f"[step {step}] JSON parse failed.")
                print(f"[step {step}] Raw completion (first 500 chars): {completions[0][:500]}")
                print(f"[step {step}] Raw completion (last 500 chars): {completions[0][-500:]}")
                print(f"[step {step}] Completion length: {len(completions[0])}")

        rewards = torch.tensor([compute_reward(c, gold) for c in completions], dtype=torch.float32)

        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantages = ((rewards - mean_r) / std_r).to(device)

        policy.train()
        loss_accum = torch.zeros(1, device=device)

        for g_idx in range(G):
            # Get the full sequence (prompt + completion) for this sample
            full_ids = gen_out[g_idx].unsqueeze(0).to(device)
            # Get only completion tokens for logprob computation
            comp_ids = full_ids[:, L_prompt:]
            adv = advantages[g_idx]

            lp_policy = completion_logprobs(policy, prompt_ids, comp_ids, dtype, device)
            with torch.no_grad():
                lp_ref = completion_logprobs(ref_model, prompt_ids, comp_ids, dtype, device)

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
