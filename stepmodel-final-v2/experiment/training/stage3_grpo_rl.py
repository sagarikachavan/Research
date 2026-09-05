"""
stage3_grpo_rl.py
==================
Stage 3 training for text-only experiment (no graph).
GRPO RL fine-tuning of Qwen LLM without graph conditioning.

Input: new_strategy + strategy_explanation
Output: step, MCP tools, and step explanation (optimized via RL)

Usage:
    python stage3_grpo_rl.py

--------------------------------------------------------------------------
FIXES vs. the original version of this file (see CHANGES for details):

1. Memory / "stuck" fix: the old code called `merge_and_unload()` on the
   Stage 2 LoRA adapter and then loaded a SECOND full copy of the 14B
   base model as `ref_model`. That means ~2x the full model resident in
   memory (~56GB in fp16) AND full fine-tuning of all 14B params instead
   of LoRA (since merging removes the adapter). This is almost certainly
   what caused Stage 3 to hang/OOM. Fix: load the base model ONCE and
   attach the Stage 2 adapter twice under two names ("policy", trainable,
   and "ref", frozen) via PEFT's multi-adapter API, switching the active
   adapter as needed. Only the small LoRA "policy" params are trainable.

2. Checkpoint-format fix: because the model is no longer merged, it stays
   a PeftModel, so `model.save_pretrained(STAGE3_ADAPTER_DIR)` now saves
   a proper LoRA adapter -- consistent with what eval/evaluate.py expects
   (`PeftModel.from_pretrained(base_model, STAGE3_ADAPTER_DIR)`). The old
   code saved a full merged model there, which evaluate.py could not load.

3. Correctness fix: the old code computed "policy_logprob" by running the
   model on the PROMPT ONLY with `labels=input_ids` (i.e. the loss of the
   model predicting its own already-known prompt tokens). That has
   nothing to do with the text that was actually generated/sampled, so
   the RL update wasn't optimizing towards the sampled completions at
   all. Fix: compute the actual sum of log-probs the model assigns to the
   GENERATED completion tokens (prompt+completion sequence, loss masked
   to the completion span), for both the policy and the reference model.

4. Visibility fix: added a tqdm progress bar over RL steps and a print
   every step (not just every 100) so the run doesn't look stuck.
--------------------------------------------------------------------------
"""

import os
import sys
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Add parent directories to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    ROOT, INPUT_TRAIN_JSON, INPUT_TEST_JSON, STAGE2_ADAPTER_DIR, STAGE3_ADAPTER_DIR,
    STEP_LABELS, MCP_LABELS, STEP2IDX, MCP2IDX, IDX2STEP, IDX2MCP,
    QWEN_MODEL_NAME,
    STAGE3_GROUP_SIZE, STAGE3_LR, STAGE3_STEPS,
    STAGE3_KL_COEF, STAGE3_PPO_CLIP, STAGE3_GRAD_ACCUM, STAGE3_GRAD_CLIP,
    RANDOM_SEED,
)


SYSTEM_PROMPT = """You are an expert penetration testing assistant. Given a strategy and explanation, determine the next step, the tools needed, and explain your reasoning.

Respond in JSON format with the following structure:
{
    "New step": "<one of the 10 step labels>",
    "MCP_tasks": {
        "<tool_name>": "<short action description>",
        ...
    },
    "Step explanation": "<detailed explanation of why this step is appropriate>"
}

Available step labels:
- Do a google search for more information
- Enumerate further on the X service to find software versions, hidden directories and file.
- Explore the suspicious files, commands and create a summary of the findings.
- Further Enumerate the website. - hidden directories, links and software
- Enumerate the domain
- Exploit the selected exploitations
- Analyze the outcomes of the previous step and find an attack path
- Ask for human assistant
- Explore the source code for vulnerabilities.
- End task and ask permission to generate the report

Available MCP tools: Nmap, Metasploit, Netcat, Dirbuster, SQLmap, Smb client, hydra, John-the-ripper, Google search, Interactive CLI, Web page interaction
"""


def build_prompt(ex: dict) -> str:
    """Build prompt from example (no graph, no hint)."""
    ctx = f"Strategy: {ex.get('new_strategy', '')}\nExplanation: {ex.get('strategy_explanation', '')}"
    lines = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>",
        f"<|im_start|>user\n{ctx}<|im_end|>",
        f"<|im_start|>assistant\n",
    ]
    return "\n".join(lines)


def parse_response(response_text: str):
    """Parse model response into step, MCP, and explanation."""
    try:
        obj = json.loads(response_text)
        step = obj.get("New step", "")
        mcp_tasks = obj.get("MCP_tasks", {})
        explanation = obj.get("Step explanation", "")
        return step, mcp_tasks, explanation
    except Exception:
        return "", {}, ""


def compute_reward(pred_step, pred_mcp, pred_expl, gold_step, gold_mcp_text, gold_expl):
    """Compute reward for GRPO."""
    reward = 0.0

    # Step similarity (exact match)
    if pred_step == gold_step:
        reward += 1.0

    # MCP F1
    gold_mcp_labels = [0] * len(MCP_LABELS)
    if gold_mcp_text:
        for i, tool in enumerate(MCP_LABELS):
            if tool.lower() in gold_mcp_text.lower():
                gold_mcp_labels[i] = 1

    pred_mcp_labels = [0] * len(MCP_LABELS)
    if pred_mcp:
        for tool in MCP_LABELS:
            if tool in pred_mcp:
                pred_mcp_labels[MCP2IDX[tool]] = 1

    if sum(gold_mcp_labels) > 0:
        tp = sum(1 for g, p in zip(gold_mcp_labels, pred_mcp_labels) if g == 1 and p == 1)
        fp = sum(1 for g, p in zip(gold_mcp_labels, pred_mcp_labels) if g == 0 and p == 1)
        fn = sum(1 for g, p in zip(gold_mcp_labels, pred_mcp_labels) if g == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        reward += f1

    # Explanation length (proxy for quality)
    if len(pred_expl) > 20:
        reward += 0.5

    return reward


def load_from_input_json(path, split):
    """Load examples from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"[{split}] Loaded {len(examples)} examples from {path}")
    return examples


def completion_logprob(model, tokenizer, prompt_ids, completion_ids, device):
    """
    Sum of per-token log-probs the model assigns to `completion_ids`
    given `prompt_ids` as context. This is what actually needs to be
    optimized for policy-gradient-style RL -- the old version of this
    file mistakenly computed a loss over the prompt tokens instead.

    prompt_ids: (1, L_prompt) LongTensor
    completion_ids: (1, L_gen) LongTensor
    Returns: scalar tensor (sum of log-probs over the L_gen completion tokens)
    """
    if completion_ids.shape[1] == 0:
        # Degenerate/empty completion: heavily penalized log-prob so it
        # doesn't crash and is naturally discouraged.
        return torch.tensor(-50.0, device=device, requires_grad=True)

    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attn = torch.ones_like(full_ids)

    out = model(input_ids=full_ids, attention_mask=attn)
    logits = out.logits  # (1, L_prompt+L_gen, V)

    L_prompt = prompt_ids.shape[1]
    L_gen = completion_ids.shape[1]

    # logits[:, L_prompt-1 : L_prompt+L_gen-1] predict completion_ids[:, 0:L_gen]
    comp_logits = logits[:, L_prompt - 1: L_prompt + L_gen - 1, :]
    log_probs = F.log_softmax(comp_logits.float(), dim=-1)
    token_lp = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)  # (1, L_gen)
    return token_lp.sum()


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no CUDA device found. Stage 3 generates + backprops through a "
              "14B-parameter model every RL step -- on CPU this will be extremely slow "
              "and may look 'stuck' even though it's just very slow. Consider a smaller "
              "STAGE3_STEPS/STAGE3_GROUP_SIZE for a CPU smoke test, or run on a GPU.")

    # Load data
    all_train_examples = load_from_input_json(INPUT_TRAIN_JSON, "train")

    # Machine-level split for validation
    all_machines = sorted(set(e['machine'] for e in all_train_examples))
    rng_split = np.random.default_rng(RANDOM_SEED + 1)
    perm_machines = rng_split.permutation(len(all_machines))
    n_val_machines = max(1, int(len(all_machines) * 0.15))
    val_machine_set = set(all_machines[i] for i in perm_machines[:n_val_machines])

    train_examples = [e for e in all_train_examples if e['machine'] not in val_machine_set]
    val_examples = [e for e in all_train_examples if e['machine'] in val_machine_set]

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")

    if not os.path.isdir(STAGE2_ADAPTER_DIR) or not os.listdir(STAGE2_ADAPTER_DIR):
        raise FileNotFoundError(
            f"Stage 2 adapter not found at {STAGE2_ADAPTER_DIR}. "
            f"Run stage2_sft_qwen.py successfully first -- Stage 3 continues training "
            f"from the Stage 2 LoRA adapter."
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Load the base model ONCE, and attach the Stage 2 adapter under two
    # names on top of it: "policy" (trainable, this is what we optimize) and
    # "ref" (frozen, this is what we compute the KL penalty against). This
    # avoids ever holding two full copies of the 14B model in memory, which
    # is what caused the old script to hang/OOM. ---
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base_model, STAGE2_ADAPTER_DIR, adapter_name="default", is_trainable=True
    )
    model.load_adapter(STAGE2_ADAPTER_DIR, adapter_name="ref", is_trainable=False)
    # is_trainable=False above already freezes the "ref" adapter's params;
    # this is just a safety net (PEFT names LoRA params like
    # "...lora_A.ref.weight" / "...lora_B.ref.weight").
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad = False

    model.set_adapter("default")

    # Only the (small) "policy" LoRA params should be trainable.
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters (policy LoRA only): {n_trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=STAGE3_LR)

    # Training loop
    best_val_reward = 0.0
    global_step = 0
    grad_accum_counter = 0
    optimizer.zero_grad()

    pbar = tqdm(range(STAGE3_STEPS), desc="Stage 3 GRPO")
    for step in pbar:
        model.set_adapter("default")
        model.train()

        # Sample batch
        group_size = min(STAGE3_GROUP_SIZE, len(train_examples))
        batch_indices = np.random.choice(len(train_examples), size=group_size, replace=False)
        batch_examples = [train_examples[i] for i in batch_indices]

        all_rewards = []
        prompt_ids_list = []
        completion_ids_list = []

        # ---- 1. Generate completions with the policy adapter active ----
        for ex in batch_examples:
            prompt = build_prompt(ex)
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = inputs['input_ids'].shape[1]
            completion_ids = outputs[:, prompt_len:]
            response = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
            pred_step, pred_mcp, pred_expl = parse_response(response)

            reward = compute_reward(
                pred_step, pred_mcp, pred_expl,
                ex.get('gold_new_step', ''),
                ex.get('gold_mcp_tasks', ''),
                ex.get('gold_step_explanation', '')
            )
            all_rewards.append(reward)
            prompt_ids_list.append(inputs['input_ids'])
            completion_ids_list.append(completion_ids)

        # ---- 2. Group-relative advantages (GRPO) ----
        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - mean_reward) / std_reward
        advantages = torch.clamp(advantages, -4.0, 4.0)

        # ---- 3. Policy gradient loss + KL penalty against the frozen
        # Stage-2 reference, using log-probs of the ACTUAL generated
        # completion tokens (this is the correctness fix). ----
        policy_loss = torch.zeros((), device=device)
        kl_loss = torch.zeros((), device=device)
        n_valid = 0

        for i, ex in enumerate(batch_examples):
            prompt_ids = prompt_ids_list[i]
            completion_ids = completion_ids_list[i]
            if completion_ids.shape[1] == 0:
                continue
            n_valid += 1

            model.set_adapter("default")
            model.train()
            lp_policy = completion_logprob(model, tokenizer, prompt_ids, completion_ids, device)

            with torch.no_grad():
                model.set_adapter("ref")
                model.eval()  # disable dropout so the reference logprob is a clean,
                              # low-variance target rather than dropout-noisy
                lp_ref = completion_logprob(model, tokenizer, prompt_ids, completion_ids, device)
            model.set_adapter("default")
            model.train()

            mean_lp_policy = lp_policy / completion_ids.shape[1]
            mean_lp_ref = lp_ref / completion_ids.shape[1]

            kl = mean_lp_policy - mean_lp_ref.detach()
            kl_loss = kl_loss + kl

            # Policy gradient: maximize advantage-weighted log-prob
            # => minimize -advantage * logprob
            policy_loss = policy_loss + (-advantages[i] * mean_lp_policy)

        if n_valid == 0:
            print(f"Step {step}: all completions were empty, skipping update.")
            continue

        policy_loss = policy_loss / n_valid
        kl_loss = kl_loss / n_valid
        total_loss = (policy_loss + STAGE3_KL_COEF * kl_loss) / STAGE3_GRAD_ACCUM

        total_loss.backward()
        grad_accum_counter += 1

        if grad_accum_counter >= STAGE3_GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(trainable_params, STAGE3_GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            grad_accum_counter = 0

        global_step += 1
        pbar.set_postfix({
            "reward": f"{mean_reward.item():.3f}",
            "pg_loss": f"{policy_loss.item():.3f}",
            "kl": f"{kl_loss.item():.4f}",
        })

        if device.type == "cuda":
            del rewards, advantages
            gc.collect()
            torch.cuda.empty_cache()

        if global_step % 100 == 0:
            print(f"Step {global_step}: Mean Reward: {mean_reward:.4f}, "
                  f"Policy Loss: {policy_loss:.4f}, KL: {kl_loss:.4f}")

            # Validation
            model.set_adapter("default")
            model.eval()
            val_rewards = []
            with torch.no_grad():
                for ex in val_examples[:32]:  # Sample for validation
                    prompt = build_prompt(ex)
                    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=False,  # Greedy for validation
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    pred_step, pred_mcp, pred_expl = parse_response(response)

                    reward = compute_reward(
                        pred_step, pred_mcp, pred_expl,
                        ex.get('gold_new_step', ''),
                        ex.get('gold_mcp_tasks', ''),
                        ex.get('gold_step_explanation', '')
                    )
                    val_rewards.append(reward)

            avg_val_reward = np.mean(val_rewards) if val_rewards else 0.0
            print(f"  Val Reward: {avg_val_reward:.4f}")

            if avg_val_reward > best_val_reward:
                best_val_reward = avg_val_reward
                # model is a PeftModel with the "policy" adapter active, so
                # this correctly saves just the (small) LoRA adapter -- the
                # same format evaluate.py expects to load.
                model.set_adapter("default")
                model.save_pretrained(STAGE3_ADAPTER_DIR, selected_adapters=["default"])
                tokenizer.save_pretrained(STAGE3_ADAPTER_DIR)
                print(f"  -> Saved best model")

    print(f"\nTraining complete. Best val reward: {best_val_reward:.4f}")


if __name__ == "__main__":
    main()
