"""
stage3_grpo_rl.py
==================
Stage 3 training for text-only experiment (no graph).
GRPO RL fine-tuning of Qwen LLM without graph conditioning.

Input: new_strategy + strategy_explanation
Output: step, MCP tools, and step explanation (optimized via RL)

Usage:
    python stage3_grpo_rl.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    except:
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


class RLDataset(Dataset):
    """Dataset for RL training."""
    
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def load_from_input_json(path, split):
    """Load examples from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"[{split}] Loaded {len(examples)} examples from {path}")
    return examples


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load Stage 2 adapter
    model = PeftModel.from_pretrained(base_model, STAGE2_ADAPTER_DIR)
    model = model.merge_and_unload()
    
    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    ref_model = PeftModel.from_pretrained(ref_model, STAGE2_ADAPTER_DIR)
    ref_model = ref_model.merge_and_unload()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE3_LR)
    
    # Training loop
    best_val_reward = 0.0
    global_step = 0
    
    for step in range(STAGE3_STEPS):
        model.train()
        
        # Sample batch
        batch_indices = np.random.choice(len(train_examples), size=STAGE3_GROUP_SIZE, replace=False)
        batch_examples = [train_examples[i] for i in batch_indices]
        
        all_rewards = []
        all_logprobs = []
        all_ref_logprobs = []
        
        # Generate completions
        for ex in batch_examples:
            prompt = build_prompt(ex)
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Policy generation
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
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred_step, pred_mcp, pred_expl = parse_response(response)
            
            # Compute reward
            reward = compute_reward(
                pred_step, pred_mcp, pred_expl,
                ex.get('gold_new_step', ''),
                ex.get('gold_mcp_tasks', ''),
                ex.get('gold_step_explanation', '')
            )
            all_rewards.append(reward)
        
        # GRPO update
        rewards = torch.tensor(all_rewards, dtype=torch.float32).to(device)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        # Compute policy loss
        policy_loss = 0.0
        kl_loss = 0.0
        
        for i, ex in enumerate(batch_examples):
            prompt = build_prompt(ex)
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Policy logprobs
            outputs = model(**inputs, labels=inputs['input_ids'])
            policy_logprob = outputs.loss
            
            # Reference logprobs
            with torch.no_grad():
                ref_outputs = ref_model(**inputs, labels=inputs['input_ids'])
                ref_logprob = ref_outputs.loss
            
            # KL penalty
            kl = policy_logprob - ref_logprob
            kl_loss += kl
            
            # Policy loss with advantage
            policy_loss += -advantages[i] * policy_logprob
        
        policy_loss = policy_loss / STAGE3_GROUP_SIZE
        kl_loss = kl_loss / STAGE3_GROUP_SIZE
        
        total_loss = policy_loss + STAGE3_KL_COEF * kl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), STAGE3_GRAD_CLIP)
        optimizer.step()
        
        global_step += 1
        
        if global_step % 100 == 0:
            print(f"Step {global_step}: Mean Reward: {mean_reward:.4f}, Policy Loss: {policy_loss:.4f}, KL Loss: {kl_loss:.4f}")
            
            # Validation
            model.eval()
            val_rewards = []
            with torch.no_grad():
                for ex in val_examples[:32]:  # Sample for validation
                    prompt = build_prompt(ex)
                    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
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
            
            avg_val_reward = np.mean(val_rewards)
            print(f"  Val Reward: {avg_val_reward:.4f}")
            
            if avg_val_reward > best_val_reward:
                best_val_reward = avg_val_reward
                model.save_pretrained(STAGE3_ADAPTER_DIR)
                tokenizer.save_pretrained(STAGE3_ADAPTER_DIR)
                print(f"  -> Saved best model")
    
    print(f"\nTraining complete. Best val reward: {best_val_reward:.4f}")


if __name__ == "__main__":
    main()
