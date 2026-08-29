"""
stage2_sft_qwen.py
==================
Stage 2 training for text-only experiment (no graph).
Supervised fine-tuning of Qwen LLM without graph conditioning.

Input: new_strategy + strategy_explanation
Output: step, MCP tools, and step explanation

Usage:
    python stage2_sft_qwen.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Add parent directories to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    ROOT, INPUT_TRAIN_JSON, INPUT_TEST_JSON, STAGE2_ADAPTER_DIR,
    STEP_LABELS, MCP_LABELS, STEP2IDX, MCP2IDX, IDX2STEP, IDX2MCP,
    QWEN_MODEL_NAME,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    STAGE2_LR, STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_GRAD_ACCUM,
    STAGE2_VAL_SPLIT, STAGE2_EARLY_STOP_PATIENCE, STAGE2_GRAD_CLIP, STAGE2_WARMUP_RATIO,
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


def build_target(ex: dict) -> str:
    """Build target JSON from example."""
    step = ex.get('gold_new_step', '')
    mcp_text = ex.get('gold_mcp_tasks', '')
    explanation = ex.get('gold_step_explanation', '')
    
    # Parse MCP tasks
    mcp_tasks = {}
    if mcp_text:
        for tool in MCP_LABELS:
            if tool.lower() in mcp_text.lower():
                mcp_tasks[tool] = f"use {tool}"
    
    target = {
        "New step": step,
        "MCP_tasks": mcp_tasks,
        "Step explanation": explanation,
    }
    return json.dumps(target, ensure_ascii=False)


class SFTDataset(Dataset):
    """Dataset for SFT training."""
    
    def __init__(self, examples, tokenizer, max_len=1536):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        prompt_text = build_prompt(ex)
        target_text = build_target(ex)
        
        full_text = prompt_text + target_text + "<|im_end|>"
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Mask prompt tokens (only compute loss on target)
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)['input_ids'])
        labels = encoding['input_ids'].clone()
        labels[:, :prompt_len] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
        }


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
    all_examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
    
    # Machine-level split for validation
    all_machines = sorted(set(e['machine'] for e in all_examples))
    rng_split = np.random.default_rng(RANDOM_SEED + 1)
    perm_machines = rng_split.permutation(len(all_machines))
    n_val_machines = max(1, int(len(all_machines) * STAGE2_VAL_SPLIT))
    val_machine_set = set(all_machines[i] for i in perm_machines[:n_val_machines])
    
    train_examples = [e for e in all_examples if e['machine'] not in val_machine_set]
    val_examples = [e for e in all_examples if e['machine'] in val_machine_set]
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples: {len(val_examples)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Datasets
    train_ds = SFTDataset(train_examples, tokenizer)
    val_ds = SFTDataset(val_examples, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=STAGE2_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=STAGE2_BATCH_SIZE, shuffle=False)
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{STAGE2_EPOCHS}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / STAGE2_GRAD_ACCUM
            
            loss.backward()
            
            if (step + 1) % STAGE2_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), STAGE2_GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * STAGE2_GRAD_ACCUM
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            model.save_pretrained(STAGE2_ADAPTER_DIR)
            tokenizer.save_pretrained(STAGE2_ADAPTER_DIR)
            print(f"  -> Saved best model (epoch {best_epoch})")
        else:
            patience_counter += 1
            if patience_counter >= STAGE2_EARLY_STOP_PATIENCE:
                print(f"  -> Early stopping (no improvement for {patience_counter} epochs)")
                break
    
    print(f"\nTraining complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
