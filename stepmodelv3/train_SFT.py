"""
train_supervised.py — Supervised fine-tuning of Qwen2.5-14B on stepmodelv3 data.

This trains the model to:
1. Predict the next step from fixed STEP_LABELS (classification)
2. Predict MCP tools from fixed MCP_LABELS (multi-label classification)
3. Generate step explanation (free-text generation)

Uses raw JSON graph input (no embeddings) for simplicity.
"""
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import data_utils
from prompts import build_chat_prompt

INPUT_TRAIN_JSON = "input/train.json"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_DIR = "/tmp/stage1_supervised"  # Use /tmp which typically has more space
MAX_PROMPT_TOKENS = 1200
MAX_NEW_TOKENS = 350
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 5e-6
NUM_EPOCHS = 3
LORA_R = 32
LORA_ALPHA = 64
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class StepDataset(Dataset):
    def __init__(self, examples, tokenizer, max_prompt_tokens, max_new_tokens):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Build prompt using existing function
        prompt_text = build_chat_prompt(ex)
        
        # Build target: classify step to fixed label, classify MCP tools to fixed labels
        # and generate explanation
        # Note: data_utils.load_json_data maps fields to different names
        target_step = ex["gold_step_category"]  # Already classified
        if target_step == "unknown":
            # If classification fails, use the original text
            target_step = ex["gold_step_text"]
        
        # Use pre-normalized MCP labels from data loading
        mcp_tasks = ex["gold_mcp_dict"]
        normalized_mcp = {}
        for tool_name, tool_desc in mcp_tasks.items():
            normalized = data_utils.normalize_tool(tool_name)
            if normalized:
                normalized_mcp[normalized] = tool_desc
        
        target = {
            "New step": target_step,
            "Step explanation": ex["gold_step_explanation"],
            "MCP_tasks": normalized_mcp
        }
        target_text = json.dumps(target, indent=2)
        
        # Combine prompt and target
        full_text = prompt_text + target_text
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_prompt_tokens + self.max_new_tokens,
            padding=False,
            return_tensors="pt"
        )
        
        # Create labels (mask prompt tokens)
        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_prompt_tokens,
            return_tensors="pt"
        )["input_ids"]
        
        labels = encoding["input_ids"].clone()
        labels[0, :prompt_ids.shape[1]] = -100  # Mask prompt tokens
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }


def main():
    print("[train] Loading data...")
    examples = data_utils.load_json_data(INPUT_TRAIN_JSON)
    # Don't filter - use all examples
    print(f"[train] Total examples: {len(examples)}")
    
    print("[train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[train] Creating dataset...")
    dataset = StepDataset(examples, tokenizer, MAX_PROMPT_TOKENS, MAX_NEW_TOKENS)
    
    print("[train] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    
    print("[train] Setting up LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("[train] Setting up trainer...")
    training_args = TrainingArguments(
        output_dir=ADAPTER_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="no",  # Disable automatic saving to avoid disk quota
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    
    print("[train] Starting training...")
    trainer.train()
    
    print("[train] Saving model...")
    # Save only essential files to avoid disk quota
    import os
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR, safe_serialization=False)  # Use pytorch format instead of safetensors
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"[train] Model saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
