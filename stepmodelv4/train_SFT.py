"""
train_SFT.py — Supervised fine-tuning of Qwen2.5-14B with graph embeddings for stepmodelv4

This trains the model to:
1. Predict the next step from fixed STEP_LABELS (classification)
2. Predict MCP tools from fixed MCP_LABELS (multi-label classification)
3. Generate step explanation (free-text generation)

Uses graph embeddings via Graph Prefix Adapter to incorporate graph structure.
"""
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import data_utils
from config import (
    INPUT_TRAIN_JSON, QWEN_MODEL_NAME, SUPERVISED_ADAPTER_DIR,
    MAX_PROMPT_TOKENS, MAX_NEW_TOKENS, BATCH_SIZE, GRAD_ACCUM,
    LR, NUM_EPOCHS, LORA_R, LORA_ALPHA, LORA_TARGETS, RANDOM_SEED,
    GNN_CKPT, ADAPTER_CKPT, PREFIX_TOKENS,
)
from prompts import build_chat_prompt
from graph_prefix_adapter import GraphPrefixAdapter, load_graph_encoder_and_adapter
from graph_encoder import Stage1Classifier

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class StepDataset(Dataset):
    def __init__(self, examples, tokenizer, graph_encoder, graph_adapter, max_prompt_tokens, max_new_tokens, device):
        self.examples = examples
        self.tokenizer = tokenizer
        self.graph_encoder = graph_encoder
        self.graph_adapter = graph_adapter
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Build prompt using existing function
        prompt_text = build_chat_prompt(ex)
        
        # Build target
        target_step = ex["gold_step_category"]
        if target_step == "unknown":
            target_step = ex["gold_step_text"]
        
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
        
        # Get graph embedding
        with torch.no_grad():
            graph = ex["graph"].to(self.device)
            field_embs = torch.tensor(ex["field_embs"], dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, graph_emb = self.graph_encoder(
                graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device),
                field_embs
            )
            soft_prompt = self.graph_adapter(graph_emb)  # (1, num_tokens, llm_dim)
        
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
            "labels": labels.squeeze(0),
            "soft_prompt": soft_prompt.squeeze(0),  # Store for later use
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")
    
    print("[train] Loading data...")
    examples = data_utils.load_json_data(INPUT_TRAIN_JSON)
    print(f"[train] Total examples: {len(examples)}")
    
    print("[train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[train] Loading graph encoder and adapter...")
    graph_encoder, graph_adapter = load_graph_encoder_and_adapter(GNN_CKPT, ADAPTER_CKPT, device)
    print("[train] Graph encoder and adapter loaded")
    
    print("[train] Creating dataset...")
    dataset = StepDataset(examples, tokenizer, graph_encoder, graph_adapter, MAX_PROMPT_TOKENS, MAX_NEW_TOKENS, device)
    
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
        output_dir=SUPERVISED_ADAPTER_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=200,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        report_to="none",
        logging_first_step=True
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
    import os
    os.makedirs(SUPERVISED_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(SUPERVISED_ADAPTER_DIR, safe_serialization=False)
    tokenizer.save_pretrained(SUPERVISED_ADAPTER_DIR)
    print(f"[train] Model saved to {SUPERVISED_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
