"""
train_grpo.py — GRPO RL fine-tuning with graph embeddings for stepmodelv4

Stage 2: Loads supervised checkpoint and refines explanation quality using GRPO.
Uses graph embeddings via Graph Prefix Adapter to incorporate graph structure.
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

from config import (
    INPUT_TRAIN_JSON, QWEN_MODEL_NAME, SUPERVISED_ADAPTER_DIR, GRPO_ADAPTER_DIR,
    GROUP_SIZE, GRPO_LR, GRPO_STEPS, KL_COEF, GRPO_GRAD_ACCUM,
    MAX_PROMPT_TOKENS, MAX_NEW_TOKENS, LORA_R, LORA_ALPHA, LORA_TARGETS,
    RANDOM_SEED, GNN_CKPT, ADAPTER_CKPT,
)
from data_utils import load_json_data, get_sentence_encoder
import data_utils
from prompts import build_chat_prompt
from graph_prefix_adapter import GraphPrefixAdapter, load_graph_encoder_and_adapter

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def build_prompt_embeds(prompt_text: str, tokenizer, embed_layer, device, dtype, max_length=MAX_PROMPT_TOKENS):
    ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
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


def compute_reward(completion: str, gold: dict) -> float:
    """Compute reward for completion against gold standard."""
    from data_utils import parse_completion, classify_step, mcp_labels_from_dict
    from sentence_transformers import SentenceTransformer
    
    # Parse completion
    parsed = parse_completion(completion)
    if not parsed:
        return 0.0
    
    # Step classification reward
    pred_step = classify_step(parsed.get("New step", ""))
    gold_step = gold["gold_step_category"]
    step_reward = 1.0 if pred_step == gold_step else 0.0
    
    # MCP classification reward
    pred_mcp = mcp_labels_from_dict(parsed.get("MCP_tasks", {}))
    gold_mcp = gold["gold_mcp_labels"]
    if len(gold_mcp) > 0:
        mcp_reward = len(set(pred_mcp) & set(gold_mcp)) / len(gold_mcp)
    else:
        mcp_reward = 1.0 if len(pred_mcp) == 0 else 0.0
    
    # Explanation similarity reward
    pred_expl = parsed.get("Step explanation", "")
    gold_expl = gold["gold_step_explanation"]
    if pred_expl and gold_expl:
        encoder = get_sentence_encoder()
        emb1 = encoder.encode(pred_expl, convert_to_numpy=True)
        emb2 = encoder.encode(gold_expl, convert_to_numpy=True)
        from sklearn.metrics.pairwise import cosine_similarity
        expl_reward = cosine_similarity([emb1], [emb2])[0][0]
    else:
        expl_reward = 0.0
    
    # Combined reward
    total_reward = 0.4 * step_reward + 0.3 * mcp_reward + 0.3 * expl_reward
    return total_reward


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Input : {INPUT_TRAIN_JSON}")
    print(f"[train] Device: {device}")

    print("[train] Loading sentence transformer for reward computation...")
    data_utils._sentence_encoder = get_sentence_encoder()
    print("[train] Sentence transformer ready")

    print("[train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train] Loading data...")
    examples = load_json_data(INPUT_TRAIN_JSON)
    print(f"[train] Loaded {len(examples)} examples")

    print("[train] Loading graph encoder and adapter...")
    graph_encoder, graph_adapter = load_graph_encoder_and_adapter(GNN_CKPT, ADAPTER_CKPT, device)
    print("[train] Graph encoder and adapter loaded")

    print("[train] Loading base model...")
    dtype = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto"
    )
    base.gradient_checkpointing_enable()

    print("[train] Setting up LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("[train] Loading supervised checkpoint...")
    if os.path.exists(SUPERVISED_ADAPTER_DIR):
        print(f"[train] Loading from {SUPERVISED_ADAPTER_DIR}")
        # Load as trainable by using from_pretrained with is_trainable=True
        policy = PeftModel.from_pretrained(base, SUPERVISED_ADAPTER_DIR, is_trainable=True)
        # Manually enable gradients for all LoRA parameters
        for name, param in policy.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        # Set to train mode
        policy.train()
        # Debug: count trainable parameters
        trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"[train] Trainable parameters after manual enable: {trainable_count:,}")
    else:
        print("[train] No supervised checkpoint found, training from scratch")
        policy = get_peft_model(base, lora_config)

    # Reference model (frozen) - create separate copy to avoid conflicts
    print("[train] Creating reference model...")
    base_ref = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_ref.gradient_checkpointing_enable()
    if os.path.exists(SUPERVISED_ADAPTER_DIR):
        ref_model = PeftModel.from_pretrained(base_ref, SUPERVISED_ADAPTER_DIR)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        ref_model = get_peft_model(base_ref, lora_config)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # Re-enable gradients for policy after reference model creation
    policy.train()
    for name, param in policy.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    # Collect trainable parameters
    trainable = [p for p in policy.parameters() if p.requires_grad]
    print(f"[train] Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=GRPO_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=GRPO_STEPS)

    print("[train] Starting GRPO training...")
    beta = KL_COEF
    G = GROUP_SIZE

    for step in range(1, GRPO_STEPS + 1):
        ex = random.choice(examples)
        gold = {
            "gold_step_text": ex["gold_step_text"],
            "gold_step_category": ex["gold_step_category"],
            "gold_mcp_labels": ex["gold_mcp_labels"],
            "gold_step_explanation": ex["gold_step_explanation"],
        }

        prompt_text = build_chat_prompt(ex)
        
        # Get graph embedding
        with torch.no_grad():
            graph = ex["graph"].to(device)
            field_embs = torch.tensor(ex["field_embs"], dtype=torch.float32).unsqueeze(0).to(device)
            _, _, graph_emb = graph_encoder(
                graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long, device=device),
                field_embs
            )
            soft_prompt = graph_adapter(graph_emb)  # (1, num_tokens, llm_dim)

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
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.05,
                num_return_sequences=G,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids_list = gen_out

        completion_only = gen_out[:, L_prompt:]
        completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_only]

        rewards = torch.tensor([compute_reward(c, gold) for c in completions], dtype=torch.float32)

        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantages = ((rewards - mean_r) / std_r).to(device)

        policy.train()
        loss_accum = torch.zeros(1, device=device)

        for g_idx in range(G):
            full_ids = gen_out[g_idx].unsqueeze(0).to(device)
            comp_ids = full_ids[:, L_prompt:]
            adv = advantages[g_idx]

            lp_policy = completion_logprobs(policy, prompt_ids, comp_ids, dtype, device)
            with torch.no_grad():
                lp_ref = completion_logprobs(ref_model, prompt_ids, comp_ids, dtype, device)

            kl = lp_policy - lp_ref.detach()
            pg_loss = -adv * lp_policy + beta * kl
            loss_accum = loss_accum + pg_loss / G

        loss_for_backward = loss_accum / GRPO_GRAD_ACCUM
        loss_for_backward.backward()

        if step % GRPO_GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if step % 100 == 0:
            print(f"step {step:4d} | reward {mean_r:.4f}±{std_r:.4f} | loss {loss_accum.item():.4f}")

        if step % 1000 == 0:
            print(f"[train] Saving checkpoint at step {step}")
            os.makedirs(GRPO_ADAPTER_DIR, exist_ok=True)
            policy.save_pretrained(GRPO_ADAPTER_DIR, safe_serialization=False)

    print("[train] Training complete")
    print(f"[train] Saving final model to {GRPO_ADAPTER_DIR}")
    os.makedirs(GRPO_ADAPTER_DIR, exist_ok=True)
    policy.save_pretrained(GRPO_ADAPTER_DIR, safe_serialization=False)
    tokenizer.save_pretrained(GRPO_ADAPTER_DIR)


if __name__ == "__main__":
    main()
