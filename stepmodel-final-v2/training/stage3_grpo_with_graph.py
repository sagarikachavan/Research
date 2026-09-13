"""
Stage 3 training with graph conditioning and GRPO RL.

This stage integrates:
- Frozen prefix adapter (from Stage 2)
- Stage 2 checkpoint (Qwen + LoRA)
- GRPO RL with comprehensive reward function
- Equal weighting for step, MCP, and explanation

Input: new_strategy + strategy_explanation + graph structure
Output: step, MCP tools, and step explanation (optimized via RL)

Usage:
    python training/stage3_grpo_with_graph.py
"""

import os
import sys
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from tqdm import tqdm

# Add parent directories to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    ROOT, INPUT_TRAIN_JSON, INPUT_TEST_JSON, STAGE2_ADAPTER_DIR, STAGE3_ADAPTER_DIR,
    STEP_LABELS, MCP_LABELS, STEP2IDX, MCP2IDX, IDX2STEP, IDX2MCP,
    QWEN_MODEL_NAME, TEXT_ENCODER_NAME,
    STAGE3_GROUP_SIZE, STAGE3_LR, STAGE3_STEPS,
    STAGE3_KL_COEF, STAGE3_PPO_CLIP, STAGE3_GRAD_ACCUM, STAGE3_GRAD_CLIP,
    RANDOM_SEED, GRAPH_PREFIX_TOKENS, GNN_OUT_DIM,
    STAGE3_DUAL_CLIP_COEF, STAGE3_KL_HARD_CAP, STAGE3_EARLY_STOP_PATIENCE,
)
from graph_encoder import Stage1Classifier
from graph_prefix_adapter import GraphPrefixAdapter
from comprehensive_evaluator import ComprehensiveEvaluator
from data_utils import get_bge_embeddings


SYSTEM_PROMPT = """You are an expert penetration testing assistant. Given a strategy, explanation, and graph context, determine the next step, the tools needed, and explain your reasoning.

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


def build_prompt(ex: dict, graph_tokens: str = "") -> str:
    """Build prompt from example with graph context."""
    ctx = f"Strategy: {ex.get('new_strategy', '')}\nExplanation: {ex.get('strategy_explanation', '')}"
    if graph_tokens:
        ctx += f"\n\nGraph Context: {graph_tokens}"
    
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


def load_from_input_json(path, split):
    """Load examples from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"[{split}] Loaded {len(examples)} examples from {path}")
    return examples


def completion_logprob(model, tokenizer, prompt_ids, completion_ids, device):
    """
    Sum of per-token log-probs the model assigns to `completion_ids`
    given `prompt_ids` as context.
    
    prompt_ids: (1, L_prompt) LongTensor
    completion_ids: (1, L_gen) LongTensor
    Returns: scalar tensor (sum of log-probs over the L_gen completion tokens)
    """
    if completion_ids.shape[1] == 0:
        return torch.tensor(-50.0, device=device, requires_grad=True)

    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attn = torch.ones_like(full_ids)

    out = model(input_ids=full_ids, attention_mask=attn)
    logits = out.logits  # (1, L_prompt+L_gen, V)

    L_prompt = prompt_ids.shape[1]
    L_gen = completion_ids.shape[1]

    comp_logits = logits[:, L_prompt - 1: L_prompt + L_gen - 1, :]
    log_probs = F.log_softmax(comp_logits.float(), dim=-1)
    token_lp = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum()


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize comprehensive evaluator
    evaluator = ComprehensiveEvaluator()
    
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
            f"Run Stage 2 training first."
        )

    # Load Stage 1 checkpoint for graph encoding
    from config import STAGE1_CKPT
    if os.path.isfile(STAGE1_CKPT):
        print("Loading Stage 1 checkpoint for graph encoding...")
        stage1_model = Stage1Classifier()
        stage1_ckpt = torch.load(STAGE1_CKPT, map_location=device)
        stage1_model.load_state_dict(stage1_ckpt)
        stage1_model.to(device)
        stage1_model.eval()
    else:
        print("Warning: Stage 1 checkpoint not found. Proceeding without graph conditioning.")
        stage1_model = None

    # Load frozen prefix adapter from Stage 2
    prefix_adapter_path = os.path.join(STAGE2_ADAPTER_DIR, "prefix_adapter.pt")
    if os.path.isfile(prefix_adapter_path):
        print("Loading frozen prefix adapter from Stage 2...")
        qwen_config = AutoConfig.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        llm_hidden = qwen_config.hidden_size
        
        prefix_adapter = GraphPrefixAdapter(
            graph_dim=GNN_OUT_DIM,
            llm_hidden=llm_hidden,
            n_tokens=GRAPH_PREFIX_TOKENS
        ).to(device)
        prefix_adapter.load_state_dict(torch.load(prefix_adapter_path, map_location=device))
        prefix_adapter.eval()
        for param in prefix_adapter.parameters():
            param.requires_grad = False
    else:
        print("Warning: Prefix adapter not found. Proceeding without graph tokens.")
        prefix_adapter = None

    # Load BGE for field embeddings if using graph
    if stage1_model is not None:
        from transformers import AutoModel
        print("Loading BGE model for field embeddings...")
        bge_model = AutoModel.from_pretrained(TEXT_ENCODER_NAME)
        bge_model.to(device)
        bge_model.eval()
    else:
        bge_model = None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model and attach Stage 2 adapter
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
    
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad = False

    model.set_adapter("default")

    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters (policy LoRA only): {n_trainable:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=STAGE3_LR)

    # Training loop
    best_val_reward = 0.0
    global_step = 0
    grad_accum_counter = 0
    patience_counter = 0
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

        # Generate completions with the policy adapter active
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

            # Compute comprehensive reward with equal weights
            reward_dict = evaluator.compute_comprehensive_reward(
                pred_step, pred_mcp, pred_expl,
                ex.get('gold_new_step', ''),
                ex.get('gold_mcp_tasks', ''),
                ex.get('gold_step_explanation', ''),
                step_weight=1.0,
                mcp_weight=1.0,
                expl_weight=1.0
            )
            reward = reward_dict['total_reward']
            all_rewards.append(reward)
            prompt_ids_list.append(inputs['input_ids'])
            completion_ids_list.append(completion_ids)

        # Group-relative advantages (GRPO)
        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - mean_reward) / std_reward
        advantages = torch.clamp(advantages, -4.0, 4.0)

        # Policy gradient loss + KL penalty
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
                model.eval()
                lp_ref = completion_logprob(model, tokenizer, prompt_ids, completion_ids, device)
            model.set_adapter("default")
            model.train()

            mean_lp_policy = lp_policy / completion_ids.shape[1]
            mean_lp_ref = lp_ref / completion_ids.shape[1]

            kl = mean_lp_policy - mean_lp_ref.detach()
            
            # KL hard cap per micro-batch
            if kl.abs() > STAGE3_KL_HARD_CAP:
                continue
                
            kl_loss = kl_loss + kl

            # Policy gradient with dual-clip for negative advantages
            log_ratio = mean_lp_policy - mean_lp_ref
            ratio = torch.clamp(log_ratio, min=-10, max=10).exp()
            
            pg_loss = -advantages[i] * mean_lp_policy
            
            # Dual-clip for negative advantages (Ye et al. 2020)
            if advantages[i] < 0:
                clipped_pg = torch.maximum(
                    pg_loss,
                    STAGE3_DUAL_CLIP_COEF * advantages[i] * mean_lp_policy
                )
                policy_loss = policy_loss + clipped_pg
            else:
                policy_loss = policy_loss + pg_loss

        if n_valid == 0:
            print(f"Step {step}: all completions were empty or KL-capped, skipping update.")
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

            # Validation with comprehensive evaluation
            model.set_adapter("default")
            model.eval()
            val_rewards = []
            val_step_rewards = []
            val_mcp_rewards = []
            val_expl_rewards = []
            
            with torch.no_grad():
                for ex in val_examples[:32]:
                    prompt = build_prompt(ex)
                    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    pred_step, pred_mcp, pred_expl = parse_response(response)

                    reward_dict = evaluator.compute_comprehensive_reward(
                        pred_step, pred_mcp, pred_expl,
                        ex.get('gold_new_step', ''),
                        ex.get('gold_mcp_tasks', ''),
                        ex.get('gold_step_explanation', ''),
                        step_weight=1.0,
                        mcp_weight=1.0,
                        expl_weight=1.0
                    )
                    
                    val_rewards.append(reward_dict['total_reward'])
                    val_step_rewards.append(reward_dict['step_reward'])
                    val_mcp_rewards.append(reward_dict['mcp_reward'])
                    val_expl_rewards.append(reward_dict['explanation_reward'])

            avg_val_reward = np.mean(val_rewards) if val_rewards else 0.0
            avg_step_reward = np.mean(val_step_rewards) if val_step_rewards else 0.0
            avg_mcp_reward = np.mean(val_mcp_rewards) if val_mcp_rewards else 0.0
            avg_expl_reward = np.mean(val_expl_rewards) if val_expl_rewards else 0.0
            
            print(f"  Val Reward: {avg_val_reward:.4f}")
            print(f"    Step: {avg_step_reward:.4f}, MCP: {avg_mcp_reward:.4f}, Expl: {avg_expl_reward:.4f}")

            if avg_val_reward > best_val_reward:
                best_val_reward = avg_val_reward
                patience_counter = 0
                model.set_adapter("default")
                model.save_pretrained(STAGE3_ADAPTER_DIR, selected_adapters=["default"])
                tokenizer.save_pretrained(STAGE3_ADAPTER_DIR)
                print(f"  -> Saved best model")
            else:
                patience_counter += 1
                if STAGE3_EARLY_STOP_PATIENCE is not None and patience_counter >= STAGE3_EARLY_STOP_PATIENCE:
                    print(f"  -> Early stopping (no improvement for {patience_counter} evals)")
                    break

    print(f"\nTraining complete. Best val reward: {best_val_reward:.4f}")


if __name__ == "__main__":
    main()
