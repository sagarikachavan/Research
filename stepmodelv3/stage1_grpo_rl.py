"""
Stage 3: Custom GRPO (Group Relative Policy Optimization) with raw text input
Uses the same reward logic and GRPO RL approach as stepmodelv2, but with raw text
input instead of graph embeddings. The graph information from stepmodelv3/input/train.json
is converted to text format and fed directly to the LLM.

Reward composition (same as stepmodelv2):
  r = 0.10 × format_ok          — valid JSON with all 3 required keys
    + 0.30 × step_exact_match   — exact match against gold step label
    + 0.30 × mcp_set_F1         — set F1 between predicted and gold tools
    + 0.30 × explanation_score  — deterministic BERTScore-style cosine sim
                                   between generated and gold explanation
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
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Configuration
INPUT_TRAIN_JSON = "input/train.json"
INPUT_TEST_JSON = "input/test.json"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
STAGE3_ADAPTER_DIR = "checkpoints/stage3_grpo_rl"
STAGE3_GROUP_SIZE = 4
STAGE3_LR = 1e-5
STAGE3_STEPS = 1000
STAGE3_KL_COEF = 0.02
MCP_LABELS = [
    "nmap", "ssh", "ftp", "smbclient", "hydra", "john", "hashcat", 
    "sqlmap", "metasploit", "netcat", "burpsuite", "gobuster", "nikto",
    "responder", "autopsy", "git-dumper", "smtp-user-enum"
]
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Graph to text conversion
# ---------------------------------------------------------------------------

def graph_to_text(graph_data: dict) -> str:
    """Convert graph JSON structure to readable text format."""
    if not graph_data or "nodes" not in graph_data:
        return "No graph data available"
    
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    
    # Group nodes by type
    agent_nodes = [n for n in nodes if n["type"] == "Agent"]
    search_nodes = [n for n in nodes if n["type"] == "Search"]
    track_nodes = [n for n in nodes if n["type"] == "Track"]
    
    text_parts = []
    text_parts.append(f"Graph for machine: {graph_data.get('machine', 'unknown')}")
    text_parts.append(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")
    text_parts.append("\n=== AGENT NODES (States) ===")
    for node in agent_nodes[:10]:  # Limit to first 10 to avoid too long
        text_parts.append(f"- {node['label']}: {node.get('title', '')[:100]}")
    
    text_parts.append("\n=== SEARCH NODES (Actions) ===")
    for node in search_nodes[:10]:
        text_parts.append(f"- {node['label']}: {node.get('title', '')[:100]}")
    
    text_parts.append("\n=== TRACK NODES (Findings) ===")
    for node in track_nodes[:10]:
        text_parts.append(f"- {node['label']}: {node.get('title', '')[:100]}")
    
    text_parts.append("\n=== RECENT EDGES ===")
    for edge in edges[-5:]:  # Last 5 edges
        text_parts.append(f"- {edge['from']} -> {edge['to']} ({edge['type']})")
    
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Load data from JSON
# ---------------------------------------------------------------------------

def load_json_data(json_path: str, split: str = "train"):
    """Load training data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        graph_text = graph_to_text(item.get("Graph", {}))
        examples.append({
            "machine": item.get("Machine", ""),
            "graph_text": graph_text,
            "step_label": item.get("step_label", ""),
            "mcp_labels": item.get("mcp_labels", []),
            "gold_step_explanation": item.get("gold_step_explanation", ""),
        })
    
    print(f"[Data] Loaded {len(examples)} examples from {json_path}")
    return examples


# ---------------------------------------------------------------------------
# Explanation quality: deterministic BERTScore via sentence transformer
# ---------------------------------------------------------------------------

def load_sentence_encoder():
    """Load sentence transformer for BERTScore computation."""
    print("[Embedding] Loading sentence transformer...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return encoder

_sentence_encoder = None

def get_sentence_encoder():
    global _sentence_encoder
    if _sentence_encoder is None:
        _sentence_encoder = load_sentence_encoder()
    return _sentence_encoder

def _explanation_bertscore(pred_expl: str, gold_expl: str,
                            min_score: float = 0.30) -> float:
    """
    Cosine similarity between sentence transformer embeddings of the generated
    and gold explanation.
    """
    if not pred_expl.strip() or not gold_expl.strip():
        return 0.0
    
    encoder = get_sentence_encoder()
    embs = encoder.encode([pred_expl[:512], gold_expl[:512]])
    cos = float(np.dot(embs[0], embs[1]))  # cosine similarity
    cos = max(0.0, cos)  # clamp negatives
    return 0.0 if cos < min_score else cos


# ---------------------------------------------------------------------------
# Reward function (same as stepmodelv2)
# ---------------------------------------------------------------------------

def _parse_completion(text: str) -> dict | None:
    """Extract the first {...} JSON block from generated text."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def compute_reward(completion: str, gold: dict,
                   w_fmt: float = 0.10,
                   w_step: float = 0.30,
                   w_mcp: float = 0.30,
                   w_exp: float = 0.30) -> float:
    """
    Composite reward — all components are deterministic (same as stepmodelv2).
    """
    obj = _parse_completion(completion)
    if obj is None or not all(k in obj for k in ("New step", "Step explanation", "MCP_tasks")):
        return 0.0

    # Format
    fmt_r = 1.0

    # Step exact match
    step_r = 1.0 if obj["New step"].strip() == gold["step_label"] else 0.0

    # MCP set F1
    mcp_val = obj.get("MCP_tasks", {})
    pred_mcp = set(mcp_val.keys() if isinstance(mcp_val, dict) else []) & set(MCP_LABELS)
    gold_mcp = set(gold["mcp_labels"])
    if not pred_mcp and not gold_mcp:
        mcp_r = 1.0
    else:
        inter = len(pred_mcp & gold_mcp)
        prec = inter / len(pred_mcp) if pred_mcp else 0.0
        rec = inter / len(gold_mcp) if gold_mcp else 0.0
        mcp_r = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Explanation BERTScore
    pred_expl = str(obj.get("Step explanation", "")).strip()
    gold_expl = gold.get("gold_step_explanation", "")
    exp_r = _explanation_bertscore(pred_expl, gold_expl)

    return w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * exp_r


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert penetration testing AI assistant. Given the current state of a penetration test represented as a graph structure, predict the next step to take, explain why, and specify which tools to use."""

def build_prompt(example: dict) -> str:
    """Build prompt from example data."""
    prompt = f"""Current penetration test state for machine: {example['machine']}

{example['graph_text']}

Based on the current graph state, predict the next step:
"""
    return prompt


# ---------------------------------------------------------------------------
# Embedding helpers (text-only, no graph prefix)
# ---------------------------------------------------------------------------

def build_prompt_embeds(prompt_text: str, tokenizer, embed_layer, device, dtype):
    """
    Tokenise prompt_text and embed the token IDs.
    Returns:
        inputs_embeds : (1, T_prompt, H)
        prompt_len    : total length — used to slice out generated portion later
    """
    ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=900,
    ).input_ids.to(device)

    token_embeds = embed_layer(ids).to(dtype)  # (1, T_prompt, H)
    return token_embeds, token_embeds.shape[1]


# ---------------------------------------------------------------------------
# Per-token log-prob of a completion given inputs_embeds
# ---------------------------------------------------------------------------

def completion_logprobs(
    model,
    inputs_embeds: torch.Tensor,   # (1, L_prompt, H)
    completion_ids: torch.Tensor,  # (1, L_gen)
    embed_layer,
    dtype,
    device,
) -> torch.Tensor:
    """
    Compute the sum of per-token log-probs for `completion_ids` given the
    prompt represented as `inputs_embeds`.
    """
    comp_embeds = embed_layer(completion_ids).to(dtype)          # (1, L_gen, H)
    full_embeds = torch.cat([inputs_embeds, comp_embeds], dim=1) # (1, L_prompt+L_gen, H)

    attn = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)
    out = model(inputs_embeds=full_embeds, attention_mask=attn)  # no labels → no loss
    logits = out.logits  # (1, L_prompt+L_gen, V)

    L_prompt = inputs_embeds.shape[1]
    comp_logits = logits[:, L_prompt - 1 : L_prompt + completion_ids.shape[1] - 1, :]
    log_probs = F.log_softmax(comp_logits, dim=-1)             # (1, L_gen, V)
    token_lp = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)  # (1, L_gen)
    return token_lp.sum()  # scalar


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 3] Training input : {INPUT_TRAIN_JSON}")
    print(f"[Stage 3] Device         : {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Policy model (base Qwen, trainable with LoRA) ───────────────────────
    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    
    # Add LoRA for efficient fine-tuning
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    policy = get_peft_model(base, lora_config)
    policy.train()
    dtype = torch.bfloat16

    # ── Reference model (base Qwen, frozen) ───────────────────────────────
    ref_base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16, device_map=None
    ).to(device)
    ref_model = get_peft_model(ref_base, lora_config)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Embedding layer (shared, read-only during generation) ────────────────
    embed_layer = policy.get_input_embeddings()

    # ── Optimizer — LoRA params only ────────────────────────────────────────
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=STAGE3_LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=STAGE3_STEPS)

    # ── Dataset ───────────────────────────────────────────────────────────────
    examples = load_json_data(INPUT_TRAIN_JSON, "train")

    # ── Output dir ────────────────────────────────────────────────────────────
    os.makedirs(STAGE3_ADAPTER_DIR, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    G = STAGE3_GROUP_SIZE   # completions per prompt
    beta = STAGE3_KL_COEF   # KL penalty weight
    grad_accum = 8          # accumulate before optimizer step
    clip_eps = 0.2          # PPO clip range

    optimizer.zero_grad()
    global_step = 0

    for step in range(1, STAGE3_STEPS + 1):

        # ── 1. Sample one training example ───────────────────────────────────
        ex = random.choice(examples)
        gold = {
            "step_label": ex["step_label"],
            "mcp_labels": ex["mcp_labels"],
            "gold_step_explanation": ex["gold_step_explanation"],
        }

        # ── 2. Build prompt embeddings (text-only, no graph prefix) ──────────
        prompt_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{build_prompt(ex)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_embeds, L_prompt = build_prompt_embeds(
            prompt_text, tokenizer, embed_layer, device, dtype
        )
        # prompt_embeds : (1, L_prompt, H)

        attn_prompt = torch.ones(1, L_prompt, dtype=torch.long, device=device)

        # ── 3. Generate G completions ─────────────────────────────────────────
        policy.eval()   # disable dropout during generation
        with torch.no_grad():
            gen_out = policy.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=G,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # gen_out: (G, L_prompt + L_gen)
        completion_ids_list = gen_out  # (G, L_gen) — already just the new tokens

        # ── 4. Decode and score completions ───────────────────────────────────
        completions = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in completion_ids_list
        ]
        rewards = torch.tensor(
            [compute_reward(c, gold) for c in completions],
            dtype=torch.float32,
        )

        # ── 5. Group-relative advantage  A_i = (r_i - mean) / (std + ε) ──────
        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantages = ((rewards - mean_r) / std_r).to(device)  # (G,)

        # ── 6. Policy gradient with clipping + KL penalty ─────────────────────
        policy.train()
        loss_accum = torch.zeros(1, device=device)

        for g_idx in range(G):
            comp_ids = completion_ids_list[g_idx].unsqueeze(0).to(device)  # (1, L_gen)
            adv = advantages[g_idx]                                         # scalar

            # Policy log-prob for this completion
            lp_policy = completion_logprobs(
                policy, prompt_embeds, comp_ids, embed_layer, dtype, device
            )

            # Reference log-prob (frozen base model)
            with torch.no_grad():
                ref_embed_layer = ref_model.get_input_embeddings()
                lp_ref = completion_logprobs(
                    ref_model, prompt_embeds.detach(), comp_ids,
                    ref_embed_layer, dtype, device
                )

            # KL penalty
            kl = lp_policy - lp_ref.detach()

            # GRPO loss
            pg_loss = -adv * lp_policy + beta * kl
            loss_accum = loss_accum + pg_loss / G

        # Scale by gradient accumulation
        loss_for_backward = loss_accum / grad_accum
        loss_for_backward.backward()

        # ── 7. Optimizer step every grad_accum steps ──────────────────────────
        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # ── 8. Logging ────────────────────────────────────────────────────────
        if step % 50 == 0:
            avg_reward = rewards.mean().item()
            fmt_ok = sum(1 for c in completions if _parse_completion(c) is not None)

            # Breakdown: avg per-component scores
            comp_scores = {"step": [], "mcp": [], "exp": []}
            for c in completions:
                obj = _parse_completion(c)
                if obj is None:
                    continue
                comp_scores["step"].append(
                    1.0 if obj.get("New step","").strip() == gold["step_label"] else 0.0
                )
                mcp_val = obj.get("MCP_tasks", {})
                pred_mcp = set(mcp_val.keys() if isinstance(mcp_val, dict) else []) & set(MCP_LABELS)
                gold_mcp = set(gold["mcp_labels"])
                if not pred_mcp and not gold_mcp:
                    comp_scores["mcp"].append(1.0)
                else:
                    inter = len(pred_mcp & gold_mcp)
                    prec = inter / len(pred_mcp) if pred_mcp else 0.0
                    rec = inter / len(gold_mcp) if gold_mcp else 0.0
                    comp_scores["mcp"].append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0)
                pred_expl = str(obj.get("Step explanation","")).strip()
                comp_scores["exp"].append(
                    _explanation_bertscore(pred_expl, gold["gold_step_explanation"])
                )

            avg_step = float(np.mean(comp_scores["step"])) if comp_scores["step"] else 0.0
            avg_mcp = float(np.mean(comp_scores["mcp"]))  if comp_scores["mcp"]  else 0.0
            avg_exp = float(np.mean(comp_scores["exp"]))  if comp_scores["exp"]  else 0.0

            print(
                f"step {step:4d}/{STAGE3_STEPS} | "
                f"avg_reward {avg_reward:.3f} | "
                f"step_r {avg_step:.3f} | "
                f"mcp_r {avg_mcp:.3f} | "
                f"exp_r {avg_exp:.3f} | "
                f"fmt_ok {fmt_ok}/{G} | "
                f"loss {loss_accum.item():.4f}"
            )

        # ── 9. Checkpoint every 200 steps ────────────────────────────────────
        if step % 200 == 0:
            ckpt_path = os.path.join(STAGE3_ADAPTER_DIR, f"step_{step}")
            policy.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  -> checkpoint saved to {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    policy.save_pretrained(STAGE3_ADAPTER_DIR)
    tokenizer.save_pretrained(STAGE3_ADAPTER_DIR)
    print(f"\nStage 3 GRPO complete. Policy saved to {STAGE3_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
