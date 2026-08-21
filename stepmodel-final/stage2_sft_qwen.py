"""
Stage 2: Supervised fine-tuning of Qwen on the FULL context plus a
graph-conditioning adapter that turns the frozen Stage-1 graph embedding
into GRAPH_PREFIX_TOKENS soft-prompt token embeddings prepended to the
Qwen input.

Training input: stepmodelv2/input/train.json
A 10% held-out validation split is used to track val loss each epoch.
The best checkpoint (lowest val loss) is saved and early stopping fires
after STAGE2_EARLY_STOP_PATIENCE epochs without improvement.

Target output per example:
    {"New step": <STEP_LABELS entry>,
     "Step explanation": <free text>,
     "MCP_tasks": {<tool>: <short action>, ...}}

Only the target tokens contribute to the loss (prompt tokens masked to -100).

Run:
    python stage2_sft_qwen.py
"""
import json
import os
import random
import csv
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, QWEN_MODEL_NAME, GRAPH_PREFIX_TOKENS, GNN_OUT_DIM,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    STAGE2_LR, STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_GRAD_ACCUM,
    STAGE2_VAL_SPLIT, STAGE2_EARLY_STOP_PATIENCE, STAGE2_GRAD_CLIP, STAGE2_WARMUP_RATIO,
    STAGE1_CKPT, STAGE2_ADAPTER_DIR,
    RANDOM_SEED, STEP_LABELS, MCP_LABELS, ROOT,
)
from data_utils import load_from_input_json, _embed_texts, CONTEXT_COLUMNS, StepLabelNormalizer, extract_mcp_labels
from graph_encoder import Stage1Classifier

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Prompt / target builders (also imported by stage3_grpo_rl.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an autonomous penetration-testing planning assistant operating "
    "strictly within an authorized lab environment. Given the current "
    "reconnaissance graph state and the previous/new strategy context, "
    "choose exactly one next-step type from the fixed taxonomy, exactly one "
    "or more tool(s) from the fixed MCP taxonomy, and explain your reasoning. "
    "IMPORTANT: Your step explanation MUST explicitly mention the chosen step "
    "type by name to justify why that specific step is appropriate."
)


def build_prompt(ex: dict) -> str:
    ctx = ex["context"]
    lines = [
        f"Machine: {ex['machine']}",
        f"New strategy: {ctx['New strategy']}",
        f"Strategy explanation: {ctx['Strategy explanation']}",
    ]
    return "\n".join(lines)


def build_target(ex: dict) -> str:
    mcp_dict = {
        label: f"Use {label} as part of: {ex['step_label']}"
        for label in ex["mcp_labels"]
    }
    return json.dumps(
        {
            "New step": ex["step_label"],
            "Step explanation": ex["gold_step_explanation"],
            "MCP_tasks": mcp_dict,
        },
        ensure_ascii=False,
    )


_MCP_PATTERNS_STG = {
    "Nmap": re.compile(r"\bnmap\b", re.I),
    "Metasploit": re.compile(r"\bmetasploit|msfconsole|msfvenom\b", re.I),
    "Netcat": re.compile(r"\bnetcat|\bnc\b", re.I),
    "Dirbuster": re.compile(r"\bdirbuster|gobuster|dirb|netexec\b", re.I),
    "SQLmap": re.compile(r"\bsqlmap\b", re.I),
    "Smb client": re.compile(r"smb\s*client|smbclient|\bsmb\b", re.I),
    "hydra": re.compile(r"\bhydra\b", re.I),
    "John-the-ripper": re.compile(r"john[\s\-]?the[\s\-]?ripper|\bjohn\b", re.I),
    "Google search": re.compile(r"google\s*search|\bgoogle\b", re.I),
    "Interactive CLI": re.compile(r"interactive\s*cli|\bssh\b|\bbash\b|\bshell\b", re.I),
    "Web page interaction": re.compile(r"web\s*page\s*interaction|\bbrowser\b|\bcurl\b", re.I),
}


def _extract_mcp_from_text_stg(text: str):
    if not text:
        return []
    return [label for label, pat in _MCP_PATTERNS_STG.items() if pat.search(text)]


def build_obj_parser():
    """Return a robust parse(text, normalizer) -> dict used by Stage2/Stage3 eval.

    Equivalent to the hardened parser in baseline_llm_eval.parse_response —
    centralised so Stage 2 and Stage 3 evaluation always apply the same logic.
    """

    def parse(text: str, normalizer):
        obj = {}
        try:
            candidates = []
            for match in re.finditer(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL
            ):
                candidates.append(match.group())
            for c in candidates:
                try:
                    obj = json.loads(c)
                    break
                except Exception:
                    continue
            if not obj:
                s = text.find("{")
                e = text.rfind("}") + 1
                if s != -1 and e > s:
                    try:
                        obj = json.loads(text[s:e])
                    except Exception:
                        pass
        except Exception:
            pass

        if not obj or "New step" not in obj:
            for pat in [
                r'"?New step"?\s*:\s*"([^"]+)"',
                r'"?new_step"?\s*:\s*"([^"]+)"',
                r'"?step"?\s*:\s*"([^"]+)"',
            ]:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    obj["New step"] = m.group(1)
                    break
        if "New step" not in obj:
            m = re.search(
                r'(?:next[_\s-]?step(?:\s*type)?|step\s*(?:type|choice)?|action)\s*[:\-–]\s*["\']?\s*([^"\':;.\n][^\n:;]{3,150}?)\s*(?:\.|,|\n|"|Tools|Tool|Reasoning|Explanation|$)',
                text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip().strip('"').strip("'").rstrip(".")
                if cand and len(cand) > 3:
                    obj["New step"] = cand

        if "Step explanation" not in obj:
            for pat in [
                r'"?Step explanation"?\s*:\s*"([^"]*)"',
                r'"?step_explanation"?\s*:\s*"([^"]*)"',
                r'"?explanation"?\s*:\s*"([^"]*)"',
            ]:
                m2 = re.search(pat, text, re.DOTALL | re.IGNORECASE)
                if m2:
                    obj["Step explanation"] = m2.group(1)
                    break
        if "Step explanation" not in obj:
            m2 = re.search(
                r'(?:reasoning|explanation|justification|why|step\s*explanation)\s*[:\-–]\s*["\']?\s*(.{5,400}?)\s*(?:\n\n|\Z|Tools|Tool:|Step|Next step)',
                text, re.DOTALL | re.IGNORECASE)
            if m2:
                obj["Step explanation"] = m2.group(1).strip().strip('"').strip("'")

        need_mcp = not isinstance(obj.get("MCP_tasks"), dict) or not obj.get("MCP_tasks")
        if need_mcp:
            m3 = re.search(
                r'"?MCP[_ ]tasks"?\s*:\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\})',
                text, re.DOTALL | re.IGNORECASE)
            if m3:
                try:
                    obj["MCP_tasks"] = json.loads(m3.group(1))
                except Exception:
                    pass
        if not isinstance(obj.get("MCP_tasks"), dict) or not obj.get("MCP_tasks"):
            m3 = re.search(
                r'"?MCP[_ ]tasks"?\s*:\s*\[([^\]]*)\]',
                text, re.DOTALL | re.IGNORECASE)
            if m3:
                keys = re.findall(r'"([^"]+)"', m3.group(1))
                if keys:
                    obj["MCP_tasks"] = {k: True for k in keys}
        if not isinstance(obj.get("MCP_tasks"), dict) or not obj.get("MCP_tasks"):
            m4 = re.search(
                r'(?:tools?|mcp(?:[_\s-]?tasks)?)\s*[:\-–]\s*(.{3,200}?)\s*(?:\n\n|\Z|Reasoning|Explanation|Step explanation|Next)',
                text, re.DOTALL | re.IGNORECASE)
            if m4:
                found = _extract_mcp_from_text_stg(m4.group(1))
                if found:
                    obj["MCP_tasks"] = {k: "" for k in found}
        if not isinstance(obj.get("MCP_tasks"), dict) or not obj.get("MCP_tasks"):
            found = _extract_mcp_from_text_stg(text)
            if found:
                obj["MCP_tasks"] = {k: "" for k in found}

        return obj

    return parse


# ---------------------------------------------------------------------------
# Graph prefix adapter (also imported by stage3_grpo_rl.py + evaluate.py)
# ---------------------------------------------------------------------------

class GraphPrefixAdapter(nn.Module):
    """
    Projects a single graph embedding into GRAPH_PREFIX_TOKENS
    soft-prompt embeddings that live in the LLM's hidden space.

    Enhanced version with LayerNorm, residual connections, and dropout for
    more stable training.

        graph_emb (B, GNN_OUT_DIM)
            → Linear(GNN_OUT_DIM, H*2) → LN → GELU → Dropout
            → Linear(H*2, H*2) → LN → GELU → Dropout
            → Linear(H*2, H*n_tokens) → reshape
            → (B, n_tokens, H)
    """

    def __init__(self, graph_dim: int, llm_hidden: int,
                 n_tokens: int = GRAPH_PREFIX_TOKENS, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.llm_hidden = llm_hidden
        self.proj = nn.Sequential(
            nn.Linear(graph_dim, llm_hidden * 2),
            nn.LayerNorm(llm_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden * 2, llm_hidden * 2),
            nn.LayerNorm(llm_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(llm_hidden * 2, llm_hidden * n_tokens),
        )
        self.output_norm = nn.LayerNorm(llm_hidden)

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        b = graph_emb.shape[0]
        raw = self.proj(graph_emb).view(b, self.n_tokens, self.llm_hidden)
        return self.output_norm(raw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_len: int = 2560):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{build_prompt(ex)}\n"
            f"<|assistant|>\n"
        )
        target_text = build_target(ex)

        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = (
            self.tok(target_text, add_special_tokens=False)["input_ids"]
            + [self.tok.eos_token_id]
        )

        input_ids = (prompt_ids + target_ids)[: self.max_len]
        # Only target tokens contribute to loss
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]

        field_embs = _embed_texts(
            [ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]
        )
        return {
            "input_ids":  torch.tensor(input_ids),
            "labels":     torch.tensor(labels),
            "graph":      ex["graph"],          # torch_geometric Data (pre-built)
            "field_embs": torch.tensor(field_embs, dtype=torch.float32),
        }


def collate_fn(batch: list, pad_id: int) -> tuple:
    from torch_geometric.data import Batch as PyGBatch

    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), -100,   dtype=torch.long)
    attn      = torch.zeros((B, max_len),         dtype=torch.long)

    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L]    = b["labels"]
        attn[i, :L]      = 1

    graphs     = PyGBatch.from_data_list([b["graph"] for b in batch])
    field_embs = torch.stack([b["field_embs"] for b in batch])
    return input_ids, attn, labels, graphs, field_embs


# ---------------------------------------------------------------------------
# Single forward pass (shared by train and val loops)
# ---------------------------------------------------------------------------

def forward_batch(input_ids, attn, labels, graphs, field_embs,
                  model, stage1_graph_encoder, stage1_context_encoder,
                  adapter, embed_layer, device, dtype):
    """
    Prepend graph prefix tokens to the token embeddings, run the model,
    and return the scalar loss.

    Enhanced: field_embs are projected through the stage1 context encoder and
    combined with the graph embedding via a gated fusion before prefix
    projection, ensuring the LLM sees both graph AND text context fused
    together in the soft-prefix (not just the graph alone).
    """
    input_ids = input_ids.to(device)
    attn      = attn.to(device)
    labels    = labels.to(device)
    graphs    = graphs.to(device)
    field_embs = field_embs.to(device)

    with torch.no_grad():
        edge_attr = getattr(graphs, 'edge_attr', None)
        graph_emb = stage1_graph_encoder(
            graphs.x, graphs.edge_index, graphs.batch, edge_attr=edge_attr
        )  # (B, GNN_OUT_DIM)
        context_emb = stage1_context_encoder(field_embs)  # (B, GNN_OUT_DIM)
        alpha = torch.sigmoid((graph_emb * context_emb).sum(-1, keepdim=True))
        combined_emb = alpha * graph_emb + (1 - alpha) * context_emb

    prefix_embeds = adapter(combined_emb.to(dtype))      # (B, n_tokens, H)
    token_embeds  = embed_layer(input_ids).to(dtype)     # (B, T, H)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    n_prefix     = prefix_embeds.shape[1]
    prefix_attn  = torch.ones(attn.shape[0], n_prefix, device=device, dtype=attn.dtype)
    attn_full    = torch.cat([prefix_attn, attn], dim=1)
    prefix_lbls  = torch.full((labels.shape[0], n_prefix), -100, device=device, dtype=labels.dtype)
    labels_full  = torch.cat([prefix_lbls, labels], dim=1)

    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_full,
        labels=labels_full,
    )
    return out.loss


# ---------------------------------------------------------------------------
# Validation loop — returns average loss over the val set
# ---------------------------------------------------------------------------

def run_validation(val_loader, model, stage1_graph_encoder, stage1_context_encoder,
                   adapter, embed_layer, device, dtype) -> float:
    model.eval()
    adapter.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for input_ids, attn, labels, graphs, field_embs in val_loader:
            loss = forward_batch(
                input_ids, attn, labels, graphs, field_embs,
                model, stage1_graph_encoder, stage1_context_encoder,
                adapter, embed_layer, device, dtype,
            )
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    adapter.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16
    os.makedirs(STAGE2_ADAPTER_DIR, exist_ok=True)

    print(f"[Stage 2] Training input  : {INPUT_TRAIN_JSON}")
    print(f"[Stage 2] Max epochs      : {STAGE2_EPOCHS}")
    print(f"[Stage 2] Val split       : {STAGE2_VAL_SPLIT:.0%}")
    print(f"[Stage 2] Early-stop pat. : {STAGE2_EARLY_STOP_PATIENCE} epochs")
    print(f"[Stage 2] Effective batch : {STAGE2_BATCH_SIZE * STAGE2_GRAD_ACCUM}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base model + LoRA ─────────────────────────────────────────────────────
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    base_model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ── Frozen Stage-1 graph encoder + context encoder ────────────────────────
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
    else:
        stage1.load_state_dict(ckpt)
    graph_encoder = stage1.graph_encoder.to(device).eval()
    context_encoder = stage1.context_encoder.to(device).eval()
    for p in graph_encoder.parameters():
        p.requires_grad_(False)
    for p in context_encoder.parameters():
        p.requires_grad_(False)
    print("[Stage 2] ✓ Frozen Stage-1 graph_encoder + context_encoder loaded")

    # ── GraphPrefixAdapter (trainable) ────────────────────────────────────────
    llm_hidden = model.config.hidden_size
    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)

    embed_layer = model.get_input_embeddings()

    # ── Dataset: load all examples, MACHINE-BASED split train / val ──────────
    all_examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
    n            = len(all_examples)

    # DATA LEAKAGE PREVENTION: split by MACHINE ID, not by example index
    # This ensures no machine's data appears in BOTH train and validation sets
    all_machines = sorted(set(e["machine"] for e in all_examples))
    rng_split = np.random.default_rng(RANDOM_SEED + 1)
    perm_machines = rng_split.permutation(len(all_machines))
    n_val_machines = max(1, int(len(all_machines) * STAGE2_VAL_SPLIT))
    val_machine_set = set(all_machines[i] for i in perm_machines[:n_val_machines])
    train_machine_set = set(all_machines) - val_machine_set

    # Overlap safety check (should never happen, but verify)
    machine_overlap = val_machine_set & train_machine_set
    if machine_overlap:
        print(f"[Stage 2] ⚠  WARNING: machine overlap detected, fixing...")
        val_machine_set = val_machine_set - machine_overlap

    train_examples = [e for e in all_examples if e["machine"] in train_machine_set]
    val_examples   = [e for e in all_examples if e["machine"] in val_machine_set]
    n_train = len(train_examples)
    n_val   = len(val_examples)

    # ── Data leakage pre-check: verify train machines don't overlap with TEST machines ──
    test_examples_precheck = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = set(e["machine"] for e in test_examples_precheck)
    train_test_overlap = train_machine_set & test_machines
    val_test_overlap = val_machine_set & test_machines
    if train_test_overlap:
        print(f"[Stage 2] ⚠  WARNING: TRAIN/TEST machine overlap: {sorted(train_test_overlap)}")
    if val_test_overlap:
        print(f"[Stage 2] ⚠  WARNING: VAL/TEST machine overlap: {sorted(val_test_overlap)}")
    if not train_test_overlap and not val_test_overlap:
        print(f"[Stage 2] ✓ No machine overlap between (train ∪ val) and test sets")
    del test_examples_precheck

    print(f"[Stage 2] Train machines  : {len(train_machine_set)}")
    print(f"[Stage 2] Val machines    : {len(val_machine_set)}")
    print(f"[Stage 2] Train examples  : {n_train}")
    print(f"[Stage 2] Val examples    : {n_val}")

    train_ds = SFTDataset(train_examples, tokenizer)
    val_ds   = SFTDataset(val_examples,   tokenizer)

    make_loader = lambda ds, shuffle: DataLoader(
        ds,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=False,
    )
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    trainable_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(adapter.parameters())
    )
    opt = torch.optim.AdamW(
        trainable_params,
        lr=STAGE2_LR,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    steps_per_epoch  = max(1, len(train_loader) // STAGE2_GRAD_ACCUM)
    total_steps      = steps_per_epoch * STAGE2_EPOCHS
    warmup_steps     = max(10, int(total_steps * STAGE2_WARMUP_RATIO))
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"[Stage 2] Steps/epoch     : {steps_per_epoch}")
    print(f"[Stage 2] Total steps     : {total_steps}  (warmup {warmup_steps}, ratio={STAGE2_WARMUP_RATIO:.0%})")

    # ── Training loop with val + early stopping ────────────────────────────────
    best_val_loss    = float("inf")
    best_epoch       = -1
    no_improve_count = 0
    global_step      = 0

    best_ckpt_dir = os.path.join(STAGE2_ADAPTER_DIR, "best")

    for epoch in range(STAGE2_EPOCHS):
        model.train()
        adapter.train()
        epoch_loss = 0.0
        opt.zero_grad()

        for i, (input_ids, attn, labels, graphs, field_embs) in enumerate(train_loader):
            loss = forward_batch(
                input_ids, attn, labels, graphs, field_embs,
                model, graph_encoder, context_encoder, adapter, embed_layer,
                device, dtype,
            )

            # Scale loss for gradient accumulation
            (loss / STAGE2_GRAD_ACCUM).backward()
            epoch_loss += loss.item()

            if (i + 1) % STAGE2_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, STAGE2_GRAD_CLIP)
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    avg = epoch_loss / (i + 1)
                    print(f"  epoch {epoch+1:02d} | step {global_step:4d} | "
                          f"train_loss {avg:.4f}")

        # ── Validation at end of each epoch ───────────────────────────────────
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss       = run_validation(
            val_loader, model, graph_encoder, context_encoder, adapter, embed_layer, device, dtype
        )

        improved = val_loss < best_val_loss
        marker   = "  ← best" if improved else ""
        print(f"epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
              f"train_loss {avg_train_loss:.4f} | "
              f"val_loss {val_loss:.4f}{marker}")

        if improved:
            best_val_loss    = val_loss
            best_epoch       = epoch + 1
            no_improve_count = 0
            # Save best checkpoint
            os.makedirs(best_ckpt_dir, exist_ok=True)
            model.save_pretrained(best_ckpt_dir)
            torch.save(adapter.state_dict(),
                       os.path.join(best_ckpt_dir, "graph_adapter.pt"))
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f"  → best checkpoint saved  (val_loss={best_val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  → no improvement for {no_improve_count}/{STAGE2_EARLY_STOP_PATIENCE} epochs")
            if no_improve_count >= STAGE2_EARLY_STOP_PATIENCE:
                print(f"\n[Stage 2] Early stopping at epoch {epoch+1}. "
                      f"Best was epoch {best_epoch} (val_loss={best_val_loss:.4f})")
                break

    # ── Copy best checkpoint to the canonical STAGE2_ADAPTER_DIR ─────────────
    # Stage 3 and evaluate.py load from STAGE2_ADAPTER_DIR directly, so the
    # best checkpoint needs to be at the top-level directory too.
    import shutil
    if os.path.isdir(best_ckpt_dir):
        for fname in os.listdir(best_ckpt_dir):
            src = os.path.join(best_ckpt_dir, fname)
            dst = os.path.join(STAGE2_ADAPTER_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"\n[Stage 2] Best checkpoint (epoch {best_epoch}, "
              f"val_loss={best_val_loss:.4f}) copied to {STAGE2_ADAPTER_DIR}")
    else:
        # Fallback: save current weights if early stopping never fired
        model.save_pretrained(STAGE2_ADAPTER_DIR)
        torch.save(adapter.state_dict(),
                   os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt"))
        tokenizer.save_pretrained(STAGE2_ADAPTER_DIR)

    print(f"[Stage 2] Training complete. Adapter at {STAGE2_ADAPTER_DIR}")
    
    # ── Evaluate on test set and save CSV ─────────────────────────────────────
    print("\n[Stage 2] Evaluating on test set...")
    test_examples = load_from_input_json(INPUT_TEST_JSON, "test")
    test_ds = SFTDataset(test_examples, tokenizer)
    test_loader = DataLoader(
        test_ds,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=False,
    )
    
    # Load best model
    model = PeftModel.from_pretrained(base_model, STAGE2_ADAPTER_DIR)
    adapter.load_state_dict(torch.load(os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt"), map_location=device))
    model.eval()
    adapter.eval()
    
    normalizer = StepLabelNormalizer()
    csv_rows = []
    
    with torch.no_grad():
        for input_ids, attn, labels, graphs, field_embs in test_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            graphs = graphs.to(device)
            field_embs = field_embs.to(device)

            # Gated fusion of graph embedding + context embedding (matching training)
            edge_attr = getattr(graphs, 'edge_attr', None)
            graph_emb = graph_encoder(graphs.x, graphs.edge_index, graphs.batch, edge_attr=edge_attr)
            ctx_emb = context_encoder(field_embs)
            alpha = torch.sigmoid((graph_emb * ctx_emb).sum(-1, keepdim=True))
            combined_emb = alpha * graph_emb + (1 - alpha) * ctx_emb

            prefix_embeds = adapter(combined_emb.to(dtype))
            token_embeds = embed_layer(input_ids).to(dtype)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            
            n_prefix = prefix_embeds.shape[1]
            prefix_attn = torch.ones(attn.shape[0], n_prefix, device=device, dtype=attn.dtype)
            attn_full = torch.cat([prefix_attn, attn], dim=1)
            
            # Generate text
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_full,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Decode generated text (skip prefix tokens)
            generated_ids = outputs[:, n_prefix:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            obj_parser = build_obj_parser()

            # Parse and collect data
            for i, gen_text in enumerate(generated_texts):
                ex_idx = len(csv_rows)
                if ex_idx < len(test_examples):
                    ex = test_examples[ex_idx]

                    obj = obj_parser(gen_text, normalizer)
                    
                    # Extract step prediction
                    pred_step_raw = obj.get("New step", "")
                    pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
                    
                    # Try multiple fallback strategies for step prediction
                    pred_step_label = "UNPARSEABLE"
                    if pred_step_norm and pred_step_norm in STEP_LABELS:
                        pred_step_label = pred_step_norm
                    elif pred_step_raw:
                        # Try direct match
                        if pred_step_raw in STEP_LABELS:
                            pred_step_label = pred_step_raw
                        else:
                            # Try fuzzy match - find closest label
                            import difflib
                            closest_match = difflib.get_close_matches(pred_step_raw, STEP_LABELS, n=1, cutoff=0.6)
                            if closest_match:
                                pred_step_label = closest_match[0]
                    
                    gold_step_label = STEP_LABELS[ex["step_idx"]]
                    
                    # Extract MCP predictions
                    pred_mcp_keys = list(obj.get("MCP_tasks", {}).keys()) if isinstance(obj.get("MCP_tasks"), dict) else []
                    pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
                    pred_mcp_tools = "|".join(pred_mcp_labels)
                    gold_mcp_tools = "|".join(ex["mcp_labels"])
                    
                    # Extract explanations
                    pred_expl = str(obj.get("Step explanation", "")).strip()
                    gold_expl = ex.get("gold_step_explanation", "")
                    
                    csv_rows.append({
                        "machine": ex.get("machine", ""),
                        "new_strategy": ex["context"].get("New strategy", ""),
                        "strategy_explanation": ex["context"].get("Strategy explanation", ""),
                        "step_prediction": pred_step_label,
                        "gold_new_step": gold_step_label,
                        "mcp_tool_prediction": pred_mcp_tools,
                        "mcp_tool_gold": gold_mcp_tools,
                        "step_explanation_predicted": pred_expl,
                        "step_explanation_gold": gold_expl,
                    })
    
    # Save CSV
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "stage2.csv")
    
    if csv_rows:
        # Compute Jaccard metrics for Stage 2 evaluation
        step_jaccards = []
        mcp_jaccards = []
        for row in csv_rows:
            step_j = 1.0 if row["step_prediction"] == row["gold_new_step"] else 0.0
            step_jaccards.append(step_j)
            pred_mcp_set = set(row["mcp_tool_prediction"].split("|")) if row["mcp_tool_prediction"] else set()
            gold_mcp_set = set(row["mcp_tool_gold"].split("|")) if row["mcp_tool_gold"] else set()
            if not pred_mcp_set and not gold_mcp_set:
                mcp_j = 1.0
            else:
                union = pred_mcp_set | gold_mcp_set
                mcp_j = len(pred_mcp_set & gold_mcp_set) / len(union) if union else 0.0
            mcp_jaccards.append(mcp_j)
            row["step_jaccard"] = step_j
            row["mcp_jaccard"] = mcp_j

        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Stage 2] Evaluation CSV saved to: {csv_path}")
        print(f"[Stage 2] Total test samples evaluated: {len(csv_rows)}")

        step_acc = float(np.mean(step_jaccards))
        mcp_jac = float(np.mean(mcp_jaccards))
        combined = (step_acc + mcp_jac) / 2.0
        step_pass = sum(1 for j in step_jaccards if j == 1.0)
        mcp_pass = sum(1 for j in mcp_jaccards if j >= 0.5)

        print(f"\n[Stage 2] ═══════════ TEST SET RESULTS ═══════════")
        print(f"  Step Exact Match     : {step_pass}/{len(csv_rows)}  ({step_acc*100:.2f}%)")
        print(f"  MCP Jaccard ≥0.5      : {mcp_pass}/{len(csv_rows)}  ({mcp_pass/len(csv_rows)*100:.2f}%)")
        print(f"  Mean Step Jaccard    : {step_acc:.4f}")
        print(f"  Mean MCP Jaccard     : {mcp_jac:.4f}")
        print(f"  Combined (Step+MCP)/2 : {combined:.4f}")
        print(f"  (Compare to Stage 1 GNN — should show significant improvement)")
        print(f"[Stage 2] ═══════════════════════════════════════")
    else:
        print("[Stage 2] Warning: No CSV rows generated")


if __name__ == "__main__":
    main()
