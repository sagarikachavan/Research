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
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

# ── Path bootstrap (folder was restructured into core/ data_prep/ training/ eval/) ──
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "core"), _os.path.join(_ROOT, "data_prep"), _os.path.join(_ROOT, "training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, QWEN_MODEL_NAME, GRAPH_PREFIX_TOKENS, GNN_OUT_DIM,
    FUSION_HIDDEN, MCP_DECISION_THRESHOLD,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    STAGE2_LR, STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_GRAD_ACCUM,
    STAGE2_VAL_SPLIT, STAGE2_EARLY_STOP_PATIENCE, STAGE2_GRAD_CLIP, STAGE2_WARMUP_RATIO,
    STAGE1_CKPT, STAGE2_ADAPTER_DIR,
    RANDOM_SEED, STEP_LABELS, MCP_LABELS, IDX2STEP, IDX2MCP, ROOT,
)

# Dimensionality of the representation handed to GraphPrefixAdapter.
# Was GNN_OUT_DIM (raw pooled graph embedding); now the fused,
# classification-calibrated representation from Stage1Classifier.encode_and_predict
# (see graph_encoder.py). Kept as a module constant so stage3_grpo_rl.py and
# evaluate.py can import it and stay dimensionally consistent with whatever
# checkpoint this file produces.
GRAPH_PREFIX_SRC_DIM = FUSION_HIDDEN // 2
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
    "reconnaissance graph state and the new strategy context, "
    "choose exactly one next-step type from the fixed taxonomy, exactly one "
    "or more tool(s) from the fixed MCP taxonomy, and explain your reasoning. "
    "IMPORTANT: Your step explanation MUST explicitly mention the chosen step "
    "type by name to justify why that specific step is appropriate."
)


def build_prompt(ex: dict, mask_hint: bool = False) -> str:
    """
    Enhanced prompt building based on research from GTA and ReFT papers.
    Structured prompt with clear sections for better reasoning guidance.

    Input contract: machine + graph (fed separately via the graph-prefix
    # adapter) + new_strategy + strategy_explanation ONLY. No previous-step
    # fields -- see CONTEXT_COLUMNS / EXTRA_OUTPUT_KEYS in data_utils.py and
    # build_input_json.py for why they were removed.

    Args:
        ex: Example dictionary with context and optional stage1_hint
        mask_hint: If True, omit the Stage 1 hint to force learning from graph tokens
    """
    ctx = ex["context"]
    lines = [
        "# Context",
        f"Machine: {ex['machine']}",
        "",
        "# Strategy",
        f"New strategy: {ctx['New strategy']}",
        f"Strategy explanation: {ctx['Strategy explanation']}",
        "",
        "# Task",
        "Based on the machine and strategy above, determine the next step, the tools needed, and explain your reasoning.",
    ]
    # Stage-1 classifier hint (optional -- set by precompute_stage1_hints()).
    # WHY: the graph-prefix soft tokens already encode Stage 1's fused
    # representation, but a 7B model with LoRA has no guarantee of reliably
    # DECODING a specific classification decision out of 16 continuous
    # vectors purely from language-modeling loss on ~1.5k rows. Spelling
    # the classifier's own (possibly wrong) top prediction out as TEXT gives
    # the model a floor roughly equal to Stage 1's accuracy for free, and
    # lets it spend its capacity on: (a) rendering the exact canonical
    # label string correctly, (b) writing a good explanation, (c) refining
    # MCP tool selection, and (d) OVERRIDING the hint on the examples where
    # the fuller strategy text makes the graph-only classifier's guess
    # wrong. It is explicitly labeled as fallible so the model isn't
    # trained to treat it as ground truth.
    #
    # HINT MASKING: During training, we randomly mask the hint (mask_hint=True)
    # to force the model to learn from the graph prefix tokens directly.
    # This prevents the model from simply copying the hint and ignoring the
    # graph conditioning.
    if not mask_hint and ex.get("stage1_hint"):
        lines.append("")
        lines.append("# Suggested Step (verify against strategy above)")
        lines.append(ex["stage1_hint"])
    return "\n".join(lines)


def format_stage1_hint(step_label: str, mcp_labels: list) -> str:
    tools = ", ".join(mcp_labels) if mcp_labels else "none confident"
    return (
        f"Classifier signal (a graph-only model's best guess, may be wrong -- "
        f"verify against the strategy above and correct it if needed): "
        f"most likely next step = \"{step_label}\"; likely tool(s) = {tools}."
    )


def precompute_stage1_hints(examples: list, stage1, device, dtype) -> None:
    """
    Runs the frozen Stage-1 classifier once over every example and attaches
    ex["stage1_hint"] (a text string; see build_prompt / format_stage1_hint).
    Mutates `examples` in place. One-time cost (~seconds for ~1.5k rows),
    done once in main() before dataset/prompt construction so
    __getitem__ doesn't need model access.
    """
    from torch_geometric.data import Batch as PyGBatch
    stage1.eval()
    with torch.no_grad():
        for ex in examples:
            graph = PyGBatch.from_data_list([ex["graph"]]).to(device)
            field_embs = torch.tensor(
                _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
                dtype=torch.float32,
            ).unsqueeze(0).to(device)
            edge_attr = getattr(graph, "edge_attr", None)
            _, step_logits, mcp_logits = stage1.encode_and_predict(
                graph.x, graph.edge_index, graph.batch, field_embs, edge_attr=edge_attr
            )
            step_pred = IDX2STEP[int(step_logits.argmax(-1).item())]
            mcp_probs = torch.sigmoid(mcp_logits).squeeze(0)
            mcp_pred = [IDX2MCP[i] for i in range(len(MCP_LABELS))
                        if mcp_probs[i].item() >= MCP_DECISION_THRESHOLD]
            ex["stage1_hint"] = format_stage1_hint(step_pred, mcp_pred)


def build_target(ex: dict) -> str:
    """
    Enhanced target building with better structure for learning.
    Based on research from GTA and ReFT for better reasoning guidance.
    """
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
        indent=2  # Better formatting for easier parsing
    )


# NOTE: this file previously carried its own copy of the MCP tool regex
# patterns (_MCP_PATTERNS_STG) and its own free-text extraction helper
# (_extract_mcp_from_text_stg), duplicated from data_utils.py and allowed
# to drift out of sync with the (fixed) canonical version there. Removed --
# use extract_mcp_labels (imported from data_utils above) everywhere a
# free-text MCP_tasks-like string needs to be turned into canonical
# MCP_LABELS. It applies the same key-only extraction (dict keys, or the
# text before each ';'-separated segment's first ':') that fixed the
# ~24.5% spurious-label rate the old whole-cell regex scan had on
# "Interactive CLI" / "Web page interaction" (e.g. matching "ssh"/"curl"
# inside an unrelated tool's own description text).
def _extract_mcp_from_text_stg(text: str):
    if not text:
        return []
    return extract_mcp_labels(text)


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
    def __init__(self, examples: list, tokenizer, max_len: int = 1536, mask_hint_prob: float = 0.0, is_training: bool = True):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len
        self.mask_hint_prob = mask_hint_prob
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{build_prompt(ex, mask_hint=(np.random.random() < self.mask_hint_prob) if self.is_training else True)}\n"
            f"<|assistant|>\n"
        )
        target_text = build_target(ex)

        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        try:
            target_enc = self.tok(target_text, add_special_tokens=False, return_offsets_mapping=True)
        except (NotImplementedError, ValueError):
            # Slow (non-"Fast") tokenizer fallback: offset_mapping isn't
            # available. Don't crash training over a checkpoint-selection
            # nicety -- just tokenize normally and mark the step span as
            # "not found" below, which run_validation already treats as
            # "skip this row" rather than misattributing garbage tokens.
            target_enc = {"input_ids": self.tok(target_text, add_special_tokens=False)["input_ids"],
                          "offset_mapping": []}
        target_ids = target_enc["input_ids"] + [self.tok.eos_token_id]

        input_ids = (prompt_ids + target_ids)[: self.max_len]
        # Only target tokens contribute to loss
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]

        # ── Step-value token span (for checkpoint-selection metric) ────────
        # Locate the "New step" value's character range inside target_text
        # (build_target's json.dumps puts it right after `{"New step": "`),
        # then map that to a token range using the offset_mapping the
        # tokenizer itself returns for target_text's actual tokenization.
        #
        # NOTE: an earlier version of this computed the span by re-tokenizing
        # target_text[:char_start] and target_text[:char_start+len(step_val)]
        # independently and diffing token counts. That's unsound: a BPE/word
        # boundary token can merge characters across the cut point (e.g. a
        # trailing `"` before the value gets fused with the value's first
        # word when tokenized as part of the full string, but becomes its
        # own separate token when the prefix is tokenized in isolation),
        # silently shifting the recovered span by a token and corrupting the
        # very metric this is meant to fix. offset_mapping reports each
        # token's real character span from the SAME tokenize call used to
        # build target_ids, so it can't disagree with itself this way.
        step_val = ex["step_label"]
        char_start = target_text.find(step_val)
        step_tok_start, step_tok_end = 0, 0  # default: span not found -> excluded from metric
        if char_start >= 0:
            char_end = char_start + len(step_val)
            offsets = target_enc["offset_mapping"]
            found_start = None
            found_end = None
            for tok_i, (a, b) in enumerate(offsets):
                if a < char_end and b > char_start:  # this token overlaps the value's char range
                    if found_start is None:
                        found_start = tok_i
                    found_end = tok_i + 1
            if found_start is not None:
                step_tok_start = len(prompt_ids) + found_start
                step_tok_end = min(len(prompt_ids) + found_end, self.max_len)
                step_tok_start = min(step_tok_start, step_tok_end)

        field_embs = _embed_texts(
            [ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]
        )
        return {
            "input_ids":  torch.tensor(input_ids),
            "labels":     torch.tensor(labels),
            "graph":      ex["graph"],          # torch_geometric Data (pre-built)
            "field_embs": torch.tensor(field_embs, dtype=torch.float32),
            "step_span":  torch.tensor([step_tok_start, step_tok_end], dtype=torch.long),
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
    step_spans = torch.stack([b["step_span"] for b in batch])  # (B, 2) = [start, end)
    return input_ids, attn, labels, graphs, field_embs, step_spans


# ---------------------------------------------------------------------------
# Single forward pass (shared by train and val loops)
# ---------------------------------------------------------------------------

def forward_batch(input_ids, attn, labels, graphs, field_embs,
                  model, stage1, adapter, embed_layer, device, dtype, return_logits=False):
    """
    Prepend graph prefix tokens to the token embeddings, run the model,
    and return the scalar loss (and, if return_logits=True, the raw logits
    plus n_prefix -- needed by run_validation to score just the "New step"
    value span, since the graph-prefix tokens shift every index by
    n_prefix relative to the un-prefixed input_ids/labels/step_spans this
    function was called with).

    FIX: previously this recombined the frozen Stage-1 graph_encoder and
    context_encoder outputs via `sigmoid((graph_emb*context_emb).sum(-1))`
    -- a hand-written, PARAMETER-FREE heuristic (not even a learned gate)
    that discarded Stage-1's actual trained graph_gate/context_gate/fusion
    stack. Now calls stage1.encode_and_predict(...) directly, so the
    GraphPrefixAdapter is conditioned on exactly the representation Stage
    1's step_head/mcp_head were trained against (see graph_encoder.py's
    encode_and_predict docstring for the full rationale).
    """
    input_ids = input_ids.to(device)
    attn      = attn.to(device)
    labels    = labels.to(device)
    graphs    = graphs.to(device)
    field_embs = field_embs.to(device)

    with torch.no_grad():
        edge_attr = getattr(graphs, 'edge_attr', None)
        combined_emb, _, _ = stage1.encode_and_predict(
            graphs.x, graphs.edge_index, graphs.batch, field_embs, edge_attr=edge_attr
        )  # (B, FUSION_HIDDEN // 2)

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
    if return_logits:
        return out.loss, out.logits, n_prefix
    return out.loss


# ---------------------------------------------------------------------------
# Validation loop — returns (avg_loss, step_field_accuracy) over the val set
# ---------------------------------------------------------------------------
#
# FIX: previously this returned only avg_loss (cross-entropy averaged over
# EVERY target token: the "New step" value, the free-text explanation, and
# every MCP_tasks JSON key/value/punctuation token combined). Checkpoint
# selection and early stopping picked whichever epoch minimized that blended
# average -- which is dominated by the much-longer explanation text, and is
# not the same thing evaluate.py measures ("Step Exact Match" / step
# accuracy). An epoch can lower the blended loss (e.g. by getting more
# confident/fluent on explanation text or the majority classes) while
# getting WORSE at a specific, less-frequent step class -- exactly the
# failure mode a class going from strong recall to 0/22 correct looks like.
#
# step_field_accuracy is a teacher-forced argmax-vs-gold accuracy computed
# ONLY over the token span the "New step" value occupies (see SFTDataset's
# step_span computation) -- i.e. "if the model had to predict each of these
# specific tokens one at a time with the correct history so far, how often
# does it pick the right one". It's not identical to greedy-decode exact
# match (that needs actual generation, done separately in evaluate.py /
# the post-training test-set loop below), but it isolates the signal that
# actually matters for checkpoint selection instead of drowning it in
# explanation-text loss.
def run_validation(val_loader, model, stage1, adapter, embed_layer, device, dtype):
    model.eval()
    adapter.eval()
    total_loss = 0.0
    n_batches  = 0
    step_correct = 0
    step_total   = 0
    with torch.no_grad():
        for input_ids, attn, labels, graphs, field_embs, step_spans in val_loader:
            loss, logits, n_prefix = forward_batch(
                input_ids, attn, labels, graphs, field_embs,
                model, stage1, adapter, embed_layer, device, dtype, return_logits=True,
            )
            total_loss += loss.item()
            n_batches  += 1

            # logits[b, t] predicts the token at position t+1 of the
            # PREFIXED sequence; step_spans are defined relative to the
            # un-prefixed input_ids, so shift by n_prefix, and by -1 to
            # align a prediction position with the label it's predicting.
            preds = logits.argmax(-1)  # (B, n_prefix + T)
            B = input_ids.shape[0]
            for b in range(B):
                s, e = step_spans[b, 0].item(), step_spans[b, 1].item()
                if e <= s:
                    continue  # span not found for this row (see SFTDataset) -- skip
                lo = n_prefix + s - 1
                hi = n_prefix + e - 1
                lo = max(lo, 0)
                if hi <= lo:
                    continue
                gold_tok = input_ids[b, s:e].to(device)
                pred_tok = preds[b, lo:hi]
                if pred_tok.shape[0] != gold_tok.shape[0]:
                    continue  # truncated by max_len -- skip rather than misalign
                step_correct += (pred_tok == gold_tok).all().item()  # whole-span exact match
                step_total   += 1

    model.train()
    adapter.train()
    avg_loss = total_loss / max(n_batches, 1)
    step_field_acc = step_correct / max(step_total, 1)
    return avg_loss, step_field_acc


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

    # ── Frozen Stage-1 classifier (whole model, not just the two encoders) ────
    # FIX: previously only `stage1.graph_encoder` / `stage1.context_encoder`
    # were kept, discarding stage1.graph_gate / context_gate / fusion --
    # i.e. exactly the trained layers that turn those two encoder outputs
    # into the representation step_head/mcp_head actually use. Keep the
    # whole frozen model so forward_batch can call
    # stage1.encode_and_predict(...) and reuse that trained fusion.
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
    else:
        stage1.load_state_dict(ckpt)
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad_(False)
    print("[Stage 2] ✓ Frozen Stage-1 classifier (encoders + gates + fusion) loaded")

    # ── GraphPrefixAdapter (trainable) ────────────────────────────────────────
    # Input dim changed from GNN_OUT_DIM (raw graph embedding) to
    # GRAPH_PREFIX_SRC_DIM = FUSION_HIDDEN // 2 (the fused representation
    # from stage1.encode_and_predict). Checkpoints trained before this fix
    # are NOT compatible -- retrain Stage 2 from scratch after this change.
    llm_hidden = model.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).to(dtype)

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

    # ── REMOVED: Precompute Stage-1 classifier hints ─────────────────────────
    # Critical fix: Force model to learn from graph prefix tokens instead of
    # copying Stage 1 predictions. This is essential for Stage 2 to actually
    # improve over Stage 1 performance.

    train_ds = SFTDataset(train_examples, tokenizer, mask_hint_prob=0.5, is_training=True)
    val_ds   = SFTDataset(val_examples,   tokenizer, mask_hint_prob=1.0, is_training=False)

    # ── Class-balanced sampling for training ──────────────────────────────
    # step_label support is heavily skewed (e.g. "Exploit the selected
    # exploitations" ~92 vs "Analyze the outcomes..." ~3 in the eval split;
    # training data is similarly skewed). Plain shuffle=True lets the model
    # minimize token-level loss mostly by getting good at the majority
    # class, which is consistent with the low recall on rare classes (e.g.
    # "Do a google search for more information" recall 0.05). Stage 1's GNN
    # avoids this via focal loss + explicit class weights (see
    # graph_encoder.Stage1Classifier.loss); Stage 2/3 are next-token SFT so
    # the equivalent lever is a weighted *sampler* — inverse-frequency
    # per-example weights so every step class is seen roughly equally often
    # per epoch, without discarding any majority-class examples.
    # NOTE: a plain 1/count inverse-frequency weight is TOO aggressive here —
    # with e.g. "Exploit the selected exploitations" at ~90+ examples vs
    # "Analyze the outcomes..." at ~3, raw inverse frequency gives the rare
    # class ~30x the sampling weight of the majority class per epoch. That
    # overshoots: the model starts over-predicting the formerly-rare classes
    # (e.g. "Do a google search" recall going to 100% but precision crashing
    # to ~34%) while the majority class's own recall collapses (85% -> 38%).
    # sqrt(1/count) is the standard, much gentler correction (used e.g. in
    # class-balanced loss / effective-number weighting): it upweights rare
    # classes without inverting the imbalance. Additionally clip the
    # weight ratio to a max of 4x the smallest per-class weight so no single
    # class can dominate or vanish from a batch.
    train_step_idxs = [e["step_idx"] for e in train_examples]
    step_counts = np.bincount(train_step_idxs, minlength=len(STEP_LABELS)).astype(np.float64)
    step_counts[step_counts == 0] = 1.0  # guard against unseen classes in this split
    inv_freq = 1.0 / np.sqrt(step_counts)
    inv_freq = np.clip(inv_freq, inv_freq.max() / 4.0, inv_freq.max())
    sample_weights = np.array([inv_freq[i] for i in train_step_idxs], dtype=np.float64)
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_examples),
        replacement=True,
    )

    make_loader = lambda ds, shuffle, sampler=None: DataLoader(
        ds,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=False,
    )
    train_loader = make_loader(train_ds, shuffle=False, sampler=train_sampler)
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
    # FIX: selection metric changed from raw val_loss to step_field_acc (see
    # run_validation docstring) -- picking "lowest blended token loss" was
    # optimizing a different thing than step-classification correctness,
    # which is very plausibly why an early checkpoint with a collapsed class
    # (0/22 correct on one step type) could still look like the "best"
    # checkpoint by loss. val_loss is still tracked and used as a tiebreaker
    # when step_field_acc ties, so this doesn't ignore explanation/MCP
    # quality entirely -- it just stops letting them outvote step accuracy.
    best_val_loss     = float("inf")
    best_step_acc      = -1.0
    best_epoch         = -1
    no_improve_count   = 0
    global_step        = 0

    best_ckpt_dir = os.path.join(STAGE2_ADAPTER_DIR, "best")

    for epoch in range(STAGE2_EPOCHS):
        model.train()
        adapter.train()
        epoch_loss = 0.0
        opt.zero_grad()

        for i, (input_ids, attn, labels, graphs, field_embs, _step_spans) in enumerate(train_loader):
            loss = forward_batch(
                input_ids, attn, labels, graphs, field_embs,
                model, stage1, adapter, embed_layer,
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
        val_loss, step_field_acc = run_validation(
            val_loader, model, stage1, adapter, embed_layer, device, dtype
        )

        # Primary: step_field_acc (higher is better). Tiebreak: lower val_loss.
        improved = (step_field_acc > best_step_acc) or (
            step_field_acc == best_step_acc and val_loss < best_val_loss
        )
        marker   = "  ← best" if improved else ""
        print(f"epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
              f"train_loss {avg_train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val_step_field_acc {step_field_acc:.4f}{marker}")

        if improved:
            best_val_loss     = val_loss
            best_step_acc      = step_field_acc
            best_epoch         = epoch + 1
            no_improve_count   = 0
            # Save best checkpoint
            os.makedirs(best_ckpt_dir, exist_ok=True)
            model.save_pretrained(best_ckpt_dir)
            torch.save(adapter.state_dict(),
                       os.path.join(best_ckpt_dir, "graph_adapter.pt"))
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f"  → best checkpoint saved  (val_step_field_acc={best_step_acc:.4f}, val_loss={best_val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  → no improvement for {no_improve_count}/{STAGE2_EARLY_STOP_PATIENCE} epochs")
            if no_improve_count >= STAGE2_EARLY_STOP_PATIENCE:
                print(f"\n[Stage 2] Early stopping at epoch {epoch+1}. "
                      f"Best was epoch {best_epoch} (val_step_field_acc={best_step_acc:.4f}, val_loss={best_val_loss:.4f})")
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
    precompute_stage1_hints(test_examples, stage1, device, dtype)
    test_ds = SFTDataset(test_examples, tokenizer)
    test_loader = DataLoader(
        test_ds,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=False,
    )
    
    # Load best model
    # ── FIX: load the saved adapter onto a FRESH base model, not `base_model` ──
    # `base_model` above was already wrapped in-place by get_peft_model() and
    # trained all the way to the early-stopping epoch (epoch 6 in the observed
    # run), not the best epoch (epoch 3) that was actually saved to disk.
    # Calling PeftModel.from_pretrained(base_model, ...) on that already-
    # wrapped, already-trained object is what produced the
    # "Already found a peft_config attribute in the model... multiple
    # adapters" warning -- it stacks a second adapter on top of the
    # in-memory (overfit, wrong-epoch) weights instead of cleanly giving you
    # just the best-epoch checkpoint. Reloading a clean base model guarantees
    # eval actually reflects the saved best checkpoint and nothing else.
    del model
    del base_model
    if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
        torch.cuda.empty_cache()
    eval_base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    model = PeftModel.from_pretrained(eval_base_model, STAGE2_ADAPTER_DIR)
    adapter.load_state_dict(torch.load(os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt"), map_location=device))
    model.eval()
    adapter.eval()
    
    normalizer = StepLabelNormalizer()
    csv_rows = []
    
    with torch.no_grad():
        for input_ids, attn, labels, graphs, field_embs, _step_spans in test_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            graphs = graphs.to(device)
            field_embs = field_embs.to(device)

            # Fused Stage-1 representation (matching training -- see
            # forward_batch / encode_and_predict). Was an ad hoc
            # parameter-free graph/context blend that did NOT match what
            # forward_batch used during training; both now call the same
            # stage1.encode_and_predict(...) so train and eval-time
            # generation see the identical distribution.
            edge_attr = getattr(graphs, 'edge_attr', None)
            with torch.no_grad():
                combined_emb, _, _ = stage1.encode_and_predict(
                    graphs.x, graphs.edge_index, graphs.batch, field_embs, edge_attr=edge_attr
                )

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
            
            # ── FIX: outputs already contains ONLY the newly generated tokens ──
            # When model.generate() is called with ONLY inputs_embeds (no
            # input_ids), HF has no token-ID representation of the prompt/prefix
            # to prepend to the returned sequence, so `outputs` IS the
            # completion -- there is nothing to slice off. The previous
            # `outputs[:, n_prefix:]` chopped off the first n_prefix (16)
            # tokens of the actual generated response (where the opening
            # `{"New step": ...` JSON almost always lives), which is why the
            # eval below was calling nearly every row "UNPARSEABLE". This is
            # the same bug already identified and fixed in stage3_grpo_rl.py
            # (see the header comment there) -- ported the fix here.
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

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

                    prompt = build_prompt(ex, mask_hint=True)
                    csv_rows.append({
                        "machine": ex.get("machine", ""),
                        "new_strategy": ex.get("new strategy", ""),
                        "strategy_explanation": ex.get("new strategy explanation", ""),
                        "step_prediction": pred_step_label,
                        "gold_new_step": gold_step_label,
                        "mcp_tool_prediction": pred_mcp_tools,
                        "mcp_tool_gold": gold_mcp_tools,
                        "step_explanation_predicted": pred_expl,
                        "step_explanation_gold": gold_expl,
                        "prompt": prompt,
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