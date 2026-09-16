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
import math
import os
import random
import csv
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, QWEN_MODEL_NAME, GRAPH_PREFIX_TOKENS, GNN_OUT_DIM, GNN_HIDDEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    STAGE2_LR, STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_GRAD_ACCUM,
    STAGE2_VAL_SPLIT, STAGE2_EARLY_STOP_PATIENCE, STAGE2_GRAD_CLIP, STAGE2_WARMUP_RATIO,
    STAGE2_WEIGHT_DECAY, STAGE2_STEP_TOKEN_LOSS_WEIGHT, STAGE2_ADAPTER_LR_MULT,
    STAGE1_CKPT, STAGE2_ADAPTER_DIR,
    RANDOM_SEED, STEP_LABELS, MCP_LABELS, ROOT,
)

# Stage-2 consumes ONLY the raw 512-d GINE graph representation from the
# frozen Stage-1 checkpoint. The private Stage-1 fusion/classifier vector is
# never used by the prefix adapter.
GRAPH_PREFIX_SRC_DIM = GNN_OUT_DIM
from data_utils import load_from_input_json, StepLabelNormalizer, extract_mcp_labels
from graph_encoder import load_graph_encoder

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


def build_prompt(ex: dict) -> str:
    """Stage-2 user prompt: the strategy pair, and nothing else.

    The graph reaches the model as 16 soft-prompt tokens produced by
    GraphPrefixAdapter from Stage 1's 512-d graph embedding, NOT as text --
    so the textual prompt carries only what the graph cannot: the new strategy
    and its explanation.

    REMOVED: the `Machine:` line (identity, not evidence -- it invited the
    model to memorise per-machine answers rather than read the strategy), and
    a `hint` parameter carrying Stage 1's prediction. That hint was dead code:
    `make_stage1_hint()` was never called from anywhere in the repo and
    `ex["stage1_pred"]` was never populated, so every prompt was identical no
    matter what the STAGE2_USE_STAGE1_HINT config claimed. Both the parameter
    and the config flag are gone rather than left to mislead an ablation table.
    """
    ctx = ex["context"]
    return "\n".join([
        "# Strategy",
        f"New strategy: {ctx['New strategy']}",
        f"Strategy explanation: {ctx['Strategy explanation']}",
        "",
        "# Task",
        "Based on the graph context and strategy above, determine the next "
        "step, the tools needed, and explain your reasoning.",
    ])


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
    Turns the frozen Stage-1 GINE output into GRAPH_PREFIX_TOKENS soft-prompt
    embeddings in the LLM's hidden space, using a learned-query cross-attention
    resampler over the graph's PER-NODE states.

    ---------------------------------------------------------------------
    REWRITTEN (see STAGE2_STAGE3_IMPROVEMENTS.md). The previous version was:

        graph_emb (B, 512)
          -> Linear(512, H*2) -> Linear(H*2, H*2) -> Linear(H*2, H*n_tokens)
          -> reshape (B, n_tokens, H)

    Two serious problems with that:

    1. INFORMATION. Every one of the 8 output tokens was a slice of a single
       expansion of ONE pooled 512-d vector. The tokens could not carry
       independent graph content -- the whole PTT graph (node titles, types,
       statuses, degrees, depths, edge types, topology) was squeezed through
       one global summary before the LLM saw anything. The GNN already
       computes per-node hidden states (`GraphEncoder.forward_with_nodes`),
       and they were being discarded at exactly the point the LLM needed them.

    2. PARAMETERS. With llm_hidden=5120 and n_tokens=8 that stack is
       Linear(512,10240) + Linear(10240,10240) + Linear(10240,40960)
       ~= 530M trainable parameters -- on ~1.5k training examples. That is a
       severe overfitting liability and by far the largest trainable block in
       Stage 2 (much larger than the LoRA itself).

    This version instead uses the standard mechanism for feeding a
    variable-size encoder output into a frozen LLM: a small set of LEARNED
    QUERY VECTORS that cross-attend over the encoder's token set, as in
    Flamingo's Perceiver Resampler (Alayrac et al., NeurIPS 2022) and BLIP-2's
    Q-Former (Li et al., ICML 2023). Each output token is free to attend to a
    different part of the graph, and the global pooled vector is prepended to
    the key/value set so it is always available (and so every row has at least
    one valid key even if a graph somehow has no nodes).

        queries (n_tokens, d)  --cross-attn-->  [global ; nodes] (B, 1+N, d)
          -> residual + FFN -> Linear(d, H) -> (B, n_tokens, H)

    At d_model=1024 / n_tokens=16 this is ~15M parameters: ~36x smaller than
    the old stack while passing twice as many, genuinely distinct, tokens.
    """

    def __init__(self, graph_dim: int, llm_hidden: int,
                 n_tokens: int = GRAPH_PREFIX_TOKENS, node_dim: int = GNN_HIDDEN,
                 d_model: int = 1024, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.llm_hidden = llm_hidden
        self.graph_dim = graph_dim
        self.node_dim = node_dim
        self.d_model = d_model

        self.queries = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.global_proj = nn.Linear(graph_dim, d_model)
        self.node_proj = nn.Linear(node_dim, d_model)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, llm_hidden),
            nn.LayerNorm(llm_hidden),
        )
        # SCALE-MATCH THE SOFT TOKENS TO REAL TOKEN EMBEDDINGS.
        # A plain LayerNorm forces unit per-element RMS, i.e. per-token L2 norm
        # == sqrt(llm_hidden) == ~71.6 at Qwen3-14B's hidden size of 5120.
        # Real Qwen embed_tokens rows have L2 norm ~1.0 (measured: mean 1.023,
        # median 1.015 on Qwen2.5-1.5B-Instruct), so the soft tokens entered
        # the residual stream ~70x oversized. Qwen is a PRE-NORM transformer:
        # every block reads RMSNorm(h) but writes an O(1) update back into h.
        # Against a norm-71 residual that update is ~70x too weak to move the
        # token, so the graph prefix passed through all 40 layers essentially
        # UNCHANGED -- never contextualized, and never integrated with the
        # text. Initializing the final LayerNorm's gain to 1/sqrt(llm_hidden)
        # puts the output at per-token L2 norm ~1.0, matching the embedding
        # table the LLM was actually trained on. It stays learnable (and
        # per-channel), so training can still adjust it; this only fixes the
        # starting scale. Done via the existing LayerNorm weight rather than a
        # new parameter so the state_dict keys are unchanged and previously
        # saved graph_adapter.pt files still load.
        with torch.no_grad():
            self.out_proj[-1].weight.fill_(1.0 / math.sqrt(llm_hidden))

    def forward(self, graph_emb: torch.Tensor,
                node_states: torch.Tensor | None = None,
                node_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        graph_emb   : (B, graph_dim)          pooled Stage-1 GINE output
        node_states : (B, N, node_dim) or None  per-node GINE hidden states
        node_mask   : (B, N) bool or None       True where a node is real

        Returns (B, n_tokens, llm_hidden).

        `node_states=None` falls back to global-only conditioning (the
        queries attend to the single global token), so any caller that only
        has the pooled vector still works.
        """
        B = graph_emb.shape[0]
        g = self.global_proj(graph_emb).unsqueeze(1)            # (B, 1, d)

        key_padding_mask = None
        if node_states is not None and node_states.shape[1] > 0:
            kv = torch.cat([g, self.node_proj(node_states)], dim=1)   # (B, 1+N, d)
            if node_mask is not None:
                gmask = torch.ones(B, 1, dtype=torch.bool, device=node_mask.device)
                # The global token is always valid, so no row is ever fully
                # masked -- which is what keeps softmax from producing NaN
                # on a graph whose nodes are all padding.
                key_padding_mask = ~torch.cat([gmask, node_mask], dim=1)
        else:
            kv = g

        q = self.queries.unsqueeze(0).expand(B, -1, -1)         # (B, n_tokens, d)
        attended, _ = self.attn(
            self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
            key_padding_mask=key_padding_mask,
        )
        h = q + attended
        h = h + self.ffn(self.ln_ffn(h))
        return self.out_proj(h)                                 # (B, n_tokens, H)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_len: int = 1536, is_training: bool = True):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_len
        self.is_training = is_training

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

        # IMPORTANT NUMERICAL FIX: if the prompt itself is >= max_len, the
        # old code truncated away the entire target, leaving labels == -100
        # for every position. Hugging Face causal-LM loss then has no valid
        # targets and returns NaN. This was the direct cause of intermittent
        # `train_loss nan` on long strategy/explanation rows.
        # Always reserve room for at least a meaningful target prefix and EOS.
        # Keep the END of the prompt (strategy/task instructions) rather than
        # the beginning if truncation is necessary.
        min_target_tokens = min(len(target_ids), max(32, min(256, len(target_ids))))
        max_prompt_len = max(1, self.max_len - min_target_tokens)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        available_target = max(1, self.max_len - len(prompt_ids))
        target_ids = target_ids[:available_target]
        if not target_ids:
            raise RuntimeError("Stage 2 example has no target tokens after truncation")

        input_ids = prompt_ids + target_ids
        # Only target tokens contribute to loss. This construction guarantees
        # at least one non-masked target token for every example.
        labels = ([-100] * len(prompt_ids)) + target_ids

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

        return {
            "input_ids":  torch.tensor(input_ids),
            "labels":     torch.tensor(labels),
            "graph":      ex["graph"],          # torch_geometric Data (pre-built)
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
    step_spans = torch.stack([b["step_span"] for b in batch])  # (B, 2) = [start, end)
    return input_ids, attn, labels, graphs, step_spans


# ---------------------------------------------------------------------------
# Single forward pass (shared by train and val loops)
# ---------------------------------------------------------------------------

def forward_batch(input_ids, attn, labels, graphs,
                  model, graph_encoder, adapter, embed_layer, device, dtype, return_logits=False,
                  step_spans=None, step_token_weight=1.0):
    """
    Prepend graph prefix tokens to the token embeddings, run the model,
    and return the scalar loss (and, if return_logits=True, the raw logits
    plus n_prefix -- needed by run_validation to score just the "New step"
    value span, since the graph-prefix tokens shift every index by
    n_prefix relative to the un-prefixed input_ids/labels/step_spans this
    function was called with).

    Stage 1 is frozen. Only its raw GINE encoder output (512-d) is
    consumed here; the private Stage-1 fusion and classifier heads are not
    part of the Stage-2/3 graph-conditioning interface.

    CLEANUP (architecture re-audit): this used to also take a `field_embs`
    tensor (a BGE embedding of the New-strategy/Strategy-explanation text,
    computed per example in SFTDataset.__getitem__) and thread it through
    every call site (collate_fn, run_validation, the train/test loops) --
    but the body here never consumed it beyond `.to(device)`, so it was
    computing and moving a real embedding-model output every batch for zero
    effect on the forward pass. Removed end-to-end.
    """
    input_ids = input_ids.to(device)
    attn      = attn.to(device)
    labels    = labels.to(device)
    graphs    = graphs.to(device)

    with torch.no_grad():
        edge_attr = getattr(graphs, 'edge_attr', None)
        # Stage 2 graph-prefix contract: ONLY the frozen Stage-1 GINE
        # representation enters the prefix adapter. The Stage-1 checkpoint is
        # a set of weights, not an input tensor; classifier/fusion outputs are
        # deliberately not exposed to Qwen.
        #
        # Now uses forward_with_nodes() so the adapter's resampler can attend
        # to PER-NODE graph states, not just the single pooled vector (see
        # GraphPrefixAdapter's docstring). Still purely the GINE encoder --
        # no fusion/classifier output crosses this boundary.
        graph_emb, node_states, node_mask = graph_encoder.forward_with_nodes(
            graphs.x, graphs.edge_index, graphs.batch, edge_attr=edge_attr
        )  # (B, 512), (B, N, GNN_HIDDEN), (B, N)

    prefix_embeds = adapter(
        graph_emb.float(), node_states.float(), node_mask
    ).to(dtype)  # (B, n_tokens, H)
    token_embeds  = embed_layer(input_ids).to(dtype)          # (B, T, H)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    n_prefix     = prefix_embeds.shape[1]
    prefix_attn  = torch.ones(attn.shape[0], n_prefix, device=device, dtype=attn.dtype)
    attn_full    = torch.cat([prefix_attn, attn], dim=1)
    prefix_lbls  = torch.full((labels.shape[0], n_prefix), -100, device=device, dtype=labels.dtype)
    labels_full  = torch.cat([prefix_lbls, labels], dim=1)

    # When we're going to compute a step-weighted loss ourselves, don't ask
    # the model to also compute its uniform `out.loss` (we ignore it) -- but
    # we still need the logits either way.
    want_weighted = (step_spans is not None) and (step_token_weight is not None) and (step_token_weight != 1.0)
    hf_labels = None if want_weighted else labels_full

    # Qwen forward in bf16 autocast.  The adapter itself is fp32; only its
    # final prefix representation is cast to the model input dtype.
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_full,
                labels=hf_labels,
            )
    else:
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_full,
            labels=hf_labels,
        )

    if want_weighted:
        # Manual next-token cross-entropy with a higher weight on the "New
        # step" value span (see STAGE2_STEP_TOKEN_LOSS_WEIGHT in config.py).
        # Coordinate frames: `labels` is un-prefixed (prompt+target);
        # labels_full = [prefix(-100) | labels], so un-prefixed index j sits
        # at full index n_prefix+j. Next-token loss predicts label at full
        # index k from position k-1, so the token at un-prefixed index j is
        # scored at shift index (n_prefix + j - 1). step_spans[b] = [s, e)
        # in un-prefixed coords therefore maps to shift indices
        # [n_prefix+s-1, n_prefix+e-1).
        logits = out.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels_full[:, 1:]
        B, Tm1, V = shift_logits.shape
        ce = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(B, Tm1)
        valid = (shift_labels != -100).float()
        weights = valid.clone()  # 1.0 on every real target token
        for b in range(B):
            s, e = int(step_spans[b][0].item()), int(step_spans[b][1].item())
            if e > s:
                lo = max(0, n_prefix + s - 1)
                hi = min(Tm1, n_prefix + e - 1)
                if hi > lo:
                    weights[b, lo:hi] = valid[b, lo:hi] * step_token_weight
        denom = weights.sum().clamp_min(1.0)
        loss = (ce * weights).sum() / denom
    else:
        loss = out.loss.float()

    if not torch.isfinite(loss):
        raise FloatingPointError("Stage 2 produced a non-finite loss")
    if return_logits:
        return loss, out.logits, n_prefix
    return loss


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
def run_validation(val_loader, model, graph_encoder, adapter, embed_layer, device, dtype,
                   tokenizer=None, val_examples=None, max_new_tokens=64):
    """Validate Stage 2 using leakage-free greedy generation.

    The primary checkpoint metric is exact accuracy of the generated ``New step``
    after normalization to the fixed 10-class taxonomy.  This replaces the old
    teacher-forced token-span exact match, where one wrong token made the whole
    field incorrect.

    IMPORTANT: validation generation uses ONLY the prompt tokens.  The SFT batch
    also contains the gold target because those tokens are needed for LM loss;
    feeding them into ``generate`` would leak the answer and invalidate the metric.
    """
    model.eval()
    adapter.eval()
    total_loss = 0.0
    n_batches = 0
    step_correct = 0
    step_total = 0

    normalizer = StepLabelNormalizer()
    obj_parser = build_obj_parser()

    with torch.no_grad():
        for batch_idx, (input_ids, attn, labels, graphs, _step_spans) in enumerate(val_loader):
            # 1) Normal validation loss on the complete prompt+target sequence.
            loss = forward_batch(
                input_ids, attn, labels, graphs,
                model, graph_encoder, adapter, embed_layer, device, dtype, return_logits=False,
            )
            total_loss += loss.item()
            n_batches += 1

            # 2) Build prompt-only sequences for generation.  The first non--100
            # label marks the beginning of the gold target.
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            graphs = graphs.to(device)

            B = input_ids.shape[0]
            prompt_lens = []
            for b in range(B):
                valid = torch.nonzero(labels[b] != -100, as_tuple=False)
                prompt_lens.append(int(valid[0].item()) if valid.numel() else int(attn[b].sum().item()))
            max_prompt_len = max(prompt_lens)

            prompt_ids = input_ids[:, :max_prompt_len].clone()
            prompt_attn = torch.zeros((B, max_prompt_len), device=device, dtype=attn.dtype)
            for b, plen in enumerate(prompt_lens):
                prompt_attn[b, :plen] = 1
                if plen < max_prompt_len:
                    prompt_ids[b, plen:] = tokenizer.pad_token_id

            edge_attr = getattr(graphs, 'edge_attr', None)
            graph_emb, node_states, node_mask = graph_encoder.forward_with_nodes(
                graphs.x, graphs.edge_index, graphs.batch, edge_attr=edge_attr
            )
            prefix_embeds = adapter(graph_emb.float(), node_states.float(), node_mask).to(dtype)
            token_embeds = embed_layer(prompt_ids).to(dtype)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            n_prefix = prefix_embeds.shape[1]
            prefix_attn = torch.ones((B, n_prefix), device=device, dtype=prompt_attn.dtype)
            attn_full = torch.cat([prefix_attn, prompt_attn], dim=1)

            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_full,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            batch_start = batch_idx * val_loader.batch_size
            for i, gen_text in enumerate(generated_texts):
                if val_examples is None:
                    continue
                ex_idx = batch_start + i
                if ex_idx >= len(val_examples):
                    continue

                gold_step = normalizer.normalize(val_examples[ex_idx].get("step_label", ""))
                obj = obj_parser(gen_text, normalizer)
                pred_step = normalizer.normalize(obj.get("New step", ""))

                # Count every validation example.  Unparseable/missing steps are
                # genuine generation failures and therefore count as incorrect.
                step_total += 1
                step_correct += int(pred_step is not None and pred_step == gold_step)

    model.train()
    adapter.train()
    avg_loss = total_loss / max(n_batches, 1)
    step_acc = step_correct / max(step_total, 1)
    return avg_loss, step_acc


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
    # Qwen is kept in bf16 for memory efficiency, but all trainable LoRA and
    # graph-adapter parameters are kept in fp32.  The previous implementation
    # let the graph adapter participate in bf16 optimisation directly; on the
    # DGX run this produced intermittent NaNs after a few optimizer steps.
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    base_model.config.use_cache = False
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

    # Keep LoRA parameters in fp32.  Qwen itself remains bf16.  The forward
    # pass below uses CUDA bf16 autocast, so this is compatible with the bf16
    # base model while giving AdamW fp32 optimizer states for the trainable
    # parameters.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    model.print_trainable_parameters()

    # ── Frozen graph encoder (ONLY the graph encoder) ────────────────────────
    # Stage 2 receives the 512-d graph embedding and nothing else from Stage 1.
    # load_graph_encoder extracts the `graph_encoder.*` sub-tree and refuses to
    # load the rest, so the fusion MLP, step head, MCP head, phase head, text
    # tower and graph gates are never even resident here. Stage 1's step/MCP
    # PREDICTIONS do not influence Stage 2 in any form.
    graph_encoder = load_graph_encoder(STAGE1_CKPT, device)
    print("[Stage 2] ✓ Graph encoder loaded (frozen); 512-d graph embedding is "
          "the only thing crossing the Stage-1 boundary")

    # ── GraphPrefixAdapter (trainable) ────────────────────────────────────────
    # Prefix input is the raw 512-d GINE graph representation. Stage-1
    # checkpoints from the previous architecture are NOT compatible; retrain
    # Stage 2 from scratch after the final Stage-1 change.
    llm_hidden = model.config.hidden_size
    # IMPORTANT: do not train this projection in bf16.  Its output is cast to
    # the Qwen dtype only immediately before concatenation with token embeds.
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).float()

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

    train_ds = SFTDataset(train_examples, tokenizer, is_training=True)
    val_ds   = SFTDataset(val_examples,   tokenizer, is_training=False)

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
    lora_params    = [p for p in model.parameters() if p.requires_grad]
    adapter_params = [p for p in adapter.parameters() if p.requires_grad]
    # Flat list is what the grad-clip / non-finite guards below iterate over;
    # the optimizer gets the same parameters split into two LR groups.
    trainable_params = lora_params + adapter_params
    # A conservative LR is intentional: Stage 2 only trains LoRA + the
    # graph-to-prefix projector while the 14B base is frozen.  The previous
    # 1e-5 setting was capable of producing a non-finite update in this
    # manual bf16 training loop.
    # CLEANUP (architecture re-audit): config.py's STAGE2_LR/STAGE2_WEIGHT_DECAY
    # were imported (or, for STAGE2_LR, imported but silently ignored) in
    # favor of these two independently hardcoded literal defaults -- editing
    # config.py had zero effect on the actual LR/weight-decay used. config.py
    # has been updated to the values actually proven in practice (2e-6 /
    # 1e-4, matching what was hardcoded here), and both env vars now fall
    # back to it, restoring config.py as the real source of truth with no
    # change to today's behavior.
    stage2_lr = float(os.environ.get("STAGE2_SAFE_LR", str(STAGE2_LR)))
    stage2_wd = float(os.environ.get("STAGE2_SAFE_WEIGHT_DECAY", str(STAGE2_WEIGHT_DECAY)))
    # SEPARATE LR GROUP FOR THE GRAPH PREFIX ADAPTER.
    # WHY: `lora_params` are low-rank deltas on an already-pretrained 14B model
    # and genuinely want a tiny LR; `adapter_params` are a ~14.6M-parameter
    # cross-attention resampler being trained FROM RANDOM INIT. AdamW moves a
    # parameter by ~lr per step, so at 2e-6 over ~744 steps the adapter's
    # learned queries travel ~1.5e-3 against a randn*0.02 init -- i.e. they
    # stay random, attend near-uniformly, and emit GRAPH_PREFIX_TOKENS nearly
    # identical soft tokens. That regressed Stage 2 to 0.6151 val_step_acc /
    # 66.42% test Step Exact Match. Same bug class (and same fix) as Stage 1's
    # graph-gate param group at STAGE1_GRAPH_GATE_LR_MULT.
    adapter_lr_mult = float(os.environ.get("STAGE2_SAFE_ADAPTER_LR_MULT",
                                          str(STAGE2_ADAPTER_LR_MULT)))
    adapter_lr = stage2_lr * adapter_lr_mult
    opt = torch.optim.AdamW(
        [
            {"params": lora_params,    "lr": stage2_lr},
            {"params": adapter_params, "lr": adapter_lr},
        ],
        lr=stage2_lr,
        weight_decay=stage2_wd,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
    )

    # Use ceil because the final partial accumulation window is also flushed.
    steps_per_epoch  = max(1, (len(train_loader) + STAGE2_GRAD_ACCUM - 1) // STAGE2_GRAD_ACCUM)
    total_steps      = steps_per_epoch * STAGE2_EPOCHS
    warmup_steps     = max(20, int(total_steps * STAGE2_WARMUP_RATIO))
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"[Stage 2] Steps/epoch     : {steps_per_epoch}")
    print(f"[Stage 2] Total steps     : {total_steps}  (warmup {warmup_steps})")
    print(f"[Stage 2] Safe LR          : {stage2_lr:.2e} (LoRA, {sum(p.numel() for p in lora_params)/1e6:.1f}M) | "
          f"{adapter_lr:.2e} (adapter x{adapter_lr_mult:g}, {sum(p.numel() for p in adapter_params)/1e6:.1f}M)")
    print(f"[Stage 2] Regularization   : weight_decay={stage2_wd:.1e} | fp32 trainables + bf16 Qwen")

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
        finite_batches = 0
        skipped_batches = 0
        accum_count = 0
        opt.zero_grad(set_to_none=True)

        for i, (input_ids, attn, labels, graphs, step_spans) in enumerate(train_loader):
            try:
                loss = forward_batch(
                    input_ids, attn, labels, graphs,
                    model, graph_encoder, adapter, embed_layer,
                    device, dtype,
                    step_spans=step_spans,
                    step_token_weight=STAGE2_STEP_TOKEN_LOSS_WEIGHT,
                )
            except FloatingPointError:
                skipped_batches += 1
                opt.zero_grad(set_to_none=True)
                accum_count = 0
                continue

            if not torch.isfinite(loss):
                skipped_batches += 1
                opt.zero_grad(set_to_none=True)
                accum_count = 0
                continue

            (loss / STAGE2_GRAD_ACCUM).backward()
            accum_count += 1
            epoch_loss += float(loss.detach().cpu())
            finite_batches += 1

            should_step = (accum_count >= STAGE2_GRAD_ACCUM) or (i == len(train_loader) - 1)
            if should_step:
                # Check gradients BEFORE clipping.  Clipping a NaN/Inf gradient
                # does not repair it and would otherwise poison the optimizer.
                grads_finite = True
                for p in trainable_params:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break

                if not grads_finite:
                    skipped_batches += accum_count
                    opt.zero_grad(set_to_none=True)
                    accum_count = 0
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, STAGE2_GRAD_CLIP, error_if_nonfinite=True
                )
                if not torch.isfinite(grad_norm):
                    skipped_batches += accum_count
                    opt.zero_grad(set_to_none=True)
                    accum_count = 0
                    continue

                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0

                # Verify that the optimizer did not create non-finite trainable
                # parameters. If it did, restore the just-skipped update by
                # stopping rather than continuing with a corrupted policy.
                params_finite = all(
                    torch.isfinite(p).all().item()
                    for p in trainable_params
                )
                if not params_finite:
                    raise RuntimeError(
                        "Stage 2 optimizer produced non-finite trainable parameters. "
                        "Lower STAGE2_SAFE_LR (currently %.2e)." % stage2_lr
                    )

                if global_step % 20 == 0:
                    avg = epoch_loss / max(finite_batches, 1)
                    print(f"  epoch {epoch+1:02d} | step {global_step:4d} | "
                          f"train_loss {avg:.4f} | skipped {skipped_batches}")

        if finite_batches == 0:
            raise RuntimeError(
                "Stage 2 encountered no finite training batches. "
                "Check the Qwen/bf16 environment and graph checkpoint."
            )

        # ── Validation at end of each epoch ───────────────────────────────────
        avg_train_loss = epoch_loss / max(finite_batches, 1)
        # BUG FIX: this was max_new_tokens=32 -- 15.6x smaller than the final
        # test-time evaluation's max_new_tokens=500. Measured against
        # STEP_LABELS: the single longest label alone needs 23 GPT-2 tokens
        # just to close the `"New step": "..."` field (before any JSON
        # syntax overhead or preamble the model might emit before starting
        # the JSON), leaving almost no margin in a 32-token budget. A
        # generation cut off mid-label produces an unparseable/incomplete
        # "New step" value, scoring as wrong even when the model predicted
        # correctly -- this systematically underestimates val accuracy
        # (observed: val plateaued at 0.74 while the SAME checkpoint scored
        # 0.88 at final test time with the larger budget) and, worse, feeds
        # a biased signal into checkpoint selection and early stopping.
        # 64 tokens comfortably covers the longest label plus JSON overhead
        # and a real margin for preamble, while staying far cheaper than the
        # full 500-token budget (which also has to cover the free-text
        # explanation and MCP dict that this step-only check doesn't need).
        val_loss, step_field_acc = run_validation(
            val_loader, model, graph_encoder, adapter, embed_layer, device, dtype,
            tokenizer=tokenizer, val_examples=val_examples, max_new_tokens=64
        )

        # Primary: step_field_acc (higher is better). Tiebreak: lower val_loss.
        improved = (step_field_acc > best_step_acc) or (
            step_field_acc == best_step_acc and val_loss < best_val_loss
        )
        marker   = "  ← best" if improved else ""
        print(f"epoch {epoch+1:02d}/{STAGE2_EPOCHS} | "
              f"train_loss {avg_train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val_generated_step_acc {step_field_acc:.4f}{marker}")

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
    #
    # BUG FIX: this used to copy2() the best-epoch files INTO
    # STAGE2_ADAPTER_DIR without ever clearing whatever was already there.
    # copy2 only overwrites a file of the SAME name -- it does nothing to a
    # stale file left behind by an earlier run under a DIFFERENT name (e.g.
    # an old adapter_model.bin coexisting with today's
    # adapter_model.safetensors, or a stale index/shard file from a run that
    # predates this checkpoint layout). PeftModel.from_pretrained() can then
    # silently pick up the STALE file instead of the one just trained --
    # exactly the failure signature observed: val_generated_step_acc=0.80 in
    # memory during training, but the reloaded-from-disk test-set eval
    # scoring near zero on BOTH step and MCP simultaneously (a collapse
    # consistent with generating from an untrained/wrong adapter, not with
    # a real generalization gap).
    #
    # Fix: wipe every TOP-LEVEL file in STAGE2_ADAPTER_DIR (never touching
    # the "best/" subdirectory, which is what we are about to copy FROM)
    # before copying today's checkpoint in, so no file from a previous run
    # can ever coexist with -- or be mistaken for -- today's.
    import shutil
    if os.path.isdir(STAGE2_ADAPTER_DIR):
        stale = [f for f in os.listdir(STAGE2_ADAPTER_DIR)
                if os.path.isfile(os.path.join(STAGE2_ADAPTER_DIR, f))]
        for fname in stale:
            os.remove(os.path.join(STAGE2_ADAPTER_DIR, fname))
        if stale:
            print(f"[Stage 2] Removed {len(stale)} stale top-level file(s) from "
                  f"{STAGE2_ADAPTER_DIR} before installing today's checkpoint: {stale}")

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
    embed_layer = model.get_input_embeddings()  # must belong to the fresh eval model
    adapter.load_state_dict(torch.load(os.path.join(STAGE2_ADAPTER_DIR, "graph_adapter.pt"), map_location=device))
    adapter = adapter.float()
    model.eval()
    adapter.eval()
    
    normalizer = StepLabelNormalizer()
    csv_rows = []
    
    with torch.no_grad():
        for input_ids, attn, labels, graphs, _step_spans in test_loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            graphs = graphs.to(device)

            # Fused Stage-1 representation (matching training -- see
            # forward_batch / encode_and_predict). Was an ad hoc
            # parameter-free graph/context blend that did NOT match what
            # forward_batch used during training; both now call the same
            # graph_encoder.forward_with_nodes(...) so train and eval-time
            # generation see the identical distribution.
            edge_attr = getattr(graphs, 'edge_attr', None)
            with torch.no_grad():
                graph_emb, node_states, node_mask = graph_encoder.forward_with_nodes(
                    graphs.x, graphs.edge_index, graphs.batch, edge_attr=edge_attr
                )  # (B, 512), (B, N, GNN_HIDDEN), (B, N)

            prefix_embeds = adapter(graph_emb.float(), node_states.float(), node_mask).to(dtype)
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

                    prompt = build_prompt(ex)
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