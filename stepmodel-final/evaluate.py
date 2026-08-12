"""
evaluate.py — Comprehensive evaluation for all three training stages.

Metrics reported for every model:

  STEP CLASSIFICATION
    Accuracy, Macro-F1, Weighted-F1, per-class precision/recall/F1,
    confusion matrix

  MCP TOOL CLASSIFICATION  (multi-label)
    Subset (exact-match) accuracy, Micro-F1, Macro-F1, Samples-F1,
    per-label precision/recall/F1

  STEP EXPLANATION QUALITY  (LLM stages only — GNN doesn't generate text)
    BLEU-1/2/4          — n-gram overlap with gold explanation
    ROUGE-L             — longest common subsequence recall
    BERTScore F1        — semantic similarity via frozen sentence encoder
                          (BAAI/bge-small-en-v1.5, reused from training)
    Step Alignment      — do explanation keywords match the predicted step?
    Context Grounding   — does the explanation reference observed findings?
    Reasoning Density   — causal/justification language fraction
    Avg / P25 / P75 token length — completeness sanity checks
    Empty rate          — fraction of blank/missing explanations

Usage:
    python evaluate.py                         # evaluate all available models
    python evaluate.py --model gnn             # GNN only
    python evaluate.py --model llm             # best available LLM adapter
    python evaluate.py --model llm \\
        --adapter-dir checkpoints/stage3_qwen_grpo
    python evaluate.py --threshold 0.5         # override MCP threshold
    python evaluate.py --save-explanations out.csv   # dump predictions to CSV
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

from config import (
    INPUT_TEST_JSON, STAGE1_CKPT, STEP_LABELS, MCP_LABELS, MCP_DECISION_THRESHOLD,
    QWEN_MODEL_NAME, ROOT,
)
from data_utils import (
    load_from_input_json, _embed_texts, CONTEXT_COLUMNS,
    mcp_multihot, StepLabelNormalizer, extract_mcp_labels,
)
from graph_encoder import Stage1Classifier
from mcp_threshold_search import predict_with_per_class_thresholds
from llm_judge import batch_evaluate_explanations, print_llm_judge_results


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_stage1_checkpoint(ckpt_path: str, device: str):
    """
    Handles both checkpoint formats:
      - New (Improvement 2): dict with 'model_state_dict' + 'mcp_thresholds'
      - Legacy: plain state dict
    Returns (model, mcp_thresholds).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        mcp_thresholds = ckpt.get(
            "mcp_thresholds", [MCP_DECISION_THRESHOLD] * len(MCP_LABELS)
        )
        print(
            f"[eval] Loaded checkpoint "
            f"(epoch={ckpt.get('best_epoch','?')}, "
            f"score={ckpt.get('best_score','?'):.4f})"
            if isinstance(ckpt.get("best_score"), float)
            else f"[eval] Loaded checkpoint (epoch={ckpt.get('best_epoch','?')})"
        )
        print(
            f"[eval] Per-class MCP thresholds: "
            f"{[round(t, 2) for t in mcp_thresholds]}"
        )
    else:
        state_dict = ckpt
        mcp_thresholds = [MCP_DECISION_THRESHOLD] * len(MCP_LABELS)
        print("[eval] Legacy checkpoint — using uniform threshold=0.5 for all MCP labels.")

    model = Stage1Classifier().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, mcp_thresholds


# ---------------------------------------------------------------------------
# Explanation quality metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def bleu_n(hypothesis: list[str], reference: list[str], n: int) -> float:
    """Sentence-level BLEU-n (no brevity penalty, just n-gram precision)."""
    if len(hypothesis) < n or len(reference) < n:
        return 0.0
    ref_ngrams: dict[tuple, int] = defaultdict(int)
    for i in range(len(reference) - n + 1):
        ref_ngrams[tuple(reference[i : i + n])] += 1
    clipped = 0
    for i in range(len(hypothesis) - n + 1):
        ng = tuple(hypothesis[i : i + n])
        if ref_ngrams.get(ng, 0) > 0:
            clipped += 1
            ref_ngrams[ng] -= 1
    return clipped / max(len(hypothesis) - n + 1, 1)


def rouge_l(hypothesis: list[str], reference: list[str]) -> float:
    """ROUGE-L F1 via LCS dynamic programming."""
    m, n = len(reference), len(hypothesis)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    recall    = lcs / m
    precision = lcs / max(n, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bertscore_f1_batch(
    hypotheses: list[str],
    references: list[str],
    batch_size: int = 64,
) -> list[float]:
    """
    Sentence-level BERTScore using BAAI/bge-small-en-v1.5 (already loaded).
    Returns cosine similarity between sentence embeddings — fast, no extra
    model download needed.
    """
    scores = []
    for i in range(0, len(hypotheses), batch_size):
        hyp_batch = [h if h.strip() else "empty" for h in hypotheses[i : i + batch_size]]
        ref_batch = [r if r.strip() else "empty" for r in references[i : i + batch_size]]
        hyp_emb = _embed_texts(hyp_batch)
        ref_emb = _embed_texts(ref_batch)
        scores.extend((hyp_emb * ref_emb).sum(axis=1).tolist())
    return scores


# ── Step-explanation-specific metrics ────────────────────────────────────────

# Keywords associated with each STEP_LABEL index — used for step alignment.
_STEP_KEYWORDS: list[list[str]] = [
    ["google", "search", "googled", "web", "query", "look", "find"],    # 0 google search
    ["enumerate", "enumerat", "version", "hidden", "directory", "service", "scan", "probe"],  # 1 enumerate further
    ["explore", "suspicious", "file", "command", "summary", "finding", "examine", "check"],  # 2 explore files
    ["website", "web", "dirb", "gobuster", "directory", "link", "url", "http", "https"],        # 3 enumerate website
    ["domain", "dns", "subdomain", "ldap", "host", "hostname"],                             # 4 enumerate domain
    ["exploit", "exploitation", "payload", "metasploit", "shell", "attack", "vulnerability"],      # 5 exploit
    ["analyz", "outcome", "attack", "path", "vector", "assess", "result", "review"],        # 6 analyze outcomes
    ["human", "assist", "help", "stuck", "unclear", "manual", "guidance"],                    # 7 ask human
    ["source", "code", "review", "vuln", "injection", "xss", "sqli", "script", "function"],  # 8 source code
    ["report", "end", "task", "complet", "finish", "document", "summary"],                     # 9 end task
]

# Causal / reasoning connectors — presence signals genuine justification.
_REASONING_TOKENS: set[str] = {
    "because", "since", "therefore", "thus", "hence", "as", "so",
    "indicates", "suggests", "shows", "reveals", "found", "identified",
    "discovered", "confirm", "allows", "enables", "in order", "to", "which",
    "given", "based", "due", "result", "led", "lead", "indicate",
}


def step_alignment_score(explanation: str, step_idx: int) -> float:
    """
    Fraction of step-specific keywords present in the explanation.
    Measures whether the model actually explains the step it claimed to take.

    Returns 0.0 – 1.0.  A score above ~0.3 is a reasonable signal.
    """
    if step_idx < 0 or step_idx >= len(_STEP_KEYWORDS):
        return 0.0
    tokens = set(_tokenize(explanation))
    keywords = _STEP_KEYWORDS[step_idx]
    if not keywords:
        return 0.0
    hits = sum(
        1 for kw in keywords
        if any(tok.startswith(kw) for tok in tokens)
    )
    return hits / len(keywords)


def context_grounding_score(explanation: str, prev_step_result: str) -> float:
    """
    Measures how much the explanation references content from the previous
    step result — i.e. factual grounding in what was actually observed.

    Implementation: unigram recall of "content words" (length ≥ 4, not
    stop-words) from prev_step_result that appear in the explanation.

    Returns 0.0 – 1.0.
    """
    _STOP = {
        "that", "this", "with", "from", "have", "been", "were", "they",
        "their", "will", "would", "could", "should", "about", "into",
        "which", "when", "also", "more", "some", "such", "than", "then",
        "there", "these", "those", "your", "what", "where", "here",
    }
    if not prev_step_result or not prev_step_result.strip():
        return 0.0
    result_toks  = {t for t in _tokenize(prev_step_result) if len(t) >= 4 and t not in _STOP}
    expl_toks    = set(_tokenize(explanation))
    if not result_toks:
        return 0.0
    overlap = result_toks & expl_toks
    return len(overlap) / len(result_toks)


def reasoning_density_score(explanation: str) -> float:
    """
    Fraction of tokens in the explanation that are reasoning/causal connectors.
    A completely generic filler sentence will score near 0; a well-justified
    explanation that says "because X was found, Y is the next step" scores
    higher.

    Returns 0.0 – 1.0.
    """
    tokens = _tokenize(explanation)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _REASONING_TOKENS)
    return hits / len(tokens)


def compute_explanation_metrics(
    pred_explanations: list[str],
    gold_explanations: list[str],
    step_preds: list[int] | None = None,
    prev_step_results: list[str] | None = None,
) -> dict:
    """
    Compute all explanation quality metrics.

    Core metrics (require gold explanation):
      BLEU-1/2/4, ROUGE-L, BERTScore-F1

    Intrinsic metrics (computed on predictions only):
      Step Alignment     — keyword overlap with predicted step type
      Context Grounding  — reference to previous step result terms
      Reasoning Density  — fraction of causal/justification tokens
      Length (avg/p25/p75/empty_rate)
    """
    bleu1, bleu2, bleu4, rougeL = [], [], [], []
    valid_preds, valid_refs = [], []
    lengths = []
    alignment, grounding, reasoning = [], [], []
    empty_count = 0

    for idx, (pred, gold) in enumerate(zip(pred_explanations, gold_explanations)):
        pred_str = pred.strip()

        # ── Length & empty rate ───────────────────────────────────────────
        tok_len = len(pred_str.split())
        lengths.append(tok_len)
        if tok_len == 0:
            empty_count += 1

        # ── Intrinsic metrics (no gold needed) ────────────────────────────
        s_idx = step_preds[idx] if step_preds is not None else -1
        alignment.append(step_alignment_score(pred_str, s_idx))

        prev_res = prev_step_results[idx] if prev_step_results is not None else ""
        grounding.append(context_grounding_score(pred_str, prev_res))

        reasoning.append(reasoning_density_score(pred_str))

        # ── Reference-based metrics (skip if gold empty) ──────────────────
        if not gold.strip():
            continue
        hyp_tok = _tokenize(pred_str)
        ref_tok = _tokenize(gold)
        bleu1.append(bleu_n(hyp_tok, ref_tok, 1))
        bleu2.append(bleu_n(hyp_tok, ref_tok, 2))
        bleu4.append(bleu_n(hyp_tok, ref_tok, 4))
        rougeL.append(rouge_l(hyp_tok, ref_tok))
        valid_preds.append(pred_str)
        valid_refs.append(gold)

    bert_scores = bertscore_f1_batch(valid_preds, valid_refs) if valid_preds else []

    def avg(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    lengths_arr = np.array(lengths) if lengths else np.array([0])

    return {
        # Reference-based
        "bleu1":            avg(bleu1),
        "bleu2":            avg(bleu2),
        "bleu4":            avg(bleu4),
        "rouge_l":          avg(rougeL),
        "bertscore_f1":     avg(bert_scores),
        # Intrinsic
        "step_alignment":   avg(alignment),
        "ctx_grounding":    avg(grounding),
        "reasoning_density": avg(reasoning),
        # Length / completeness
        "avg_length":       avg(lengths),
        "p25_length":       float(np.percentile(lengths_arr, 25)),
        "p75_length":       float(np.percentile(lengths_arr, 75)),
        "empty_rate":       empty_count / max(len(pred_explanations), 1),
        # Coverage
        "n_scored":         len(valid_preds),
        "n_total":          len(pred_explanations),
    }


# ---------------------------------------------------------------------------
# GNN evaluation  (classification only — no text generation)
# ---------------------------------------------------------------------------

def eval_gnn(threshold_override=None, auto_save_csv=False) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Test input: {INPUT_TEST_JSON}")
    examples = load_from_input_json(INPUT_TEST_JSON, "test")

    if not os.path.exists(STAGE1_CKPT):
        print(f"[eval] Checkpoint not found at {STAGE1_CKPT}. Run stage1_gnn_train.py first.")
        return

    model, ckpt_thresholds = load_stage1_checkpoint(STAGE1_CKPT, device)
    use_thresholds = (
        [float(threshold_override)] * len(MCP_LABELS)
        if threshold_override is not None
        else ckpt_thresholds
    )

    graphs, field_embs_list, step_gold, mcp_gold = [], [], [], []
    for ex in examples:
        # Graph is already a torch_geometric Data object from load_from_input_json
        graphs.append(ex["graph"])
        field_embs_list.append(
            _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS])
        )
        step_gold.append(ex["step_idx"])
        mcp_gold.append(ex["mcp_vec"])

    step_preds, mcp_preds = [], []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            from torch_geometric.data import Batch as PyGBatch
            batch_graphs = PyGBatch.from_data_list(graphs[i : i + bs]).to(device)
            batch_fe = torch.tensor(
                np.stack(field_embs_list[i : i + bs]), dtype=torch.float32
            ).to(device)
            step_logits, mcp_logits, _ = model(
                batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch, batch_fe
            )
            step_preds.append(step_logits.argmax(-1).cpu().numpy())
            probs = torch.sigmoid(mcp_logits).cpu().numpy()
            mcp_preds.append(predict_with_per_class_thresholds(probs, use_thresholds))

    step_preds = np.concatenate(step_preds)
    mcp_preds  = np.concatenate(mcp_preds)
    step_gold  = np.array(step_gold)
    mcp_gold   = np.stack(mcp_gold)

    report_classification(step_preds, step_gold, mcp_preds, mcp_gold)

    # GNN has no text generation — note this explicitly
    print("\n  [Explanation quality: N/A — GNN is a classifier, not a text generator]")

    # Print thresholds used
    print("\n[eval] MCP thresholds used:")
    for label, thr in zip(MCP_LABELS, use_thresholds):
        marker = "  <-- non-default" if abs(thr - 0.5) > 0.05 else ""
        print(f"  {label:<22}  {thr:.2f}{marker}")

    # ── CSV dump for GNN predictions ────────────────────────────────────────
    if auto_save_csv:
        import csv
        
        # Create output directory
        output_dir = os.path.join(ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "stage1_gnn_predictions.csv")
        
        # Build CSV rows (without explanation fields)
        csv_rows = []
        for i, ex in enumerate(examples):
            pred_step_label = STEP_LABELS[step_preds[i]] if step_preds[i] >= 0 else "UNPARSEABLE"
            gold_step_label = STEP_LABELS[ex["step_idx"]]
            
            # Convert MCP vectors to label lists
            pred_mcp_labels = [MCP_LABELS[j] for j, val in enumerate(mcp_preds[i]) if val == 1]
            gold_mcp_labels = [MCP_LABELS[j] for j, val in enumerate(mcp_gold[i]) if val == 1]
            
            csv_row = {
                "machine": ex["machine"],
                "new_strategy": ex["context"].get("New strategy", ""),
                "strategy_explanation": ex["context"].get("Strategy explanation", ""),
                "gold_new_step": gold_step_label,
                "predicted_new_step": pred_step_label,
                "gold_mcp_tasks": "|".join(gold_mcp_labels),
                "predicted_mcp_tasks": "|".join(pred_mcp_labels),
                "step_correct": int(step_preds[i] == ex["step_idx"]),
            }
            csv_rows.append(csv_row)
        
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[eval] GNN prediction CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# LLM evaluation  (classification + explanation quality)
# ---------------------------------------------------------------------------

def eval_llm(adapter_dir: str, threshold_override=None,
             max_new_tokens: int = 200,
             save_explanations: str | None = None) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from torch_geometric.data import Batch as PyGBatch
    from stage2_sft_qwen import (
        build_prompt, SYSTEM_PROMPT, GraphPrefixAdapter,
    )
    from graph_encoder import Stage1Classifier

    # MCP thresholds — not used for LLM (tools come from parsed JSON text),
    # but loaded for reporting consistency.
    if threshold_override is not None:
        use_thresholds = [float(threshold_override)] * len(MCP_LABELS)
    elif os.path.exists(STAGE1_CKPT):
        _, use_thresholds = load_stage1_checkpoint(STAGE1_CKPT, "cpu")
        print("[eval] MCP thresholds loaded from Stage-1 checkpoint (for reference).")
    else:
        use_thresholds = [MCP_DECISION_THRESHOLD] * len(MCP_LABELS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16
    print(f"[eval] Test input: {INPUT_TEST_JSON}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype
    ).to(device)
    llm_model = PeftModel.from_pretrained(base, adapter_dir).eval()

    # ── Graph prefix adapter ──────────────────────────────────────────────
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
    else:
        stage1.load_state_dict(ckpt)
    graph_encoder = stage1.graph_encoder.to(device).eval()
    for p in graph_encoder.parameters():
        p.requires_grad_(False)

    from config import GNN_OUT_DIM, GRAPH_PREFIX_TOKENS
    llm_hidden = llm_model.config.hidden_size
    adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(device).to(dtype)
    adapter_ckpt = os.path.join(adapter_dir, "graph_adapter.pt")
    if os.path.exists(adapter_ckpt):
        adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
        print(f"[eval] Loaded GraphPrefixAdapter from {adapter_ckpt}")
    else:
        print(f"[eval] WARNING: graph_adapter.pt not found in {adapter_dir} — "
              f"using randomly initialised adapter (results will be worse)")
    adapter.eval()

    embed_layer = llm_model.get_input_embeddings()

    examples = load_from_input_json(INPUT_TEST_JSON, "test")
    normalizer = StepLabelNormalizer()

    step_preds, mcp_preds, step_gold, mcp_gold       = [], [], [], []
    pred_explanations, gold_explanations               = [], []
    prev_step_results: list[str]                       = []
    parse_failures                                     = 0
    # rows saved to CSV if --save-explanations is set
    csv_rows: list[dict]                               = []

    for ex in tqdm(examples, desc="Generating", unit="sample"):
        prompt = build_prompt(ex)
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )

        with torch.no_grad():
            pyg_batch = PyGBatch.from_data_list([ex["graph"]]).to(device)
            graph_emb = graph_encoder(
                pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch
            )
            prefix_embeds = adapter(graph_emb.to(dtype))
            ids = tokenizer(
                full_prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=900,
            ).input_ids.to(device)
            token_embeds  = embed_layer(ids).to(dtype)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            out = llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # ── Parse JSON ────────────────────────────────────────────────────
        obj = {}
        try:
            # Try to find and parse JSON object
            start = gen_text.index("{")
            end   = gen_text.rindex("}") + 1
            obj   = json.loads(gen_text[start:end])
        except Exception:
            parse_failures += 1
            # Enhanced regex fallback with multiple patterns
            # Pattern 1: "New step": "value"
            m = re.search(r'"?New step"?\s*:\s*"([^"]+)"', gen_text, re.IGNORECASE)
            if m:
                obj["New step"] = m.group(1)
            # Pattern 2: "new_step": "value" (underscore variant)
            m = re.search(r'"?new_step"?\s*:\s*"([^"]+)"', gen_text, re.IGNORECASE)
            if m and "New step" not in obj:
                obj["New step"] = m.group(1)
            # Pattern 3: "step": "value"
            m = re.search(r'"?step"?\s*:\s*"([^"]+)"', gen_text, re.IGNORECASE)
            if m and "New step" not in obj:
                obj["New step"] = m.group(1)
            
            # Step explanation patterns
            m2 = re.search(r'"?Step explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL | re.IGNORECASE)
            if m2:
                obj["Step explanation"] = m2.group(1)
            # Pattern 2: "step_explanation": "value"
            m2 = re.search(r'"?step_explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL | re.IGNORECASE)
            if m2 and "Step explanation" not in obj:
                obj["Step explanation"] = m2.group(1)
            # Pattern 3: "explanation": "value"
            m2 = re.search(r'"?explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL | re.IGNORECASE)
            if m2 and "Step explanation" not in obj:
                obj["Step explanation"] = m2.group(1)
            
            # MCP tasks patterns - try to extract tool names
            mcp_pattern = re.search(r'"?MCP[_ ]tasks"?\s*:\s*\[([^\]]*)\]', gen_text, re.DOTALL | re.IGNORECASE)
            if mcp_pattern:
                # Extract quoted tool names from the array
                tools = re.findall(r'"([^"]+)"', mcp_pattern.group(1))
                if tools:
                    obj["MCP_tasks"] = {tool: True for tool in tools}
            # Alternative MCP pattern: object format
            mcp_obj_pattern = re.search(r'"?MCP[_ ]tasks"?\s*:\s*\{([^}]+)\}', gen_text, re.DOTALL | re.IGNORECASE)
            if mcp_obj_pattern and "MCP_tasks" not in obj:
                # Extract keys from object format
                tools = re.findall(r'"([^"]+)"\s*:', mcp_obj_pattern.group(1))
                if tools:
                    obj["MCP_tasks"] = {tool: True for tool in tools}

        # ── Step classification ───────────────────────────────────────────
        pred_step_raw  = obj.get("New step", "")
        pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
        s_idx = (
            STEP_LABELS.index(pred_step_norm)
            if pred_step_norm in STEP_LABELS
            else -1
        )
        step_preds.append(s_idx)
        step_gold.append(ex["step_idx"])

        # ── MCP classification ────────────────────────────────────────────
        pred_mcp_keys   = (
            list(obj.get("MCP_tasks", {}).keys())
            if isinstance(obj.get("MCP_tasks"), dict)
            else []
        )
        pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
        mcp_preds.append(mcp_multihot(pred_mcp_labels))
        mcp_gold.append(ex["mcp_vec"])

        # ── Explanation ───────────────────────────────────────────────────
        pred_expl = str(obj.get("Step explanation", "")).strip()
        gold_expl = ex.get("gold_step_explanation", "")
        if not isinstance(gold_expl, str):
            gold_expl = ""
        pred_explanations.append(pred_expl)
        gold_explanations.append(gold_expl)

        # Keep the previous step result for context-grounding metric
        prev_step_results.append(
            ex["context"].get("Previous step result", "") or ""
        )

        # Accumulate CSV row
        if save_explanations:
            csv_rows.append({
                "machine":          ex["machine"],
                "gold_step":        STEP_LABELS[ex["step_idx"]],
                "pred_step":        pred_step_norm or "UNPARSEABLE",
                "step_correct":     int(s_idx == ex["step_idx"]),
                "gold_mcp":         "|".join(ex["mcp_labels"]),
                "pred_mcp":         "|".join(pred_mcp_labels),
                "gold_explanation": gold_expl,
                "pred_explanation": pred_expl,
            })

    if parse_failures:
        print(
            f"\n[eval] Note: {parse_failures}/{len(examples)} responses had no "
            f"parseable JSON — regex fallback applied where possible."
        )

    step_preds_arr = np.array(step_preds)
    step_gold_arr  = np.array(step_gold)
    mcp_preds_arr  = np.stack(mcp_preds)
    mcp_gold_arr   = np.stack(mcp_gold)

    # ── Classification report ─────────────────────────────────────────────
    report_classification(step_preds_arr, step_gold_arr, mcp_preds_arr, mcp_gold_arr)

    # ── Explanation quality report ────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("STEP EXPLANATION QUALITY")
    print("=" * 60)
    print("Computing explanation metrics ...")
    expl_metrics = compute_explanation_metrics(
        pred_explanations,
        gold_explanations,
        step_preds=step_preds_arr,
        prev_step_results=prev_step_results,
    )
    report_explanation(expl_metrics)

    # ── LLM judge evaluation (if requested) ───────────────────────────────
    if args.use_llm_judge:
        print("\n\n" + "=" * 60)
        print("LLM JUDGE EVALUATION")
        print("=" * 60)
        print("Using LLM to evaluate explanation quality...")
        
        # Prepare examples for LLM judge
        llm_examples = []
        for i, ex in enumerate(examples):
            if i < len(pred_explanations):
                llm_examples.append({
                    "pred_explanation": pred_explanations[i],
                    "gold_explanation": gold_explanations[i],
                    "pred_step": STEP_LABELS[step_preds[i]] if step_preds[i] >= 0 else "UNPARSEABLE",
                    "gold_step": STEP_LABELS[ex["step_idx"]],
                    "context": ex["context"],
                    "machine": ex["machine"],
                })
        
        # Run LLM judge evaluation
        llm_results = batch_evaluate_explanations(
            examples=llm_examples,
            model=args.llm_judge_model,
            max_samples=args.llm_judge_samples,
            verbose=True,
        )
        
        # Print LLM judge results
        print_llm_judge_results(llm_results)

    # ── Optional CSV dump ─────────────────────────────────────────────────
    if (save_explanations or args.auto_save_csv) and csv_rows:
        import csv
        
        # Create output directory
        output_dir = os.path.join(ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        if save_explanations:
            csv_path = save_explanations
        else:
            # Auto-generate path based on adapter directory name
            if adapter_dir:
                stage_name = os.path.basename(adapter_dir)
                csv_path = os.path.join(output_dir, f"{stage_name}_predictions.csv")
            else:
                csv_path = os.path.join(output_dir, "llm_predictions.csv")
        
        # Enhance CSV rows with all requested fields
        enhanced_rows = []
        for i, row in enumerate(csv_rows):
            enhanced_row = {
                "machine": row["machine"],
                "new_strategy": examples[i]["context"].get("New strategy", ""),
                "strategy_explanation": examples[i]["context"].get("Strategy explanation", ""),
                "gold_new_step": row["gold_step"],
                "predicted_new_step": row["pred_step"],
                "gold_step_explanation": row["gold_explanation"],
                "predicted_step_explanation": row["pred_explanation"],
                "gold_mcp_tasks": row["gold_mcp"],
                "predicted_mcp_tasks": row["pred_mcp"],
                "step_correct": row["step_correct"],
            }
            enhanced_rows.append(enhanced_row)
        
        fieldnames = list(enhanced_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced_rows)
        print(f"\n[eval] Prediction CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def report_classification(
    step_preds: np.ndarray,
    step_gold: np.ndarray,
    mcp_preds: np.ndarray,
    mcp_gold: np.ndarray,
) -> None:
    print("\n" + "=" * 60)
    print("STEP CLASSIFICATION")
    print("=" * 60)
    print(f"  Accuracy      : {accuracy_score(step_gold, step_preds):.4f}")
    print(f"  Macro F1      : {f1_score(step_gold, step_preds, average='macro',    zero_division=0):.4f}")
    print(f"  Weighted F1   : {f1_score(step_gold, step_preds, average='weighted', zero_division=0):.4f}")

    labels_present = sorted(
        set(step_gold.tolist()) | set(int(p) for p in step_preds if p >= 0)
    )
    print("\n  Per-class report:")
    print(
        classification_report(
            step_gold, step_preds,
            labels=labels_present,
            target_names=[
                STEP_LABELS[i] if 0 <= i < len(STEP_LABELS) else "UNPARSEABLE"
                for i in labels_present
            ],
            zero_division=0,
        )
    )
    print("  Confusion matrix (rows=gold, cols=pred):")
    cm = confusion_matrix(step_gold, step_preds, labels=list(range(len(STEP_LABELS))))
    print(cm)

    print("\n" + "=" * 60)
    print("MCP TOOL CLASSIFICATION  (multi-label)")
    print("=" * 60)
    print(f"  Subset (exact-match) accuracy : {accuracy_score(mcp_gold, mcp_preds):.4f}")
    print(f"  Micro F1                      : {f1_score(mcp_gold, mcp_preds, average='micro',   zero_division=0):.4f}")
    print(f"  Macro F1                      : {f1_score(mcp_gold, mcp_preds, average='macro',   zero_division=0):.4f}")
    print(f"  Samples F1                    : {f1_score(mcp_gold, mcp_preds, average='samples', zero_division=0):.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(
        mcp_gold, mcp_preds, average=None, zero_division=0
    )
    print("\n  Per-label metrics:")
    print(f"  {'Label':<22}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Sup':>5}")
    print("  " + "-" * 52)
    for i, label in enumerate(MCP_LABELS):
        flag = "  ← low recall" if rec[i] < 0.3 and support[i] > 0 else ""
        print(
            f"  {label:<22}  {prec[i]:>6.3f}  {rec[i]:>6.3f}  "
            f"{f1[i]:>6.3f}  {int(support[i]):>5}{flag}"
        )


def report_explanation(metrics: dict) -> None:
    n_scored = metrics["n_scored"]
    n_total  = metrics["n_total"]

    if n_scored < n_total:
        print(
            f"  (Reference-based metrics scored on {n_scored}/{n_total} pairs — "
            f"{n_total - n_scored} skipped because gold explanation was empty)"
        )

    # ── Reference-based ──────────────────────────────────────────────────────
    print(f"\n  {'Metric':<26}  {'Score':>8}  Interpretation")
    print("  " + "-" * 62)

    def _bar(score: float, thresholds: tuple, icons: tuple = ("✗", "~", "✓")) -> str:
        if score >= thresholds[1]:
            return icons[2]
        if score >= thresholds[0]:
            return icons[1]
        return icons[0]

    b1 = metrics["bleu1"]
    b2 = metrics["bleu2"]
    b4 = metrics["bleu4"]
    rl = metrics["rouge_l"]
    bs = metrics["bertscore_f1"]
    sa = metrics["step_alignment"]
    cg = metrics["ctx_grounding"]
    rd = metrics["reasoning_density"]

    print(f"  {'BLEU-1':<26}  {b1:>8.4f}  {_bar(b1,(0.20,0.35))}  n-gram overlap with gold")
    print(f"  {'BLEU-2':<26}  {b2:>8.4f}  {_bar(b2,(0.10,0.20))}")
    print(f"  {'BLEU-4':<26}  {b4:>8.4f}  {_bar(b4,(0.03,0.10))}")
    print(f"  {'ROUGE-L':<26}  {rl:>8.4f}  {_bar(rl,(0.20,0.35))}  longest common subsequence")
    print(f"  {'BERTScore-F1':<26}  {bs:>8.4f}  {_bar(bs,(0.60,0.75))}  semantic similarity to gold")

    # ── Intrinsic ─────────────────────────────────────────────────────────────
    print()
    print(f"  {'Step Alignment':<26}  {sa:>8.4f}  {_bar(sa,(0.20,0.35))}  "
          f"explanation mentions step keywords")
    print(f"  {'Context Grounding':<26}  {cg:>8.4f}  {_bar(cg,(0.05,0.15))}  "
          f"references previous step findings")
    print(f"  {'Reasoning Density':<26}  {rd:>8.4f}  {_bar(rd,(0.03,0.07))}  "
          f"causal/justification language")

    # ── Length / completeness ─────────────────────────────────────────────────
    avg_l = metrics["avg_length"]
    p25_l = metrics["p25_length"]
    p75_l = metrics["p75_length"]
    empty = metrics["empty_rate"]
    print()
    print(f"  {'Avg length (tokens)':<26}  {avg_l:>8.1f}")
    print(f"  {'P25 length':<26}  {p25_l:>8.1f}")
    print(f"  {'P75 length':<26}  {p75_l:>8.1f}")
    print(f"  {'Empty rate':<26}  {empty:>8.1%}  "
          f"{'⚠ high — JSON parsing likely failing' if empty > 0.1 else 'OK'}")

    # ── Summary diagnosis ─────────────────────────────────────────────────────
    print()
    print("  Diagnosis:")
    issues = []
    if b1 < 0.20:
        issues.append("  • BLEU-1 < 0.20 — low lexical overlap; model paraphrases heavily "
                      "or generates off-topic text")
    if bs < 0.60:
        issues.append("  • BERTScore < 0.60 — explanations semantically distant from gold; "
                      "consider more SFT epochs")
    if sa < 0.20:
        issues.append("  • Step Alignment < 0.20 — explanation doesn't mention the predicted "
                      "step keywords; model may be ignoring its own step choice")
    if cg < 0.05:
        issues.append("  • Context Grounding < 0.05 — explanation barely references observed "
                      "findings; model is not anchoring to evidence")
    if rd < 0.03:
        issues.append("  • Reasoning Density < 0.03 — very few causal connectors; "
                      "explanations read as assertions, not justifications")
    if empty > 0.1:
        issues.append(f"  • Empty rate {empty:.0%} — many blank explanations; "
                      "check JSON format compliance (parse_failures above)")
    if avg_l < 20:
        issues.append(f"  • Avg length {avg_l:.0f} tokens is very short; "
                      "model may be truncating — increase max_new_tokens")

    if not issues:
        print("  ✓ All metrics within acceptable ranges")
    else:
        for issue in issues:
            print(issue)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def check_model_availability() -> list[tuple[str, str | None]]:
    available = []
    if os.path.exists(STAGE1_CKPT):
        available.append(("gnn", None))
    ckpt_dir = os.path.dirname(STAGE1_CKPT)
    for subdir, label in [
        ("stage2_qwen_lora", "Stage 2 SFT"),
        ("stage3_qwen_grpo", "Stage 3 GRPO"),
    ]:
        d = os.path.join(ckpt_dir, subdir)
        if os.path.exists(d):
            available.append(("llm", d))
    return available


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate stepmodelv2 — step classification, MCP tools, explanation quality"
    )
    parser.add_argument(
        "--model", choices=["gnn", "llm", "all"], default="all",
        help="Which model(s) to evaluate (default: all available)",
    )
    parser.add_argument(
        "--adapter-dir", default=None,
        help="Specific LLM adapter directory. Ignored when --model=all.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override MCP threshold (single float). "
             "Omit to use per-class thresholds from Stage-1 checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=500,
        help="Max tokens to generate per sample in LLM mode (default: 500).",
    )
    parser.add_argument(
        "--save-explanations", default=None, metavar="PATH",
        help="Save per-sample explanation predictions to a CSV file at PATH.",
    )
    parser.add_argument(
        "--auto-save-csv", action="store_true", default=None,
        help="Automatically save prediction CSVs for LLM models (stage 2/3). "
             "Automatically enabled for LLM models. Use --no-auto-save-csv to disable.",
    )
    parser.add_argument(
        "--no-auto-save-csv", dest="auto_save_csv", action="store_false",
        help="Disable automatic CSV saving for LLM models.",
    )
    parser.add_argument(
        "--use-llm-judge", action="store_true", default=None,
        help="Use LLM judge to evaluate explanation quality (requires OPENAI_API_KEY). "
             "Automatically enabled for LLM models (stage 2/3). Use --no-llm-judge to disable.",
    )
    parser.add_argument(
        "--no-llm-judge", dest="use_llm_judge", action="store_false",
        help="Disable automatic LLM judge evaluation for LLM models.",
    )
    parser.add_argument(
        "--llm-judge-model", default="gpt-4o",
        help="OpenAI model to use for LLM judge (default: gpt-4o).",
    )
    parser.add_argument(
        "--llm-judge-samples", type=int, default=None,
        help="Maximum number of samples to evaluate with LLM judge (for testing).",
    )
    args = parser.parse_args()

    # Auto-enable LLM judge and CSV saving for LLM models unless explicitly disabled
    if args.use_llm_judge is None and args.model in ["llm", "all"]:
        args.use_llm_judge = True
    if args.auto_save_csv is None and args.model in ["llm", "all"]:
        args.auto_save_csv = True

    if args.model == "all":
        available = check_model_availability()
        if not available:
            print("[eval] No trained models found. Run the pipeline first.")
            sys.exit(1)

        print(f"[eval] Found {len(available)} model(s) to evaluate:")
        for mtype, adir in available:
            label = "Stage 1 GNN" if mtype == "gnn" else adir
            print(f"  • {label}")
        print()

        for mtype, adir in available:
            header = "Stage 1 GNN" if mtype == "gnn" else adir
            print(f"\n{'═' * 60}")
            print(f"  MODEL: {header}")
            print(f"{'═' * 60}")
            if mtype == "gnn":
                eval_gnn(threshold_override=args.threshold, auto_save_csv=args.auto_save_csv)
            else:
                eval_llm(
                    adir,
                    threshold_override=args.threshold,
                    max_new_tokens=args.max_new_tokens,
                    save_explanations=args.save_explanations,
                )

    elif args.model == "gnn":
        if not os.path.exists(STAGE1_CKPT):
            print(f"[eval] Stage-1 checkpoint not found: {STAGE1_CKPT}")
            sys.exit(1)
        print(f"\n{'═' * 60}\n  MODEL: Stage 1 GNN\n{'═' * 60}")
        eval_gnn(threshold_override=args.threshold, auto_save_csv=args.auto_save_csv)

    else:  # llm
        adapter = args.adapter_dir
        if adapter is None:
            ckpt_dir = os.path.dirname(STAGE1_CKPT)
            for subdir in ["stage3_qwen_grpo", "stage2_qwen_lora"]:
                d = os.path.join(ckpt_dir, subdir)
                if os.path.exists(d):
                    adapter = d
                    break
        if adapter is None or not os.path.exists(adapter):
            print("[eval] No LLM adapter found. Train Stage 2/3 first, or pass --adapter-dir.")
            sys.exit(1)
        print(f"\n{'═' * 60}\n  MODEL: {adapter}\n{'═' * 60}")
        eval_llm(
            adapter,
            threshold_override=args.threshold,
            max_new_tokens=args.max_new_tokens,
            save_explanations=args.save_explanations,
        )
