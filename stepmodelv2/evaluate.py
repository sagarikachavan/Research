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
    BERTScore F1        — semantic similarity via sentence embeddings
                          (uses the same frozen encoder already loaded,
                           no extra model download needed)
    Avg token length    — sanity check that the model isn't truncating

Usage:
    python evaluate.py                         # evaluate all available models
    python evaluate.py --model gnn             # GNN only
    python evaluate.py --model llm             # best available LLM adapter
    python evaluate.py --model llm \\
        --adapter-dir checkpoints/stage3_qwen_grpo
    python evaluate.py --threshold 0.5         # override MCP threshold
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
    TEST_CSV, STAGE1_CKPT, STEP_LABELS, MCP_LABELS, MCP_DECISION_THRESHOLD,
    QWEN_MODEL_NAME,
)
from data_utils import (
    load_and_clean, load_graph, _embed_texts, CONTEXT_COLUMNS,
    mcp_multihot, StepLabelNormalizer, extract_mcp_labels,
)
from graph_encoder import Stage1Classifier
from mcp_threshold_search import predict_with_per_class_thresholds


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
    """ROUGE-L recall via LCS dynamic programming."""
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
    recall = lcs / m
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
    Compute BERTScore-style F1 using the already-loaded sentence encoder
    (BAAI/bge-small-en-v1.5).  This avoids downloading a separate BERTScore
    model and is fast because it reuses the encoder already in memory.

    Score = cosine similarity between the sentence embeddings of hypothesis
    and reference.  This is equivalent to BERTScore with sentence-level
    pooling (not token-level), which is faster and sufficient for ranking.
    """
    scores = []
    for i in range(0, len(hypotheses), batch_size):
        hyp_batch = hypotheses[i : i + batch_size]
        ref_batch = references[i : i + batch_size]
        # Filter empty strings to avoid NaN embeddings
        hyp_safe = [h if h.strip() else "empty" for h in hyp_batch]
        ref_safe = [r if r.strip() else "empty" for r in ref_batch]
        hyp_emb = _embed_texts(hyp_safe)   # (B, D) normalised
        ref_emb = _embed_texts(ref_safe)   # (B, D) normalised
        cos_sims = (hyp_emb * ref_emb).sum(axis=1).tolist()  # dot of unit vecs = cosine
        scores.extend(cos_sims)
    return scores


def compute_explanation_metrics(
    pred_explanations: list[str],
    gold_explanations: list[str],
) -> dict:
    """
    Aggregate BLEU-1/2/4, ROUGE-L, BERTScore-F1, and avg length over all
    (prediction, reference) pairs.  Pairs where gold is empty are skipped.
    """
    bleu1_scores, bleu2_scores, bleu4_scores, rougeL_scores = [], [], [], []
    valid_preds, valid_refs = [], []
    lengths = []

    for pred, gold in zip(pred_explanations, gold_explanations):
        lengths.append(len(pred.split()))
        if not gold.strip():
            continue
        hyp_tok = _tokenize(pred)
        ref_tok = _tokenize(gold)
        bleu1_scores.append(bleu_n(hyp_tok, ref_tok, 1))
        bleu2_scores.append(bleu_n(hyp_tok, ref_tok, 2))
        bleu4_scores.append(bleu_n(hyp_tok, ref_tok, 4))
        rougeL_scores.append(rouge_l(hyp_tok, ref_tok))
        valid_preds.append(pred)
        valid_refs.append(gold)

    bert_scores = (
        bertscore_f1_batch(valid_preds, valid_refs) if valid_preds else []
    )

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        "bleu1":        avg(bleu1_scores),
        "bleu2":        avg(bleu2_scores),
        "bleu4":        avg(bleu4_scores),
        "rouge_l":      avg(rougeL_scores),
        "bertscore_f1": avg(bert_scores),
        "avg_length":   avg(lengths),
        "n_scored":     len(valid_preds),
        "n_total":      len(pred_explanations),
    }


# ---------------------------------------------------------------------------
# GNN evaluation  (classification only — no text generation)
# ---------------------------------------------------------------------------

def eval_gnn(threshold_override=None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = load_and_clean(TEST_CSV, "test")

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
        graphs.append(load_graph(ex["machine"], ex["row_id"], ex["ptt"], "test"))
        field_embs_list.append(
            _embed_texts([ex["context"][c] or "empty" for c in CONTEXT_COLUMNS])
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


# ---------------------------------------------------------------------------
# LLM evaluation  (classification + explanation quality)
# ---------------------------------------------------------------------------

def eval_llm(adapter_dir: str, threshold_override=None, max_new_tokens: int = 200) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from stage2_sft_qwen import build_prompt, SYSTEM_PROMPT

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
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)
    llm_model = PeftModel.from_pretrained(base, adapter_dir).eval()

    examples = load_and_clean(TEST_CSV, "test")
    normalizer = StepLabelNormalizer()

    step_preds, mcp_preds, step_gold, mcp_gold = [], [], [], []
    pred_explanations, gold_explanations = [], []
    parse_failures = 0

    for ex in tqdm(examples, desc="Generating", unit="sample"):
        prompt = build_prompt(ex)
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )
        ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            out = llm_model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # ── Parse JSON ────────────────────────────────────────────────────
        obj = {}
        try:
            start = gen_text.index("{")
            end   = gen_text.rindex("}") + 1
            obj   = json.loads(gen_text[start:end])
        except Exception:
            parse_failures += 1
            # Fallback: try to extract step label with a regex even without JSON
            m = re.search(r'"?New step"?\s*:\s*"([^"]+)"', gen_text)
            if m:
                obj["New step"] = m.group(1)
            m2 = re.search(r'"?Step explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL)
            if m2:
                obj["Step explanation"] = m2.group(1)

        # ── Step classification ───────────────────────────────────────────
        pred_step_raw  = obj.get("New step", "")
        pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
        step_idx = (
            STEP_LABELS.index(pred_step_norm)
            if pred_step_norm in STEP_LABELS
            else -1
        )
        step_preds.append(step_idx)
        step_gold.append(ex["step_idx"])

        # ── MCP classification ────────────────────────────────────────────
        pred_mcp_keys  = (
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
        pred_explanations.append(pred_expl)
        gold_explanations.append(gold_expl if isinstance(gold_expl, str) else "")

    if parse_failures:
        print(
            f"\n[eval] Note: {parse_failures}/{len(examples)} responses had no "
            f"parseable JSON — regex fallback applied where possible."
        )

    step_preds = np.array(step_preds)
    step_gold  = np.array(step_gold)
    mcp_preds  = np.stack(mcp_preds)
    mcp_gold   = np.stack(mcp_gold)

    # ── Classification report ─────────────────────────────────────────────
    report_classification(step_preds, step_gold, mcp_preds, mcp_gold)

    # ── Explanation quality report ────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("STEP EXPLANATION QUALITY")
    print("=" * 60)
    print("Computing explanation metrics (BLEU / ROUGE-L / BERTScore) ...")
    expl_metrics = compute_explanation_metrics(pred_explanations, gold_explanations)
    report_explanation(expl_metrics)


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
    coverage = f"{n_scored}/{n_total}"
    if n_scored < n_total:
        print(
            f"  (Scored {coverage} pairs — {n_total - n_scored} skipped "
            f"because gold explanation was empty)"
        )

    print(f"\n  {'Metric':<20}  {'Score':>8}")
    print("  " + "-" * 32)
    print(f"  {'BLEU-1':<20}  {metrics['bleu1']:>8.4f}")
    print(f"  {'BLEU-2':<20}  {metrics['bleu2']:>8.4f}")
    print(f"  {'BLEU-4':<20}  {metrics['bleu4']:>8.4f}")
    print(f"  {'ROUGE-L':<20}  {metrics['rouge_l']:>8.4f}")
    print(f"  {'BERTScore-F1':<20}  {metrics['bertscore_f1']:>8.4f}")
    print(f"  {'Avg pred length':<20}  {metrics['avg_length']:>8.1f}  tokens")
    print()

    # Interpretation guide
    print("  Interpretation:")
    b1 = metrics["bleu1"]
    rl = metrics["rouge_l"]
    bs = metrics["bertscore_f1"]
    if b1 >= 0.35:
        print("  ✓ BLEU-1 ≥ 0.35 — good word-level overlap with gold explanations")
    elif b1 >= 0.20:
        print("  ~ BLEU-1 0.20–0.35 — moderate overlap; explanations are paraphrased")
    else:
        print("  ✗ BLEU-1 < 0.20 — low lexical overlap; explanations diverge from gold")

    if bs >= 0.75:
        print("  ✓ BERTScore ≥ 0.75 — explanations are semantically close to gold")
    elif bs >= 0.60:
        print("  ~ BERTScore 0.60–0.75 — moderate semantic similarity")
    else:
        print("  ✗ BERTScore < 0.60 — explanations are semantically distant from gold")


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
        "--max-new-tokens", type=int, default=200,
        help="Max tokens to generate per sample in LLM mode (default: 200).",
    )
    args = parser.parse_args()

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
                eval_gnn(threshold_override=args.threshold)
            else:
                eval_llm(
                    adir,
                    threshold_override=args.threshold,
                    max_new_tokens=args.max_new_tokens,
                )

    elif args.model == "gnn":
        if not os.path.exists(STAGE1_CKPT):
            print(f"[eval] Stage-1 checkpoint not found: {STAGE1_CKPT}")
            sys.exit(1)
        print(f"\n{'═' * 60}\n  MODEL: Stage 1 GNN\n{'═' * 60}")
        eval_gnn(threshold_override=args.threshold)

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
        )
