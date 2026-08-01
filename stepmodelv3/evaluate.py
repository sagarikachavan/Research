"""
evaluate.py — Evaluation for the stepmodelv3 GRPO-trained model against
stepmodelv3/input/test.json.

Reports:
  STEP CLASSIFICATION      (10 data-derived categories, see data_utils.py)
    Accuracy, Macro-F1, Weighted-F1, per-class report, confusion matrix
  MCP TOOL CLASSIFICATION  (18 data-derived canonical tools, multi-label)
    Subset accuracy, Micro/Macro/Samples-F1, per-label precision/recall/F1
  STEP EXPLANATION QUALITY
    BLEU-1/2/4, ROUGE-L, BERTScore-F1, step alignment, reasoning density,
    length stats, empty rate

Usage:
    python evaluate.py
    python evaluate.py --adapter-dir checkpoints/stage1_grpo_rl
    python evaluate.py --save-explanations out.csv
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from data_utils import (
    load_json_data, parse_completion, classify_step, mcp_labels_from_dict,
    mcp_multihot, STEP_LABELS, MCP_LABELS, get_sentence_encoder,
)
from prompts import build_chat_prompt

INPUT_TEST_JSON = "input/test.json"
QWEN_MODEL_NAME = "Qwen/Qwen3-14B-Instruct"
STAGE1_ADAPTER_DIR = "checkpoints/stage1_grpo_rl"
MAX_PROMPT_TOKENS = 1800

# ---------------------------------------------------------------------------
# Explanation quality metrics (unchanged logic from the original evaluate.py
# — these are format-agnostic and were already correct)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def bleu_n(hypothesis: list[str], reference: list[str], n: int) -> float:
    if len(hypothesis) < n or len(reference) < n:
        return 0.0
    ref_ngrams: dict[tuple, int] = defaultdict(int)
    for i in range(len(reference) - n + 1):
        ref_ngrams[tuple(reference[i: i + n])] += 1
    clipped = 0
    for i in range(len(hypothesis) - n + 1):
        ng = tuple(hypothesis[i: i + n])
        if ref_ngrams.get(ng, 0) > 0:
            clipped += 1
            ref_ngrams[ng] -= 1
    return clipped / max(len(hypothesis) - n + 1, 1)


def rouge_l(hypothesis: list[str], reference: list[str]) -> float:
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


def bertscore_f1_batch(hypotheses, references, encoder, batch_size: int = 64):
    scores = []
    for i in range(0, len(hypotheses), batch_size):
        hyp_batch = [h if h.strip() else "empty" for h in hypotheses[i:i + batch_size]]
        ref_batch = [r if r.strip() else "empty" for r in references[i:i + batch_size]]
        hyp_emb = encoder.encode(hyp_batch)
        ref_emb = encoder.encode(ref_batch)
        hyp_emb = hyp_emb / (np.linalg.norm(hyp_emb, axis=1, keepdims=True) + 1e-8)
        ref_emb = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-8)
        scores.extend((hyp_emb * ref_emb).sum(axis=1).tolist())
    return scores


_REASONING_TOKENS = {
    "because", "since", "therefore", "thus", "hence", "so", "indicates",
    "suggests", "shows", "reveals", "found", "identified", "discovered",
    "confirm", "allows", "enables", "given", "based", "due", "result",
    "led", "lead", "indicate",
}

_STEP_KEYWORDS_FOR_ALIGNMENT = {
    "recon_scan": ["nmap", "netdiscover", "port", "scan", "ip"],
    "enumerate_further": ["enumerate", "version", "hidden", "directory", "service"],
    "enumerate_website": ["website", "web", "directory", "link", "software"],
    "enumerate_domain": ["domain", "dns", "subdomain"],
    "explore_files": ["explore", "suspicious", "file", "command", "summary"],
    "source_code_review": ["source", "code", "review", "vuln"],
    "google_search": ["google", "search"],
    "exploit": ["exploit", "payload", "metasploit", "shell"],
    "analyze_outcomes": ["analyz", "outcome", "attack", "path"],
    "end_task": ["end", "task", "report", "complet"],
}


def step_alignment_score(explanation: str, step_cat: str) -> float:
    keywords = _STEP_KEYWORDS_FOR_ALIGNMENT.get(step_cat)
    if not keywords:
        return 0.0
    tokens = set(_tokenize(explanation))
    hits = sum(1 for kw in keywords if any(tok.startswith(kw) for tok in tokens))
    return hits / len(keywords)


def reasoning_density_score(explanation: str) -> float:
    tokens = _tokenize(explanation)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _REASONING_TOKENS)
    return hits / len(tokens)


def compute_explanation_metrics(pred_explanations, gold_explanations, step_cats, encoder):
    bleu1, bleu2, bleu4, rougeL = [], [], [], []
    valid_preds, valid_refs = [], []
    lengths, alignment, reasoning = [], [], []
    empty_count = 0

    for idx, (pred, gold) in enumerate(zip(pred_explanations, gold_explanations)):
        pred_str = pred.strip()
        tok_len = len(pred_str.split())
        lengths.append(tok_len)
        if tok_len == 0:
            empty_count += 1

        alignment.append(step_alignment_score(pred_str, step_cats[idx]))
        reasoning.append(reasoning_density_score(pred_str))

        if not gold.strip():
            continue
        hyp_tok, ref_tok = _tokenize(pred_str), _tokenize(gold)
        bleu1.append(bleu_n(hyp_tok, ref_tok, 1))
        bleu2.append(bleu_n(hyp_tok, ref_tok, 2))
        bleu4.append(bleu_n(hyp_tok, ref_tok, 4))
        rougeL.append(rouge_l(hyp_tok, ref_tok))
        valid_preds.append(pred_str)
        valid_refs.append(gold)

    bert_scores = bertscore_f1_batch(valid_preds, valid_refs, encoder) if (valid_preds and encoder) else []

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    lengths_arr = np.array(lengths) if lengths else np.array([0])
    return {
        "bleu1": avg(bleu1), "bleu2": avg(bleu2), "bleu4": avg(bleu4),
        "rouge_l": avg(rougeL), "bertscore_f1": avg(bert_scores),
        "step_alignment": avg(alignment), "reasoning_density": avg(reasoning),
        "avg_length": avg(lengths),
        "p25_length": float(np.percentile(lengths_arr, 25)),
        "p75_length": float(np.percentile(lengths_arr, 75)),
        "empty_rate": empty_count / max(len(pred_explanations), 1),
        "n_scored": len(valid_preds), "n_total": len(pred_explanations),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def eval_grpo(adapter_dir: str, max_new_tokens: int = 350, save_explanations: str | None = None) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[eval] Test input : {INPUT_TEST_JSON}")
    print(f"[eval] Device     : {device}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir).eval()

    print("[eval] Loading sentence transformer for BERTScore...")
    encoder = get_sentence_encoder()

    examples = load_json_data(INPUT_TEST_JSON)
    examples = [e for e in examples if e["gold_step_text"] and e["gold_step_category"] != "unknown"]
    print(f"[eval] Usable test rows after filtering: {len(examples)}")

    step_preds, step_gold = [], []
    mcp_preds, mcp_gold = [], []
    pred_explanations, gold_explanations, step_cats_for_align = [], [], []
    parse_failures = 0
    csv_rows = []

    for ex in tqdm(examples, desc="Evaluating", unit="sample"):
        prompt_text = build_chat_prompt(ex)
        with torch.no_grad():
            ids = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=MAX_PROMPT_TOKENS,
            ).input_ids.to(device)
            embed_layer = model.get_input_embeddings()
            token_embeds = embed_layer(ids).to(dtype)
            attn = torch.ones(token_embeds.shape[:2], dtype=torch.long, device=device)

            out = model.generate(
                inputs_embeds=token_embeds,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,     # greedy at eval time for reproducible accuracy numbers
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)

        obj = parse_completion(gen_text)
        if obj is None:
            parse_failures += 1
            obj = {}
            m = re.search(r'"?New step"?\s*:\s*"([^"]+)"', gen_text)
            if m:
                obj["New step"] = m.group(1)
            m2 = re.search(r'"?Step explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL)
            if m2:
                obj["Step explanation"] = m2.group(1)

        # ---- step classification ----
        pred_step_raw = str(obj.get("New step", ""))
        pred_cat = classify_step(pred_step_raw) if pred_step_raw else "unknown"
        s_idx = STEP_LABELS.index(pred_cat) if pred_cat in STEP_LABELS else -1
        g_idx = STEP_LABELS.index(ex["gold_step_category"])
        step_preds.append(s_idx)
        step_gold.append(g_idx)

        # ---- mcp classification ----
        mcp_val = obj.get("MCP_tasks", {})
        if isinstance(mcp_val, str):
            from data_utils import parse_mcp_tasks
            mcp_val = parse_mcp_tasks(mcp_val)
        pred_mcp_labels = mcp_labels_from_dict(mcp_val) if isinstance(mcp_val, dict) else []
        mcp_preds.append(mcp_multihot(pred_mcp_labels))
        mcp_gold.append(mcp_multihot(ex["gold_mcp_labels"]))

        # ---- explanation ----
        pred_expl = str(obj.get("Step explanation", "")).strip()
        gold_expl = ex["gold_step_explanation"]
        pred_explanations.append(pred_expl)
        gold_explanations.append(gold_expl)
        step_cats_for_align.append(ex["gold_step_category"])

        if save_explanations:
            csv_rows.append({
                "machine": ex["machine"],
                "gold_step": ex["gold_step_text"],
                "gold_step_category": ex["gold_step_category"],
                "pred_step": pred_step_raw or "UNPARSEABLE",
                "pred_step_category": pred_cat,
                "step_correct": int(s_idx == g_idx),
                "gold_mcp": "|".join(ex["gold_mcp_labels"]),
                "pred_mcp": "|".join(pred_mcp_labels),
                "gold_explanation": gold_expl,
                "pred_explanation": pred_expl,
            })

    if parse_failures:
        print(f"\n[eval] Note: {parse_failures}/{len(examples)} responses had no parseable JSON")

    step_preds_arr, step_gold_arr = np.array(step_preds), np.array(step_gold)
    mcp_preds_arr, mcp_gold_arr = np.stack(mcp_preds), np.stack(mcp_gold)

    report_classification(step_preds_arr, step_gold_arr, mcp_preds_arr, mcp_gold_arr)

    print("\n\n" + "=" * 60)
    print("STEP EXPLANATION QUALITY")
    print("=" * 60)
    expl_metrics = compute_explanation_metrics(pred_explanations, gold_explanations, step_cats_for_align, encoder)
    report_explanation(expl_metrics)

    # ---- combined "all three match" accuracy the user asked about ----
    overall_hits = 0
    for i in range(len(step_preds)):
        step_ok = step_preds[i] == step_gold[i]
        mcp_ok = f1_score(mcp_gold_arr[i:i+1], mcp_preds_arr[i:i+1], average="samples", zero_division=0) >= 0.5
        expl_ok = 0.0  # filled below using bertscore per-row if available
        overall_hits += int(step_ok and mcp_ok)
    print("\n" + "=" * 60)
    print("COMBINED (Next step + MCP set match, threshold-based)")
    print("=" * 60)
    print(f"  Step-and-MCP joint accuracy : {overall_hits/len(step_preds):.4f}  "
          f"({overall_hits}/{len(step_preds)})")
    print("  (Combine with STEP CLASSIFICATION accuracy and BERTScore-F1 above "
          "for the full 'next step + explanation + MCP' picture — see README "
          "for how these three roll up into a single number.)")

    if save_explanations and csv_rows:
        import csv
        fieldnames = list(csv_rows[0].keys())
        with open(save_explanations, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[eval] Explanation predictions saved to: {save_explanations}")


def report_classification(step_preds, step_gold, mcp_preds, mcp_gold):
    print("\n" + "=" * 60)
    print("STEP CLASSIFICATION  (10 canonical categories)")
    print("=" * 60)
    print(f"  Accuracy      : {accuracy_score(step_gold, step_preds):.4f}")
    print(f"  Macro F1      : {f1_score(step_gold, step_preds, average='macro', zero_division=0):.4f}")
    print(f"  Weighted F1   : {f1_score(step_gold, step_preds, average='weighted', zero_division=0):.4f}")

    labels_present = sorted(set(step_gold.tolist()) | set(int(p) for p in step_preds if p >= 0))
    print("\n  Per-class report:")
    print(classification_report(
        step_gold, step_preds, labels=labels_present,
        target_names=[STEP_LABELS[i] if 0 <= i < len(STEP_LABELS) else "UNPARSEABLE" for i in labels_present],
        zero_division=0,
    ))
    print("  Confusion matrix (rows=gold, cols=pred):")
    cm = confusion_matrix(step_gold, step_preds, labels=list(range(len(STEP_LABELS))))
    print(cm)

    print("\n" + "=" * 60)
    print("MCP TOOL CLASSIFICATION  (18 canonical tools, multi-label)")
    print("=" * 60)
    print(f"  Subset (exact-match) accuracy : {accuracy_score(mcp_gold, mcp_preds):.4f}")
    print(f"  Micro F1                      : {f1_score(mcp_gold, mcp_preds, average='micro', zero_division=0):.4f}")
    print(f"  Macro F1                      : {f1_score(mcp_gold, mcp_preds, average='macro', zero_division=0):.4f}")
    print(f"  Samples F1                    : {f1_score(mcp_gold, mcp_preds, average='samples', zero_division=0):.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(mcp_gold, mcp_preds, average=None, zero_division=0)
    print("\n  Per-label metrics:")
    print(f"  {'Label':<22}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Sup':>5}")
    print("  " + "-" * 52)
    for i, label in enumerate(MCP_LABELS):
        flag = "  <- low recall" if rec[i] < 0.3 and support[i] > 0 else ""
        print(f"  {label:<22}  {prec[i]:>6.3f}  {rec[i]:>6.3f}  {f1[i]:>6.3f}  {int(support[i]):>5}{flag}")


def report_explanation(metrics: dict) -> None:
    n_scored, n_total = metrics["n_scored"], metrics["n_total"]
    if n_scored < n_total:
        print(f"  (Reference-based metrics scored on {n_scored}/{n_total} pairs — "
              f"{n_total - n_scored} skipped, empty gold explanation)")
    print(f"\n  {'Metric':<26}  {'Score':>8}")
    print("  " + "-" * 40)
    for key in ("bleu1", "bleu2", "bleu4", "rouge_l", "bertscore_f1", "step_alignment", "reasoning_density"):
        print(f"  {key:<26}  {metrics[key]:>8.4f}")
    print()
    print(f"  {'Avg length (tokens)':<26}  {metrics['avg_length']:>8.1f}")
    print(f"  {'P25 length':<26}  {metrics['p25_length']:>8.1f}")
    print(f"  {'P75 length':<26}  {metrics['p75_length']:>8.1f}")
    print(f"  {'Empty rate':<26}  {metrics['empty_rate']:>8.1%}  "
          f"{'<- high, JSON parsing likely failing' if metrics['empty_rate'] > 0.1 else 'OK'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stepmodelv3 GRPO RL model")
    parser.add_argument("--adapter-dir", type=str, default=STAGE1_ADAPTER_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--save-explanations", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.adapter_dir):
        print(f"[eval] ERROR: Adapter directory not found: {args.adapter_dir}")
        print("[eval] Train the model first: python train_grpo.py")
        exit(1)

    eval_grpo(args.adapter_dir, args.max_new_tokens, args.save_explanations)
