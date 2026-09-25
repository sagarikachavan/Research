"""
Score step-explanation quality for MULTIPLE models with MULTIPLE judges,
so `explanation_judge_accuracy` isn't just self-consistent with one local
judge -- Prof. Suranga's concern -- and to follow Yasod's guidance (see
Pen-Strategist paper, arxiv.org/pdf/2605.04499, Table 2 / Section 5.5):
cross-check explanation quality across several commercial-LLM judges, then
validate those judges against manual human scoring
(manual_validation_sample.py).

Reuses the EXACT rubric prompt, parsing, and is_correct gate from
core/llm_judge.py (JUDGE_SYSTEM_PROMPT / _build_user_prompt / _parse_rubric /
_is_correct_from_rubric) so every judge -- local Qwen or commercial -- is
scoring against the identical 4-criteria rubric. Only the generation call
differs per judge backend.

Usage:
    python multi_judge_explanation_eval.py \
        --models stage2:output/stage2.csv stage3:output/stage3.csv commercial_gpt-5:output/commercial_gpt-5.csv \
        --judges local_qwen gpt-4o claude-sonnet \
        --max-samples 268

Output:
    output/multi_judge_comparison.csv   -- one row per (model, judge): accuracy, mean correctness
    output/multi_judge_per_row.csv      -- one row per (model, judge, test-row): full rubric,
                                            needed by judge_correlation.py against manual scores
"""
import argparse
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "data_prep"), os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import ROOT, LLM_JUDGE_MODEL_NAME
from llm_judge import (
    JUDGE_SYSTEM_PROMPT, RUBRIC_DIMS, _build_user_prompt, _parse_rubric,
    _score_from_rubric, _is_correct_from_rubric,
)
from commercial_llm import generate_text, MODEL_REGISTRY

LOCAL_JUDGE_KEY = "local_qwen"


def _load_local_judge():
    """Loads the project's existing local judge model (core/llm_judge.py's
    own path via evaluate.py) so local_qwen scores here are directly
    comparable to explanation_judge_accuracy in comparison_report.py."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llm_judge import set_llm_judge_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(LLM_JUDGE_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLM_JUDGE_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    set_llm_judge_model(model, tok, device)


def _score_with_local_judge(pred_expl, gold_expl, pred_step, context):
    from llm_judge import evaluate_explanation_with_llm
    scores, _raw = evaluate_explanation_with_llm(
        pred_expl, gold_expl, pred_step, context, use_cache=True,
    )
    return scores


def _score_with_commercial_judge(pred_expl, gold_expl, pred_step, context, judge_key):
    prompt = _build_user_prompt(pred_expl, gold_expl, pred_step, context)
    raw = generate_text(
        "Respond with ONLY the requested JSON object. No commentary.",
        f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt}",
        judge_key, max_tokens=300,
    )
    rubric = _parse_rubric(raw)
    if rubric is None:
        return {"correctness_score": None, "is_correct": None, "rubric": None,
                "judge_error": True}
    return {
        "correctness_score": _score_from_rubric(rubric),
        "is_correct": _is_correct_from_rubric(rubric),
        "rubric": {d: rubric[d] for d in RUBRIC_DIMS},
        "judge_error": False,
    }


def load_model_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                     help="name:path/to/predictions.csv pairs, e.g. stage2:output/stage2.csv")
    ap.add_argument("--judges", nargs="+", default=[LOCAL_JUDGE_KEY],
                     help=f"{LOCAL_JUDGE_KEY} and/or any MODEL_REGISTRY key ({sorted(MODEL_REGISTRY.keys())})")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    model_paths = {}
    for spec in args.models:
        name, path = spec.split(":", 1)
        model_paths[name] = path

    if LOCAL_JUDGE_KEY in args.judges:
        print(f"[multi_judge] Loading local judge ({LLM_JUDGE_MODEL_NAME}) ...")
        _load_local_judge()

    per_row_rows = []
    agg = {}  # (model, judge) -> {n, n_correct, n_error, sum_score}

    for model_name, path in model_paths.items():
        rows = load_model_csv(path)
        if args.max_samples:
            rows = rows[: args.max_samples]

        for judge_key in args.judges:
            key = (model_name, judge_key)
            agg[key] = {"n": 0, "n_correct": 0, "n_error": 0, "sum_score": 0.0}

            for i, row in enumerate(rows):
                pred_expl = row.get("predicted_step_explanation", "")
                gold_expl = row.get("gold_step_explanation", "")
                pred_step = row.get("predicted_new_step", "")
                context = {
                    "New strategy": row.get("new_strategy", ""),
                    "Strategy explanation": row.get("strategy_explanation", ""),
                }

                if judge_key == LOCAL_JUDGE_KEY:
                    scores = _score_with_local_judge(pred_expl, gold_expl, pred_step, context)
                else:
                    try:
                        scores = _score_with_commercial_judge(pred_expl, gold_expl, pred_step, context, judge_key)
                    except Exception as e:
                        print(f"  [{model_name}/{judge_key}] row {i}: judge call failed: {e}")
                        scores = {"correctness_score": None, "is_correct": None, "rubric": None, "judge_error": True}

                a = agg[key]
                a["n"] += 1
                if scores.get("judge_error"):
                    a["n_error"] += 1
                else:
                    a["n_correct"] += int(bool(scores["is_correct"]))
                    a["sum_score"] += scores["correctness_score"]

                rubric = scores.get("rubric") or {}
                per_row_rows.append({
                    "model": model_name, "judge": judge_key, "row_idx": i,
                    "machine": row.get("machine", ""),
                    "correctness_score": scores.get("correctness_score"),
                    "is_correct": scores.get("is_correct"),
                    "judge_error": scores.get("judge_error"),
                    **{f"rubric_{d}": rubric.get(d) for d in RUBRIC_DIMS},
                })
            print(f"[{model_name} / {judge_key}] "
                  f"{agg[key]['n_correct']}/{agg[key]['n'] - agg[key]['n_error']} correct "
                  f"({agg[key]['n_error']} judge errors)")

    out_dir = os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)

    comparison_path = os.path.join(out_dir, "multi_judge_comparison.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "judge", "n_scored", "n_errors", "accuracy", "mean_correctness_score"])
        for (model_name, judge_key), a in agg.items():
            scored = a["n"] - a["n_error"]
            acc = a["n_correct"] / scored if scored else 0.0
            mean_score = a["sum_score"] / scored if scored else 0.0
            writer.writerow([model_name, judge_key, scored, a["n_error"], f"{acc:.4f}", f"{mean_score:.4f}"])
    print(f"\n[multi_judge] Aggregate comparison saved to {comparison_path}")

    per_row_path = os.path.join(out_dir, "multi_judge_per_row.csv")
    with open(per_row_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_row_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_row_rows)
    print(f"[multi_judge] Per-row scores saved to {per_row_path}")


if __name__ == "__main__":
    main()
