"""
evaluate_structure_impact_on_step_task.py
============================================

Answers "Next step #2" from the original report: does training on graph
structure tasks help, hurt, or not affect the REAL step-prediction task?

This is a thin wrapper, not a reimplementation: it calls eval/evaluate.py's
own `eval_llm(...)` function directly on whichever checkpoint you point it
at, so the step-prediction numbers it produces are computed by EXACTLY the
same code as your existing Stage 2 / Stage 3 numbers -- no risk of a
second, subtly-different evaluation path producing numbers that aren't
actually comparable.

What it does:
  1. Runs eval_llm() on the Stage-2 checkpoint (baseline)
  2. Runs eval_llm() on checkpoints/graph_structure/best (structure-only)
  3. Runs eval_llm() on checkpoints/multitask/best (joint), if present
  4. Prints a side-by-side diff table of the headline metrics
  5. Also runs run_reliability_suite.py + analyze_results.py's scoring
     logic on all THREE checkpoints so you get the structure-side answer
     and the step-prediction-side answer in one place.

Usage:
    python graph_adapter_experiments/evaluate_structure_impact_on_step_task.py
    # or point at specific dirs:
    python graph_adapter_experiments/evaluate_structure_impact_on_step_task.py \
        --checkpoints checkpoints/stage2_qwen_lora checkpoints/graph_structure/best checkpoints/multitask/best
"""
import os
import sys
import io
import contextlib
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CKPT_DIR, RESULTS_DIR


def run_eval_llm_capture(adapter_dir: str, max_new_tokens: int = 300):
    """Calls eval/evaluate.py's eval_llm() and captures its printed metrics
    block, since it doesn't return a structured object -- reparses the
    printed report the same way a human reading the console output would.
    We keep the LLM judge OFF here (use_llm_judge=False) to keep this
    comparison fast and deterministic; re-run eval/evaluate.py directly with
    --use-llm-judge if you also want judge scores for these checkpoints."""
    import evaluate as eval_module  # eval/evaluate.py, via the path bootstrap

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eval_module.eval_llm(
            adapter_dir,
            max_new_tokens=max_new_tokens,
            use_llm_judge=False,
            auto_save_csv=False,
        )
    text = buf.getvalue()
    return text, _parse_headline_metrics(text)


def _parse_headline_metrics(text: str) -> dict:
    import re
    out = {}
    patterns = {
        "step_accuracy": r"Accuracy\s*:\s*([\d.]+)",
        "step_macro_f1": r"Macro F1\s*:\s*([\d.]+)",
        "mcp_subset_accuracy": r"Subset \(exact-match\) accuracy\s*:\s*([\d.]+)",
        "mcp_micro_f1": r"Micro F1\s*:\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=None,
                     help="Adapter dirs to compare. Default: Stage-2 baseline + "
                          "graph_structure/best + multitask/best (whichever exist).")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    args = ap.parse_args()

    import config
    if args.checkpoints:
        checkpoints = args.checkpoints
    else:
        candidates = [
            config.STAGE2_ADAPTER_DIR,
            os.path.join(CKPT_DIR, "graph_structure", "best"),
            os.path.join(CKPT_DIR, "multitask", "best"),
        ]
        checkpoints = [c for c in candidates if os.path.exists(c)]
        missing = [c for c in candidates if not os.path.exists(c)]
        if missing:
            print("[evaluate_structure_impact] Skipping missing checkpoints (train them first): "
                  + ", ".join(missing))

    if not checkpoints:
        raise SystemExit("No checkpoints found. Run train_structure_adapter.py and/or "
                          "train_multitask_adapter.py first, or pass --checkpoints explicitly.")

    results = {}
    for ckpt in checkpoints:
        label = os.path.basename(os.path.dirname(ckpt)) if os.path.basename(ckpt) == "best" \
            else os.path.basename(ckpt)
        print(f"\n{'=' * 70}\n[evaluate_structure_impact] Evaluating: {label}  ({ckpt})\n{'=' * 70}")
        full_text, metrics = run_eval_llm_capture(ckpt, args.max_new_tokens)
        print(full_text[-2000:])  # tail of the real eval_llm output, for sanity
        results[label] = metrics

    # ── Side-by-side diff table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP-PREDICTION IMPACT SUMMARY  (all numbers from eval/evaluate.py's "
          "own eval_llm(), unmodified)")
    print("=" * 70)
    metric_keys = ["step_accuracy", "step_macro_f1", "mcp_subset_accuracy", "mcp_micro_f1"]
    header = "| Checkpoint | " + " | ".join(metric_keys) + " |"
    sep = "|---" * (len(metric_keys) + 1) + "|"
    print(header)
    print(sep)
    baseline_label = list(results.keys())[0]
    for label, m in results.items():
        row = [label] + [f"{m.get(k, float('nan')):.4f}" for k in metric_keys]
        print("| " + " | ".join(row) + " |")
    print()

    baseline = results[baseline_label]
    for label, m in list(results.items())[1:]:
        deltas = {k: (m.get(k, float("nan")) - baseline.get(k, float("nan"))) for k in metric_keys}
        verdict_bits = []
        for k, d in deltas.items():
            if abs(d) < 0.005:
                continue
            verdict_bits.append(f"{k} {'+' if d > 0 else ''}{d:.4f}")
        verdict = "; ".join(verdict_bits) if verdict_bits else "no meaningful change"
        print(f"[evaluate_structure_impact] {label} vs {baseline_label}: {verdict}")

    out_path = os.path.join(RESULTS_DIR, "step_task_impact_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate_structure_impact] Full results saved -> {out_path}")
    print("[evaluate_structure_impact] NOTE: this only reports step_accuracy/macro_f1/mcp "
          "metrics (LLM judge disabled for speed). To also see explanation-quality / LLM-judge "
          "scores per checkpoint, run eval/evaluate.py directly with --adapter-dir <ckpt> "
          "--use-llm-judge for each one.")


if __name__ == "__main__":
    main()
