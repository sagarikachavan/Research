"""
"Step Explanation (LLM judge)" bar chart, extended to include commercial-LLM
baselines -- same visual style, same rubric/gate as the original
explanation_comparison.png (comparison_report.py's create_visualizations):
core/llm_judge.py's project rubric (relevance>=2 AND technical_accuracy>=2
AND completeness>=1), NOT the new G-Eval rubric. Uses ONE judge (default
local_qwen -- the same Qwen2.5-7B-Instruct model evaluate.py's own
explanation_judge_accuracy uses) so the numbers are directly comparable to
the original chart's 57.9%/64.1% for stage2/stage3, not a different metric
wearing the same name.

Requires output/multi_judge_comparison.csv to have been generated with
--rubric project (NOT geval -- the default multi_judge run in run.py uses
geval, so re-run with the project rubric first):

    python multi_judge_explanation_eval.py \
        --models stage2:output/stage2.csv stage3:output/stage3.csv \
                 commercial_gpt-5:output/commercial_gpt-5.csv \
                 commercial_gpt-5-mini:output/commercial_gpt-5-mini.csv \
        --judges local_qwen gpt-4o \
        --rubric project

Usage:
    python plot_explanation_judge_accuracy.py
    python plot_explanation_judge_accuracy.py --judge gpt-4o --out explanation_comparison_gpt4o.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(_ROOT, "output", "multi_judge_comparison.csv"))
    ap.add_argument("--judge", default="local_qwen",
                     help="Which judge's accuracy to plot -- default matches the "
                          "Qwen2.5-7B-Instruct model the original explanation_comparison.png used.")
    ap.add_argument("--out", default=os.path.join(_ROOT, "output", "explanation_comparison_with_commercial.png"))
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Not found: {args.csv} -- run multi_judge_explanation_eval.py first.")

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["judge"] == args.judge]

    if not rows:
        judges_seen = set()
        with open(args.csv, newline="", encoding="utf-8") as f:
            judges_seen = {r["judge"] for r in csv.DictReader(f)}
        raise SystemExit(f"No rows for judge={args.judge!r} in {args.csv}. "
                          f"Judges present: {sorted(judges_seen)}")

    rubric_mode = None
    per_row_path = os.path.join(os.path.dirname(args.csv), "multi_judge_per_row.csv")
    if os.path.exists(per_row_path):
        with open(per_row_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rubric_mode = r.get("rubric_mode")
                break
    if rubric_mode == "geval":
        print("[WARNING] multi_judge_per_row.csv was generated with --rubric geval, "
              "not 'project' -- these accuracy numbers use a DIFFERENT rubric/threshold "
              "than the original explanation_comparison.png (57.9%/64.1%) and are NOT "
              "directly comparable to it. Re-run multi_judge_explanation_eval.py with "
              "--rubric project first if you want numbers on the same scale.")

    models = [r["model"] for r in rows]
    accs = [float(r["accuracy"]) for r in rows]

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 5))
    bars = ax.bar(models, accs, color="skyblue")
    for bar, val in zip(bars, accs):
        ax.annotate(f"{val * 100:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_title("explanation judge accuracy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    fig.suptitle(f"Step Explanation (LLM judge: {args.judge})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {args.out}")


if __name__ == "__main__":
    main()
