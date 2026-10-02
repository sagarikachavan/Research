"""
Bar charts from output/multi_judge_comparison.csv -- one grouped bar chart
each for mean_correctness_score (the number directly comparable to the
Pen-Strategist paper's Table 2 "Explanation" column) and accuracy (this
project's own >=0.6 binarization, not something the paper reports).

Reads whatever models/judges are actually in the CSV, so it stays correct
as you add more (e.g. once a gpt-4o judge run is added to a local_qwen-only
result) without needing to edit this script.

Usage:
    python plot_multi_judge_comparison.py
    python plot_multi_judge_comparison.py --csv output/multi_judge_comparison.csv --out-dir output
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display on a GPU box / headless cluster
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_comparison(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    models = sorted({r["model"] for r in rows})
    judges = sorted({r["judge"] for r in rows})
    return rows, models, judges


def grouped_bar(rows, models, judges, value_key, title, ylabel, out_path):
    by_key = {(r["model"], r["judge"]): float(r[value_key]) for r in rows}

    x = np.arange(len(models))
    width = 0.8 / max(len(judges), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 5))

    for i, judge in enumerate(judges):
        vals = [by_key.get((m, judge), 0.0) for m in models]
        bars = ax.bar(x + i * width - (len(judges) - 1) * width / 2, vals, width, label=judge)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend(title="judge")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(_ROOT, "output", "multi_judge_comparison.csv"))
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "output"))
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"Not found: {args.csv} -- run eval/multi_judge_explanation_eval.py first.")

    rows, models, judges = load_comparison(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    grouped_bar(
        rows, models, judges, "mean_correctness_score",
        "Explanation quality (G-Eval mean score, paper-comparable)",
        "mean_correctness_score (0-1)",
        os.path.join(args.out_dir, "multi_judge_mean_score.png"),
    )
    grouped_bar(
        rows, models, judges, "accuracy",
        "Explanation \"correct\" rate (score >= 0.6, project's own threshold)",
        "accuracy",
        os.path.join(args.out_dir, "multi_judge_accuracy.png"),
    )


if __name__ == "__main__":
    main()
