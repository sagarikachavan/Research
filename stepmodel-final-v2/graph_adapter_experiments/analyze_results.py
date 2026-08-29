"""
analyze_results.py
====================

Turns run_reliability_suite.py's raw_results_*.jsonl into the actual
research answer: does the Graph Prefix Adapter carry usable, graph-specific
structural information, and does the LLM use it?

For every task, compares the `real` condition against every control
condition using a PAIRED permutation test (same items, different prefix
embedding -- so any difference is attributable to the intervention, not to
which items happened to be sampled). Reports mean score + 95% bootstrap CI
per condition, the paired real-vs-control gap, and a p-value.

Verdict logic per task (see RESEARCH_PLAN.md for the full reasoning):
  - "Evidence of graph understanding" if real is significantly (p<0.05)
    and meaningfully (>0.10 absolute) better than BOTH zero AND wrong_graph.
  - "Possible text/prior shortcut, not graph-specific" if real beats zero
    but does NOT beat wrong_graph -- the model may be using the fused
    context-text signal or its own priors rather than actual structure.
  - "No evidence of graph understanding for this task" otherwise.

Usage:
    python graph_adapter_experiments/analyze_results.py \
        --results results/raw_results_best.jsonl
    # or compare multiple checkpoints in one report:
    python graph_adapter_experiments/analyze_results.py \
        --results results/raw_results_stage2_qwen_lora.jsonl results/raw_results_best.jsonl
"""
import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, bootstrap_ci, paired_permutation_test

import numpy as np


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze_one_checkpoint(rows, label):
    # index: (task, condition) -> list of scores ; and paired index:
    # (task, machine, row_id) -> {condition: score}
    by_task_cond = defaultdict(list)
    paired = defaultdict(dict)
    for r in rows:
        by_task_cond[(r["task"], r["condition"])].append(r["score"])
        paired[(r["task"], r["machine"], r["row_id"])][r["condition"]] = r["score"]

    tasks = sorted(set(t for t, _ in by_task_cond.keys()))
    lines = [f"## Checkpoint: `{label}`\n"]

    verdicts = {}
    for task in tasks:
        lines.append(f"### Task: `{task}`\n")
        lines.append("| Condition | Mean | 95% CI | n |")
        lines.append("|---|---|---|---|")
        conditions_here = sorted(c for t, c in by_task_cond.keys() if t == task)
        for cond in conditions_here:
            vals = by_task_cond[(task, cond)]
            mean, lo, hi = bootstrap_ci(vals)
            lines.append(f"| {cond} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | {len(vals)} |")
        lines.append("")

        # Paired comparisons: real vs each control, on items that have BOTH scored.
        real_key = "real"
        controls = [c for c in conditions_here if c != real_key]
        sig_rows = []
        for control in controls:
            paired_a, paired_b = [], []
            for item_key, cond_scores in paired.items():
                if item_key[0] != task:
                    continue
                if real_key in cond_scores and control in cond_scores:
                    paired_a.append(cond_scores[real_key])
                    paired_b.append(cond_scores[control])
            if len(paired_a) < 5:
                continue
            gap = float(np.mean(paired_a) - np.mean(paired_b))
            p = paired_permutation_test(paired_a, paired_b)
            sig_rows.append((control, gap, p, len(paired_a)))

        if sig_rows:
            lines.append("**Paired real vs. control** (same items, permutation test):\n")
            lines.append("| vs. condition | real − control | p-value | n pairs |")
            lines.append("|---|---|---|---|")
            for control, gap, p, n in sig_rows:
                star = "**significant**" if p < 0.05 else "not significant"
                lines.append(f"| {control} | {gap:+.3f} | {p:.4f} ({star}) | {n} |")
            lines.append("")

        # Verdict for this task
        gap_vs_zero = next((g for c, g, p, n in sig_rows if c == "zero"), None)
        p_vs_zero = next((p for c, g, p, n in sig_rows if c == "zero"), None)
        gap_vs_wrong = next((g for c, g, p, n in sig_rows if c == "wrong_graph"), None)
        p_vs_wrong = next((p for c, g, p, n in sig_rows if c == "wrong_graph"), None)

        if gap_vs_zero is not None and gap_vs_wrong is not None:
            beats_zero = p_vs_zero < 0.05 and gap_vs_zero > 0.10
            beats_wrong = p_vs_wrong < 0.05 and gap_vs_wrong > 0.10
            if beats_zero and beats_wrong:
                verdict = ("✅ Evidence of GRAPH-SPECIFIC understanding: real beats both "
                           "'no graph at all' and 'a different real graph'.")
            elif beats_zero and not beats_wrong:
                verdict = ("⚠ Evidence the model uses SOME signal in the fused embedding "
                           "(beats zero), but NOT that it's reading the specific graph's "
                           "structure (does not beat a wrong graph) -- likely leaning on the "
                           "fused context-text component or a generic prior instead.")
            else:
                verdict = "❌ No evidence of graph understanding for this task."
            verdicts[task] = verdict
            lines.append(f"**Verdict:** {verdict}\n")
        else:
            lines.append("**Verdict:** insufficient conditions run to judge (need `zero` "
                          "and `wrong_graph`).\n")

    return "\n".join(lines), verdicts


def maybe_plot(rows_by_label, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze_results] matplotlib not installed -- skipping charts "
              "(pip install matplotlib to enable).")
        return []

    paths = []
    for label, rows in rows_by_label.items():
        by_task_cond = defaultdict(list)
        for r in rows:
            by_task_cond[(r["task"], r["condition"])].append(r["score"])
        tasks = sorted(set(t for t, _ in by_task_cond.keys()))
        for task in tasks:
            conds = sorted(c for t, c in by_task_cond.keys() if t == task)
            means = [np.mean(by_task_cond[(task, c)]) for c in conds]
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(conds, means, color="#4C72B0")
            if "real" in conds:
                bars[conds.index("real")].set_color("#55A868")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Mean score")
            ax.set_title(f"{label} — {task}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"chart_{label}_{task}.png")
            fig.savefig(out_path, dpi=130)
            plt.close(fig)
            paths.append(out_path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "REPORT.md"))
    args = ap.parse_args()

    rows_by_label = {}
    report_sections = []
    for path in args.results:
        label = os.path.basename(path).replace("raw_results_", "").replace(".jsonl", "")
        rows = load_rows(path)
        rows_by_label[label] = rows
        section, verdicts = analyze_one_checkpoint(rows, label)
        report_sections.append(section)

    chart_paths = maybe_plot(rows_by_label, RESULTS_DIR)

    header = (
        "# Graph Prefix Adapter Reliability Report\n\n"
        "Generated by `analyze_results.py`. Every number below comes from "
        "`run_reliability_suite.py`'s causal-intervention experiment: the SAME "
        "questions asked under different graph-prefix conditions (real graph, "
        "wrong graph, no graph, etc.), so a gap between `real` and a control "
        "condition is directly attributable to that intervention.\n\n"
        "See RESEARCH_PLAN.md for what each condition means and how to read "
        "the verdicts below.\n\n"
    )
    if chart_paths:
        header += "Charts:\n" + "\n".join(f"- `{os.path.relpath(p, RESULTS_DIR)}`" for p in chart_paths) + "\n\n"

    with open(args.out, "w") as f:
        f.write(header + "\n---\n\n".join(report_sections))

    print(f"[analyze_results] Report written -> {args.out}")
    if chart_paths:
        print(f"[analyze_results] {len(chart_paths)} chart(s) written -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
