"""
Aggregate the Stage-1 modality ablation (training/run_stage1_ablation.py) into
the evidence for one claim:

    Step classification depends mainly on the strategy TEXT;
    MCP tool selection depends mainly on the GRAPH.

That is a double dissociation, so the report is organised around it:

  1. Results per variant (fusion / text_only / graph_only): mean +- std over
     seeds, using the SAME test metrics stage1_gnn_train.py reports as its
     headline numbers (read from each run's checkpoint).
  2. What removing each modality costs, seed-matched against fusion, with an
     explicit, falsifiable check of the two halves of the hypothesis.
  3. Paired bootstrap over the test rows for the key comparisons, so the claim
     does not rest on point estimates from a 268-row test set. The interval
     reflects test-set sampling of the seed-averaged models; across-seed
     variability is the +- in table 1.
  4. A bar chart (output/ablations/stage1_modality_ablation.png).

Everything printed is also written to output/ablations/stage1_ablation_report.md.

Usage:
    python eval/aggregate_stage1_ablation.py
    python eval/aggregate_stage1_ablation.py --ckpt-root checkpoints/ablations --out-root output/ablations
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODES = ["fusion", "text_only", "graph_only"]
TAG_RE = re.compile(r"^(fusion|text_only|graph_only)_seed(\d+)$")

# All reported; the four HEADLINE metrics are what the hypothesis check and plot use.
METRICS = [
    ("step_accuracy", "Step acc"),
    ("step_macro_f1", "Step macro-F1"),
    ("mcp_samples_f1", "MCP samples-F1"),
    ("mcp_macro_f1", "MCP macro-F1"),
    ("mcp_micro_f1", "MCP micro-F1"),
    ("mcp_subset_accuracy", "MCP subset acc"),
]
HEADLINE = ["step_accuracy", "step_macro_f1", "mcp_samples_f1", "mcp_macro_f1"]
LABEL = dict(METRICS)


def load_runs(ckpt_root, out_root):
    import torch
    runs = {}
    for path in sorted(glob.glob(os.path.join(ckpt_root, "*", "stage1_gnn_classifier.pt"))):
        tag = os.path.basename(os.path.dirname(path))
        m = TAG_RE.match(tag)
        if not m:
            print(f"[warn] ignoring {tag!r}: not <mode>_seed<N>")
            continue
        mode, seed = m.group(1), int(m.group(2))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        tm = ck.get("test_metrics")
        if not tm:
            print(f"[warn] {tag}: checkpoint has no test_metrics (run did not finish?) -> skipped")
            continue
        if ck.get("ablation") not in (None, mode):
            print(f"[warn] {tag}: checkpoint says ablation={ck.get('ablation')!r}, tag says {mode!r} -> skipped")
            continue
        csv_path = os.path.join(out_root, tag, "stage1.csv")
        runs[(mode, seed)] = {
            "metrics": {k: float(v) for k, v in tm.items()},
            "csv": csv_path if os.path.exists(csv_path) else None,
        }
    return runs


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else float("nan"))


def cell(vals):
    m, s = mean_std(vals)
    return f"{m:.3f} ± {s:.3f}" if len(vals) > 1 else f"{m:.3f}"


def per_row(csv_path):
    """Per-row step correctness and MCP F1 from a run's predictions CSV."""
    steps, mcps, gold_steps, gold_mcps = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            P = set(filter(None, r["mcp_tool_prediction"].split("|")))
            G = set(filter(None, r["mcp_tool_gold"].split("|")))
            denom = len(P) + len(G)
            mcps.append(2 * len(P & G) / denom if denom else 0.0)   # sklearn 'samples', zero_division=0
            steps.append(1.0 if r["step_prediction"] == r["gold_new_step"] else 0.0)
            gold_steps.append(r["gold_new_step"])
            gold_mcps.append(r["mcp_tool_gold"])
    return np.array(steps), np.array(mcps), gold_steps, gold_mcps


def bootstrap_diff(a, b, n_boot=5000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=os.path.join(os.environ.get("CKPT_DIR", os.path.join(ROOT, "checkpoints")), "ablations"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "output", "ablations"))
    args = ap.parse_args()

    runs = load_runs(args.ckpt_root, args.out_root)
    if not runs:
        sys.exit(f"No finished ablation runs under {args.ckpt_root}. "
                 f"Run training/run_stage1_ablation.py first.")
    os.makedirs(args.out_root, exist_ok=True)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    by_mode = {m: sorted(s for (mm, s) in runs if mm == m) for m in MODES}
    out("# Stage-1 modality ablation\n")
    out("Runs found: " + ", ".join(f"{m}: seeds {by_mode[m] or '-'}" for m in MODES) + "\n")

    # ---- 1. results per variant ------------------------------------------
    out("## 1. Test metrics per variant (mean ± std over seeds)\n")
    out("| variant | n | " + " | ".join(LABEL[k] for k, _ in METRICS) + " |")
    out("|---|---|" + "---|" * len(METRICS))
    summary_rows = []
    for m in MODES:
        if not by_mode[m]:
            continue
        cells, row = [], {"variant": m, "n_seeds": len(by_mode[m])}
        for k, _ in METRICS:
            vals = [runs[(m, s)]["metrics"][k] for s in by_mode[m] if k in runs[(m, s)]["metrics"]]
            cells.append(cell(vals) if vals else "n/a")
            if vals:
                mu, sd = mean_std(vals)
                row[f"{k}_mean"], row[f"{k}_std"] = round(mu, 4), round(sd, 4)
        out(f"| {m} | {len(by_mode[m])} | " + " | ".join(cells) + " |")
        summary_rows.append(row)
    with open(os.path.join(args.out_root, "stage1_ablation_summary.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in summary_rows for k in r}, key=lambda k: (k not in ("variant", "n_seeds"), k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)

    # ---- 2. cost of removing each modality --------------------------------
    drops = {}
    if by_mode["fusion"]:
        out("\n## 2. What removing a modality costs (fusion − variant, seed-matched; positive = performance lost)\n")
        out("| removed | seeds | " + " | ".join(LABEL[k] for k in HEADLINE) + " |")
        out("|---|---|" + "---|" * len(HEADLINE))
        for variant, what in (("text_only", "graph removed (fusion − text_only)"),
                              ("graph_only", "text removed (fusion − graph_only)")):
            common = sorted(set(by_mode["fusion"]) & set(by_mode[variant]))
            if not common:
                out(f"| {what} | none in common | " + " | ".join("n/a" for _ in HEADLINE) + " |")
                continue
            drops[variant] = {k: float(np.mean([runs[("fusion", s)]["metrics"][k] - runs[(variant, s)]["metrics"][k]
                                                for s in common])) for k in HEADLINE}
            out(f"| {what} | {len(common)} | " + " | ".join(f"{drops[variant][k]:+.3f}" for k in HEADLINE) + " |")

        if len(drops) == 2:
            out("\n**Hypothesis check (point estimates; CIs in section 3):**\n")
            h1 = drops["graph_only"]["step_accuracy"] > drops["text_only"]["step_accuracy"]
            h2 = drops["text_only"]["mcp_samples_f1"] > drops["graph_only"]["mcp_samples_f1"]
            out(f"- Step is text-driven: removing the text costs more step accuracy "
                f"({drops['graph_only']['step_accuracy']:+.3f}) than removing the graph "
                f"({drops['text_only']['step_accuracy']:+.3f})  ->  {'SUPPORTED' if h1 else 'NOT supported'}")
            out(f"- MCP is graph-driven: removing the graph costs more MCP samples-F1 "
                f"({drops['text_only']['mcp_samples_f1']:+.3f}) than removing the text "
                f"({drops['graph_only']['mcp_samples_f1']:+.3f})  ->  {'SUPPORTED' if h2 else 'NOT supported'}")
            out(f"- Double dissociation (both): **{'YES' if h1 and h2 else 'NO'}**")
    else:
        out("\n(no fusion runs found -- section 2 needs them as the reference)")

    # ---- 3. paired bootstrap over test rows --------------------------------
    out("\n## 3. Paired bootstrap over the test rows (95% CI; * = interval excludes 0)\n")
    per = {}
    ref_gold = None
    ok = True
    for (m, s), r in runs.items():
        if not r["csv"]:
            out(f"(no predictions CSV for {m}_seed{s}; skipping bootstrap)")
            ok = False
            break
        st, mc, gs, gm = per_row(r["csv"])
        if ref_gold is None:
            ref_gold = (gs, gm)
        elif (gs, gm) != ref_gold:
            out("(test rows differ between runs -- cannot pair; skipping bootstrap)")
            ok = False
            break
        for k, vec, off in (("step_accuracy", st, 0.0), ("mcp_samples_f1", mc, 0.0)):
            if abs(vec.mean() - r["metrics"][k]) > 1e-3:
                out(f"[warn] {m}_seed{s}: CSV-recomputed {k} {vec.mean():.4f} != checkpoint {r['metrics'][k]:.4f}")
        per.setdefault(m, {"step": [], "mcp": []})
        per[m]["step"].append(st)
        per[m]["mcp"].append(mc)
    if ok and per:
        avg = {m: {"step": np.mean(v["step"], axis=0), "mcp": np.mean(v["mcp"], axis=0)} for m, v in per.items()}
        comps = [("fusion − text_only  (value of the graph)", "fusion", "text_only"),
                 ("fusion − graph_only (value of the text)", "fusion", "graph_only"),
                 ("text_only − graph_only", "text_only", "graph_only")]
        out("| comparison | Δ step accuracy [CI] | Δ MCP samples-F1 [CI] |")
        out("|---|---|---|")
        for name, a, b in comps:
            if a not in avg or b not in avg:
                continue
            cells = []
            for key in ("step", "mcp"):
                d, lo, hi = bootstrap_diff(avg[a][key], avg[b][key])
                star = "*" if (lo > 0 or hi < 0) else ""
                cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
            out(f"| {name} | {cells[0]} | {cells[1]} |")
        out("\nExpected under the hypothesis: row 3 positive on step, negative on MCP; "
            "row 1 small on step, large on MCP.")
        out(f"(Models averaged over seeds per variant: "
            + ", ".join(f"{m} n={len(per[m]['step'])}" for m in per) + "; n_boot=5000.)")

    # ---- 4. plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        present = [m for m in MODES if by_mode[m]]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        w = 0.8 / len(present)
        x = np.arange(len(HEADLINE))
        for i, m in enumerate(present):
            vals = np.array([[runs[(m, s)]["metrics"][k] for s in by_mode[m]] for k in HEADLINE])  # (metrics, seeds)
            mu = vals.mean(axis=1)
            sd = vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros(len(HEADLINE))
            pos = x + (i - (len(present) - 1) / 2) * w
            ax.bar(pos, mu, w, yerr=sd, capsize=3, label=f"{m} (n={vals.shape[1]})")
            for j in range(vals.shape[1]):
                ax.scatter(pos, vals[:, j], color="k", s=9, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[k] for k in HEADLINE])
        ax.set_ylim(0, 1)
        ax.set_ylabel("test score")
        ax.set_title("Stage-1 modality ablation (bars = mean over seeds, dots = seeds)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=len(present), frameon=False)
        fig.tight_layout()
        png = os.path.join(args.out_root, "stage1_modality_ablation.png")
        fig.savefig(png, dpi=200)
        plt.close(fig)
        out(f"\nPlot: {png}")
    except Exception as e:  # plotting must never lose the tables above
        out(f"\n(plot skipped: {e})")

    report = os.path.join(args.out_root, "stage1_ablation_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport: {report}\nSummary CSV: {os.path.join(args.out_root, 'stage1_ablation_summary.csv')}")


if __name__ == "__main__":
    main()
