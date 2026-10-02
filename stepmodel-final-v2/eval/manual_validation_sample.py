"""
Sample rows for manual (human) scoring, and compute how well each judge's
automated score agrees with your manual score -- this is the "evaluate it
manually by giving scores yourself" step Yasod described, mirroring the
Pen-Strategist paper's Section 5.5 human-expert study at a scale one person
can actually do.

Step 1 -- sample:
    python manual_validation_sample.py sample \
        --per-row output/multi_judge_per_row.csv --n 40
    -> writes output/manual_scoring_template.csv for you to fill in by hand
       (same 4-criteria rubric the LLM judges use, so scores are directly
       comparable: relevance/technical_accuracy/completeness/clarity, 0-3).

Step 2 -- after you've filled in the `manual_*` columns yourself:
    python manual_validation_sample.py correlate \
        --template output/manual_scoring_template.csv \
        --per-row output/multi_judge_per_row.csv
    -> for each judge, reports Spearman correlation between that judge's
       correctness_score and your manual score on the same rows -- the
       judge with the highest correlation is the one most trustworthy to
       report as `explanation_judge_accuracy` going forward.
"""
import argparse
import csv
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import ROOT, RANDOM_SEED

MANUAL_COLS = ["manual_relevance", "manual_technical_accuracy", "manual_completeness", "manual_clarity"]


def cmd_sample(args):
    with open(args.per_row, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # One row per (model, row_idx) -- judges disagree, but you're scoring the
    # SAME prediction, not the same prediction-judge pair, so dedupe first.
    seen = {}
    for r in rows:
        key = (r["model"], r["row_idx"])
        if key not in seen:
            seen[key] = r
    unique_rows = list(seen.values())

    random.seed(RANDOM_SEED)
    sample = random.sample(unique_rows, min(args.n, len(unique_rows)))

    out_path = os.path.join(ROOT, "output", "manual_scoring_template.csv")
    fieldnames = ["model", "row_idx", "machine"] + MANUAL_COLS + ["manual_notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample:
            writer.writerow({
                "model": r["model"], "row_idx": r["row_idx"], "machine": r.get("machine", ""),
                **{c: "" for c in MANUAL_COLS}, "manual_notes": "",
            })

    print(f"[manual_sample] Wrote {len(sample)} rows to {out_path}")
    print("[manual_sample] Fill in the manual_* columns by hand (0-3 each, same rubric as the LLM judges):")
    print("  relevance: does the explanation justify the SAME step as the reference?")
    print("  technical_accuracy: are the technical claims correct/consistent with the reference?")
    print("  completeness: does it cover the reference's key reasoning?")
    print("  clarity: is it well-written and easy to follow?")
    print("[manual_sample] NOTE: you'll need to look up the actual predicted/gold explanation text yourself")
    print("  (e.g. from output/stage2.csv, output/commercial_<model>.csv) using model+row_idx to find it --")
    print("  this template intentionally doesn't duplicate the full text to keep it easy to scroll through.")


def _spearman(xs, ys):
    """Spearman rank correlation, no scipy dependency."""
    n = len(xs)
    if n < 2:
        return None
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def cmd_correlate(args):
    with open(args.template, newline="", encoding="utf-8") as f:
        manual_rows = [r for r in csv.DictReader(f) if r.get("manual_relevance", "").strip() != ""]
    if not manual_rows:
        print("[correlate] No filled-in rows found in the template yet -- fill in the manual_* columns first.")
        return

    manual_score = {}
    for r in manual_rows:
        vals = [float(r[c]) for c in MANUAL_COLS]
        manual_score[(r["model"], r["row_idx"])] = sum(vals) / (3.0 * len(vals))  # normalize to 0-1, same scale as correctness_score

    with open(args.per_row, newline="", encoding="utf-8") as f:
        per_row = list(csv.DictReader(f))

    by_judge = {}
    for r in per_row:
        key = (r["model"], r["row_idx"])
        if key not in manual_score or r["correctness_score"] in ("", None):
            continue
        by_judge.setdefault(r["judge"], []).append((manual_score[key], float(r["correctness_score"])))

    print(f"[correlate] {len(manual_rows)} manually-scored rows matched against each judge:\n")
    print(f"{'Judge':20} {'N':>5} {'Spearman r':>12}")
    for judge, pairs in by_judge.items():
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r = _spearman(xs, ys)
        r_str = f"{r:.3f}" if r is not None else "n/a"
        print(f"{judge:20} {len(pairs):>5} {r_str:>12}")
    print("\n[correlate] Higher Spearman r = that judge's scores track your manual judgment more closely.")
    print("[correlate] Report explanation_judge_accuracy using whichever judge scores highest here --")
    print("  that's your evidence to Prof. Suranga that the judge is measuring something real.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sample")
    s1.add_argument("--per-row", required=True)
    s1.add_argument("--n", type=int, default=40)
    s1.set_defaults(func=cmd_sample)

    s2 = sub.add_parser("correlate")
    s2.add_argument("--template", required=True)
    s2.add_argument("--per-row", required=True)
    s2.set_defaults(func=cmd_correlate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
