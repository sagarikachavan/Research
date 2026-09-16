"""Run the Stage-1 ablation matrix and emit the comparison table.

WHY THIS EXISTS
---------------
Stage 1 accumulated ~15 mechanisms on ~1.9k training rows without any of them
being ablated. The single most important missing experiment is not "add
another loss" -- it is the table that says WHICH INFORMATION SOURCE ACTUALLY
CARRIES THE PREDICTION:

    text only   |  graph only  |  fusion

If text-only ~= fusion, the graph is decorative for that head and the research
claim has to be stated accordingly. If fusion clearly beats both, the graph
contribution is proven. Either outcome is a publishable result; not knowing is
not.

Each configuration is a SEPARATE PROCESS with different environment flags, so
module-level config reads cannot leak between runs. Every row is one full
Stage-1 K-fold training run, which is expensive -- use --configs to run a
subset, and --dry-run to print the plan first.

USAGE
    python eval/ablation_table.py --dry-run
    python eval/ablation_table.py --configs text_only,graph_only,fusion
    python eval/ablation_table.py --all
    python eval/ablation_table.py --report-only     # rebuild table from saved JSON
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "core"))

RESULTS_PATH = _ROOT / "output" / "ablation_results.json"
STAGE1 = _ROOT / "training" / "stage1_gnn_train.py"

# name -> (description, env overrides)
# Ordered so the three rows that settle the graph question come first.
CONFIGS: dict[str, tuple[str, dict[str, str]]] = {
    "fusion": (
        "Current architecture (graph + text, cross-attention fusion)",
        {},
    ),
    "text_only": (
        "Graph ablated entirely — does the graph add anything?",
        {"STAGE1_ABLATE_GRAPH": "1"},
    ),
    "graph_only": (
        "Text ablated entirely — is the graph informative on its own?",
        {"STAGE1_ABLATE_TEXT": "1"},
    ),
    "simple_fusion": (
        "[t, g, t*g, |t-g|] instead of cross-attention",
        {"STAGE1_SIMPLE_FUSION": "1"},
    ),
    "no_mixup": (
        "Manifold mixup off",
        {"STAGE1_ABLATE_MIXUP": "1"},
    ),
    "no_supcon": (
        "SupCon off",
        {"STAGE1_ABLATE_SUPCON": "1"},
    ),
    "natural_sampling": (
        "Natural sampling (rebalance only in the decoupled classifier stage)",
        {"STAGE1_NATURAL_SAMPLING": "1"},
    ),
    "separate_towers": (
        "Independent Step/MCP semantic CNNs (Pen-Strategist sec 4.2.2)",
        {"STAGE1_SEPARATE_TOWERS": "1"},
    ),
    "no_prototypes": (
        "Label prototypes off",
        {"STAGE1_USE_PROTOTYPES": "0"},
    ),
    "no_step_cond_mcp": (
        "Step-conditioned MCP off",
        {"STAGE1_STEP_COND_MCP": "0"},
    ),
    "no_phase_head": (
        "Auxiliary phase head off",
        {"STAGE1_PHASE_LOSS_WEIGHT": "0"},
    ),
    "no_structured_smoothing": (
        "Uniform label smoothing instead of similarity-structured",
        {"STAGE1_USE_STRUCTURED_SMOOTHING": "0"},
    ),
    "drop_dead_classes": (
        "9-way softmax (dead class removed from normalization)",
        {"STAGE1_DROP_DEAD_CLASSES": "1"},
    ),
    "gine": (
        "GINE instead of GATv2",
        {"STAGE1_GNN_TYPE": "gine"},
    ),
}

METRICS = [
    "step_accuracy", "step_macro_f1",
    "mcp_samples_f1", "mcp_micro_f1", "mcp_subset_accuracy",
]


def load_results() -> dict:
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_results(res: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=2))


def parse_stage1_output(text: str) -> dict:
    """Pull the PRIMARY test block's metrics out of a Stage-1 run's stdout.

    Prefers the blended/primary block; falls back to the ensemble block. Reads
    the LAST occurrence so a later, more-final block wins.
    """
    out: dict[str, float] = {}
    lines = text.splitlines()
    primary_at = None
    for i, ln in enumerate(lines):
        if "[PRIMARY]" in ln:
            primary_at = i
    if primary_at is None:
        for i, ln in enumerate(lines):
            if "FOLD ENSEMBLE" in ln:
                primary_at = i
    if primary_at is None:
        return out
    for ln in lines[primary_at: primary_at + 20]:
        for m in METRICS + ["combined_score"]:
            if ln.strip().startswith(m):
                try:
                    out[m] = float(ln.split(":")[-1].strip())
                except ValueError:
                    pass
    return out


def run_config(name: str, env_over: dict, timeout_s: int) -> dict:
    env = dict(os.environ)
    env.update(env_over)
    env.setdefault("PYTHONUNBUFFERED", "1")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(STAGE1)],
        capture_output=True, text=True, env=env, cwd=str(_ROOT),
        timeout=timeout_s,
    )
    dur = time.time() - t0
    metrics = parse_stage1_output(proc.stdout)
    log_path = _ROOT / "output" / f"ablation_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n===== STDERR =====\n" + proc.stderr)
    return {
        "env": env_over,
        "metrics": metrics,
        "returncode": proc.returncode,
        "seconds": round(dur, 1),
        "log": str(log_path.relative_to(_ROOT)),
        "ok": proc.returncode == 0 and bool(metrics),
    }


def render_table(res: dict) -> str:
    rows = [n for n in CONFIGS if n in res and res[n].get("metrics")]
    if not rows:
        return "(no completed configurations yet)"
    head = f"{'config':<26}" + "".join(f"{m.replace('step_','').replace('mcp_','mcp '):>15}" for m in METRICS)
    sep = "-" * len(head)
    base = res.get("fusion", {}).get("metrics", {})
    out = [head, sep]
    for n in rows:
        m = res[n]["metrics"]
        line = f"{n:<26}"
        for k in METRICS:
            v = m.get(k)
            line += f"{v:>15.4f}" if isinstance(v, float) else f"{'--':>15}"
        out.append(line)
        if base and n != "fusion":
            d = f"{'  vs fusion':<26}"
            for k in METRICS:
                v, b = m.get(k), base.get(k)
                d += f"{v - b:>+15.4f}" if isinstance(v, float) and isinstance(b, float) else f"{'':>15}"
            out.append(d)
    out += ["", "READING THIS TABLE:",
            "  text_only ~= fusion   -> the graph is not contributing to that metric",
            "  graph_only >> chance  -> the graph is informative on its own",
            "  fusion > both         -> the graph contribution is proven",
            "",
            "Paper reference (Pen-Strategist Step Model, arXiv 2605.04499):",
            "  step_accuracy 0.8287 | mcp samples-F1 0.64 | mcp subset acc 0.4888"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", help="comma-separated config names")
    ap.add_argument("--all", action="store_true", help="run every configuration")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--report-only", action="store_true", help="rebuild the table from saved results")
    ap.add_argument("--timeout", type=int, default=6 * 60 * 60, help="per-config timeout (s)")
    ap.add_argument("--force", action="store_true", help="re-run configs that already have results")
    args = ap.parse_args()

    res = load_results()

    if args.report_only:
        print(render_table(res))
        return

    if args.all:
        names = list(CONFIGS)
    elif args.configs:
        names = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        names = ["fusion", "text_only", "graph_only"]
        print("No --configs given; defaulting to the three that settle the graph question.\n")

    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}\nAvailable: {list(CONFIGS)}")

    todo = [n for n in names if args.force or n not in res or not res[n].get("ok")]
    skip = [n for n in names if n not in todo]

    print(f"Plan: {len(todo)} configuration(s) to run"
          + (f", {len(skip)} already done (use --force to redo): {skip}" if skip else ""))
    for n in todo:
        desc, env = CONFIGS[n]
        print(f"  {n:<26} {desc}")
        print(f"  {'':<26} env: {env or '(defaults)'}")
    print("\nEach row is a full Stage-1 K-fold run. Budget hours, not minutes.")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    for n in todo:
        desc, env = CONFIGS[n]
        print(f"\n=== {n} — {desc} ===", flush=True)
        try:
            r = run_config(n, env, args.timeout)
        except subprocess.TimeoutExpired:
            r = {"env": env, "metrics": {}, "returncode": -1,
                 "seconds": args.timeout, "ok": False, "log": None}
            print(f"  TIMEOUT after {args.timeout}s")
        res[n] = r
        save_results(res)
        if r["ok"]:
            m = r["metrics"]
            print(f"  done in {r['seconds']}s  "
                  f"step={m.get('step_accuracy', float('nan')):.4f}  "
                  f"mcp_samples_f1={m.get('mcp_samples_f1', float('nan')):.4f}")
        else:
            print(f"  FAILED (rc={r['returncode']}) — see {r.get('log')}")

    print("\n" + render_table(res))
    print(f"\nResults JSON: {RESULTS_PATH.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
