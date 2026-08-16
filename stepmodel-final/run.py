"""
run.py — Full pipeline runner for stepmodelv2.

Executes each stage in order:
  1. generate_graphs.py       — build processed_data/{train,test} graph JSONs
  2. build_input_json.py      — build input/train.json and input/test.json
  3. stage1_gnn_train.py      — train GNN + context-fusion classifier
  4. stage2_sft_qwen.py       — supervised fine-tune Qwen with LoRA + graph prefix
  5. stage3_grpo_rl.py        — GRPO reinforcement learning fine-tune
  6. evaluate.py              — evaluate GNN model on test split (fast, deterministic)

Each stage runs as a subprocess so imports/GPU memory are fully isolated
between stages. If a stage fails, the pipeline stops and prints the error.

Usage:
    python run.py                        # run all stages
    python run.py --start-from stage2   # resume from a specific stage
    python run.py --only generate_graphs build_input_json  # run specific stages

Available stage names:
    generate_graphs, build_input_json, stage1, stage2, stage3, evaluate
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Stage definitions ──────────────────────────────────────────────────────────
# Each entry: (name, script_filename, extra_args)
STAGES = [
    # ("generate_graphs",  "generate_graphs.py",   []),
    # ("build_input_json", "build_input_json.py",  []),
    # ("stage1",           "stage1_gnn_train.py",  []),
    # ("stage2",           "stage2_sft_qwen.py",   []),
    # ("stage3",           "stage3_grpo_rl.py",    []),
    # ("evaluate",         "evaluate.py",           []),
    ("baseline_zeroshot", "baseline_llm_eval.py", ["--num_shots", "0"]),
    ("baseline_3shot",   "baseline_llm_eval.py", ["--num_shots", "3"]),
    ("baseline_5shot",   "baseline_llm_eval.py", ["--num_shots", "5"]),
    ("comparison",       "comparison_report.py", []),
]

STAGE_NAMES = [s[0] for s in STAGES]

BASE_DIR = Path(__file__).parent


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_stage(name: str, script: str, extra_args: list[str]) -> bool:
    """
    Run a single stage as a subprocess using the same Python interpreter.
    Returns True on success, False on failure.
    """
    script_path = BASE_DIR / script
    cmd = [sys.executable, str(script_path)] + extra_args

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  STAGE: {name}  →  {script}")
    if extra_args:
        print(f"  ARGS:  {' '.join(extra_args)}")
    print(f"{sep}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n✓  {name} completed in {fmt_duration(elapsed)}")
        return True
    else:
        print(f"\n✗  {name} FAILED (exit code {result.returncode}) after {fmt_duration(elapsed)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the full stepmodelv2 pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available stage names: {', '.join(STAGE_NAMES)}",
    )
    parser.add_argument(
        "--start-from",
        metavar="STAGE",
        choices=STAGE_NAMES,
        default=None,
        help="Skip all stages before this one and resume from here.",
    )
    parser.add_argument(
        "--only",
        metavar="STAGE",
        nargs="+",
        choices=STAGE_NAMES,
        default=None,
        help="Run only the listed stages (in their natural order).",
    )
    args = parser.parse_args()

    # Determine which stages to run
    stages_to_run = list(STAGES)

    if args.only:
        only_set = set(args.only)
        stages_to_run = [s for s in STAGES if s[0] in only_set]
    elif args.start_from:
        start_idx = STAGE_NAMES.index(args.start_from)
        stages_to_run = STAGES[start_idx:]

    if not stages_to_run:
        print("No stages selected. Exiting.")
        sys.exit(0)

    print("\n" + "═" * 70)
    print("  stepmodelv2 — Full Pipeline Runner")
    print("═" * 70)
    print(f"  Stages to run ({len(stages_to_run)}):")
    for i, (name, script, _) in enumerate(stages_to_run, 1):
        print(f"    {i}. {name}  ({script})")
    print()

    pipeline_start = time.time()
    results = {}

    for name, script, extra_args in stages_to_run:
        success = run_stage(name, script, extra_args)
        results[name] = success
        if not success:
            print(f"\n Pipeline aborted at stage '{name}'.")
            print("  Fix the error above and re-run with --start-from", name)
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print("\n" + "═" * 70)
    print("  Pipeline Summary")
    print("═" * 70)
    for name, ok in results.items():
        status = "✓  done" if ok else "✗  FAILED"
        print(f"    {status:10s}  {name}")
    skipped = [s[0] for s in stages_to_run if s[0] not in results]
    for name in skipped:
        print(f"    {'—  skipped':10s}  {name}")
    print(f"\n  Total time: {fmt_duration(total)}")
    print("═" * 70 + "\n")

    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
