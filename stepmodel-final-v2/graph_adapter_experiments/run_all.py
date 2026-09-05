"""
run_all.py
==========
The one script to run. Chains the three steps in order:

    1. build_probe_tasks.py            (build the task JSONLs, fast, no GPU)
    2. train_adapter.py                (train GNN + adapter [+ optional LoRA])
    3. eval_right_vs_wrong_graph.py    (the right-graph vs wrong-graph report)

Every flag below just forwards to the corresponding underlying script, so
`--help` on any of them still works if you want to run a step by hand later
(e.g. to re-run just the eval against a checkpoint you already trained).

Requires STANDALONE_INPUT_TRAIN_JSON / STANDALONE_INPUT_TEST_JSON to be set
(or the default `<repo_root>/input/{train,test}.json` to exist) — see
standalone_config.py. Only the "graph" field of those files is read.

Usage:
    python run_all.py
    python run_all.py --steps 3000 --use_lora
    python run_all.py --skip_build --skip_train --checkpoint standalone_checkpoints/run1/best
"""
import argparse
import os
import subprocess
import sys

from standalone_config import TASKS_DIR, CKPT_DIR, TRAIN_STEPS, TRAIN_EVAL_EVERY, TRAIN_GRAD_ACCUM, TRAIN_LR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=THIS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(CKPT_DIR, "run1"),
                     help="where train_adapter.py saves checkpoints")
    ap.add_argument("--model_name", default=None, help="override standalone_config.LLM_MODEL_NAME")
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--grad_accum", type=int, default=TRAIN_GRAD_ACCUM)
    ap.add_argument("--lr", type=float, default=TRAIN_LR)
    ap.add_argument("--eval_every", type=int, default=TRAIN_EVAL_EVERY)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--real_frac", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.2, help="build_probe_tasks.py machine-level split")
    ap.add_argument("--max_eval_items", type=int, default=150)
    ap.add_argument("--skip_build", action="store_true", help="reuse existing standalone_tasks/*.jsonl")
    ap.add_argument("--skip_train", action="store_true", help="go straight to eval on --checkpoint")
    ap.add_argument("--checkpoint", default=None,
                     help="only needed with --skip_train; defaults to <out_dir>/best")
    args = ap.parse_args()

    py = sys.executable

    if not args.skip_build:
        run([py, "build_probe_tasks.py", "--val_frac", str(args.val_frac)])
    else:
        print("Skipping build_probe_tasks.py (--skip_build) — reusing existing standalone_tasks/*.jsonl")

    if not args.skip_train:
        cmd = [
            py, "train_adapter.py",
            "--out_dir", args.out_dir,
            "--steps", str(args.steps),
            "--grad_accum", str(args.grad_accum),
            "--lr", str(args.lr),
            "--eval_every", str(args.eval_every),
            "--real_frac", str(args.real_frac),
        ]
        if args.model_name:
            cmd += ["--model_name", args.model_name]
        if args.use_lora:
            cmd += ["--use_lora"]
        run(cmd)
        checkpoint = args.checkpoint or os.path.join(args.out_dir, "best")
    else:
        if not args.checkpoint:
            raise SystemExit("--skip_train requires --checkpoint <path to a saved checkpoint dir>")
        checkpoint = args.checkpoint
        print(f"Skipping train_adapter.py (--skip_train) — evaluating {checkpoint}")

    eval_cmd = [py, "eval_right_vs_wrong_graph.py", "--checkpoint", checkpoint,
                "--max_items", str(args.max_eval_items)]
    if args.model_name:
        eval_cmd += ["--model_name", args.model_name]
    run(eval_cmd)


if __name__ == "__main__":
    main()
