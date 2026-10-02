"""
Run the Stage-1 modality ablation: for each seed, train
    fusion      -- text + graph (the real model, retrained under identical
                   code/config so the comparison is controlled)
    text_only   -- graph tower output zeroed
    graph_only  -- text tower output zeroed
as three separate subprocesses of training/stage1_gnn_train.py.

Why this exists: the claim "step depends on the strategy text, MCP depends on
the graph" needs a double dissociation -- remove the graph and step barely
moves while MCP drops; remove the text and step collapses while MCP drops less.
Training each variant from scratch (rather than zeroing an input at test time)
rules out the objection that the fused model simply co-adapted to both towers.

Every run is tagged <mode>_seed<seed> and writes ONLY under
    checkpoints/ablations/<tag>/   and   output/ablations/<tag>/
(see config.STAGE1_RUN_TAG), so the real Stage-1 checkpoint and
output/stage1.csv that Stage 2/3 and the comparison report use are never
touched. Runs whose final checkpoint already exists are skipped, so an
interrupted sweep resumes where it stopped.

The seed controls weight init AND the machine-grouped train/val split; the
test set is fixed. So the across-seed spread includes split variance.

Usage (on the GPU machine):
    python training/run_stage1_ablation.py                    # 3 modes x seeds 42 1 2
    python training/run_stage1_ablation.py --seeds 42         # quick single-seed pass
    python training/run_stage1_ablation.py --modes text_only graph_only
    python training/run_stage1_ablation.py --dry-run          # just list what would run

Then:  python eval/aggregate_stage1_ablation.py
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(ROOT, "training", "stage1_gnn_train.py")
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(ROOT, "checkpoints"))

MODE_ENV = {
    "fusion": {},
    "text_only": {"STAGE1_ABLATE_GRAPH": "1"},   # graph removed
    "graph_only": {"STAGE1_ABLATE_TEXT": "1"},   # text removed
}


def final_ckpt(tag):
    return os.path.join(CKPT_DIR, "ablations", tag, "stage1_gnn_classifier.pt")


def fmt(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--modes", nargs="+", choices=list(MODE_ENV), default=list(MODE_ENV))
    ap.add_argument("--force", action="store_true", help="retrain even if the run's checkpoint exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Seed-major order: a partially finished sweep still has complete
    # (fusion, text_only, graph_only) triples for the earlier seeds.
    plan = [(mode, seed) for seed in args.seeds for mode in args.modes]
    print(f"[ablation] {len(plan)} run(s) planned: "
          + ", ".join(f"{m}_seed{s}" for m, s in plan))

    results = []
    t_all = time.time()
    for mode, seed in plan:
        tag = f"{mode}_seed{seed}"
        if os.path.exists(final_ckpt(tag)) and not args.force:
            print(f"[ablation] {tag}: already trained -> skipping")
            results.append((tag, "skipped"))
            continue
        if args.dry_run:
            print(f"[ablation] {tag}: would train (dry run)")
            results.append((tag, "dry-run"))
            continue

        env = os.environ.copy()
        for k in ("STAGE1_ABLATE_GRAPH", "STAGE1_ABLATE_TEXT", "STAGE1_CKPT"):
            env.pop(k, None)               # never inherit a stray flag into a run
        env.update(MODE_ENV[mode])
        env["RANDOM_SEED"] = str(seed)
        env["STAGE1_RUN_TAG"] = tag

        print(f"\n{'=' * 70}\n[ablation] training {tag}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, TRAIN_SCRIPT], cwd=ROOT, env=env).returncode
        dt = time.time() - t0
        if rc != 0:
            print(f"[ablation] {tag} FAILED (exit {rc}) after {fmt(dt)} -- stopping. "
                  f"Fix the error and re-run; finished runs are skipped.")
            results.append((tag, f"FAILED({rc})"))
            break
        print(f"[ablation] {tag} done in {fmt(dt)}")
        results.append((tag, f"done {fmt(dt)}"))

    print(f"\n[ablation] summary (total {fmt(time.time() - t_all)}):")
    for tag, status in results:
        print(f"    {tag:22s} {status}")
    if all(not r[1].startswith("FAILED") for r in results) and not args.dry_run:
        print("[ablation] next: python eval/aggregate_stage1_ablation.py")
    sys.exit(1 if any(r[1].startswith("FAILED") for r in results) else 0)


if __name__ == "__main__":
    main()
