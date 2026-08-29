"""
run_experiment.py
=================
Orchestration script to run the complete text-only experiment pipeline.

This script runs:
1. Data preparation (build input JSON files)
2. Stage 2 training (LLM SFT without graph conditioning)
3. Stage 3 training (GRPO RL without graph conditioning)
4. Evaluation (Stage 2 and Stage 3)

Usage:
    python run_experiment.py
    python run_experiment.py --skip-data  # Skip data prep if already done
    python run_experiment.py --skip-stage2  # Skip Stage 2 if already trained
    python run_experiment.py --skip-stage3  # Skip Stage 3 if already trained
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {cwd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n✅ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 training")
    parser.add_argument("--skip-stage3", action="store_true", help="Skip Stage 3 training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()
    
    # Get paths
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    data_prep_dir = script_dir / "data_prep"
    training_dir = script_dir / "training"
    eval_dir = script_dir / "eval"
    
    print(f"Experiment root: {script_dir}")
    print(f"Main pipeline root: {root_dir}")
    
    # Step 1: Data preparation
    if not args.skip_data:
        run_command(
            ["python", "build_input_json.py"],
            cwd=data_prep_dir,
            description="Data preparation (build input JSON files)"
        )
    else:
        print("\n⏭️  Skipping data preparation (--skip-data flag set)")
    
    # Step 2: Stage 2 training
    if not args.skip_stage2:
        run_command(
            ["python", "stage2_sft_qwen.py"],
            cwd=training_dir,
            description="Stage 2 training (LLM SFT without graph conditioning)"
        )
    else:
        print("\n⏭️  Skipping Stage 2 training (--skip-stage2 flag set)")
    
    # Step 3: Stage 3 training
    if not args.skip_stage3:
        run_command(
            ["python", "stage3_grpo_rl.py"],
            cwd=training_dir,
            description="Stage 3 training (GRPO RL without graph conditioning)"
        )
    else:
        print("\n⏭️  Skipping Stage 3 training (--skip-stage3 flag set)")
    
    # Step 4: Evaluation
    if not args.skip_eval:
        run_command(
            ["python", "evaluate.py", "--stage", "all"],
            cwd=eval_dir,
            description="Evaluation (Stage 2 and Stage 3)"
        )
    else:
        print("\n⏭️  Skipping evaluation (--skip-eval flag set)")
    
    print(f"\n{'='*60}")
    print("🎉 Experiment pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {script_dir / 'output'}")
    print(f"Checkpoints saved to: {script_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
