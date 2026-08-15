#!/usr/bin/env python3
"""
Master Script for Graph Structure Training and Evaluation Pipeline

This script runs the complete graph structure training and evaluation pipeline:
1. Train graph structure models (individual tasks and multi-task)
2. Test trained models on all graph structure tasks
3. Investigate overall performance impact

Usage:
    python run_graph_structure_pipeline.py --mode full
    python run_graph_structure_pipeline.py --mode train_only
    python run_graph_structure_pipeline.py --mode test_only
    python run_graph_structure_pipeline.py --mode evaluate_only
"""
import os
import sys
import argparse
import subprocess
from typing import List

from config import ROOT, CKPT_DIR

# Default paths
DEFAULT_OUTPUT_DIR = os.path.join(CKPT_DIR, "graph_structure")
DEFAULT_MULTI_TASK_DIR = os.path.join(CKPT_DIR, "graph_structure_multi_task")
DEFAULT_EVAL_DIR = os.path.join(ROOT, "results", "graph_structure_investigation")


def run_command(cmd: List[str], description: str):
    """Run a command and print its output in real-time."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run command with real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nError: {description} failed with return code {process.returncode}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def train_graph_structure_tasks(tasks: List[str], epochs: int = 5, output_dir: str = DEFAULT_OUTPUT_DIR):
    """Train graph structure models for specified tasks."""
    print(f"\n{'=' * 60}")
    print("TRAINING GRAPH STRUCTURE MODELS")
    print(f"{'=' * 60}")
    
    for task in tasks:
        if task == "multi_task":
            task_output_dir = DEFAULT_MULTI_TASK_DIR
        else:
            task_output_dir = os.path.join(DEFAULT_OUTPUT_DIR, task)
        
        cmd = [
            "python", "train_graph_structure.py",
            "--task", task,
            "--epochs", str(epochs),
            "--output_dir", task_output_dir
        ]
        
        success = run_command(cmd, f"Training {task} model")
        if not success:
            print(f"Warning: {task} training failed, continuing with other tasks")


def test_graph_structure_models(checkpoint_dirs: List[str], test_types: List[str]):
    """Test trained models on graph structure tasks."""
    print(f"\n{'=' * 60}")
    print("TESTING GRAPH STRUCTURE MODELS")
    print(f"{'=' * 60}")
    
    for checkpoint_dir in checkpoint_dirs:
        if not os.path.exists(checkpoint_dir):
            print(f"Warning: Checkpoint not found: {checkpoint_dir}, skipping")
            continue
        
        for test_type in test_types:
            cmd = [
                "python", "test_graph_prefix_adapter.py",
                "--mode", "trained",
                "--checkpoint", checkpoint_dir,
                "--test_type", test_type
            ]
            
            success = run_command(cmd, f"Testing {test_type} with {os.path.basename(checkpoint_dir)}")
            if not success:
                print(f"Warning: {test_type} test failed for {checkpoint_dir}")


def evaluate_performance(baseline_checkpoint: str, graph_structure_checkpoint: str, 
                        multi_task_checkpoint: str, output_dir: str = DEFAULT_EVAL_DIR):
    """Run performance investigation."""
    print(f"\n{'=' * 60}")
    print("PERFORMANCE INVESTIGATION")
    print(f"{'=' * 60}")
    
    cmd = [
        "python", "investigate_performance.py",
        "--output_dir", output_dir
    ]
    
    if baseline_checkpoint:
        cmd.extend(["--baseline_checkpoint", baseline_checkpoint])
    if graph_structure_checkpoint:
        cmd.extend(["--graph_structure_checkpoint", graph_structure_checkpoint])
    if multi_task_checkpoint:
        cmd.extend(["--multi_task_checkpoint", multi_task_checkpoint])
    
    success = run_command(cmd, "Performance investigation")
    if not success:
        print("Warning: Performance investigation failed")


def main():
    parser = argparse.ArgumentParser(description="Master script for graph structure pipeline")
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "train_only", "test_only", "evaluate_only"],
                       help="Pipeline mode: full (train+test+evaluate), train_only, test_only, evaluate_only")
    parser.add_argument("--tasks", type=str, nargs="+", 
                       default=["adjacency", "edge_type", "path", "multi_task"],
                       help="Graph structure tasks to train (node_type skipped due to missing data)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--test_types", type=str, nargs="+",
                       default=["adjacency", "node_type", "edge_type", "path", "step_prediction"],
                       help="Test types to run")
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                       help="Path to baseline Stage 2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for trained models")
    parser.add_argument("--eval_dir", type=str, default=DEFAULT_EVAL_DIR,
                       help="Output directory for evaluation results")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRAPH STRUCTURE TRAINING AND EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Tasks to train: {args.tasks}")
    print(f"Epochs: {args.epochs}")
    print(f"Test types: {args.test_types}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evaluation directory: {args.eval_dir}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # Step 1: Train graph structure models
    if args.mode in ["full", "train_only"]:
        train_graph_structure_tasks(args.tasks, args.epochs, args.output_dir)
    
    # Step 2: Test graph structure models
    if args.mode in ["full", "test_only"]:
        # Collect checkpoint directories
        checkpoint_dirs = []
        for task in args.tasks:
            if task == "multi_task":
                checkpoint_dir = DEFAULT_MULTI_TASK_DIR
            else:
                checkpoint_dir = os.path.join(args.output_dir, task)
            checkpoint_dirs.append(checkpoint_dir)
        
        test_graph_structure_models(checkpoint_dirs, args.test_types)
    
    # Step 3: Evaluate performance
    if args.mode in ["full", "evaluate_only"]:
        # Determine checkpoint paths
        graph_structure_checkpoint = os.path.join(args.output_dir, "adjacency")
        multi_task_checkpoint = DEFAULT_MULTI_TASK_DIR
        
        evaluate_performance(
            args.baseline_checkpoint,
            graph_structure_checkpoint,
            multi_task_checkpoint,
            args.eval_dir
        )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  Trained models: {args.output_dir}")
    print(f"  Evaluation results: {args.eval_dir}")


if __name__ == "__main__":
    main()
