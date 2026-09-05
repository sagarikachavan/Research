"""
Ensemble training for Stage 1 GNN with multiple random seeds.
Trains multiple models and saves checkpoints for ensemble evaluation.
"""
import os
import sys
import subprocess
import random
import numpy as np
import torch
from pathlib import Path

# Ensemble configuration
ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]  # 5 different random seeds
ENSEMBLE_CKPT_DIR = "checkpoints/ensemble_stage1"

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_single_model(seed):
    """Train a single Stage 1 model with given seed."""
    print(f"\n{'='*60}")
    print(f"Training ensemble model with seed {seed}")
    print(f"{'='*60}\n")
    
    # Modify config to use ensemble checkpoint directory
    os.environ['STAGE1_CKPT'] = f"{ENSEMBLE_CKPT_DIR}/seed_{seed}.pt"
    
    # Set random seed before importing (affects all imports)
    import numpy as np
    import torch
    set_random_seed(seed)
    
    # Import and run training
    from stage1_gnn_train import main
    
    try:
        main()
        print(f"✓ Successfully trained model with seed {seed}")
        return True
    except Exception as e:
        print(f"✗ Failed to train model with seed {seed}: {e}")
        return False

def main():
    """Train all ensemble models."""
    # Create ensemble checkpoint directory
    Path(ENSEMBLE_CKPT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"ENSEMBLE TRAINING: Stage 1 GNN")
    print(f"{'='*60}")
    print(f"Seeds: {ENSEMBLE_SEEDS}")
    print(f"Checkpoint directory: {ENSEMBLE_CKPT_DIR}")
    print(f"Number of models: {len(ENSEMBLE_SEEDS)}")
    print(f"{'='*60}\n")
    
    # Train each model
    results = []
    for seed in ENSEMBLE_SEEDS:
        success = train_single_model(seed)
        results.append((seed, success))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*60}")
    for seed, success in results:
        status = "✓" if success else "✗"
        print(f"  Seed {seed}: {status}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\nSuccessful: {successful}/{len(results)}")
    print(f"Checkpoints saved to: {ENSEMBLE_CKPT_DIR}/")
    print(f"{'='*60}\n")
    
    if successful == len(results):
        print("✓ All ensemble models trained successfully!")
        print("Next step: Run ensemble evaluation with:")
        print("  python evaluate_ensemble_stage1.py")
    else:
        print("✗ Some models failed. Check logs above.")

if __name__ == "__main__":
    main()
