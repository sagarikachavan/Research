#!/usr/bin/env python3
"""
Verify all training checkpoints exist and are complete before pushing to GitHub.

This script checks that:
1. Stage 1 GNN checkpoint exists with required keys
2. Stage 2 LoRA adapter exists with required files (adapter_model.safetensors)
3. Stage 3 GRPO adapter exists with required files (if trained)

Run this before `git push` to ensure all trained weights are committed.
"""
import os
import sys
from pathlib import Path

# Add core directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "core"))

from config import (
    ROOT,
    STAGE1_CKPT,
    STAGE2_ADAPTER_DIR,
    STAGE3_ADAPTER_DIR,
)


def check_stage1_checkpoint():
    """Verify Stage 1 GNN checkpoint exists and has required keys."""
    print(f"\n[1/3] Checking Stage 1 checkpoint: {STAGE1_CKPT}")
    
    if not os.path.exists(STAGE1_CKPT):
        print(f"  ❌ FAIL: Stage 1 checkpoint not found!")
        print(f"     Run: python training/stage1_gnn_train.py")
        return False
    
    # Try to load and verify structure
    try:
        import torch
        ckpt = torch.load(STAGE1_CKPT, map_location="cpu", weights_only=False)
        
        if not isinstance(ckpt, dict):
            print(f"  ❌ FAIL: Checkpoint is not a dictionary")
            return False
        
        required_keys = ["model_state_dict", "best_epoch", "best_score"]
        missing = [k for k in required_keys if k not in ckpt]
        if missing:
            print(f"  ❌ FAIL: Missing required keys: {missing}")
            return False
        
        print(f"  ✓ Stage 1 checkpoint valid (epoch={ckpt['best_epoch']}, score={ckpt['best_score']:.4f})")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: Could not load checkpoint: {e}")
        return False


def check_stage2_checkpoint():
    """Verify Stage 2 LoRA adapter exists with required files."""
    print(f"\n[2/3] Checking Stage 2 adapter: {STAGE2_ADAPTER_DIR}")
    
    if not os.path.isdir(STAGE2_ADAPTER_DIR):
        print(f"  ❌ FAIL: Stage 2 adapter directory not found!")
        print(f"     Run: python training/stage2_sft_qwen.py")
        return False
    
    # Check for required files
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",  # CRITICAL: this was missing in your case
        "graph_adapter.pt",
        "tokenizer.json",
    ]
    
    missing = []
    for f in required_files:
        path = os.path.join(STAGE2_ADAPTER_DIR, f)
        if not os.path.exists(path):
            missing.append(f)
    
    if missing:
        print(f"  ❌ FAIL: Missing required files: {missing}")
        print(f"     The most critical missing file is adapter_model.safetensors")
        print(f"     This indicates Stage 2 training did not complete successfully.")
        print(f"     Re-run: python training/stage2_sft_qwen.py")
        return False
    
    # Verify safetensors is not empty
    safetensors_path = os.path.join(STAGE2_ADAPTER_DIR, "adapter_model.safetensors")
    if os.path.getsize(safetensors_path) < 1000:  # Less than 1KB is suspicious
        print(f"  ❌ FAIL: adapter_model.safetensors is too small ({os.path.getsize(safetensors_path)} bytes)")
        return False
    
    print(f"  ✓ Stage 2 adapter complete with all required files")
    return True


def check_stage3_checkpoint():
    """Verify Stage 3 GRPO adapter exists (optional - only if trained)."""
    print(f"\n[3/3] Checking Stage 3 adapter: {STAGE3_ADAPTER_DIR}")
    
    if not os.path.isdir(STAGE3_ADAPTER_DIR):
        print(f"  ⚠️  Stage 3 not trained yet (optional)")
        return True  # Stage 3 is optional
    
    # Check for required files in main directory or best/ subdirectory
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "graph_adapter.pt",
        "value_head.pt",
        "tokenizer.json",
    ]
    
    # Check both main directory and best/ subdirectory
    check_dirs = [STAGE3_ADAPTER_DIR]
    best_dir = os.path.join(STAGE3_ADAPTER_DIR, "best")
    if os.path.isdir(best_dir):
        check_dirs.append(best_dir)
    
    for check_dir in check_dirs:
        missing = []
        for f in required_files:
            path = os.path.join(check_dir, f)
            if not os.path.exists(path):
                missing.append(f)
        
        if not missing:
            print(f"  ✓ Stage 3 adapter complete in {os.path.basename(check_dir)}/")
            return True
    
    print(f"  ❌ FAIL: Stage 3 directory exists but missing required files")
    print(f"     Checked: {check_dirs}")
    print(f"     Missing files: {required_files}")
    return False


def main():
    """Run all checkpoint checks."""
    print("=" * 60)
    print("CHECKPOINT VERIFICATION SCRIPT")
    print("Run this before git push to ensure all weights are committed")
    print("=" * 60)
    
    results = {
        "stage1": check_stage1_checkpoint(),
        "stage2": check_stage2_checkpoint(),
        "stage3": check_stage3_checkpoint(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for stage, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {stage.upper():8s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All checkpoints verified. Safe to push to GitHub.")
        return 0
    else:
        print("\n❌ Some checkpoints are missing or incomplete.")
        print("   Train the missing stages before pushing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
