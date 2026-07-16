# Verification Checklist for stepmodel-new

Run through this checklist to verify all fixes are working correctly.

## ✓ Pre-Flight Checks

- [ ] You are in the correct directory: `/path/to/stepmodel-new`
- [ ] `train_gnn_rl.py` exists and has recent modifications
- [ ] `config.json` exists and is valid JSON
- [ ] Data files exist: `data/training_data.csv` and `data/test_data.csv`
- [ ] Graph embeddings exist: `embeddings_data/train/` and `embeddings_data/test/`

```bash
# Verify files exist
ls train_gnn_rl.py config.json data/*.csv
ls -d embeddings_data/train embeddings_data/test

# Check Python packages
python -c "import torch, transformers; print('✓ Dependencies OK')"
```

## ✓ Code Fixes Verification

Run these commands to verify the fixes are in place:

### 1. ZeroDivisionError Fix
```bash
grep -n "max(num_samples, 1)" train_gnn_rl.py
# Should show multiple matches around lines 1406-1409
```

Expected output:
```
1406:        avg_epoch_loss = total_loss / max(num_samples, 1)
1408:        avg_epoch_step_loss = total_step_loss / max(num_samples, 1)
1409:        avg_epoch_mcp_loss = total_mcp_loss / max(num_samples, 1)
```

### 2. Progress Logging Safety
```bash
grep -n "num_samples > 0 and num_samples %" train_gnn_rl.py
# Should show match around line 1400
```

Expected output:
```
1400:                if num_samples > 0 and num_samples % 50 == 0:
```

### 3. Phase Announcements
```bash
grep -n "PHASE 1: SUPERVISED WARMUP TRAINING" train_gnn_rl.py
grep -n "PHASE 2: GRPO REINFORCEMENT LEARNING" train_gnn_rl.py
# Both should show matches
```

### 4. Configuration Summary
```bash
grep -n "TRAINING CONFIGURATION" train_gnn_rl.py
# Should show match
```

### 5. Warning Messages
```bash
grep -n "WARNING: All samples" train_gnn_rl.py
# Should show match around line 1412
```

## ✓ Test Run (Dry Run)

Perform a quick test run to verify everything works:

```bash
# Create a minimal test config
cat > config_verify.json << 'EOF'
{
  "paths": {
    "data_dir": "data",
    "output_dir": "output_verify",
    "log_dir": "runs_verify"
  },
  "model": {
    "llm_name": "microsoft/phi-2",
    "trust_remote_code": true,
    "torch_dtype": "float16",
    "load_in_4bit": false,
    "use_lora": false,
    "pooling_strategy": "hybrid"
  },
  "training": {
    "num_supervised_epochs": 1,
    "num_grpo_epochs": 0,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "num_warmup_steps": 10,
    "num_generations_per_sample": 2,
    "clip_eps": 0.2,
    "generate_max_new_tokens": 512,
    "generate_temperature": 0.9,
    "generate_top_p": 0.95,
    "patience": 3,
    "validation_split": 0.1,
    "max_seq_length": 512,
    "step_loss_weight": 4.0,
    "mcp_loss_weight": 1.0,
    "explanation_loss_weight": 0.0,
    "seed": 42
  }
}
EOF

# Run verification test
python train_gnn_rl.py --config config_verify.json 2>&1 | tee verify_output.log
```

### Expected Output Sections:

1. **Training Configuration Section:**
```
==============================================================
TRAINING CONFIGURATION
==============================================================
Model: microsoft/phi-2
Device: cuda
...
```

2. **Phase 1 Announcement:**
```
==============================================================
PHASE 1: SUPERVISED WARMUP TRAINING
==============================================================
```

3. **Progress Updates:**
```
[Supervised] Epoch 1/1, Sample 50/1603, Avg Loss: X.XXXX, ...
```

4. **Epoch Summary:**
```
==============================================================
Supervised Epoch 1/1 Complete!
==============================================================
  Training Metrics:
    Avg Loss: X.XXXX
    ...
```

5. **No Crashes:**
- No ZeroDivisionError
- No unhandled exceptions
- Clean completion

## ✓ Output Verification

After the test run, verify the outputs:

```bash
# Check output directory created
ls -la output_verify/
# Should contain: best_supervised_checkpoint.pt, supervised_checkpoint_epoch_1.pt

# Check logs created
ls -la runs_verify/
# Should contain TensorBoard log directory

# Check verification log
tail -n 50 verify_output.log
# Should show training completion message
```

## ✓ Helper Scripts

Test the helper script:

```bash
# Make executable (if not already)
chmod +x run_training.sh

# Test with verification config
./run_training.sh config_verify.json
```

Should produce:
- Clean output to terminal
- Log file in `logs/training_TIMESTAMP.log`
- Exit cleanly

## ✓ Documentation

Verify all documentation files exist:

```bash
ls -1 *.md *.sh
```

Should show:
```
FIXES_APPLIED.md
QUICK_START.md
README_UPDATES.md
VERIFICATION_CHECKLIST.md
run_training.sh
```

## ✓ Clean Up Test Files

After successful verification:

```bash
# Remove test outputs
rm -rf output_verify runs_verify
rm config_verify.json verify_output.log

# Or keep for reference
mkdir -p verification_results
mv output_verify runs_verify verify_output.log verification_results/
```

## ✓ Full Training Test

If everything above passes, run a full training:

```bash
# Use your actual config
./run_training.sh config.json

# Or with custom logging
python train_gnn_rl.py --config config.json 2>&1 | tee logs/full_training.log
```

## Common Issues During Verification

### Issue: "grep: No such file"
**Solution:** You're not in the stepmodel-new directory
```bash
cd /path/to/stepmodel-new
```

### Issue: "python: command not found"
**Solution:** Use python3 or activate your environment
```bash
python3 train_gnn_rl.py --config config.json
# or
conda activate your_env
```

### Issue: "torch not found"
**Solution:** Install dependencies
```bash
pip install torch transformers tensorboard
```

### Issue: "Config not found"
**Solution:** Create or check config.json path
```bash
ls config.json
# If missing, copy from stepmodel directory
```

### Issue: Still getting errors
**Solution:** Check the log file for details
```bash
cat verify_output.log | grep -i error
cat verify_output.log | grep -i traceback
```

## Final Checklist

Before running production training, confirm:

- [ ] All grep commands above found expected patterns
- [ ] Test run completed without crashes
- [ ] Output directory contains checkpoints
- [ ] TensorBoard logs were created
- [ ] Helper script runs successfully
- [ ] All documentation files are present
- [ ] You understand the output format
- [ ] You know how to monitor training
- [ ] You know where checkpoints are saved
- [ ] You reviewed QUICK_START.md

## Success Criteria

✓ **All checks passed** - You're ready to run full training!

```bash
./run_training.sh config.json
```

Monitor with:
- Terminal output (real-time)
- TensorBoard: `tensorboard --logdir=runs`
- Log file: `tail -f logs/training_TIMESTAMP.log`
- GPU usage: `watch -n 1 nvidia-smi`

## Troubleshooting

If any check fails:
1. Re-read the error message carefully
2. Check FIXES_APPLIED.md for details
3. Review QUICK_START.md for usage examples
4. Verify you're using the updated train_gnn_rl.py
5. Check that all dependencies are installed
6. Try the minimal test config first

## Report Issues

If you find bugs not covered by these fixes:
1. Save the error log
2. Note the configuration used
3. Note which step failed
4. Check if it's a data or code issue
5. Document and report

---

**Version:** 2026-07-16  
**Status:** Ready for production use after passing all checks
