# Quick Start Guide for stepmodel-new

## Running Training

### Option 1: Using the Helper Script (Recommended)
```bash
cd /path/to/stepmodel-new
./run_training.sh config.json
```

This will:
- Run training with the specified config
- Save logs to `logs/training_TIMESTAMP.log`
- Show output in terminal in real-time
- Save output to log file simultaneously

### Option 2: Direct Python Execution
```bash
cd /path/to/stepmodel-new
python train_gnn_rl.py --config config.json
```

### Option 3: With Custom Logging
```bash
python train_gnn_rl.py --config config.json 2>&1 | tee logs/my_training.log
```

## What You'll See

### 1. Configuration Summary
```
==============================================================
TRAINING CONFIGURATION
==============================================================
Model: microsoft/phi-2
Device: cuda
Pooling Strategy: hybrid
...
```

### 2. Phase 1: Supervised Training
```
==============================================================
PHASE 1: SUPERVISED WARMUP TRAINING
==============================================================

[Supervised] Epoch 1/5, Sample 50/1603, Avg Loss: 12.5432, ...
[Supervised] Epoch 1/5, Sample 100/1603, Avg Loss: 12.2341, ...
...
  Evaluating on validation set...

==============================================================
Supervised Epoch 1/5 Complete!
==============================================================
  Training Metrics:
    Avg Loss: 12.1030
    Step CE Loss: 2.2690
    MCP BCE Loss: 0.7880
  ...
```

### 3. Phase 2: GRPO Training
```
==============================================================
PHASE 2: GRPO REINFORCEMENT LEARNING FINE-TUNING
==============================================================

[GRPO] Epoch 1/10, Update 1/401, Avg Reward: 0.2345
...
```

### 4. Final Test Results
```
==============================================================
FINAL TEST EVALUATION
==============================================================

==============================================================
TEST RESULTS
==============================================================
  Step Metrics:
    Accuracy: 0.5432
    Micro F1: 0.5123
  ...
```

## Monitoring Training

### Check GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Monitor Log File in Real-Time
```bash
tail -f logs/training_TIMESTAMP.log
```

### Check TensorBoard (if enabled)
```bash
tensorboard --logdir=runs
# Open browser to http://localhost:6006
```

## Important Files

- **config.json** - Training configuration
- **train_gnn_rl.py** - Main training script
- **output/** - Checkpoints saved here
  - `best_supervised_checkpoint.pt` - Best model from supervised phase
  - `best_checkpoint.pt` - Best overall model
  - `final_checkpoint.pt` - Final model after all training
- **runs/** - TensorBoard logs
- **logs/** - Training output logs (if using run_training.sh)

## Checkpoints Explained

1. **best_supervised_checkpoint.pt**
   - Saved after supervised training phase
   - Best model based on validation selection score
   - Use if GRPO phase doesn't improve results

2. **supervised_checkpoint_epoch_N.pt**
   - Saved after each supervised epoch
   - Useful for debugging or resuming training

3. **best_checkpoint.pt**
   - Overall best model (includes GRPO training)
   - This is typically the one you want to use
   - Automatically loaded for final test evaluation

4. **grpo_checkpoint_epoch_N.pt**
   - Saved after each GRPO epoch
   - Useful for analyzing GRPO training progression

5. **final_checkpoint.pt**
   - Saved at the very end
   - Includes test metrics
   - May not be the best model (use best_checkpoint.pt instead)

## Common Issues & Solutions

### 1. ZeroDivisionError
**Status:** ✓ FIXED
This has been completely resolved. If you still see it, check that you're running the updated code.

### 2. All Samples Skipped
```
WARNING: All samples in supervised epoch 2 were skipped due to non-finite losses!
```

**Solutions:**
- Lower the learning rate in config.json
- Check for data corruption
- Reduce batch size
- Try disabling AMP (set model dtype to float32)

### 3. Out of Memory (OOM)
**Solutions:**
- Reduce batch_size in config.json
- Increase gradient_accumulation_steps
- Enable 4-bit quantization: `"load_in_4bit": true`
- Reduce max_seq_length

### 4. Very Low Validation Accuracy
**Possible causes:**
- Learning rate too high or too low
- Loss weight imbalance (adjust step_loss_weight, mcp_loss_weight)
- Data quality issues
- Model size mismatch for task complexity

### 5. TensorBoard Not Available
```
Warning: TensorBoard not available, skipping logging.
```

**Solution:**
```bash
pip install tensorboard
```

## Configuration Tips

### For Faster Training (Development)
```json
{
  "training": {
    "num_supervised_epochs": 2,
    "num_grpo_epochs": 2,
    "batch_size": 4,
    "gradient_accumulation_steps": 2
  }
}
```

### For Better Quality (Production)
```json
{
  "training": {
    "num_supervised_epochs": 5,
    "num_grpo_epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5
  }
}
```

### For Low Memory Systems
```json
{
  "model": {
    "load_in_4bit": true,
    "use_lora": true,
    "lora_r": 8
  },
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 512
  }
}
```

## Stopping Training

- **Ctrl+C** - Gracefully stops training
  - Current epoch will complete
  - Checkpoints are saved
  
- **Early Stopping** - Automatic
  - Triggered when no improvement for `patience` epochs
  - Message: "⚠ Early stopping triggered..."

## Resuming Training

Training cannot be automatically resumed from checkpoints yet. To continue training:
1. Load the checkpoint manually in code
2. Adjust config to skip already-completed epochs
3. Or run a new training session with different config

## Getting Help

1. Check the log files in `logs/` directory
2. Review the FIXES_APPLIED.md document
3. Check TensorBoard for training curves
4. Examine the checkpoint metadata:
   ```python
   import torch
   ckpt = torch.load('output/best_checkpoint.pt', map_location='cpu')
   print(ckpt.keys())
   ```

## Next Steps After Training

1. **Evaluate the model:**
   ```bash
   python evaluate.py --checkpoint output/best_checkpoint.pt
   ```

2. **Test on specific samples:**
   Check the evaluate.py script or create custom inference code

3. **Analyze results:**
   - Look at TensorBoard curves
   - Compare supervised vs GRPO performance
   - Check which samples the model gets wrong

4. **Tune hyperparameters:**
   - Adjust loss weights
   - Try different learning rates
   - Experiment with pooling strategies
