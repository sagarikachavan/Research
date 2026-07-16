# Logging Improvements - Enhanced Progress Tracking

**Date:** July 16, 2026  
**Issue:** Training appeared "stuck" with no visible progress  
**Status:** ✅ FIXED - Comprehensive logging added

## Problem

The training script appeared stuck after showing the configuration summary:
- No progress updates for a long time
- Users couldn't tell if training was actually running or frozen
- First progress message only appeared after 50 samples (could take 10-30 minutes!)

## Solution

Added **comprehensive real-time progress logging** throughout the training:

### 1. Supervised Training Progress

**Added:**
- Startup message confirming training begins
- **Real-time sample counter** updating every 10 samples initially
- Progress reports every 50 samples with:
  - Current sample count
  - Average losses
  - **Samples per second**
  - **Estimated time remaining (ETA)**
- Epoch completion time
- Non-finite loss warnings

**Example Output:**
```
==============================================================
PHASE 1: SUPERVISED WARMUP TRAINING
==============================================================

Starting training with 1603 samples...
Progress will be shown every 10 samples (initially) and every 50 samples thereafter.

Starting Supervised Epoch 1/5...
  Processing sample 1/1603...
  Processing sample 2/1603...
  Processing sample 3/1603...
  ...
  Processing sample 10/1603...
  Processing sample 20/1603...
  ...

[Supervised] Epoch 1/5, Sample 50/1603, Avg Loss: 12.5432, Step CE: 2.2341, MCP BCE: 0.7654, Speed: 2.34 samples/sec, ETA: 11.2m

[Supervised] Epoch 1/5, Sample 100/1603, Avg Loss: 12.2341, Step CE: 2.1234, MCP BCE: 0.7321, Speed: 2.41 samples/sec, ETA: 10.4m

  WARNING: Non-finite loss at sample 125, skipping...

[Supervised] Epoch 1/5, Sample 150/1603, Avg Loss: 12.1030, Step CE: 2.0890, MCP BCE: 0.7123, Speed: 2.38 samples/sec, ETA: 10.2m

  Epoch 1 training completed in 11.2 minutes
  Evaluating on validation set...
```

### 2. GRPO Training Progress

**Added:**
- Epoch start message
- **Real-time batch counter** during rollout generation
- Update-by-update progress with:
  - Current update count
  - Average reward
  - **Updates per second**
  - **Estimated time remaining (ETA)**
- Epoch completion time

**Example Output:**
```
==============================================================
PHASE 2: GRPO REINFORCEMENT LEARNING FINE-TUNING
==============================================================

RL auxiliary supervised weight: 0.300
Starting GRPO training with 1603 batches per epoch...

Starting GRPO Epoch 1/5...
  Processing batch 1/1603, generating rollouts...

[GRPO] Epoch 1/5, Update 1/1603, Avg Reward: 0.2345, Speed: 0.15 updates/sec, ETA: 178.5m

  Processing batch 2/1603, generating rollouts...

[GRPO] Epoch 1/5, Update 2/1603, Avg Reward: 0.2456, Speed: 0.16 updates/sec, ETA: 166.4m
```

### 3. Visual Indicators

**Progress Indicators:**
- `Processing sample X/Y...` - Shows real-time progress (updates in place with `\r`)
- `[Supervised]` / `[GRPO]` - Clear phase prefixes
- `ETA: X.Xm` - Estimated time remaining in minutes
- `Speed: X.XX samples/sec` - Processing speed
- `\n` for important messages - Ensures they're not overwritten

### 4. Time Tracking

**Added tracking for:**
- Per-epoch elapsed time
- Samples/updates per second
- Estimated time to completion
- Total epoch duration

## Changes Made

**File:** `train_gnn_rl.py`

### Supervised Training Loop (~line 1340)
```python
# Before
for epoch in range(num_supervised_epochs):
    for batch_samples in train_loader:
        for sample in batch_samples:
            # ... training code ...

# After
for epoch in range(num_supervised_epochs):
    print(f"Starting Supervised Epoch {epoch+1}/{num_supervised_epochs}...")
    epoch_start_time = time.time()
    
    for batch_idx, batch_samples in enumerate(train_loader):
        for sample in batch_samples:
            # Show real-time progress
            if num_samples < 10 or num_samples % 10 == 0:
                print(f"  Processing sample {num_samples + 1}/{len(train_dataset)}...", 
                      end='\r', flush=True)
            
            # ... training code ...
            
            # Detailed progress every 50 samples
            if num_samples > 0 and num_samples % 50 == 0:
                elapsed = time.time() - epoch_start_time
                samples_per_sec = num_samples / elapsed
                eta_minutes = (len(train_dataset) - num_samples) / samples_per_sec / 60
                print(f"\n[Supervised] ... Speed: {samples_per_sec:.2f} samples/sec, ETA: {eta_minutes:.1f}m")
    
    print(f"\n  Epoch {epoch+1} completed in {(time.time() - epoch_start_time) / 60:.1f} minutes")
```

### GRPO Training Loop (~line 1530)
```python
# Before
for epoch in range(num_grpo_epochs):
    for batch_samples in train_loader:
        # ... generate rollouts ...
        print(f"[GRPO] Epoch {epoch+1}, Update {num_updates + 1}, Avg Reward: {avg_reward:.4f}")

# After
for epoch in range(num_grpo_epochs):
    print(f"Starting GRPO Epoch {epoch+1}/{num_grpo_epochs}...")
    epoch_start_time = time.time()
    
    for batch_idx, batch_samples in enumerate(train_loader):
        print(f"  Processing batch {batch_idx + 1}/{len(train_loader)}, generating rollouts...", 
              end='\r', flush=True)
        
        # ... generate rollouts ...
        
        elapsed = time.time() - epoch_start_time
        updates_per_sec = (num_updates + 1) / elapsed
        eta_minutes = (len(train_loader) - num_updates - 1) / updates_per_sec / 60
        print(f"\n[GRPO] Epoch {epoch+1}, Update {num_updates + 1}/{len(train_loader)}, "
              f"Avg Reward: {avg_reward:.4f}, Speed: {updates_per_sec:.2f} updates/sec, ETA: {eta_minutes:.1f}m")
```

## Benefits

### Before
- ❌ No feedback for 10-30 minutes
- ❌ Couldn't tell if training was stuck or running
- ❌ No idea how long it would take
- ❌ No warning when samples were skipped

### After
- ✅ **Immediate feedback** - See progress every 10 samples
- ✅ **Real-time updates** - Counter updates continuously
- ✅ **Time estimates** - Know exactly how long to wait
- ✅ **Performance metrics** - See samples/updates per second
- ✅ **Clear warnings** - Know when samples are skipped
- ✅ **Completion times** - See how long each epoch took

## Performance Impact

- Minimal - Progress printing is lightweight
- `end='\r'` overwrites lines instead of creating new ones
- Only shows detailed stats every 50 samples
- Real-time counter updates efficiently

## Expected Behavior

### First 10 Samples (Initial Phase)
```
  Processing sample 1/1603...
  Processing sample 2/1603...
  Processing sample 3/1603...
```
Updates after each sample

### After 10 Samples
```
  Processing sample 20/1603...
  Processing sample 30/1603...
  Processing sample 40/1603...
```
Updates every 10 samples

### Every 50 Samples
```
[Supervised] Epoch 1/5, Sample 50/1603, Avg Loss: 12.54, Speed: 2.34 samples/sec, ETA: 11.2m
```
Detailed progress with statistics

## Monitoring Tips

### Training Speed
- **Good:** 2-5 samples/sec (supervised), 0.1-0.3 updates/sec (GRPO)
- **Slow:** <1 sample/sec (supervised), <0.05 updates/sec (GRPO)
- **Fast:** >10 samples/sec (supervised), >0.5 updates/sec (GRPO)

### ETA Accuracy
- First estimate may be inaccurate (based on few samples)
- Becomes more accurate after 100+ samples
- Can vary based on sample complexity

### What to Watch
- Loss should generally decrease
- Speed should stabilize after first few samples
- ETA should decrease steadily
- No frequent "Non-finite loss" warnings

## Troubleshooting

### Still No Output?
1. Check if Python is buffering output: `python -u train_gnn_rl.py --config config.json`
2. Ensure you're watching the correct terminal
3. Check if the process is actually running: `nvidia-smi` (should show GPU usage)

### Progress Stopped?
1. Check GPU memory: `nvidia-smi`
2. Look for error messages
3. Check if process is still alive: `ps aux | grep train_gnn`

### Very Slow Progress?
1. **Expected:** First few samples are slower (model initialization)
2. Check GPU utilization: Should be high (>70%)
3. Consider reducing `max_seq_length` or enabling `load_in_4bit`

## Additional Notes

- Real-time counter uses `\r` to overwrite same line
- Detailed progress uses `\n` to create new lines
- All timestamps are in local time
- ETA is estimated and may vary
- Speed metrics are calculated from epoch start

---

**Result:** Training progress is now fully visible and transparent! No more wondering if it's stuck or frozen.
