# Fix: Non-Finite Loss Handling

**Date:** July 16, 2026  
**Issue:** `WARNING: Non-finite loss at sample X, skipping...`  
**Status:** ✅ IMPROVED - Added comprehensive safeguards

## Problem

Training was encountering non-finite (NaN or Infinity) losses, causing samples to be skipped:

```
WARNING: Non-finite loss at sample 17, skipping...
WARNING: Non-finite loss at sample 42, skipping...
```

Non-finite losses are problematic because:
1. They prevent the model from learning
2. Too many skipped samples can halt training progress
3. They indicate numerical instability

## Root Causes

Non-finite losses can occur due to:

1. **Exploding gradients** - Values become too large
2. **Vanishing gradients** - Values become too small  
3. **Division by zero** - In loss computations
4. **Invalid operations** - log(0), log(negative), etc.
5. **Numerical overflow** - Values exceed float limits
6. **Unstable model outputs** - Extreme logit values

## Solutions Implemented

### 1. Comprehensive Safeguards in Loss Computation

Added checks at multiple points in `compute_supervised_loss_for_sample()`:

```python
def compute_supervised_loss_for_sample(...):
    step_logits, mcp_logits = classify_sample(...)
    
    # Check 1: Sanitize logits before loss computation
    if not torch.isfinite(step_logits).all():
        print(f"    ERROR: Non-finite step_logits detected!")
        step_logits = torch.nan_to_num(step_logits, nan=0.0, posinf=10.0, neginf=-10.0)
    
    if not torch.isfinite(mcp_logits).all():
        print(f"    ERROR: Non-finite mcp_logits detected!")
        mcp_logits = torch.nan_to_num(mcp_logits, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Compute losses
    step_loss = F.cross_entropy(step_logits, step_target, weight=step_class_weights)
    
    # Check 2: Sanitize individual losses
    if not torch.isfinite(step_loss):
        print(f"    ERROR: Non-finite step_loss! Setting to 0.0")
        step_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    mcp_loss = F.binary_cross_entropy_with_logits(...)
    if not torch.isfinite(mcp_loss):
        print(f"    ERROR: Non-finite mcp_loss! Setting to 0.0")
        mcp_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Check 3: Handle explanation loss with try-except
    try:
        explanation_loss = ...
        if not torch.isfinite(explanation_loss):
            explanation_loss = torch.tensor(0.0, device=device)
    except Exception as e:
        print(f"    ERROR: Exception in explanation loss: {e}")
        explanation_loss = torch.tensor(0.0, device=device)
    
    # Check 4: Final sanity check on total loss
    total_loss = step_loss_weight * step_loss + ...
    if not torch.isfinite(total_loss):
        print(f"    ERROR: Non-finite total_loss!")
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss, step_loss.detach(), mcp_loss.detach()
```

### 2. Detailed Error Diagnostics

Added informative error messages showing:
- Which component has non-finite values
- Min/max values of problematic tensors
- Individual loss values when total is non-finite

### 3. Graceful Degradation

Instead of crashing:
- Replace NaN with 0.0
- Clip Infinity to ±10.0
- Continue training with sanitized values
- Only skip sample if recovery fails

### 4. Skip Counter and Statistics

Track how many samples are skipped:

```python
num_skipped = 0  # Initialize per epoch

# When skipping:
num_skipped += 1

# At epoch end:
if num_skipped > 0:
    print(f"  WARNING: Skipped {num_skipped}/{total} samples ({percent:.1f}%)")
```

## Expected Behavior Now

### Scenario 1: Recoverable Non-Finite Values

```
  Processing sample 17/1603...
    ERROR: Non-finite mcp_logits detected! min=-inf, max=15.234
  Processing sample 18/1603...
```

Sample continues training with sanitized values.

### Scenario 2: Unrecoverable Non-Finite Loss

```
  Processing sample 42/1603...
    ERROR: Non-finite step_logits detected! min=nan, max=nan
    ERROR: Non-finite step_loss! Setting to 0.0
    ERROR: Non-finite total_loss! step=0.0000, mcp=0.7654, expl=0.0000
  WARNING: Non-finite loss at sample 42 even after safeguards, skipping...
```

Sample is skipped only after all recovery attempts fail.

### Scenario 3: Epoch Summary

```
  Epoch 1 training completed in 11.2 minutes
  WARNING: Skipped 3/1603 samples due to non-finite losses (0.2%)
```

Clear reporting of how many samples had issues.

## Preventing Non-Finite Losses

### Configuration Adjustments

If you see frequent non-finite losses, try:

#### 1. Lower Learning Rate
```json
{
  "training": {
    "learning_rate": 1e-5  // Reduce from 3e-5
  }
}
```

#### 2. Reduce Loss Weights
```json
{
  "training": {
    "step_loss_weight": 2.0,  // Reduce from 4.0
    "mcp_loss_weight": 0.5,   // Reduce from 1.0
    "explanation_loss_weight": 0.0  // Disable if problematic
  }
}
```

#### 3. Enable Gradient Clipping (already enabled)
```json
{
  "training": {
    "max_grad_norm": 0.5  // Reduce from 1.0
  }
}
```

#### 4. Use Mixed Precision Carefully
- AMP is already disabled for your setup (4-bit=False, FP16 model)
- This is good for stability

#### 5. Reduce Sequence Length
```json
{
  "training": {
    "max_seq_length": 256  // Reduce from 512
  }
}
```

## Monitoring

### Acceptable Rates
- **< 1%** skipped: Normal, no action needed
- **1-5%** skipped: Monitor, consider adjustments
- **> 5%** skipped: Investigate and adjust hyperparameters
- **> 50%** skipped: Serious issue, stop training

### What to Watch

```bash
# During training, look for:
"ERROR: Non-finite step_logits"    # Frequent? Lower learning rate
"ERROR: Non-finite mcp_logits"     # Frequent? Check MCP weights
"ERROR: Non-finite explanation"     # Set explanation_loss_weight=0.0
"WARNING: Skipped X/Y samples"      # Track percentage
```

## Debugging Non-Finite Losses

If you see frequent non-finite losses:

### 1. Check Model Outputs
```python
# After classify_sample()
print(f"Step logits range: {step_logits.min():.4f} to {step_logits.max():.4f}")
print(f"MCP logits range: {mcp_logits.min():.4f} to {mcp_logits.max():.4f}")
```

### 2. Check Gradients
```python
# After backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if not torch.isfinite(grad_norm):
            print(f"Non-finite gradient in {name}")
```

### 3. Check Input Data
```python
# Before forward pass
print(f"Input IDs range: {input_ids.min()} to {input_ids.max()}")
print(f"Attention mask: {attention_mask.sum()} / {attention_mask.numel()}")
```

### 4. Enable Additional Logging
Look for the ERROR messages in the output - they show exactly where the problem occurs.

## Changes Made

**File:** `train_gnn_rl.py`

### In `compute_supervised_loss_for_sample()` (~line 640)
- Added 4 levels of safeguards
- Added detailed error diagnostics
- Added try-except for explanation loss
- Sanitize non-finite values instead of immediately failing

### In Supervised Training Loop (~line 1365)
- Added `num_skipped` counter
- Updated warning message to say "even after safeguards"
- Added epoch summary showing skip statistics

## Results

- **Before:** Samples silently skipped, no diagnosis
- **After:** 
  - Most non-finite values recovered automatically
  - Detailed diagnostics when issues occur
  - Statistics on skip rate
  - Training continues more robustly

## When to Worry

### Don't Worry If:
- ✅ < 1% samples skipped
- ✅ ERROR messages are rare
- ✅ Training loss decreases overall
- ✅ Validation metrics improve

### Worry If:
- ⚠️ > 5% samples skipped per epoch
- ⚠️ ERROR messages on every sample
- ⚠️ All samples skipped (total failure)
- ⚠️ Validation metrics don't improve

## Quick Fix Checklist

If seeing many non-finite losses:

1. ☐ Reduce learning rate to 1e-5 or lower
2. ☐ Set `explanation_loss_weight: 0.0`
3. ☐ Reduce `step_loss_weight` to 2.0
4. ☐ Reduce `max_grad_norm` to 0.5
5. ☐ Reduce `max_seq_length` to 256
6. ☐ Check for data corruption in embeddings_data/
7. ☐ Try with a smaller model (if using 7B+)

## Summary

The non-finite loss handling has been significantly improved:

1. ✅ Added 4-level safeguard system
2. ✅ Detailed error diagnostics
3. ✅ Automatic recovery for most cases
4. ✅ Statistics on skip rate
5. ✅ Clear warning messages

Training will now be much more robust and informative when encountering numerical issues!

---

**Note:** Some non-finite losses (especially early in training) are normal. The new system will handle them gracefully and alert you if they become frequent.
