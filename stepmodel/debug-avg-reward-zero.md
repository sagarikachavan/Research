[OPEN] Avg reward stuck at 0.00

## Session
- session_id: avg-reward-zero
- started_at: 2026-07-07
- repo: /home/sagarika/Research/stepmodel

## Symptom
- Training logs report `Avg Reward 0.00`.

## Hypotheses
- Reward shaping always returns zero because predicted tags never match canonical labels.
- Reward values are computed but truncated, rounded, or reset before logging.
- The RL branch is not active, so the metric is emitted from an empty reward list.
- Parsing of generated step/MCP tags fails, causing reward inputs to be blank.
- The environment batch or dataloader feeds malformed targets during RL training.

## Plan
- Inspect reward computation and logging path.
- Add instrumentation only around reward collection and parsing.
- Reproduce the issue and confirm which hypothesis matches runtime evidence.
- Apply a minimal fix and verify the metric changes.

## Evidence
- Confirmed: `generate()` returns continuation-only tokens when called with `inputs_embeds`.
- Confirmed: rollout metadata stored `prompt_len = 111` while `outputs.sequences.shape[1] = 64`.
- Confirmed: the GRPO loss masked out all shifted tokens with `mask[:, :prompt_len - 1] = False`, leaving `kept_tokens = 0`.
- Confirmed: `old_log_prob` collapsed to `0.0`, so the RL objective had no usable continuation log-prob signal.

## Root Cause
- The RL code treated continuation-only generated tokens as if they still contained the prompt.
- This made both rollout scoring and loss recomputation mis-handle the prompt boundary.
- As a result, GRPO effectively optimized an empty token slice, so policy updates during RL were inert and average reward stayed near the tiny initialization baseline.

## Fix
- Store prompt token ids separately in each rollout.
- Keep generated token ids as continuation-only ids.
- Reconstruct the full prompt + continuation sequence inside `compute_grpo_loss()`.
- Score all continuation transition scores for `old_log_prob`.
- Mask only the prompt portion during the recomputed policy loss.

## Verification
- Before fix: `prompt_len=111`, `sequence_len=64`, `kept_tokens=0`, `old_log_prob=0.0`.
- After fix: `prompt_len=111`, `generated_len=64`, `full_seq_len=175`, `kept_tokens=64`, `old_log_prob=-63.7274`.
- Post-fix smoke test: `compute_grpo_loss(...)` returns a finite loss (`0.2825`) on a 2-rollout sample.
