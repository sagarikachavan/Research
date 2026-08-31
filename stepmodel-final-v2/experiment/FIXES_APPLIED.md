# Why `experiment/` was getting stuck at Stage 2 → Stage 3, and what was fixed

I compared your uploaded zip against `github.com/sagarikachavan/Research`
(branch `restructured_code`) — the code is identical between the two, so
this applies to both.

## Root causes

### 1. `run_experiment.py` — silent output buffering (the "Stage 2 looks stuck" part)

The orchestrator ran each stage with
`subprocess.run(cmd, capture_output=True, ...)`. This buffers **all**
stdout — including the tqdm progress bar — until the subprocess exits.
For a training stage that can run for hours, that means the terminal
shows **nothing at all** for the entire stage. It wasn't actually hung,
you just couldn't see it working.

**Fix:** switched to `subprocess.Popen` with line-by-line streaming, so
progress prints live as it happens.

### 2. `stage3_grpo_rl.py` — the real hang/crash risk (the "then Stage 3" part)

Three compounding bugs:

- It called `merge_and_unload()` on the Stage 2 LoRA adapter and then
  loaded a **second full copy** of the 14B base model as `ref_model`.
  That's ~2x the full model resident in memory at once (~56GB in fp16),
  *and*, because merging removes the adapter, Stage 3 was silently doing
  **full fine-tuning of all 14B parameters** instead of LoRA. This is
  almost certainly what actually hangs/OOMs.
- The RL "policy log-prob" was computed by running the model on the
  **prompt only**, with the prompt as its own label — i.e. it never even
  looked at the text it had generated. The RL update wasn't optimizing
  the sampled completions at all.
- Because the model was merged (not a `PeftModel` anymore),
  `model.save_pretrained(STAGE3_ADAPTER_DIR)` wrote a full merged model
  to disk, but `evaluate.py` expects that directory to be a LoRA adapter
  (`PeftModel.from_pretrained(base, dir)`). Even a successful training
  run would have failed at evaluation.
- No progress output between steps other than every 100 steps — with the
  above problems, a very long silent gap.

**Fix:** load the base model **once**, attach the Stage 2 adapter under
two names on top of it — `"default"` (trainable, this is what gets
optimized) and `"ref"` (frozen, used for the KL penalty) — using PEFT's
multi-adapter API, and switch the active adapter as needed. Only the
small LoRA `"default"` parameters are trainable; there is only ever one
copy of the 14B weights in memory. Log-probs are now computed correctly
over the generated completion tokens for both policy and reference.
Saving now produces a proper flat LoRA-adapter directory compatible with
`evaluate.py`. Added a tqdm bar with a live reward/loss/KL postfix, an
upfront check that the Stage 2 adapter exists, and a warning if no CUDA
device is present.

### 3. `evaluate.py` — two more hang risks

- It never checked that the Stage 2/3 adapter directory exists before
  calling `PeftModel.from_pretrained(base_model, adapter_dir)`. If the
  directory is missing (e.g. Stage 3 hadn't produced a checkpoint yet
  because of bug #2 above), PEFT silently treats that path as a Hugging
  Face Hub repo ID and tries to resolve it **over the network** — which
  can hang for a long time before eventually failing with a confusing
  error.
- With `--stage all` (what the orchestrator uses), it loaded a full 14B
  model for Stage 2 evaluation, then loaded **another** full 14B model
  for Stage 3 evaluation without ever freeing the first one — risking an
  OOM right at the very last step of the pipeline.

**Fix:** added an existence check (skips a stage with a clear message
instead of hanging on a network call), and explicit `del` +
`torch.cuda.empty_cache()` between stages.

## Files changed

- `experiment/run_experiment.py`
- `experiment/training/stage3_grpo_rl.py` (major rewrite)
- `experiment/training/stage2_sft_qwen.py` (minor: CUDA warning)
- `experiment/eval/evaluate.py`

## What I verified

I don't have a GPU or Qwen3-14B access in this environment, so I
couldn't run the actual pipeline end-to-end. What I *did* verify with
real HuggingFace/PEFT models (GPT-2-sized, offline) is the specific
mechanism that was broken and is now fixed:

- Multi-adapter save (Stage 2 style) → load as `"default"` + `"ref"` →
  switch between them.
- Gradients from the policy-gradient loss only flow into the `"default"`
  adapter's parameters, never into `"ref"`.
- The policy adapter's weights measurably diverge from the frozen `"ref"`
  adapter after a single optimizer step (confirming the KL term and
  gradient flow behave as intended).
- The reference log-prob is deterministic across repeated calls (dropout
  correctly disabled for that forward pass).
- Saving the Stage-3-style checkpoint produces a flat directory
  (`adapter_config.json` at the root, not in a subfolder — this is a real
  PEFT quirk for non-`"default"`-named adapters that I had to work around),
  and it reloads and merges cleanly the same way `evaluate.py` does.

I'd still recommend a short smoke test on your actual hardware — e.g.
temporarily set `STAGE3_STEPS = 5` and `STAGE3_GROUP_SIZE = 2` in
`config.py` — before committing to a full run, just to confirm timing
and memory headroom on your specific GPU(s).
