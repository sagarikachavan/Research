# Stage 2 & Stage 3 Architecture Audit — Findings and Changes

**Scope:** `training/stage2_sft_qwen.py`, `core/graph_prefix_adapter.py`,
`eval/evaluate.py` (Stage 2); `training/stage3_grpo_rl.py`, `core/llm_judge.py`,
`core/comprehensive_evaluator.py` (Stage 3). Goal: the same architecture
review `STAGE1_IMPROVEMENTS.md` did for Stage 1, extended to the two stages
that document explicitly deferred a deep pass pending Stage 1 (see that
file's §7 "Stage 2 / Stage 3: a preliminary look").

**How this was produced:** full reads of every file in scope (two Explore
agents read Stage 2 and Stage 3 independently and in parallel, cross-checked
against a direct read of the exact current code before any fix was written),
plus `grep -rn` across the whole repo before every deletion/signature change
to confirm nothing else depended on the removed code. Every fix below was
verified with `python -m py_compile` on every edited file plus, where the
change was non-trivial numerical logic (the dual-clip PPO objective, the
adapter dtype fix), a synthetic-tensor smoke test — the same verification
standard `STAGE1_IMPROVEMENTS.md` used. **None of this has been validated by
a real training/eval run** — this session has no GPU or the real dataset;
training happens on a separate machine. See "What's been verified" at the
end.

None of these are speculative tuning. Each is either a silent correctness bug
that directly affects the numbers being compared to the paper, pure wasted
compute, or a documented fix for a real observed incident that was written up
in `config.py` but never actually wired into the training loop.

---

## Stage 2

### 1. Eval-time adapter dtype mismatch (real correctness bug)

`eval/evaluate.py`'s `eval_llm()` used to build the `GraphPrefixAdapter`
directly in bf16 (`.to(device).to(dtype)`) and only afterward call
`adapter.load_state_dict(...)` with the fp32-trained checkpoint. Training
(`stage2_sft_qwen.py`'s `forward_batch`) deliberately keeps the adapter
itself in fp32 for the entire forward pass and casts only its *output* to
bf16 right before concatenation — an explicit comment there documents this as
the fix for "intermittent NaNs" observed when the adapter was trained
directly in bf16 on the DGX run.

Reproduced directly: building a small linear module in bf16 and loading an
fp32 state dict into it silently **downcasts the loaded weights in place** —
`bad.lin.weight.dtype` came back `torch.bfloat16` after `load_state_dict`,
vs. `torch.float32` when the module is built in fp32 first. So evaluation
was silently running the adapter's LayerNorms/GELUs/matmuls in a precision it
was deliberately trained to avoid, with no error raised anywhere — exactly
the kind of mismatch that can degrade or destabilize generated outputs
without being visible in any log, and it directly affects the Stage 2 (and,
since Stage 3 loads Stage 2's adapter, indirectly Stage 3) numbers being
compared to the paper.

**Fix:** `eval/evaluate.py` now builds the adapter in fp32
(`.to(device)`, no `.to(dtype)`), loads the checkpoint, and casts only the
adapter's *output* (`adapter(graph_h.float()).to(dtype)`) — exactly matching
`forward_batch`'s pattern. Checked `training/stage3_grpo_rl.py`'s own adapter
loading for the same bug: it already does this correctly
(`.to(device).float()` before `load_state_dict`, output-only cast at the
call site) — the bug was isolated to `evaluate.py`.

Also removed a `context_texts` variable computed immediately above this code
and never used, plus the comment above it claiming Stage 2/3 were trained
from "the classification-calibrated fused Stage-1 representation" — that's
stale/wrong relative to the code beneath it, which has always called
`stage1.graph_encoder(...)` (the raw 512-d GINE output), matching training's
actual interface exactly. The *behavior* was already correct; only the
comment was misleading.

### 2. Dead Stage-1-hint mechanism, removed

`build_prompt(ex, mask_hint=...)`, `format_stage1_hint()`, and
`precompute_stage1_hints()` implemented an optional mechanism to splice a
text hint from Stage 1's own prediction into the Stage 2 prompt, with
`mask_hint` meant to randomly hide it during training so the model has to
learn from the graph-prefix tokens directly (`STAGE2_HINT_MASK_PROB` in
`config.py` set the masking probability).

`precompute_stage1_hints()` was never called from anywhere in the repo
(confirmed by `grep`; `main()` has an explicit `# REMOVED: Precompute Stage-1
classifier hints` comment) — so `ex["stage1_hint"]` never existed on any real
example, and `build_prompt()`'s body never actually read `mask_hint` or the
hint field regardless of what was passed. The prompt was identical no matter
what `mask_hint` was set to; `SFTDataset.__getitem__`'s
`np.random.random() < self.mask_hint_prob` draw and the `STAGE2_HINT_MASK_PROB`
env-var read were computing a value that was then discarded. An older audit
note (`CHANGES_AND_FINDINGS.md` §5) describing this as intentional train/eval
behavior is stale relative to current code — it was evidently written before
hint precomputation was removed from `main()`.

**Fix:** removed `mask_hint` from `build_prompt()`'s signature (and its
docstring paragraph describing it), deleted `format_stage1_hint()` and
`precompute_stage1_hints()` entirely, removed `mask_hint_prob` from
`SFTDataset`, and removed `STAGE2_HINT_MASK_PROB` from `config.py` (nothing
reads it once the feature is gone). Fixed every call site
(`stage2_sft_qwen.py`'s dataset construction and test-loop CSV-writing call,
plus one call in the separate experimental script
`graph_adapter_experiments/legacy_core_coupled/train_multitask_adapter.py`,
which imports `build_prompt` from this module and would otherwise have broken
on the signature change even though it's outside this audit's primary scope
— `run.py` never touches that directory, but leaving an import error behind
for anyone who does run it would be a worse outcome than a one-line fix).

### 3. Dead `field_embs` computation, removed (real wasted compute)

`SFTDataset.__getitem__` called `_embed_texts(...)` (an embedding-model
forward pass) for every single training/val/test example to build a
`field_embs` tensor, which `collate_fn` stacked and every downstream
function (`forward_batch`, `run_validation`, the train loop, the test loop)
accepted and moved to device — but `forward_batch`'s actual model-building
logic only ever used `graphs.x/edge_index/batch/edge_attr` via
`stage1.graph_encoder(...)`; `field_embs` was never consumed. This is
vestigial from an earlier fused-encoder prompt design and was silently
burning real embedding-model compute every batch for zero training effect.

**Fix:** removed `field_embs` end-to-end — from `SFTDataset.__getitem__`,
`collate_fn`'s return tuple, `forward_batch`'s signature, and every call site
(`run_validation`, the train loop, the test loop). Also removed the now-fully
-unused `_embed_texts`/`CONTEXT_COLUMNS` imports. Same fix applied to the one
external call site in `graph_adapter_experiments/legacy_core_coupled/
train_multitask_adapter.py` (see above).

### 4. Dead config wiring for `STAGE2_LR`/`STAGE2_WEIGHT_DECAY`

`config.py` imported `STAGE2_LR = 1e-5` with a comment "Reduced for FP16
stability," but `stage2_sft_qwen.py`'s `main()` never actually referenced
the imported constant — it read `os.environ.get("STAGE2_SAFE_LR", "2e-6")`
instead, an independently hardcoded literal that doesn't match config's
value at all. `STAGE2_WEIGHT_DECAY` was similarly read via
`os.environ.get("STAGE2_WEIGHT_DECAY", "1e-4")` — same *name* as the config
constant, but never actually importing/reading it (its default happened to
coincidentally match config's `1e-4`). Editing `config.py`'s `STAGE2_LR` had
literally zero effect on any real run; whoever wrote the "Reduced for FP16
stability" comment made that change directly in the training script and
never back-ported it here.

**Fix:** `config.py`'s `STAGE2_LR` corrected to `2e-6` (the value actually
exercised by every real run so far). `stage2_sft_qwen.py`'s `main()` now
reads `os.environ.get("STAGE2_SAFE_LR", str(STAGE2_LR))` and
`os.environ.get("STAGE2_SAFE_WEIGHT_DECAY", str(STAGE2_WEIGHT_DECAY))` —
`config.py` is the real fallback default again, with **zero change** to
today's actual runtime values.

### 5. Dead file: `core/graph_prefix_adapter.py`, deleted

A second, divergent `GraphPrefixAdapter` implementation existed in
`core/graph_prefix_adapter.py`, with a docstring claiming it "implements the
prefix adapter... as specified in the multi-stage training pipeline." `grep
-rn` across the entire repo for any import of this module (`from
graph_prefix_adapter import`, `from core.graph_prefix_adapter import`,
`core.graph_prefix_adapter`) returned **zero hits** anywhere — every actual
consumer (`stage2_sft_qwen.py`, `stage3_grpo_rl.py`, `evaluate.py`) uses the
locally-defined `GraphPrefixAdapter` class inside `stage2_sft_qwen.py`
instead. The two implementations also architecturally diverge (this file
hardcoded `llm_hidden=5120` and sized its hidden layer as
`max(llm_hidden, graph_dim*2)`; the real one always uses `llm_hidden*2` and
reads `llm_hidden` from the live model config) — a checkpoint saved by the
real training script would fail to `load_state_dict` into this one with a
shape mismatch, a landmine for anyone who later "fixes" an import to point
here. Deleted; nothing referenced it.

---

## Stage 3

### 6. Misleading module docstring — fixed

The module docstring's "Reward composition" section claimed the explanation
reward was "0.30 × LLM judge correctness score (0.0-1.0) using GPT-4o," with
a "WHY LLM JUDGE FOR EXPLANATION" section citing caching to make repeated API
calls feasible. **No OpenAI/GPT-4o call exists anywhere in this file** —
confirmed by `grep`. The actual, live explanation reward is
`_deterministic_explanation_score()`: `0.60 * BGE cosine similarity + 0.20 *
lexical (difflib) ratio + 0.10 * step-keyword support + 0.10 * predicted/gold
tool-set overlap`. Its own docstring is explicit that this is deliberate:
"not the LLM judge used at test time, preventing a noisy evaluator from
becoming the optimization target" — a defensible anti-reward-hacking design
choice, just not what the module header described.

**Fix:** rewrote the docstring's reward-composition section to describe the
actual deterministic reward and explain (correctly, this time) why it's kept
separate from the test-time judge (`core/llm_judge.py`'s rubric gate, used
only by `eval/evaluate.py`) — using the same evaluator as both the RL reward
and the reported test metric risks the policy gaming the judge's specific
quirks rather than the underlying explanation quality.

### 7. Dead code: `ValueHead`, LLM-judge caching path — removed

Two more fully-dead blocks found while fixing the docstring above:

- **`ValueHead`** (a learned value-function head for baseline-reduction),
  defined but never instantiated anywhere in the file. The actual advantage
  computation is GRPO's group-relative z-score, not a learned baseline.
- **`LLM_JUDGE_SYSTEM_PROMPT`, `_get_cache_key`, `set_llm_judge_model`,
  `_explanation_llm_judge_cached`** (an `@lru_cache`-wrapped in-loop LLM-judge
  call, matching what the stale docstring described). `set_llm_judge_model`
  was never called from anywhere in the repo (confirmed by `grep`), so
  `_llm_judge_model`/`_llm_judge_tokenizer` were always `None` and every real
  call to `_explanation_llm_judge_cached` would have silently fallen through
  to the length-bucket heuristic branch (`< 20 chars -> 0.3`, etc.) rather
  than ever actually calling a judge model — confirmed dead, not merely
  unused.

**Fix:** deleted both blocks and their now-unused `hashlib`/`functools.lru_cache`
imports.

### 8. All ten `STAGE3_*`/GRPO config constants were dead — wired in

`stage3_grpo_rl.py` imported `STAGE3_GROUP_SIZE`, `STAGE3_LR`,
`STAGE3_STEPS`, `STAGE3_KL_COEF`, `STAGE3_PPO_CLIP`, `STAGE3_GRAD_ACCUM`,
`STAGE3_GRAD_CLIP`, `STAGE3_DUAL_CLIP_COEF`, `STAGE3_KL_HARD_CAP`, and
`STAGE3_EARLY_STOP_PATIENCE` from `config.py` (confirmed by reading the
import block) but **never referenced any of them again anywhere in the file**
(confirmed by `grep`) — `main()` re-declared its own hyperparameters from
`os.environ.get("STAGE3_SAFE_*", "<hardcoded literal>")` instead. Editing
`config.py`'s `STAGE3_*` constants had zero effect on a real run. Two values
had already silently drifted apart: `STAGE3_GRAD_CLIP=1.0` in `config.py` vs.
`torch.nn.utils.clip_grad_norm_(trainable, 0.5)` hardcoded directly in the
training loop.

**Fix:** `config.py`'s `STAGE3_GRAD_CLIP` corrected to `0.5` (the value
actually exercised by every real run so far). `main()` now reads every
`STAGE3_SAFE_*` env var with `config.py`'s corresponding `STAGE3_*` constant
as the fallback default (`os.environ.get("STAGE3_SAFE_LR", str(STAGE3_LR))`,
etc.) instead of an independently hardcoded literal — restores `config.py` as
the real source of truth, with **zero change** to today's runtime values for
every constant except the two described next.

### 9. Documented dual-clip PPO fix, written up but never wired in — implemented

This is the most consequential fix in this audit. `config.py` contains a
detailed, citation-backed comment (Ye et al. 2020 dual-clip PPO) describing a
**real observed training collapse**: `pg_loss 127 -> 2897 -> 8255 -> 8175` at
steps 350/800/850/1000, held-out val reward collapsing `0.454 -> 0.29`, fmt
compliance dropping to 1/4. Root cause per that comment: for a
negative-advantage sample whose importance ratio has drifted far above 1,
standard single-clip PPO's `min(surr1, surr2)` objective is **not bounded
from below** in that quadrant — only the "good news" direction is capped.
With `log_ratio` clamped at ±4 (`ratio` up to `e^4 ≈ 55`) and advantage
clamped to ±3, an exploding-ratio, negative-advantage completion can produce
`|pg_loss|` orders of magnitude larger than a normal term, dominating the
batch loss and producing exactly the single-step gradient spike in the log.
`config.py` presents `STAGE3_DUAL_CLIP_COEF=3.0` (floor the objective at
`dual_clip_coef * advantage` when advantage is negative, per Ye et al.) and
`STAGE3_KL_HARD_CAP=4.0` (raised from an over-tight 1.0 that was discarding
~28% of micro-batches) as the fix.

**Neither was actually implemented.** The real PPO loss was plain
`pg = -torch.minimum(surr1, surr2)`, no third branch; the real KL gate was a
flat `if mean_kl > 1.0: skip`, not a 4.0 per-micro-batch cap reading the
config constant. `STAGE1_IMPROVEMENTS.md`'s §7 preliminary-look section cited
this same `config.py` comment block as evidence Stage 3's RL instability was
"iterated carefully... not neglected" — that characterization was accurate
only for `config.py`'s comments, not for what the training loop actually
executed.

**Fix, implemented in `training/stage3_grpo_rl.py`:**
- The PPO loss now computes `clip_obj = min(surr1, surr2)` (identical to
  before) and, only when `adv < 0`, additionally floors it at
  `dual_clip_obj = max(clip_obj, SAFE_DUAL_CLIP * adv)`, selected via
  `torch.where(adv < 0, dual_clip_obj, clip_obj)`. For `adv >= 0` the
  objective is byte-identical to the old single-clip behavior — dual-clip
  changes *nothing* about the positive-advantage branch by construction.
- The per-micro-batch KL check now compares `mean_kl` against
  `SAFE_KL_HARD_CAP` (config's `STAGE3_KL_HARD_CAP`, default 4.0) instead of
  a hardcoded `1.0`.
- `torch.nn.utils.clip_grad_norm_` now uses `SAFE_GRAD_CLIP` (config's
  corrected `STAGE3_GRAD_CLIP=0.5`) instead of the hardcoded `0.5` literal —
  same value, now actually driven by config.

**Verified with a synthetic smoke test** (no GPU/model needed — this is pure
tensor arithmetic): reproduced the incident's own numbers (`ratio≈22000,
adv=-4`) and confirmed the old objective gives `pg_loss=88000` while the new
one gives `pg_loss=12` — a ~7300x reduction in exactly the failure case the
comment describes. Confirmed over 2000 random trials that (a) for `adv >= 0`
the new and old objectives are numerically identical (dual-clip is a true
no-op there), and (b) for `adv < 0` the new objective is always bounded at
`dual_clip_coef * adv`, i.e. never falls below the documented floor. Also
confirmed the new `SAFE_KL_HARD_CAP=4.0` threshold now lets `mean_kl` values
in `(1.0, 4.0]` through an update that the old `1.0` threshold would have
discarded, matching `config.py`'s own claim about why 1.0 was too tight
(~28% of micro-batches discarded for no real safety benefit once dual-clip
already bounds pg_loss).

This restores a real, previously-documented safety mechanism rather than
introducing a new experimental one — it's gated as "always on" (no separate
opt-out flag) the same way the original comment intended it as *the* fix, not
an A/B option, but it has **not been validated by a real training run** since
this session has no GPU access; see below.

---

## Files changed

- `eval/evaluate.py` — fp32 adapter build/load, output-only bf16 cast;
  removed dead `context_texts` + stale comment.
- `training/stage2_sft_qwen.py` — removed `mask_hint`/`format_stage1_hint`/
  `precompute_stage1_hints`/`mask_hint_prob`; removed `field_embs` end-to-end;
  `STAGE2_SAFE_LR`/`STAGE2_SAFE_WEIGHT_DECAY` now fall back to `config.py`;
  removed now-dead imports (`FUSION_HIDDEN`, `MCP_DECISION_THRESHOLD`,
  `IDX2STEP`, `IDX2MCP`, `_embed_texts`, `CONTEXT_COLUMNS`).
- `training/stage3_grpo_rl.py` — fixed module docstring; removed dead
  `ValueHead`/LLM-judge-caching code + unused imports; every `STAGE3_SAFE_*`
  env var now falls back to `config.py`; implemented dual-clip PPO + the
  documented KL hard cap; grad-clip now config-driven.
- `core/config.py` — `STAGE2_LR` corrected to `2e-6`; removed dead
  `STAGE2_HINT_MASK_PROB`; `STAGE3_GRAD_CLIP` corrected to `0.5`; comments on
  `STAGE3_DUAL_CLIP_COEF`/`STAGE3_KL_HARD_CAP` updated to reflect that they're
  now actually wired in.
- `core/graph_prefix_adapter.py` — deleted (dead file, zero importers).
- `graph_adapter_experiments/legacy_core_coupled/train_multitask_adapter.py`
  — updated its two `stage2_sft_qwen` call sites (`build_prompt`,
  `forward_batch`) to match the new signatures, so this out-of-pipeline
  experimental script (not touched by `run.py`) doesn't break as a side
  effect of the Stage 2 cleanup above.

## What's been verified vs. what hasn't

Every fix above was checked with `python -m py_compile` (all touched files,
plus a repo-wide `grep` for stray references to anything renamed/removed —
clean). The dual-clip PPO objective and the adapter fp32/bf16 dtype behavior
were additionally verified with synthetic-tensor smoke tests reproducing the
documented incident numbers, following the same verification standard used
throughout `STAGE1_IMPROVEMENTS.md`. **None of this has been validated by a
real Stage 2 or Stage 3 training/eval run** — this session has no GPU or
access to the real dataset. Recommended order once synced to the GPU machine:
run Stage 1 (k-fold) first per `STAGE1_IMPROVEMENTS.md` §9, then Stage 2 SFT,
then Stage 3 GRPO, and compare against the paper (step accuracy 0.8287, step
micro-F1 0.80, MCP micro-F1 0.64, MCP subset accuracy 0.4888) and against
this project's own prior validated numbers.
