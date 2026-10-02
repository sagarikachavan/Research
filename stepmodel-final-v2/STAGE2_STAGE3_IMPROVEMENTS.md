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

## 10. First real Stage 2 run: a real bug found and fixed (validation token budget)

First real Stage 2 SFT run (`training/stage2_sft_qwen.py`) landed at a
striking result: per-epoch validation (`val_generated_step_acc`) plateaued
around 0.74 and never exceeded 0.7406 across 8 epochs, while the **same
best checkpoint** scored **0.8843** ("Step Exact Match") on the final test
set. A 14-point jump from validation to test, in the direction test beating
validation, is not what normal generalization looks like on splits drawn
from the same dataset — it's the signature of the two numbers not measuring
the same thing.

**Root cause, confirmed by reading the code**: the per-epoch validation call
(`run_validation(...)`, called from the training loop) generated with
`max_new_tokens=32`, while the final test-set evaluation later in the same
file generated with `max_new_tokens=500` — a **15.6x** difference, with no
documented reason for the gap. Measured against the actual `STEP_LABELS`
taxonomy (GPT-2 tokenizer as a proxy — Qwen's own tokenizer wasn't available
in this environment, but the order of magnitude holds): the single longest
label alone (`"Enumerate further on the X service to find software
versions, hidden directories and file."`) needs 23 tokens just to close the
`"New step": "..."` field, before any JSON syntax overhead or preamble the
model might emit before starting the JSON. A 32-token budget leaves almost
no margin — a generation cut off mid-label produces an incomplete/
unparseable `"New step"` value, which `build_obj_parser()`'s regex fallback
can't recover (no closing quote to match), so the row scores as wrong even
when the model's prediction was correct. This **systematically
underestimated** validation accuracy for the whole run, and — more
seriously — fed that biased signal directly into checkpoint selection
(`step_field_acc > best_step_acc`) and early stopping, both of which drove
real decisions during training, not just a diagnostic printout.

**Fix**: `run_validation()`'s call site and default both changed from
`max_new_tokens=32`/`48` to **64** — comfortably covers the longest label
plus JSON overhead and a real margin for preamble, while staying far
cheaper per validation pass than the full 500-token budget (which also has
to cover the free-text explanation and MCP dict that this step-only check
doesn't need, so it was never the right number to match anyway). No other
generation call site in the pipeline has this kind of budget mismatch —
checked `training/stage3_grpo_rl.py` (300 and `MAX_NEW_TOKENS`, default
260, consistent throughout) and `eval/evaluate.py` (`args.max_new_tokens`
used consistently everywhere) — this was isolated to Stage 2's validation
loop.

**What this means for the run that already completed**: the checkpoint it
selected (epoch 6) still scored well at final test time (88.43%), so this
specific run's outcome is likely fine — but the *process* that selected it
was working from a biased signal, so there's no guarantee epoch 6 was
actually the best of the 8 epochs trained, only the best *as measured by a
truncated metric*. Not yet validated by a real re-run with the fix.

## 11. `eval/evaluate.py`'s `eval_gnn()` was never actually runnable — real bug, fixed

Running `python evaluate.py --model gnn` (recommended in this same session as
a diagnostic to get Stage 1's confusion matrix) crashed immediately:

```
ValueError: Stage 1 requires semantic_tokens and semantic_mask built from
New strategy + Strategy explanation.
```

**Root cause**: `eval_gnn()` built a `field_embs` tensor (a BGE embedding of
`CONTEXT_COLUMNS`) and passed it as `Stage1Classifier.forward()`'s 4th
positional argument. That parameter exists on the signature for backward
compatibility but is **never read inside `encode_and_predict()`** — what the
model actually requires (and raises `ValueError` without) is
`semantic_tokens`/`semantic_mask`: frozen-GPT-2 token embeddings of `"New
strategy"` + `"Strategy explanation"`, built via `precompute_semantic_tokens()`.
`eval_gnn()` never called that function at all. This is not something
introduced this session — nothing in this engagement touched `eval_gnn()`'s
call site or `encode_and_predict()`'s required-argument contract before now
— this code path had apparently never been exercised end-to-end until this
session asked for it as a diagnostic.

**Fix**: `eval_gnn()` now builds `ex["semantic_text"]` for every example
(identical logic to `training/stage1_gnn_train.py`'s `Stage1Dataset`) and
calls `precompute_semantic_tokens()` once up front, then pads each batch's
variable-length token sequences into a `(B, max_len, D)` tensor + validity
mask before calling the model — the same collate logic
`stage1_gnn_train.py`'s `collate()` uses. The dead `field_embs`/
`CONTEXT_COLUMNS`/`_embed_texts` computation is removed entirely rather than
kept as an unused side computation.

**Verified end-to-end** (not just `py_compile`): ran the actual fixed code
path with real GPT-2 tokenization (not synthetic/mocked) against a freshly
-initialized `Stage1Classifier`, including the empty-text edge case (both
context fields blank, matching the "empty empty" placeholder text
`SemanticCNNEncoder`'s own NaN-guard from `STAGE1_IMPROVEMENTS.md` §2 was
built for) — produced correctly-shaped, finite logits with no crash.

## 12. Stage 3's own reward/validation used a much weaker JSON parser than Stage 2's final eval — the actual RL signal, not just a number

A first real Stage 3 run's own printed baseline was hard to reconcile with
Stage 2's numbers: `evaluate_policy_on_val()` scored the **same** Stage-2
checkpoint at `step=0.6318` on val, while that checkpoint's own Stage-2
test-set evaluation (§10/§11 context) reported `0.8843`. A ~25-point gap
between val and test on the same checkpoint, both using generous generation
budgets (this file's `evaluate_policy_on_val` already uses
`max_new_tokens=300`, not the Stage-2 bug from §10), is too large to be
ordinary split noise. The training log itself gave a second, independent
tell: `fmt 3/8` at step 2 — only 3 of 8 completions from a model *freshly
SFT-trained specifically to emit this format* parsed as valid JSON at all.

**Root cause, confirmed by reading the code**: `training/stage3_grpo_rl.py`
had two different completion parsers. `stage2_sft_qwen.py::build_obj_parser()`
— a hardened, multi-layer-fallback parser (balanced-brace regex candidates,
then a naive first-`{`/last-`}` slice, then per-field regex extraction for
`"New step"`/`"Step explanation"`/`"MCP_tasks"` independently if full JSON
parsing fails entirely) — was already imported into this file
(`training/stage3_grpo_rl.py:119`), but only ever wired into the very last
test-CSV export loop (`:1149`, post-fix line numbers). Everywhere it
actually mattered — `compute_reward()` (`:254`, **the actual RL reward
signal GRPO trains against**), `evaluate_policy_on_val()` (`:546`, the
baseline/periodic-validation score), and the `fmt` diagnostic counter
(`:1014`) — used a bare-bones local `_parse_completion()`: a single
`text.index("{")` / `text.rindex("}")` slice fed straight to `json.loads`,
with **no fallback at all**. Any stray brace anywhere in the generated text
(e.g. inside a technical explanation describing a JSON-like config, or a
completion that runs out of tokens before the object closes) fails the
*entire* parse, scoring that row as a total miss even when the step
prediction was sitting right there in the text.

This is not merely a misleading-number bug like §10/§11 — `compute_reward()`
feeds directly into GRPO's policy gradient. A parser that spuriously
zeroes out otherwise-correct completions **is a corrupted training signal**,
not just a corrupted diagnostic; a model can get penalized for a formatting
quirk unrelated to whether its actual step/tool/explanation prediction was
right.

**Fix**: `_parse_completion()` now delegates to
`build_obj_parser()`(instantiated once at module scope as `_obj_parser`),
preserving its original `text -> dict | None` contract (empty-result cases
still return `None`) so every caller downstream needed zero changes.

**Verified** with a direct old-vs-new comparison on four realistic
completion shapes (clean JSON, a stray brace inside the explanation text,
preamble text before the JSON, and — the case that matters most — a
completion truncated mid-explanation with no closing brace at all): the old
parser failed outright (`None`) on the truncated case, silently losing a
perfectly extractable `"New step"` value; the new parser recovered it
correctly via `build_obj_parser()`'s per-field fallback, matched the old
parser exactly on the three cases where the old parser already succeeded
(no regression), never fabricated a value where nothing was extractable.
**Not yet validated by a real run** — the in-progress Stage 3 run that
surfaced this bug was training against the old, corrupted signal; restarting
Stage 3 with this fix is recommended rather than letting that run continue.

## 13. Second real run: two evaluate.py train/eval mismatches + Stage 3 dead LR + Stage 2 majority-class collapse

A full `run.py` pass plus `python evaluate.py --model llm` surfaced four
distinct problems, in decreasing severity:

### 13.1 `evaluate.py` was under-reporting Stage 2 by ~16 points (two train/eval mismatches)

The SAME Stage-2 checkpoint scored **88.43%** step-exact-match in
`stage2_sft_qwen.py`'s own final test loop but only **72.39%** in
`eval/evaluate.py --model llm`. Reading both generation paths, the
difference is two eval-only decoding settings that exist NOWHERE in training
or in Stage 2's own (trusted) test loop:

1. **`repetition_penalty=1.1`** — actively harmful here. The model must
   reproduce a long *canonical* `STEP_LABELS` string verbatim, and those
   labels repeat tokens that already appear in the prompt/taxonomy
   ("Enumerate", "further", "the"...). A repetition penalty pushes the
   decoder away from exactly the label text that exact-match scoring
   requires. Removed (along with a meaningless `temperature=1.0` that only
   emitted a warning under `do_sample=False`).
2. **`truncation=True, max_length=900` keeping the FIRST 900 tokens** —
   `build_prompt` puts `Machine` first and the actual `# Strategy` text +
   `# Task` instruction LAST, so a >900-token prompt was evaluated with its
   most important content deleted. Training (`SFTDataset`) truncates the
   other way (`prompt_ids[-max_prompt_len:]`, keeping the end). Changed
   `evaluate.py` to keep the end and use the same 1536 budget.

Stage 2's own test loop (`stage2_sft_qwen.py:1085`) already used clean
greedy decoding (no penalty, correct truncation) — so **88.43% is the
trustworthy number and 72.39% was an `evaluate.py` artifact**. This fix
aligns the two. (This is the same class of bug as §10 but on the opposite
side — §10 was validation *under*-generating during training; this is the
final evaluator *mis*-generating.)

### 13.2 Stage 3 LR was 20x too small — the policy never moved

The user correctly flagged the Stage 3 run as "not right." The log proves
it: `kl=0.0000` at steps 1, 50, 100 — the policy literally never changed
from the Stage-2 start. Cause: `STAGE3_LR=1e-7` with `STAGE3_STEPS=600`
(and grad_accum 4 = 150 real updates) is far too little to move a 128M-param
LoRA. The project's `main` branch ran Stage 3 at `LR 2e-6` / `3000 steps`,
which actually trains. Restored `STAGE3_LR=2e-6` and set
`STAGE3_STEPS=1500` (user-requested); `STAGE3_GROUP_SIZE` stays 8. The
low-reward-variance skips the user saw at step 150 were a compound symptom
of this dead LR plus §12's broken parser zeroing rewards — both now fixed.

### 13.3 Stage 2 majority-class collapse — step-token loss weighting

Even at 88% overall, the confusion matrix shows a real structural weakness:
Stage 2 predicts "Exploit" (the majority class, n=526 train) ~136 times when
gold=92, and rare classes 4/6/8 get **zero** correct — step macro-F1
collapsed to 0.468. Root cause: Stage 2's SFT loss is HuggingFace's uniform
average over the ENTIRE JSON target (~200-300 tokens, dominated by the
free-text explanation); the "New step" canonical label is only ~10-20 of
those tokens, so the classification signal is heavily diluted and the model
minimizes loss via fluent prose + the majority default.

Fix: `STAGE2_STEP_TOKEN_LOSS_WEIGHT=5.0` (config) upweights the loss on the
"New step" value span — which `SFTDataset` already computes as `step_span`
but never used for loss — via a manual next-token weighted cross-entropy in
`forward_batch` (train path only; validation keeps the plain loss for
comparability). Verified numerically in isolation: the coordinate-frame math
(un-prefixed span → prefix-shifted → next-token-shifted indices) upweights
exactly the two step-value tokens 5x and leaves explanation tokens at 1x,
prompt/prefix tokens masked at 0; the resulting loss is finite, differs from
uniform, and backprops cleanly. **Not yet validated by a real Stage 2
retrain** — this changes the training objective, so it needs a real run to
confirm it lifts the rare-class recall without hurting the majority class.

### Recommended next steps (in order)
1. `python eval/evaluate.py --model llm --adapter-dir ../checkpoints/stage2_qwen_lora`
   with NO retrain — the eval fixes (§13.1) alone should lift the *reported*
   Stage-2 numbers from 72% toward the ~88% Stage 2's own loop already sees.
2. Retrain Stage 2 (`python training/stage2_sft_qwen.py`) with the
   step-token weighting (§13.3) to attack the rare-class collapse, then
   re-eval.
3. Run Stage 3 (`python training/stage3_grpo_rl.py`) with the fixed LR
   (§13.2) on top of the improved Stage 2.

## 14. Stage-1 -> Stage-2 interface rewritten (the real bottleneck)

The `GraphPrefixAdapter` was:

```
graph_emb (B,512) -> Linear(512,10240) -> Linear(10240,10240)
                  -> Linear(10240, 8*5120) -> reshape (B,8,5120)
```

Two serious problems:

1. **Information.** All 8 tokens were slices of a single expansion of ONE
   pooled 512-d vector. They could not carry independent graph content — the
   entire PTT graph (node titles/types/statuses, degrees, depths, edge types,
   topology) was squeezed through one global summary before Qwen saw
   anything. Meanwhile `GraphEncoder.forward_with_nodes()` already computes
   per-node hidden states and Stage 1 uses them internally for its own
   cross-attention — they were simply discarded at the boundary.
2. **Parameters.** That stack is **~530M trainable parameters** (measured:
   529.5M at llm_hidden=5120) to expand one 512-d vector, trained on ~1.5k
   examples — larger than the LoRA itself and a severe overfitting liability.

**Fix:** replaced with a learned-query cross-attention resampler — the
standard mechanism for feeding a variable-size encoder output into a frozen
LLM (Flamingo's Perceiver Resampler, Alayrac et al. NeurIPS 2022; BLIP-2's
Q-Former, Li et al. ICML 2023). `GRAPH_PREFIX_TOKENS` learned queries
cross-attend over `[global_pooled ; per-node states]`, then residual + FFN +
projection to the LLM hidden size. `GRAPH_PREFIX_TOKENS` raised 8 -> 16.

Measured: **529.5M -> 14.6M parameters (36.3x smaller) with 2x the tokens**,
each free to attend to a different part of the graph. Prepending the global
token to the key/value set also guarantees every row has at least one valid
key, so a graph whose nodes are all padding cannot produce a softmax NaN.

All four call sites now pass per-node states (`forward_with_nodes`):
`stage2_sft_qwen.forward_batch`, Stage 2's validation and test loops,
`stage3_grpo_rl.build_prefix_embeds`, and `evaluate.py::eval_llm` (the
graph-conditioning ablation script inherits it via `build_prefix_embeds`).
The two `adapter.proj[0].in_features` dimension checks were updated to
`adapter.graph_dim` since `.proj` no longer exists.

**Verified** (synthetic, no GPU): output shape `(B, 16, H)`; finite output
including a row whose nodes are ALL padding; tokens carry distinct content
(max pairwise diff 3.08, not degenerate copies); output genuinely changes
when per-node states change (delta 5.86 — it is not ignoring them); the
global-only fallback path still works for any caller without node states;
learned queries and node projection both receive gradient; and the full
frozen-GINE -> resampler path handles ragged node counts (6/11/3) correctly.
**Not yet validated by a real training run — this changes the Stage-2
interface, so Stage 2 (and Stage 3 after it) must be retrained.**

---

## 15. Stage-2 regression post-mortem: the adapter rewrite could not train

### Symptom

The `GraphPrefixAdapter` rewrite (§ earlier — learned-query cross-attention
resampler) made Stage 2 *worse*, not better:

| Metric | Before rewrite | After rewrite |
|---|---|---|
| `val_step_acc` @ epoch 7 | 0.7950 | **0.6151** |
| `train_loss` @ epoch 2 | 1.6653 | **2.4312** |
| Step Exact Match (test) | 88.43% | **66.42%** |
| Mean MCP Jaccard | 0.8675 | **0.7602** |

Stage 1 improved over the same period (step_accuracy 0.7761 → 0.7948), so the
regression was isolated to Stage 2, and the adapter rewrite was the only change.

### Root cause 1 — the resampler shared LoRA's learning rate

The optimizer put LoRA and the adapter in a single param group at
`STAGE2_LR = 2e-6`. That is right for LoRA (low-rank deltas on an already
pretrained 14B) and badly wrong for the adapter (14.6M parameters trained
**from random init**).

AdamW's per-step update magnitude is ~`lr` (the gradient is normalized by
`sqrt(v)`), so over the run's 744 optimizer steps a parameter can travel at
most `744 x 2e-6 = 1.5e-3`. The learned queries are initialized at
`randn * 0.02`, so they move only ~7% of their init scale and stay effectively
**random**. Random queries attend near-uniformly over the node set, which
collapses all 16 prefix tokens toward the same vector (the mean node state) —
strictly *less* informative than the old adapter's 8 distinct fixed random
projections.

Measured directly (744 AdamW steps, same init, same target):

| LR | query movement `|dq|/|q0|` | mean inter-token cosine | final loss |
|---|---|---|---|
| 2e-6 (shared) | 3.38% | +0.157 (partially collapsed) | 1.1314 |
| 1e-4 (x50) | 11.20% | **-0.001** (fully distinct) | **0.0045** |

**Fix:** give the adapter its own AdamW param group at
`STAGE2_LR * STAGE2_ADAPTER_LR_MULT` (50x → 1e-4, the standard LR for
training a projector/resampler from scratch on a frozen LLM; Flamingo and
BLIP-2 both train their resamplers at 1e-4). LoRA keeps 2e-6.
`get_cosine_schedule_with_warmup` is a `LambdaLR`, so it scales each group's
own `base_lr` — the 50x ratio is verified preserved at steps 20/372/744.

This is the **same bug class** as Stage 1's graph gates, fixed the same way
(`STAGE1_GRAPH_GATE_LR_MULT`). `trainable_params` is still built as a flat
list so the existing grad-clip and non-finite guards are untouched.

### Root cause 2 — soft tokens entered the residual stream ~70x oversized

The adapter ended in `nn.LayerNorm(llm_hidden)`, which forces unit per-element
RMS, i.e. per-token L2 norm `sqrt(5120) = 71.6`. Real Qwen `embed_tokens` rows
have L2 norm **~1.0** (measured on the locally cached Qwen2.5-1.5B-Instruct:
mean 1.023, median 1.015, p99 1.259 — Qwen3-14B weights are not cached on this
machine, so this is a same-family proxy).

Qwen is a **pre-norm** transformer: each block reads `RMSNorm(h)` but writes an
O(1) update back into `h`. Against a norm-71 residual that update is ~70x too
weak to move the token, so the graph prefix passed through all 40 layers
essentially **unchanged** — never contextualized by the LLM, never integrated
with the text.

**Fix:** initialize the final LayerNorm's gain to `1/sqrt(llm_hidden)`, putting
the output at per-token L2 norm ~1.0 (verified: 1.0000). It stays learnable and
per-channel, so training can still adjust it — this only fixes the *starting*
scale. Done via the existing LayerNorm weight rather than a new parameter, so
**state_dict keys are unchanged** and previously saved `graph_adapter.pt` files
still load `strict=True` (verified: 25 tensors, clean load).

### Verification

- Adapter output: `(B, 16, 5120)`, per-token norm 1.0000, 14.6M params, finite.
- NaN safety retained: all-nodes-masked and no-`node_states` paths both finite
  (the always-valid global token keeps every softmax row populated).
- LR-group ratio held at exactly 50.0x across warmup, mid-run, and decay.
