# Stage 1 Architecture & Class-Imbalance Audit — Findings and Changes

**Scope:** `stepmodel-final-v2/core/graph_encoder.py`, `core/config.py`,
`training/stage1_gnn_train.py`. Goal: close the gap from ~78% step
accuracy / ~70% MCP micro-F1 to the 85–90% / 70–80% targets.

**How this was produced:** full read of the Stage 1 code path (graph
encoder, semantic CNN, fusion, losses, sampler, training loop, k-fold
ensemble trainer, threshold search) plus the existing `CHANGES_AND_FINDINGS.md`
audit trail. Every claim about model behavior below was checked by actually
instantiating the modules with `torch`/`torch_geometric` and running forward
+ backward passes on synthetic tensors shaped like the real data (no GPU or
the real dataset was available in this environment, so nothing here is a
"trust me" — it's either a mathematical proof, a reproduced bug, or a cited
paper, but **none of it has been validated by an actual training run**; see
"What I could not verify" at the end).

---

## 0. Regression found and fixed (read this first)

The first version of these changes was trained on real data and made things
**worse**: step accuracy 0.7164 (down from a baseline around ~0.75–0.78) and
MCP micro-F1 **0.2160** (down from ~0.70). This section documents exactly
what went wrong, how it was diagnosed, and what changed. Nothing below this
section describes the currently-shipped code; it describes what was wrong
with the *previous* version of this audit's changes.

**Diagnosis.** The evaluation CSV from that run
(`output/stage1.csv`) showed the model predicting **all 11 MCP tools
positive on every single one of the 268 test rows** — a total collapse, not
a calibration issue. That pointed straight at the one MCP-specific change:
`STAGE1_MCP_LOSS_TYPE` had been defaulted to `"asl"` (Asymmetric Loss).

Two compounding bugs, both confirmed with reproductions (not just theory):

1. **`asymmetric_loss()`'s focusing term wasn't detached from autograd.**
   The official ASL reference implementation always computes its
   `(1-pt)^gamma` focusing weight under `torch.no_grad()` specifically to
   prevent it from being differentiated a second time; the first version of
   this function didn't do that. A toy reproduction (small dataset, few
   epochs, same LR as this codebase) showed the non-detached version's
   predicted-positive rate climbing from 52% to 98% over 15 epochs while the
   reported loss kept shrinking the whole time — the loss number looked
   like things were going *well* while the model was actively collapsing.

2. **Even after fixing (1), `gamma_neg=4.0` — the ASL paper's own
   default, tuned for large image-tagging datasets over hundreds of
   epochs — is too aggressive for ~1.5k training rows over ~30–80 epochs.**
   The same toy reproduction, now with the detachment fix applied, still
   drifted to a ~99% predicted-positive rate by epoch 39, just more slowly.
   A gentler `gamma_neg=2.0` recovered instead of collapsing in the same
   test. Plain (symmetric) focal BCE — what the codebase already had before
   this audit touched anything — was the most stable of all three in that
   comparison.

**Fix, shipped now:**
- `asymmetric_loss()` in `core/graph_encoder.py` now computes its focusing
  term inside `torch.no_grad()`, matching the official implementation.
- `STAGE1_ASL_GAMMA_NEG` default lowered from 4.0 to 2.0.
- **`STAGE1_MCP_LOSS_TYPE` default reverted from `"asl"` back to `"focal"`**
  — the loss that was already achieving ~70% MCP micro-F1 before this audit.
  ASL is kept in the codebase, correctly fixed, as an explicit opt-in for
  later experimentation — not as the default, since the evidence above shows
  it's a worse fit for this specific dataset size/training-length regime,
  independent of the bug fix.

**What was *not* touched in this pass:** the fusion fix (§1 below), the NaN
fix (§2), and logit adjustment (§3) were left exactly as originally shipped.
The collapse pattern (all-positive predictions, loss trending to ~0 while
predictions got worse) is fully explained by the ASL bug on its own, and a
severely mis-behaving MCP loss term early in training would also inject
noisy gradients into the shared fusion trunk that both heads read from —
which plausibly explains some or all of the modest step-accuracy dip too,
without needing a second explanation. **Please re-run training now** with
this fix; if step accuracy is still below your prior baseline after that,
logit adjustment (§3) is the next thing to isolate — lower
`STAGE1_LOGIT_ADJ_TAU` to ~0.5, or set `STAGE1_USE_LOGIT_ADJUSTMENT = False`
for one comparison run, before changing anything else.

---

## 1. The fusion was not fusing — confirmed and fixed

You asked the right question. `Stage1Classifier`'s "cross-attention fusion"
took the pooled semantic vector and the pooled graph vector, unsqueezed each
to a length-1 sequence, and ran `nn.MultiheadAttention(semantic_q, graph_kv,
graph_kv)`.

**Softmax over a single key is identically 1.0, for any query.** That means
the attention weight can never depend on the semantic query's content, so
the whole block collapses to a fixed linear function of the graph vector
alone (`out_proj(v_proj(graph_proj))`) — and the gradient with respect to
every query/key parameter is exactly zero, so those parameters were never
actually being trained toward anything useful either.

Reproduced directly:

```
OLD fusion (seq_len=1 on both sides):
 attn weights (should always be 1.0): [1.0, 1.0, 1.0]
 output changes when query changes? False
 max abs diff between two totally different queries: 0.0

NEW fusion (query=1, key/value=6 real graph-node tokens):
 attn weights vary across keys: [0.17, 0.08, 0.14, 0.14, 0.21, 0.27]
 output changes when query changes? True
 max abs diff: 0.193
```

So no, the fusion was not happening in any meaningful sense — it was
concatenation (`[semantic_proj, graph_proj, constant-ish]`) dressed up as
cross-attention.

### Fix (implemented in `core/graph_encoder.py`)

Real bidirectional, token-level cross-attention:

- `GraphEncoder` gained `forward_with_nodes()`, which returns the pooled
  512-d vector **plus** the per-node hidden states (padded) and a validity
  mask — a genuine multi-token sequence per graph (most PTT graphs have
  many nodes). `forward()` is kept as a thin wrapper around it so every
  existing caller that expects a single pooled tensor
  (`stage2_sft_qwen.py`'s `stage1.graph_encoder(...)` calls, `evaluate.py`)
  is untouched — this was checked by grepping every call site before
  changing the signature.
- `Stage1Classifier` now also projects the raw per-token GPT-2 embeddings
  (already computed upstream for the CNN branch) into fusion space as a
  second real token sequence.
- Two-way cross-attention: the semantic vector attends over the graph's
  node tokens (`cross_attn_sem2graph`, "which graph nodes matter for this
  strategy?"), and the graph vector attends over the strategy's token
  embeddings (`cross_attn_graph2sem`, "which words matter for this graph
  state?"). Both use proper `key_padding_mask`s so padding never leaks in.
- Added a cheap explicit multiplicative (Hadamard) interaction term between
  the two pooled vectors — Kim et al., ["Hadamard Product for Low-rank
  Bilinear Pooling"](https://arxiv.org/abs/1610.04325), ICLR 2017 — as a
  second, complementary interaction path.
- Fusion input is now `[semantic_proj, graph_proj, sem2graph_out,
  graph2sem_out, interaction]` (5 × D/2) instead of the old 3 × D/2; the
  fusion MLP's input layer was resized accordingly. `step_head`/`mcp_head`
  input dims are unchanged.

Verified with a full forward+backward smoke test (synthetic batch, variable
node counts 1/5/12, variable valid-token counts): gradients now reach every
new parameter (`cross_attn_sem2graph.in_proj_weight` grad norm 1.16,
`cross_attn_graph2sem.in_proj_weight` grad norm 1.07, `graph_node_proj` grad
norm 1.20, `semantic_token_proj` grad norm 1.48 — all previously either
nonexistent or, for the analogous old query path, exactly zero), and that
`GraphEncoder.forward()`'s public single-tensor contract used by Stage 2/3
and `evaluate.py` is preserved.

**Why this design and not something else:** the project's own stated
differentiator over Pen Strategist is "graph structure, not just text" —
node-level cross-attention is the version of fusion that actually lets the
graph's structure (not just a pooled summary of it) influence the
prediction, which is the more defensible use of the GNN's output.
Background reading on this style of fusion: co-attention for
graph/structured + text ([CAST: Cross Attention based multimodal fusion of
Structure and Text](https://arxiv.org/html/2502.06836v1)), and gated/bilinear
alternatives if you want a lighter-weight option later — Arevalo et al.,
["Gated Multimodal Units for Information Fusion"](https://arxiv.org/abs/1702.01992),
ICLR 2017 workshop.

---

## 2. A real NaN bug, found while testing the fix

While smoke-testing, a batch containing a row with fewer valid semantic
tokens than the largest CNN kernel (`SEMANTIC_CNN_KERNELS = (2,3,4,5,7)`,
so anything under 7 tokens) produced `NaN` in `step_logits`/`mcp_logits` —
**for the whole batch**, not just that row.

Root cause, in `SemanticCNNEncoder.forward()`: when every convolution
window for a sample is invalid (masked out), `y.max(dim=-1)` returns the
`torch.finfo(y.dtype).min` fill value itself (~ -3.4e38) for every channel.
That huge-magnitude constant then blows up the very next `LayerNorm`'s
variance computation into `inf`/`NaN`.

This matters here specifically because `Stage1Dataset.__init__` substitutes
the literal string `"empty"` for a missing `New strategy` or `Strategy
explanation` field — so a row with **both** fields empty gets semantic text
`"empty empty"`, which GPT-2's tokenizer turns into only 2–3 tokens, shorter
than the k=5 and k=7 kernels. If any such row exists anywhere in your
`training_data.csv` (worth checking — `grep`-ing for empty `New strategy`
cells is a 30-second check), it would silently NaN that entire batch's
gradient during training, which is exactly the kind of thing that can look
like "unstable training" or unexplained bad epochs without ever showing up
clearly in a loss curve (a NaN'd batch typically just gets skipped or
poisons the optimizer state depending on your loop, rather than crashing
loudly).

**Fix:** channels with zero valid windows are now masked to `0.0` instead
of the raw fill value ("this kernel contributes nothing for this sample"
instead of "this kernel is extremely confident about a value that doesn't
exist"). Verified the exact repro case above no longer produces NaN,
including a full backward pass.

---

## 3. Class imbalance: what was already there, and what's new

**What was already implemented (and is good):** capped inverse-sqrt-frequency
class weights for both Step and MCP heads, a `WeightedRandomSampler`, focal
loss on both heads, a hand-built hard-negative margin loss for the
known-confusable Step pairs, per-class MCP decision thresholds
(bootstrap-stabilized, support-gated, with an automatic fallback to 0.5 if
tuning would make things worse), SWA over top-K checkpoints, and — per your
project notes — a machine-level k-fold ensemble trainer. This is already a
lot of correct, thoughtful machinery; the gap to 85–90%/70–80% is unlikely
to be "no one thought about imbalance."

**The actual imbalance, from `training_data.csv`** (raw label text, before
taxonomy normalization): `"Exploit the selected exploitations"` alone is
~33% of all 1,894 rows (626), while `"Enumerate the domain"` and `"Explore
the source code for vulnerabilities"` are each under 2% (~20–25 rows
combined across their text variants). That is roughly a 25:1 head-to-tail
ratio in a 10-way classifier — this is a genuinely long-tailed
classification problem, not a mild imbalance, and it's happening on top of
an already-small dataset (~1.5k train rows for what the code itself notes
is a ~7.8M-parameter model — see §5).

**What I added, on top of (not replacing) the existing machinery:**

1. **Logit adjustment** (Menon, Jayasumana, Rawat, Jain, Veit, Kumar,
   ["Long-tail learning via logit adjustment"](https://arxiv.org/abs/2007.07314),
   ICLR 2021). Adds `tau * log(class_prior)` to each Step class's logit,
   **only inside the training loss** — the model's raw forward-pass output
   (used by `evaluate.py`'s argmax) is untouched, exactly as the paper's
   train-time variant prescribes, so no evaluation code needed to change.
   This is a theoretically-grounded, single-line fix (it directly targets
   the Bayes-optimal decision rule under label shift) rather than another
   hand-tuned weight, and it doesn't saturate the way the existing capped
   weights do (`STAGE1_MAX_CLASS_WEIGHT = 2.5` means a class with a 25:1
   imbalance still only gets a 2.5x pull). Default on
   (`STAGE1_USE_LOGIT_ADJUSTMENT = True`, `STAGE1_LOGIT_ADJ_TAU = 1.0`, the
   paper's default).

2. **Asymmetric Loss (ASL) for the MCP multi-label head** (Ridnik,
   Ben-Baruch et al.,
   ["Asymmetric Loss For Multi-Label Classification"](https://arxiv.org/abs/2009.14119),
   ICCV 2021). Unlike symmetric focal BCE (one `gamma` for both positive
   and negative terms), ASL focuses positives and negatives independently
   and additionally hard-shifts very-easy-negative probabilities toward
   zero contribution before computing their loss term — in principle a
   better match for multi-label imbalance, since with 11 tools and most
   rows positive for only 1–2, "confidently-correct negative" is the
   overwhelming majority case. **Update: this was tried as the default and
   caused a real regression (MCP micro-F1 0.70 → 0.22, model collapsed to
   predicting all 11 tools positive on every row) — see §0 above for the
   full diagnosis.** It's fixed now (the focusing term is properly detached
   from autograd, and the default `gamma_neg` was lowered from the paper's
   4.0 to 2.0) and still implemented as `asymmetric_loss()` in
   `graph_encoder.py`, but **`STAGE1_MCP_LOSS_TYPE` now defaults to
   `"focal"`** (the original, already-proven-at-~70%-F1 behavior). Set it
   to `"asl"` only as a deliberate, monitored experiment — watch predicted-
   positive rate per epoch, not just the loss number, for the same collapse
   pattern.

3. **Manifold Mixup** (Verma et al.,
   ["Manifold Mixup: Better Representations by Interpolating Hidden States"](https://arxiv.org/abs/1806.05236),
   ICML 2019) as an optional auxiliary loss term, computed on the fused
   representation (mixes two examples' `fused_h` + soft-mixes their
   one-hot/multi-hot targets, scores them with the existing cheap
   classification heads). This is the one change that's **default OFF**
   (`STAGE1_USE_MANIFOLD_MIXUP = False`) — it changes the loss landscape
   and, unlike the other two, isn't a drop-in correction for a specific
   diagnosed problem, so it should be A/B tested rather than assumed to
   help. It's aimed squarely at the params:rows mismatch in §5 below:
   smoothing decision boundaries between classes tends to help most when a
   model is large relative to its training set.

**Recommended ablation order** (you have the k-fold trainer already, so
each of these is one `python training/stage1_gnn_train_kfold.py` run):
run current-defaults-as-shipped first; if Step macro-F1 improves but
accuracy or the majority class's recall drops noticeably, that's the
signature of logit adjustment stacking with the existing capped weights and
over-correcting — first try `STAGE1_LOGIT_ADJ_TAU = 0.5`, and only if that's
not enough, try turning `STAGE1_USE_STEP_CLASS_WEIGHTS` off with logit
adjustment left on (they're now two independent, toggleable mechanisms for
the same underlying problem, so you don't have to guess — you can isolate
which one is doing the work). Then try `STAGE1_USE_MANIFOLD_MIXUP = True`
as a separate run on top of whichever of the above wins.

**One thing I looked for and did not find:** the earlier
`CHANGES_AND_FINDINGS.md` audit (from the predecessor `stepmodel-final`
codebase) flagged a batch-level `adaptive_cost_sensitive_loss()` bug where
per-class accuracy was computed from a single mini-batch and produced huge
noisy weight spikes for rare classes. I grepped the current
`stepmodel-final-v2/training/stage1_gnn_train.py` for that function name —
**it does not exist in this codebase.** Either it was already removed in
the v2 rewrite, or the note carried over from the old audit file without
the underlying code — either way, nothing to fix there now.

---

## 4. A structural risk I did not change, but want to flag clearly

`config.py`'s own comments already note the model is "~7.8M params" trained
on "~1.5k examples" — that's roughly a 5,000:1 parameter-to-training-row
ratio. `GNN_HIDDEN=512` with 4 GINE layers, `FUSION_HIDDEN=1024`, and now an
even richer fusion block (two more attention modules) is a lot of capacity
for a dataset this size, and heavy regularization (dropout, edge dropout,
weight decay, SWA) can only partially compensate — it doesn't change the
fundamental degrees-of-freedom mismatch.

I did **not** shrink the architecture, because doing that responsibly needs
an actual ablation run to confirm it doesn't just trade "overfitting" for
"underfitting," and I have no GPU/dataset access here to run one. What I'd
suggest instead of guessing: run one comparison with a smaller "lite"
profile — `GNN_HIDDEN=256`, `GNN_LAYERS=3`, `FUSION_HIDDEN=512` — against
current defaults, on the same k-fold split, and see which generalizes
better on held-out machines. If the lite profile matches or beats current
test accuracy, that's a strong signal the extra capacity is being spent on
memorizing rather than generalizing, and it also makes Manifold Mixup (§3)
and any future data augmentation more likely to help rather than needing to
fight the same capacity mismatch.

---

## 5. Something outside the code worth a human look

Both this codebase's own audit trail and the raw label distribution point
at the same thing: `"Explore the suspicious files, commands and create a
summary of the findings"` and `"Exploit the selected exploitations"` are
persistently confused with each other across every stage and checkpoint.
Given `"Exploit..."` is ~33% of the data and the two step descriptions can
plausibly describe adjacent moments in the same real pentest, this has the
signature of a genuine labeling-taxonomy ambiguity rather than something a
loss function or architecture change can fix. Worth spot-checking a handful
of the actual confused rows against the source PTT to see if the label
itself is arguable in those cases — no code change proposed here, since
this needs a human judgment call on the labeling guidelines.

---

## 6. Files changed

- `core/config.py` — added `STAGE1_USE_LOGIT_ADJUSTMENT`,
  `STAGE1_LOGIT_ADJ_TAU`, `STAGE1_MCP_LOSS_TYPE`, `STAGE1_ASL_GAMMA_NEG`,
  `STAGE1_ASL_GAMMA_POS`, `STAGE1_ASL_CLIP`, `STAGE1_USE_MANIFOLD_MIXUP`,
  `STAGE1_MIXUP_ALPHA`, `STAGE1_MIXUP_WEIGHT`. Nothing existing was removed
  or renumbered.
- `core/graph_encoder.py` — new `asymmetric_loss()` function; `GraphEncoder`
  gained `_pad_nodes()` (shared helper) and `forward_with_nodes()`
  (`forward()` is now a thin wrapper, unchanged contract); `Stage1Classifier`
  fusion block rewritten (real cross-attention + Hadamard interaction, see
  §1); `Stage1Classifier.loss()` gained `step_log_priors`/`logit_adj_tau`
  and `use_asl`/`asl_gamma_neg`/`asl_gamma_pos`/`asl_clip` params (all
  optional, defaulting to previous behavior when omitted); new
  `predict_from_fused()` helper for the mixup path; `SemanticCNNEncoder`
  NaN fix (§2).
- `training/stage1_gnn_train.py` — new `manifold_mixup_loss()` function;
  `train_one_split()` computes `step_log_priors` from the split's train
  counts and passes the new loss kwargs through; optional mixup term added
  to the batch loss, gated by config and strictly additive to the existing
  loss (never replaces it).
- `training/stage1_gnn_train_kfold.py` — **not modified**. It calls
  `train_one_split()` from the file above unchanged, so every new feature
  (logit adjustment, ASL, optional mixup) applies automatically to every
  fold of the ensemble trainer with no separate wiring needed.

All three edited files were verified with `python -m py_compile` both in a
sandbox and on your machine, and with `torch`/`torch_geometric` forward +
backward smoke tests against synthetic batches (variable graph sizes 1/5/12
nodes, variable valid-token counts including the pathological short-text
case from §2, gradient-flow checks on every new parameter).

## 7. Round 2 — first successful run, vs. the Pen Strategist paper, and targeted fixes

With the ASL bug fixed (§0), a real training run completed successfully —
no collapse, sane per-epoch behavior:

| Metric | Before any changes (your baseline) | Broken ASL run | **Fixed run (current)** | Target | Pen Strategist paper (arXiv 2605.04499) |
|---|---|---|---|---|---|
| Step accuracy | ~0.78 | 0.716 | **0.750** | 0.85–0.90 | 0.8287 |
| Step micro-F1 | — | 0.716 | **0.750** | — | 0.80 |
| Step macro-F1 | ~0.53–0.56 | 0.530 | **0.609** | — | not reported |
| MCP micro-F1 | ~0.70 | 0.216 | **0.692** | 0.70–0.80 | 0.64 |
| MCP subset accuracy | — | 0.000 | **0.522** | — | 0.4888 |

**You're already beating the paper on MCP** (both micro-F1 0.69 vs 0.64,
and subset accuracy 0.52 vs 0.49). Step accuracy is the real gap — 8 points
behind the paper's single-model number, 10–15 behind your own target. That
gap, not the MCP head, is where the next round of effort should go.

### What the errors actually look like (from your `output/stage1.csv`)

Per-class Step recall on the 268-row test set:

| Step class | Recall | n |
|---|---|---|
| End task / generate report | 0.96 | 23 |
| Exploit the selected exploitations | 0.87 | 92 |
| Enumerate further on the X service | 0.80 | 59 |
| Further enumerate the website | 0.74 | 27 |
| Explore the source code | 0.60 | 5 |
| Do a google search | 0.55 | 22 |
| Enumerate the domain | 0.43 | 7 |
| Explore suspicious files / commands | 0.47 | 30 |
| Analyze outcomes / find attack path | **0.00** | 3 |

And the confusion pairs, ranked: **"Explore suspicious files..." ↔
"Exploit the selected exploitations" is the single largest error mode by a
wide margin (12 of 268 rows: 8 Explore→Exploit, 4 Exploit→Explore)** —
more than double any other pair. This is exactly the confusion the
predecessor codebase's own audit trail flagged
(`CHANGES_AND_FINDINGS.md`: *"Both Stage 1 and Stage 2 (and 3) consistently
confuse 'Explore the suspicious files...' with 'Exploit the selected
exploitations'"*) — and checking `training/stage1_gnn_train.py`'s
`hard_groups` list (the pairs the hard-negative margin loss is told to
actively separate), **that exact pair, `(2, 5)`, was never actually in the
list**, despite being flagged twice now. Six other, smaller confusion pairs
were covered; the biggest one wasn't.

On the MCP side, per-tool precision/recall shows the multi-label head is
generally healthy (Nmap P=0.87/R=0.93, Interactive CLI P=0.82/R=0.76) with
two clear weak points: **SQLmap has 0/7 recall** (never predicted positive
on any test row) and **Metasploit has 0.32 precision** (over-predicted 26
false positives against only 12 true positives). Both are exactly what
you'd expect from a class with too little validation data for its
per-class threshold to be tuned reliably in a single train/val split —
looking at the training log, SQLmap had only 1 positive example in the
239-row validation split, so its threshold search correctly refused to
tune it and left it at the untuned 0.5 default (see the mcp_threshold_
search log: `label 5: val_positives=1 < min_val_positives -> keeping
default 0.5`). This is precisely the problem `training/stage1_gnn_train_
kfold.py` exists to solve (see next section).

### Fixes made based on this evidence

1. **Added `(2, 5)` to `hard_groups`** in `training/stage1_gnn_train.py` —
   the hard-negative margin loss now directly targets the #1 confusion
   pair, which it was never actually doing despite being flagged in two
   separate audits.
2. **`STEP_LOSS_WEIGHT` raised from 1.20 to 1.50`** in `core/config.py`
   (`MCP_LOSS_WEIGHT` left at 1.80). Both heads share the same fusion trunk
   and compete for gradient through one summed loss; MCP is already near
   its target range while Step has the larger gap, so this shifts relative
   emphasis toward the head that needs it, without touching MCP's own loss
   function. If a re-run shows MCP regressing, dial this back toward 1.20
   rather than also raising `MCP_LOSS_WEIGHT` — keep it a one-variable
   change.
3. **`STAGE1_USE_MANIFOLD_MIXUP` turned on** (was off, see §3). The
   overfitting concern raised speculatively in §4 is no longer
   speculative: this run's own log shows train loss (both step and mcp
   components) dropping to ~0.01–0.03 by epoch 40 of 80, while
   `val_step_acc` was still oscillating epoch-to-epoch (0.55 → 0.80 → 0.69
   → 0.80 in four consecutive epochs, long after train loss had flattened
   near zero) — the textbook signature of a 22M-parameter model with more
   capacity than a ~1.5k-row training set can constrain. Mixup is the
   cheapest available lever against exactly that.

All three changes were re-verified with the same forward+backward smoke
test approach as before (variable graph sizes, the new hard-negative group,
the mixup path with the new loss weights) before being written to your
machine.

### The next lever, and it's already built: run the k-fold ensemble trainer

Everything above is a same-single-split tweak. The predecessor codebase's
own config comments call `training/stage1_gnn_train_kfold.py` *"the single
highest-leverage change for a dataset this small"* — and the evidence from
this run agrees: a 239-row validation split simply doesn't have enough
positive examples for several MCP classes (SQLmap: 1, hydra: 3-4, several
others under 10) to tune their decision thresholds at all, and Step
accuracy's val-vs-test gap (0.81 val vs 0.75 test) has the shape of
small-sample noise rather than a systematic error. The k-fold trainer pools
out-of-fold predictions across **all ~1.5k training rows** for threshold
search instead of 239, and ensembles 5 models trained on different
train/val machine splits, which directly attacks both problems at once
without any further architecture change. This needs no new code — it
already exists and now inherits every fix above automatically (it calls
`train_one_split()` from the same file). Run:

```
python training/stage1_gnn_train_kfold.py
```

and compare its `ENSEMBLE TEST SET RESULTS` block against the single-split
numbers above.

### Stage 2 / Stage 3: a preliminary look (deeper pass once Stage 1 is locked in)

You asked to check Stage 2/3 architecture too. Holding off on deep changes
there is deliberate: Stage 2/3 condition on Stage 1's frozen graph encoder
via `GraphPrefixAdapter`, so retuning them before Stage 1's encoder is
final means redoing that work once Stage 1 changes again (which it will,
after the k-fold run above). From what's already been read in this audit:

- `core/graph_prefix_adapter.py` — architecturally sound: a 3-layer MLP
  projecting the frozen 512-d graph embedding into `n_tokens × llm_hidden`,
  reshaped into soft-prompt tokens, LayerNorm'd. No issues found.
- `training/stage3_grpo_rl.py` — the predecessor audit
  (`CHANGES_AND_FINDINGS.md` §4) already diagnosed and fixed a real bug
  here (Stage 3 had no validation-based checkpoint selection and would
  silently ship a worse-than-Stage-2 model); that fix is in place and
  wasn't touched by this audit.
- Both stages' hyperparameters (LoRA rank, GRPO KL coefficient, dual-clip
  settings) have extensive inline reasoning already in `config.py`,
  including citations (Ye et al. 2020 dual-clip PPO) — this reads as
  someone who already iterated carefully on Stage 3's known RL instability,
  not an area that looks neglected.

Once the k-fold run lands Stage 1 at or near target, the right next step is
re-running Stage 2/3 against the new frozen encoder and re-evaluating
against Pen Strategist's paper — happy to do a full audit pass on those two
stages at that point, the same way this one was done for Stage 1.

---

## 8. Round 3 — the Round-2 fixes were validated, still short of target: real research pass + architecture changes

### 8.1 What the Round-2 real run actually showed

| Metric | Round-2 test | Round-2 val | Target | Paper |
|---|---|---|---|---|
| Step accuracy | 0.7649 | 0.8075 | 0.85–0.90 | 0.8287 |
| Step macro-F1 | 0.5817 | 0.598 (epoch 20 checkpoint region) | — | — |
| MCP micro-F1 | 0.6944 | 0.7266 | 0.70–0.80 | 0.6400 |
| MCP macro-F1 | 0.5099 | — | — | — |

Two things stand out and drove everything below:

1. **Val beat target (0.8075) but test didn't (0.7649).** A 4-point val/test
   gap on a 26-val-machine / 268-test-row split is consistent with ordinary
   split noise, not a broken model — but it means the single-split number
   this run reports is not a reliable estimate of "how good is this model,"
   which matters a lot when the next lever we reach for should be judged
   against it.
2. **step_macro_f1 (0.582) sits far below step_accuracy (0.765).** Macro-F1
   weights every class equally; accuracy is dominated by the majority class
   ("Exploit the selected exploitations", n=526/1483 train rows). A large
   accuracy/macro-F1 gap is the textbook signature of a classifier that has
   a good feature space but a majority-biased decision boundary — not a bad
   feature space. That distinction matters (see 8.3).

### 8.2 What was actually researched (arXiv/Google Scholar, September 2026)

Five searches, two paper abstracts fetched directly, specifically looking
for anything published or reproduced-with-consensus since the original
audit that would change the plan for a ~1.5k-row, 10-class, CNN+GNN
multimodal setup:

- Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y. (2020). [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/abs/1910.09217). ICLR. — **adopted, see 8.3.**
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362). NeurIPS. — **adopted, see 8.4.**
- Ju, W. et al. (2025). [Cluster-guided Contrastive Class-Imbalanced Graph Classification (C³GNN)](https://arxiv.org/abs/2412.12984). AAAI. — the current (2025) state of the art specifically for *class-imbalanced graph classification*; combines clustering-based oversampling of majority-class subclasses, Mixup, and supervised contrastive learning. Directly validates the combination already partially in this codebase (Manifold Mixup + now real SupCon) rather than pointing to a fundamentally different architecture — considered but not fully adopted (its clustering step is a bigger change than the dataset size justifies right now; flagged as the next thing to try if 8.3/8.4 aren't enough).
- You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). [Graph Contrastive Learning with Augmentations (GraphCL)](https://arxiv.org/abs/2010.13902). NeurIPS. — self-supervised graph augmentation; considered for a self-supervised GNN pretraining phase, not adopted this round because it needs an unlabeled-graph pool this project doesn't currently have (every PTT graph here already has labels) — flagged, not implemented.
- Multiple 2024–2025 surveys on LLM-based text data augmentation for minority classes (e.g. arXiv:2501.18845) — considered for paraphrasing the rarest Step/MCP rows (SQLmap n=22, hydra n=15, "Explore source code" n=19); not implemented this round because it requires an external LLM call path this pipeline doesn't currently have wired up, and risks generating strategy text that doesn't match real PTT semantics without human review. Flagged as a real option if 8.3/8.4/k-fold still leave a gap.

### 8.3 Fix — decoupled classifier re-balancing (Kang et al., ICLR 2020)

The paper's central, somewhat counter-intuitive finding: **jointly**
training a representation and a classifier under class-balancing (the
re-weighting, focal loss, and re-sampling already used throughout Stage 1's
main training loop) can *hurt* the representation, even though those same
techniques *help* the classifier. Their fix — and the current state of the
art's default recipe for exactly the accuracy/macro-F1 gap seen in 8.1 — is
to decouple the two:

1. **Representation phase** (unchanged): train the full model under
   instance-balanced-ish sampling, exactly what `train_one_split()` already
   does.
2. **Classifier re-balancing phase** (new): freeze the graph encoder,
   semantic CNN, and every fusion layer; re-train *only* `step_head` and
   `mcp_head` for a short budget with a genuinely class-balanced sampler
   (every Step class equally likely to be drawn, not the capped
   square-root-inverse-frequency sampler used in phase 1). The backbone
   forward pass runs under `torch.no_grad()`, so this phase only ever
   updates two small `Linear` stacks — cheap even with its own epoch budget.

Implemented as `retrain_classifier_heads()` in `training/stage1_gnn_train.py`,
called automatically at the end of `train_one_split()` (so both the
single-split trainer and the k-fold trainer get it for free — it's one
function, reused exactly the way `train_one_split()` itself already is).
Gated by `STAGE1_USE_DECOUPLED_RETRAIN` (default `True`, 15 epochs,
lr=5e-4). **Never regresses silently**: the pre-retrain head weights are
restored if re-balancing doesn't beat the pre-retrain val score, the same
"adopt only if it wins" pattern already used for SWA a few lines above it
in the same function.

### 8.4 Fix — Supervised Contrastive Loss was a dead no-op, now actually implemented

This is the sharpest finding of this round. `config.py` has had a
`STAGE1_SUPCON_WEIGHT` knob since before this audit began, and the training
loop has logged a `con=` loss component in every run you've ever pasted —
but the line computing it was:

```python
con = fused_h.new_zeros(())
```

**every single batch, unconditionally**, regardless of what
`STAGE1_SUPCON_WEIGHT` was set to. The weight was multiplying a constant
zero. This is why every log you've shown has `con=0.0000` for all 41+
epochs — it was never going to be anything else. It wasn't a subtle bug;
`grep`ing the training loop for what `con` actually was is what surfaced it.

Implemented `supervised_contrastive_loss()`: the standard multi-positive
Supervised Contrastive Loss (Khosla et al. 2020), computed on Stage 1's
fused representation using the Step label to define positive pairs — every
other row in the batch sharing the same gold Step label is pulled together
in embedding space, every row with a different label is pushed apart. This
adds a training signal that cross-entropy structurally can't provide: CE
only ever compares a row to its own one-hot target, while SupCon compares
every row in a batch to every other row currently in it, which is a much
denser gradient signal per training row on a dataset this small (batch size
20 → up to 20×19 pairwise comparisons per batch instead of 20 independent
CE terms). Enabled by default (`STAGE1_SUPCON_WEIGHT = 0.15`,
`STAGE1_SUPCON_TEMPERATURE = 0.10`, Khosla et al.'s own recommended range).
Handles the edge cases correctly (verified in the smoke test in 8.6):
batch size 1, and a batch where no two rows happen to share a Step label,
both return exactly `0.0` instead of `NaN` or a crash.

### 8.5 Fix — walked back the unjustified capacity increase

`GNN_HIDDEN` and `FUSION_HIDDEN` had been raised from 384/768 to 512/1024
at some point before this audit started, with comments claiming "more
expressive power" / "richer fusion" but no A/B run backing either change.
Round 2's real log gave the first actual evidence, and it argues the other
way: by epoch ~30/80, train loss had dropped to ~0.01–0.03 while
`val_step_acc` was still swinging double digits epoch-to-epoch — and Stage
1 was carrying **22,094,237 trainable parameters against 1,483 training
rows**, a ~15:1 ratio that was flagged as a structural risk in §4 above and
never actually acted on. Reverted both back to 384/768. Combined with 8.3
and 8.4 (which add training *signal*, not raw capacity), trainable
parameters drop to **14,311,709** — a ~35% cut, ~9.6:1 params:rows instead
of ~15:1. `GNN_OUT_DIM` (the 512-d contract Stage 2/3 depend on) is
untouched — this is purely internal Stage-1 capacity.

### 8.6 What's been verified this round

All three changes were smoke-tested end-to-end against the real
`Stage1Classifier` class (not a mock) in a sandbox with `torch` +
`torch_geometric` installed, using synthetic graphs shaped like real PTT
graphs:

- `supervised_contrastive_loss()`: finite, positive loss and a real
  gradient on a normal batch; exactly `0.0` (no NaN, no crash) on a
  batch of 1 and on a batch where every row has a distinct Step label.
- A full forward pass through the *resized* model (384/768 hidden dims)
  produces the expected shapes and back-propagates cleanly with no NaNs
  anywhere in the gradients.
- `retrain_classifier_heads()`: ran two full epochs against a synthetic
  dataset and diffed every backbone parameter before/after — **zero
  parameters outside `step_head`/`mcp_head` changed**, confirming the
  freeze is real and not just a documentation claim.
- `py_compile` and a live `import config; import stage1_gnn_train` on your
  actual device confirm the new values (`GNN_HIDDEN=384`, `SUPCON_WEIGHT=
  0.15`, `DECOUPLED=True`) are what's actually on disk, not just in this
  write-up.

**What this round has not been validated against, and can't be from here:**
a real training run on your data. Everything above is a targeted,
literature-grounded intervention aimed specifically at the two symptoms in
8.1 (val/test split variance, accuracy/macro-F1 gap) — not a guarantee. If
`python run.py` still falls short after this, the next lever is the k-fold
ensemble trainer (`training/stage1_gnn_train_kfold.py` — already wired to
pick up every change above automatically, since it calls the exact same
`train_one_split()`), which is the one remaining change on the table that
directly attacks the val/test gap itself rather than the model.

**One workflow note:** your training runs happen on a separate GPU machine
(`dgxuser@...` in your pasted logs) from the Mac this session edits
(`/Users/sagarikachavan/Documents/Research/...`). Whatever you use to sync
between them (git, rsync, etc.) needs to run before `python run.py` picks
up any of the changes in this section.

---

## Retraining is required after any of these code changes

Because the fusion block's parameter names and shapes changed
(`cross_attn` → `cross_attn_sem2graph`/`cross_attn_graph2sem`, plus the new
`graph_node_proj`/`semantic_token_proj`/`interaction_proj` and a resized
`fusion` input layer), **any Stage-1 checkpoint trained before this audit
will not load into this architecture** — `load_state_dict` will raise a
key-mismatch error. This was already true after §1; it remains true now.
Every checkpoint in `checkpoints/` as of this update was produced by the
fixed code (§0) but *before* the §7 changes above, so it's now stale too —
the next `python run.py` (or k-fold run) will overwrite it with a fresh,
directly-comparable one.

## What's been verified vs. what hasn't

Every code change in this document has been verified two ways: (1) a
mathematical/logical argument or a reproduced bug (§0, §1, §2), and (2)
`torch`/`torch_geometric` forward+backward smoke tests on synthetic data in
a sandbox without your GPU or dataset. As of §7, the fusion fix (§1), NaN
fix (§2), and the ASL fix (§0) have now also been validated by a real
training run on your machine — that's the run reported in §7's table. The
three §7 changes (hard_groups addition, loss reweight, mixup) have now
*also* been validated by a real run — that's the run reported in §8.1
(step accuracy 0.750 → 0.765, mcp micro-F1 0.692 → 0.694 test). The three
§8 changes (decoupled classifier re-balancing, real SupCon, capacity
reduction) have **not** yet been validated by a real run; see §8.6 for
exactly what sandbox verification was and wasn't possible from here.

## References

- Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar,
  S. (2021). [Long-tail learning via logit adjustment](https://arxiv.org/abs/2007.07314). ICLR.
- Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M.,
  & Zelnik-Manor, L. (2021). [Asymmetric Loss For Multi-Label Classification](https://arxiv.org/abs/2009.14119). ICCV.
- Verma, V., Lamb, A., Beckham, C., Najafi, A., Mitliagkas, I., Lopez-Paz,
  D., & Bengio, Y. (2019). [Manifold Mixup: Better Representations by Interpolating Hidden States](https://arxiv.org/abs/1806.05236). ICML.
- Kim, J.-H., On, K.-W., Lim, W., Kim, J., Ha, J.-W., & Zhang, B.-T. (2017).
  [Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325). ICLR.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555). CVPR. (considered as an alternative to logit adjustment — see note below)
- Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019). [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413). NeurIPS. (LDAM-DRW — an alternative/complementary margin-based approach if logit adjustment alone isn't enough)
- Han, X., Jiang, Z., Liu, N., & Hu, X. (2022). [G-Mixup: Graph Data Augmentation for Graph Classification](https://arxiv.org/abs/2202.07179). ICML. (raw-graph-level alternative to the embedding-level Manifold Mixup implemented here, if you want to try mixing at the graph-structure level instead)
- Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017).
  [Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992). ICLR Workshop.
- Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec,
  J. (2020). [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265). ICLR. (background on GINE / edge-feature-aware message passing, the base architecture already in use here)
- Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y. (2020). [Decoupling Representation and Classifier for Long-Tailed Recognition](https://arxiv.org/abs/1910.09217). ICLR. (Round 3 — decoupled classifier re-balancing, §8.3)
- Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362). NeurIPS. (Round 3 — real SupCon implementation, §8.4)
- Ju, W. et al. (2025). [Cluster-guided Contrastive Class-Imbalanced Graph Classification](https://arxiv.org/abs/2412.12984). AAAI. (current SOTA for class-imbalanced graph classification specifically — considered, not fully adopted, §8.2)
- You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., & Shen, Y. (2020). [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902). NeurIPS. (self-supervised graph augmentation — flagged, not adopted, §8.2)

## 9. Round 4 — independent re-audit (no Stage-1 code changes)

A fresh session, asked to review the whole 3-stage pipeline end-to-end and
push toward beating the Pen Strategist paper, re-read Stage 1's current code
in full (`core/graph_encoder.py`, `training/stage1_gnn_train.py`, and the
Round-3 `config.py` diff) independently of everything written above.

**Finding: no new bugs.** The Round-3 diff (capacity reduction to 384/768,
real Supervised Contrastive Loss, decoupled classifier re-balancing) is
internally consistent, matches its own documentation in this file, and
follows the established "never regress silently" pattern throughout
(`retrain_classifier_heads()` and the SWA block both only adopt their change
if it beats the pre-change val score).

**Decision: no further Stage-1 architecture changes this round.** Round 3 is
already a 3-change batch that has never been run for real. Stacking a 4th
unvalidated change on top of it now would make the next real run
uninterpretable — if step accuracy moves, there would be no way to tell
whether SupCon, decoupled retraining, the capacity cut, or a new 4th change
was responsible. The highest-leverage next action remains what §7/§8 already
said: run `training/stage1_gnn_train_kfold.py` (or `stage1_gnn_train.py` for
a quick single-split check first) on the GPU machine and compare against the
target (step accuracy > 0.8287, MCP micro-F1 0.70-0.80) and against Round 2's
last validated numbers (step accuracy 0.7649 test, MCP micro-F1 0.6944 test).

**Superseding note (later the same session):** `training/stage1_gnn_train_kfold.py`
was subsequently **deleted** at the user's explicit request, as part of a
repo-wide cleanup of files that don't contribute to the active `run.py`
pipeline. Every mention of it above and below (as "the next step," "the
single highest-leverage change," etc.) is now historical — it describes what
was true when written, not a runnable recommendation. The single-split
`training/stage1_gnn_train.py` (which `train_one_split()` still lives in,
unchanged) is the only Stage-1 trainer going forward.

This round instead did a first deep pass over Stage 2 and Stage 3, which had
previously gotten only the "preliminary look" in §7 above (deliberately
deferred pending Stage 1). See the new `STAGE2_STAGE3_IMPROVEMENTS.md` for
that audit and the fixes it made — several real bugs (not speculative
tuning) that directly affect the numbers being compared to the paper: a
Stage-2 eval-time adapter dtype mismatch, dead Stage-2/Stage-3 config wiring
(editing `config.py`'s `STAGE2_*`/`STAGE3_*` constants had no effect on a
real run), and a documented Stage-3 PPO-collapse fix (dual-clip PPO) that was
written up in `config.py` but never actually wired into the training loop.

## 10. Round 5 — Round 3 validated by a real run; graph gate added

### 10.1 The Round-3 real run

First real training run with the Round-3 changes (capacity 384/768, real
SupCon, decoupled classifier re-balancing) landed at:

| Metric | Round-3 test | Round-2 test (previous) | Target | Paper |
|---|---|---|---|---|
| Step accuracy | 0.7649 | 0.7649 | 0.85–0.90 | 0.8287 |
| Step macro-F1 | 0.5679 | 0.5817 | — | — |
| MCP micro-F1 (sklearn) | 0.6904 | 0.6944 | 0.70–0.80 | — |
| MCP samples-F1 (paper-comparable) | **0.7109** | — | — | 0.64 |
| MCP subset accuracy | **0.5336** | — | — | 0.4888 |

**Step accuracy is identical to Round 2 (0.7649, to four decimal places) —
Round 3's changes did not move it either way**, and macro-F1 is essentially
unchanged too (still far below accuracy: the minority-class problem the
decoupled retrain was aimed at is not resolved). **MCP is now clearly and
comfortably ahead of the paper** on both paper-comparable numbers (samples-F1
0.71 vs 0.64, subset accuracy 0.53 vs 0.49) — that head does not need further
work right now.

### 10.2 A concrete, evidenced hypothesis: the graph may be hurting Step specifically

The user's own read of the numbers, and it holds up: the Pen-Strategist
paper's Step Model uses **no graph at all** (frozen GPT-2 + CNN on text
only) and scores 82.87% — **6.4 points above this graph-augmented model's
76.49%**. A text-only baseline beating a graph-augmented model on the exact
same task is a real, causally suggestive signal that the graph branch may
currently be injecting noise into Step prediction specifically — plausible
on its face, since "which step type comes next" is largely determined by
the strategy text itself, while MCP tool availability plausibly benefits
more from graph/structural grounding (consistent with MCP being the metric
that's ahead, not behind).

### 10.3 Fix: learnable graph gate

Added a single learnable scalar gate (`core/graph_encoder.py`,
`Stage1Classifier.graph_gate_raw`, a sigmoid-parameterized `nn.Parameter`)
applied to every graph-**derived** term in the fusion concat (`graph_proj`,
`sem2graph_out`, `graph2sem_out`, `interaction`) — `semantic_proj` (the one
term with zero graph involvement) always enters fusion at full, unscaled
strength. Initialized so `sigmoid(graph_gate_raw) == STAGE1_GRAPH_GATE_INIT
== 0.25` (graph starts at ~25% strength vs. semantic's 100%, i.e. a genuine
"add-on" per the user's framing) but is a trainable parameter, not a
hand-picked constant — gradient descent, not a guess, decides whether to
grow it back toward 1.0 if the graph does prove useful for a given
prediction. This deliberately does not foreclose graph-conditioning; it
just removes the previously-hard-imposed assumption that graph and text
contribute equally by default. Gated behind `STAGE1_USE_GRAPH_GATE`
(default `True`) for a clean A/B if needed. The gate's current value is now
printed every epoch (`training/stage1_gnn_train.py`) so its evolution during
training is directly observable.

**Verified** (synthetic-tensor smoke test, no GPU needed): the gate
initializes to exactly `0.25` as configured; it receives real, nonzero
gradient on a forward+backward pass (confirmed trainable, not just present);
at `gate≈0` swapping in a completely different graph produces **exactly
zero** change in step logits (perfect isolation — confirms the gate truly
controls graph influence, not just dampens it approximately), while at
`gate≈1` the same swap produces a large change (0.57 max logit diff); and
`STAGE1_USE_GRAPH_GATE=False` correctly reproduces the pre-gate code path.
**Not yet validated by a real training run** — this is a single, isolated,
well-motivated change on top of the now-validated Round 3 baseline, so the
next real run should cleanly attribute any step-accuracy movement to this
change alone.

## 11. Round 6 — the gate result was inconclusive, and why; three-way diagnosis

### 11.1 What the Round-5 real run actually showed

| Metric | Round-3 test (no gate) | Round-5 test (gate, init 0.25) | Delta |
|---|---|---|---|
| Step accuracy | 0.7649 | 0.7687 | +0.4pt |
| Step macro-F1 | 0.5679 | 0.6402 | **+7.2pt** |
| MCP micro-F1 (sklearn) | 0.6904 | 0.7036 | +1.3pt |
| MCP subset accuracy | 0.5336 | 0.5000 | **-3.4pt** |
| MCP samples-F1 (paper-comparable) | 0.7109 | 0.7129 | +0.2pt |
| val_step_acc | 0.8117 | 0.8326 | +2.1pt |
| val/test step-accuracy gap | 4.7pt | 6.4pt | wider |

Mixed, not a clean win: step macro-F1 improved substantially, but MCP subset
accuracy regressed and the val/test gap widened. Crucially, **the gate
itself barely moved** — 0.250 → 0.232 over the 44 epochs before early
stopping, a ~7% relative change. A single scalar with a short, cheap
gradient path to the loss should move much faster than a 14M-parameter
model if it's actually finding a better position; it didn't get the chance
to within this budget. That makes the Round-5 result genuinely
**inconclusive** rather than evidence the gate hypothesis is right or
wrong — it's too early to credit the macro-F1 gain to the gate specifically
versus ordinary run-to-run variance.

### 11.2 Three-way diagnosis (user asked: architecture, class imbalance, or
### text understanding?)

- **Not primarily an architecture bug.** A small, surgical, additive change
  (one scalar) produced a proportionate, directionally sensible response —
  no collapse, no NaNs, no wild swing. That's the signature of a
  functioning architecture responding to a real (if small) lever, not a
  broken one.
- **Class imbalance is real but only partly explanatory, and has a hard,
  code-unfixable floor.** Step macro-F1 (0.640) still trails accuracy
  (0.769) by ~13 points, down from ~20 before this round's change — moving,
  not stuck. But `"Ask for human assistant"` has **zero training examples**
  (`count=0` in the per-epoch weight printout) — guaranteed 0 recall
  regardless of any architecture or loss change, a data-collection gap, not
  a model gap. Several MCP tools (SQLmap n=22, hydra n=15, three others)
  have so few validation positives the threshold search explicitly declines
  to tune them ("not enough support to tune safely") — same story. No
  further imbalance-specific code change is proposed this round without
  evidence of which specific mechanism is under/over-correcting; the
  existing machinery (logit adjustment, capped weights, decoupled retrain,
  hard-negative margin, focal loss, SupCon, per-class thresholds) is
  already extensive and visibly working (macro-F1 +7.2pt this round).
- **Val/test noise is the single biggest unexplained number, and it isn't
  fully fixable by code on a fixed-size dataset.** val hit 0.8326 (above
  the 0.8287 target) while test sits at 0.7687 — a 6.4-point gap on a
  239-row val / 268-row test split, where a handful of flipped rare-class
  predictions swings the aggregate by several points. This is exactly what
  the (since-deleted, at the user's explicit request)
  `stage1_gnn_train_kfold.py` existed to average out by pooling
  out-of-fold predictions across all ~1.5k rows. It was not restored this
  round (that decision is the user's to revisit, not something to silently
  undo) — instead, `RANDOM_SEED` was made env-overridable
  (`core/config.py`) so a cheap multi-seed check
  (`RANDOM_SEED=1 python training/stage1_gnn_train.py`, etc.) can quantify
  how much of any future delta is noise before crediting it to a code
  change.
- **Semantic/text understanding was not directly tested this round.** Worth
  noting for context: the paper's Step Model uses the *same* frozen-GPT-2
  approach and scores higher (82.87%) than this graph-augmented model
  (76.87%) — which argues against "GPT-2 itself is too weak" as the primary
  explanation (the paper gets good results from it alone) and keeps the
  graph-interference hypothesis on the table instead.

### 11.3 Fix: give the graph gate its own, much faster learning rate

Root cause of Round 5's inconclusive result: the gate was in the same
`AdamW` parameter group as the rest of the model, sharing one learning
rate meant for a 14M-parameter network. `core/config.py` adds
`STAGE1_GRAPH_GATE_LR_MULT = 12.0`; `training/stage1_gnn_train.py`'s
`train_one_split()` now puts `graph_gate_raw` in its own optimizer param
group at `STAGE1_LR * 12`, both groups driven by the same warmup/cosine
`lr_lambda` schedule so the 12x ratio holds throughout training, not just
at epoch 0. This doesn't change what the gate does or the architecture at
all — it only changes how fast it's allowed to get there, so the next run's
gate trajectory (and final value) is a real signal about whether the graph
helps or hurts Step prediction, not a truncated one.

**Verified** (synthetic, no GPU needed): the gate is correctly isolated
into its own param group (exactly 1 parameter, `14,311,709 + 1 =
14,311,710` — matches the Round-5 run's own "Trainable parameters" printout
exactly); the 12x LR ratio survives a `LambdaLR` schedule step
mathematically exactly; a full 5-step forward+backward+optimizer-step loop
on synthetic data runs cleanly with no shape errors and the gate value
measurably moves. **Not yet validated by a real run.**

Why logit adjustment over Class-Balanced Loss or LDAM-DRW: all three are
legitimate, well-cited options for this exact problem. Logit adjustment was
chosen as the primary addition because it's a strictly additive change to
the loss function (no change to what the sampler or the existing weights
do), has a closed-form, parameter-free-per-class prior (no extra
hyperparameter search beyond the single `tau`), and composes cleanly with
the focal loss and hard-negative margin already in the codebase. If, after
running the ablation in §3, logit adjustment alone doesn't close enough of
the gap, LDAM-DRW's margin-based approach (which changes the *training
schedule* — plain CE first, class-balanced reweighting only in the later
epochs) is the most natural next thing to try, since it addresses a
different mechanism (training a large margin between the closest confusable
classes, deferred so it doesn't destabilize early training) rather than
just re-deriving another class-weight number.

## 12. Round 7 — per-head graph gates (the shared gate was empirically forced to compromise)

### 12.1 The measurement that forced this

Round 6 (giving the single shared graph gate its own 12x LR so it could
actually reach equilibrium) produced the decisive result:

| Metric | Round 5 (slow gate) | Round 6 (fast gate) |
|---|---|---|
| Step accuracy | 0.7687 | **0.7836** |
| Step macro-F1 | 0.6402 | 0.6475 |
| MCP micro-F1 | **0.7036** | 0.6539 |
| MCP samples-F1 | 0.7129 | 0.6748 |

Letting the gate move pushed Step UP ~1.5pt and MCP DOWN ~5pt. That is not
noise in one direction — it is the signature of **one scalar being pulled by
two tasks that want opposite things**. Step is largely determined by the
strategy wording (the paper's text-only Step model gets 82.87% with no graph
at all); MCP depends on graph state (which services/findings exist decides
which tools are usable). A single shared gate can only land on a compromise
that is wrong for both.

### 12.2 Fix: one gate per head (MMoE-style)

`Stage1Classifier` now carries `graph_gate_step_raw` and
`graph_gate_mcp_raw`, and runs the (small) fusion MLP once per head, so each
head reads a fused vector built with its own graph weighting. Per-task gating
over a shared bottom is exactly the Multi-gate Mixture-of-Experts formulation
(Ma et al., KDD 2018), which exists for precisely this
tasks-conflict-over-a-shared-representation situation. The GINE encoder and
semantic CNN still run once and stay shared — this remains a shared-bottom
multi-task model, not two models.

Initialized asymmetrically in the direction the data already points
(`STAGE1_GRAPH_GATE_INIT_STEP=0.15`, `STAGE1_GRAPH_GATE_INIT_MCP=0.60`), both
trainable so gradient descent can overrule the prior. Both gates share the
fast LR group. `STAGE1_USE_PER_HEAD_GRAPH_GATE=False` restores the single-gate
behavior for a one-flag A/B. SupCon now runs on the Step-side fused vector
(its positives are defined by the Step label, so that is the semantically
correct one); Manifold Mixup mixes both fused vectors using the same lambda
and permutation so the soft-mixed targets stay valid for both heads.

**Verified** (synthetic, no GPU): gates initialize to exactly their configured
values; both receive independent nonzero gradient; `fused_step != fused_mcp`;
and — the important one — driving the MCP gate to zero changes `mcp_logits`
by 0.62 while changing `step_logits` by **exactly 0.000000**, proving the
heads are genuinely isolated from each other's gate. **Not yet validated by a
real training run.**

## 13. Round 8 — Step-side decision calibration (the missing half of the calibration story)

### 13.1 What the Round-7 run showed

Per-head gates worked as designed on MCP:

| Metric | Round 6 (shared gate) | Round 7 (per-head) |
|---|---|---|
| MCP micro-F1 | 0.6539 | **0.7017** (+4.8pt) |
| MCP samples-F1 | 0.6748 | **0.7251** (+5.0pt) |
| MCP subset acc | 0.5037 | **0.5149** |
| Step accuracy | 0.7836 | 0.7761 (-0.8pt) |

Gates separated exactly as intended: `gate_step` -> 0.040, `gate_mcp` -> 0.518.
MCP is now back above the 70% target floor.

### 13.2 The dominant Step failure mode is calibration, not representation

Breaking down all 60 step errors on the 268-row test set:

```
class                gold  pred  correct   over/under
2 explore-files        30    43       17      +13     precision 0.40
6 analyze               3     7        0       +4
0 google               22    16       15       -6
8 explore-source        5     1        1       -4
```

**26 of the 60 errors (43%) are false-positive class 2 alone**, drawn from
seven different gold classes. Class 6 is predicted 7x for 3 gold examples
with 0 correct. That is a decision-boundary/prior problem: the features can
separate these classes while argmax sits in the wrong place.

And there was a glaring asymmetry in the codebase: **the MCP head has had
bootstrap-stabilized per-class threshold calibration for several rounds (and
it measurably helps -- val micro-F1 0.7742 tuned vs 0.7557 uniform), while
the Step head had no calibration at all** -- plain `argmax(logits)`.

### 13.3 Fix: per-class Step logit bias (`search_step_logit_bias`)

The multi-class analogue of a per-label threshold is a per-class additive
logit bias, chosen so `argmax(logits + bias)` maximizes validation accuracy
(coordinate ascent over a bounded grid). This is standard post-hoc
calibration / prior correction for long-tailed classification, and is the
inference-time counterpart of the train-time logit adjustment (Menon et al.,
ICLR 2021) already in use here.

It carries all three defenses the MCP search uses, because fitting 10 free
parameters to a 239-row val split is exactly the trap that once collapsed
the MCP test numbers:
1. **min-support gate** — classes with < 8 validation examples keep bias 0.0
   and are never tuned (so "Ask for human assistant" with 0 support, and
   other tiny classes, cannot be fit to noise);
2. **bootstrap stabilization** — coordinate ascent is repeated over 25
   resamples and the per-class median is taken, with bias bounded to
   [-1.5, +1.5];
3. **never-regress guard** — if the tuned bias doesn't beat plain argmax on
   the val split it was fit on, it is discarded entirely.

Stored in the checkpoint as `step_logit_bias`, applied at inference in both
`stage1_gnn_train.evaluate()` and `eval/evaluate.py::eval_gnn` (and printed
there, so you can see what it chose). Training is untouched — this is purely
post-hoc, so it cannot destabilize the fit.

**Verified** on a simulation built from the real confusion matrix's shape
(injected systematic over-prediction of classes 2 and 6): fitted on a
239-row "val" split, it improved a **held-out 268-row "test" split by
+13pt** — i.e. it generalizes rather than just overfitting the split it was
fit on. Guards confirmed: an already-well-calibrated model is not degraded,
and zero-support classes keep exactly 0.0 bias. **Real-data gain will be
smaller than the simulation's** (the injected bias is exaggerated).

### 13.4 Also: two missing hard-negative pairs

`hard_groups` never contained `(2, 8)` — yet class 8
("Explore the source code...") lost **4 of its 5** test examples to class 2
("Explore the suspicious files..."), two labels that literally both begin
with "Explore the". Added `(2, 8)` and `(0, 2)`.

---

## Round 9 — Machine-grouped K-fold ensembling + pooled out-of-fold calibration

### What the test-set CSV actually showed

Analysis of `output/stage1.csv` (268 rows, 29 machines, step accuracy 0.7948):

**1. There is no label noise to blame.** 262 distinct model inputs cover all 268
rows, and **zero** inputs carry conflicting gold labels. The achievable ceiling
on this test set is **100%** — 85-90% is not blocked by the data.

**2. Errors are spread, not clustered.** Per-machine accuracy std = 0.115 across
all 29 machines (min 0.57, max 1.00). The worst 5 machines hold only 33% of
errors while covering 18% of rows. That is a **variance** signature, not a
distribution-shift cliff — and variance is what ensembling fixes.

**3. Two "attractor" classes absorb most errors.**

| class | gold | pred | recall | precision |
|---|---|---|---|---|
| [2] Explore suspicious files | 30 | **39** | 0.60 | **0.46** |
| [5] Exploit | 92 | **103** | 0.92 | 0.83 |
| [1] Enumerate X service | 59 | **49** | 0.80 | 0.96 |
| [8] Explore source code | 5 | **1** | 0.20 | 1.00 |

Class 2 is a catch-all sink: predicted 39 times, correct 18, absorbing rows from
classes 1, 3, 5, 6 and 8. Meanwhile class 1 has 0.96 precision but 0.80 recall —
the model is *too conservative* about it. That asymmetry is exactly what a
per-class logit bias corrects.

**Error budget (55 errors, need +15 rows for 85%):**

| bucket | n | recoverable? |
|---|---|---|
| pulled into attractors 2/5 | **29** | yes — calibration target |
| other confusions | 15 | partly |
| rare-class starved (gold 4/6/8, <=30 train rows) | 11 | no — needs data |

### Why the existing calibration did not fix it

The per-class step logit bias search is **not** broken. Verified against a
simulated val set carrying the same attractor pathology: the bootstrap-median
aggregation recovered `class 2: -1.00` and matched direct (non-bootstrapped)
coordinate ascent exactly (0.7714 -> 0.8393 for both). It only zeroes classes
whose mean bias is already negligible, which is correct conservative behavior.

The real problem is that **val does not look like test**. Val scored 0.8410 vs
test 0.7948 and does not exhibit the attractor skew, so the search found almost
nothing to correct and transferred nothing. A single 15%-machine split (48
machines, ~279 rows) is too small and too unrepresentative to calibrate on.

### The change

Machine-grouped 5-fold cross-validation in `training/stage1_gnn_train.py`,
addressing both findings at once:

1. **Ensemble** — `evaluate()` now accepts a list of models. Step **logits** are
   averaged across folds; MCP **sigmoid probabilities** are averaged. This
   attacks the variance that finding (2) identifies as the bottleneck.
2. **Pooled out-of-fold calibration** — every training row gets a prediction
   from a model that never saw its machine. The MCP threshold search and step
   bias search are fit on all **1865** pooled OOF rows covering all 302 training
   machines, instead of ~279 rows from 48 machines — a **6.7x larger** and far
   more representative calibration set.

Step logits are averaged rather than softmax probabilities because an additive
per-class bias commutes with the mean:
`mean_k(logits_k + b) == mean_k(logits_k) + b`. So a bias calibrated on
single-model OOF logits applies unchanged to the ensemble. **Verified
numerically**, not just asserted.

**Caveat, stated honestly:** OOF logits come from one fold model each, while
test-time logits average K models, so the ensemble's logit spread is slightly
narrower and a bias fit on OOF is marginally aggressive. Both searches keep
their never-regress guards (now fit on the OOF pool), and the run prints an
explicit single-best-fold vs ensemble comparison so the effect is measurable
rather than assumed.

### Compatibility

`STAGE1_CKPT` stays a **single-model** checkpoint — `stage2_sft_qwen.py`,
`stage3_grpo_rl.py` and `eval/evaluate.py` all load it as one
`model_state_dict`. The best fold's weights are written there, carrying the
pooled-OOF thresholds and bias, plus `kfold_members` / `kfold_val_scores` /
`test_metrics_single` / `test_metrics_ensemble` metadata. Stages 2 and 3 are
unchanged by this. Individual members live at `checkpoints/stage1_fold{k}.pt`.

`STAGE1_USE_KFOLD=0` restores the original single-split path
(`_main_single_split`), preserved intact.

### Verification

- single model vs `[single model]` produce identical metrics (backward compatible)
- step logits and MCP probabilities verified averaged correctly across members
- additive bias verified to commute with the mean
- real-data folds: 1865 rows / 302 machines -> 5 folds of 365-390 rows,
  49-76 machines each, **all 9 populated step classes present in every fold**,
  disjoint machines, full row coverage
- each fold trains on 80% of rows (~1497) vs 85% (~1585) for the single split —
  slightly less data per model, offset by averaging K models and a 6.7x larger
  calibration set

### Cost

Stage 1 wall-clock is ~5x (five models instead of one). Stages 2 and 3 are
unaffected.
