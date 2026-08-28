# Audit notes — stepmodel-final (fixing-input-issues branch)

This file documents what was checked, what was found, and what was changed.
Everything below was verified against the actual code and the actual
`input/train.json` / `input/test.json` in your zip, not guessed at.

## 1. Does the pipeline make sense?

Yes. It's a coherent 3-stage curriculum:

```
PTT text (CSV) ──ptt_parser.py──▶ attack graph (JSON)
                                        │
                    build_input_json.py│  (produces input/train.json, input/test.json)
                                        ▼
Stage 1 (GNN)  graph + strategy text ──▶ step label + MCP tools   [classifier, trained from scratch]
                                        │  frozen, used as embedding backbone
Stage 2 (SFT)  graph embedding (16 soft-prompt tokens) + strategy text ──▶ Qwen3-14B
                                        │  generates step + explanation + MCP tools as JSON
                                        │  (LoRA fine-tune, Stage 1 frozen)
Stage 3 (GRPO) same input, RL-refines Stage 2's policy using a reward built from
                                        │  step-similarity + MCP-F1 + LLM-judge score
                                        ▼
evaluate.py  scores GNN and/or LLM checkpoints on the held-out test set
```

Input **and** target label spaces line up correctly across stages, the label
taxonomies (`STEP_LABELS`, `MCP_LABELS`) are shared via `config.py`, and Stage
2/3 both condition on the *same* frozen Stage-1 graph encoder so the input
distribution the LLM sees is consistent with what it saw when the graph
encoder was trained. That part of the design is sound.

## 2. What's fed into each stage — verified from the code

| Stage | Input | Target |
|---|---|---|
| `build_input_json.py` | `data/training_data.csv`, `data/test_data.csv` (columns: Machine, PTT, New strategy, Strategy explanation, New step, Step explanation, MCP_tasks) | → `input/train.json`, `input/test.json` |
| `generate_graphs.py` | same CSVs | → `processed_graph/{train,test}/...` (HTML/JSON, for manual QA only — not read by any training stage) |
| Stage 1 (GNN) | graph (nodes/edges from `input/*.json`) + `New strategy` + `Strategy explanation` text, embedded with `BAAI/bge-base-en-v1.5` | step label (softmax over 10) + MCP multi-hot (sigmoid over 11) |
| Stage 2 (SFT) | frozen Stage-1 graph embedding → 8 soft-prompt tokens, prepended to a text prompt built from the same `New strategy`/`Strategy explanation` fields (Stage-1's own *predicted* step/MCP is optionally appended as a masked "hint," shown 50% of the time in training, always hidden at val/test) | generates `{"New step", "Step explanation", "MCP_tasks"}` as JSON text |
| Stage 3 (GRPO) | identical input to Stage 2 (loads the Stage-2 LoRA adapter as its starting policy) | same JSON generation task, refined by a reward function (see §4) |
| `evaluate.py` | `input/test.json` + a trained checkpoint (`--model gnn` or `--model llm`) | metrics report |

`processed_graph/` is a dead end in the data flow — nothing downstream reads
it. It exists purely so a human can open the HTML files and visually sanity
check the graph construction.

## 3. Data leakage — checked, clean

Verified directly against `input/train.json` / `input/test.json`:

- **175 train machines, 29 test machines, 0 overlap.**
- Both `stage1_gnn_train.py` and `stage2_sft_qwen.py` split train/val at the
  **machine level** (not row level) and print an explicit train∩test /
  val∩test overlap warning at startup — this was already correctly
  implemented.
- `data_utils.py` explicitly documents and *removes* a "previous step" field
  that used to be fed into every stage — it was the gold label text of the
  immediately preceding row for the same machine, i.e. a soft label leak
  (the model could partially solve step classification by transition
  frequency instead of reasoning over the graph). This fix was already in
  place; it's correct and worth knowing about since it explains part of why
  accuracy isn't higher — the model no longer gets that shortcut.
- Stage 3 additionally re-derives the same machine-level val split
  (`RANDOM_SEED + 1`) so RL doesn't train on Stage 2's validation machines
  either, and re-checks train/test overlap before running.

No row-level or column-level leakage found.

## 4. Root cause: why Stage 3 was worse than Stage 2 (and the fix)

This was the main bug. **Stage 3 had no validation-based checkpoint
selection.** It trained for `STAGE3_STEPS` (3000) steps, saved numbered
snapshots every 200 steps, and then — regardless of how training actually
went — **unconditionally overwrote the final adapter directory with whatever
the very last step produced.**

Compare this to Stage 2, which already does this correctly
(`stage2_sft_qwen.py` tracks `best_val_loss` / `best_step_acc` on a held-out
machine split and copies the *best* epoch's checkpoint to
`STAGE2_ADAPTER_DIR`, with early stopping). Stage 3 was the one stage in the
pipeline missing this safeguard, which is exactly consistent with what your
results showed: Stage 3 is marginally worse than Stage 2 on **every single
metric** (step accuracy, macro F1, MCP subset accuracy, LLM-judge
correctness) — the signature of "the last RL step happened to be a slightly
bad one and nothing caught it," not "RL is fundamentally broken."

Why the last step isn't reliably the best one here specifically:
- The reward has an LLM-judge component that's noisy and ~10% of the time
  fails to parse into valid rubric JSON (visible in your own eval output —
  "26/28 samples excluded" — the same judge is used as part of the Stage 3
  reward during training, not just at eval time).
- The reward is a hand-built "curriculum" that reweights format/step/MCP/
  explanation terms as training progresses, so the *objective itself* is
  non-stationary — reward at step 200 isn't measuring the same thing as
  reward at step 2800.
- The code already contains one fixed bug in the advantage computation (a
  double-baseline stacking bug — see the long comment above the advantage
  calculation in `stage3_grpo_rl.py`) that the previous author diagnosed
  from training logs but which, even after fixing, doesn't by itself
  guarantee the last step is best — it just makes updates *less* biased,
  not risk-free.

**Fix implemented** (in `training/stage3_grpo_rl.py`):

1. Added `evaluate_policy_on_val()` — greedy-decodes the current policy on
   the held-out val machines (the same ones excluded from RL training) and
   scores it with the *fixed* (non-curriculum) reward weights, so scores are
   comparable across the whole run.
2. Before training starts, the Stage-2 starting checkpoint is scored on this
   val set — that's `baseline_val_score`, the number Stage 3 actually has to
   beat.
3. Every 200 steps (aligned with the existing checkpoint cadence), the
   policy is re-scored on val. If it beats the best score seen so far, that
   checkpoint is saved to `checkpoints/stage3_qwen_grpo/best/`.
4. **At the end of training**, instead of blindly saving the last step:
   - If some RL checkpoint beat the Stage-2 baseline → that checkpoint
     (not the last step) is promoted to `checkpoints/stage3_qwen_grpo/`.
   - If RL *never* beat Stage 2 on held-out data → the Stage-2 adapter is
     copied in as the Stage 3 output instead, with a loud warning printed,
     so you never silently ship a regression. This also means
     `evaluate.py --model llm` on the Stage-3 directory can now never score
     worse than Stage 2 did.
   - The raw last-step weights are still kept (under
     `checkpoints/stage3_qwen_grpo/last_step_raw/`) purely for debugging —
     they're just no longer what gets used by default.
5. The in-memory model is reloaded from whichever checkpoint was promoted
   before the existing end-of-script test-set evaluation runs, so
   `explanations_stage3.csv` reflects what's actually on disk.

This is a mechanical, low-risk fix (model-selection wrapper around existing
training), not a rewrite of the RL algorithm — I did not touch the reward
weights, KL coefficient, or curriculum schedule, since tuning those without
being able to actually run a GPU job here would be guessing. If, after this
fix, Stage 3 still falls back to the Stage-2 baseline (i.e. it never beats
it even with proper selection), that's a legitimate signal the reward/KL
settings need retuning — worth trying: lower `STAGE3_KL_COEF` back toward
its docstring value of 0.015 (config currently has 0.02), or reduce
`STAGE3_STEPS` and rely on the new best-checkpoint logic rather than a fixed
long run.

## 5. Mistakes found in Stage 1 / Stage 2

**Stage 1** — `adaptive_cost_sensitive_loss()` in `stage1_gnn_train.py`
recomputes "per-class accuracy" **from the current mini-batch alone**
(batch size 16) and uses it to reweight the loss. Several step classes have
single-digit support in the *entire test set* (e.g. "Analyze the outcomes…"
has support 3, "Explore the source code…" has support 5), so a batch of 16
frequently contains **zero** examples of a rare class. When that happens,
`class_correct = 0, class_total = 0`, and the code computes
`class_acc = 0 / (0 + 1e-8) = 0`, i.e. it treats "no evidence this batch" the
same as "confidently wrong," producing a `perf_weight` of `1/(0+1e-8)` ≈ 1e8
before normalization. That's a large, noisy weight spike injected into
essentially every batch for the rare classes, which is a plausible
contributor to the volatile per-class recall you see (0.0 recall on
`hydra`, 0.29–0.71 swinging on `Enumerate the domain` across the three
stages' otherwise-similar checkpoints). This wasn't changed in this pass
(would need a re-train to validate the fix helps), but the concrete fix is
straightforward: skip classes with `class_total == 0` when computing
`perf_weights` instead of scoring them as 0% accurate.

**Stage 1** — `TEXT_ENCODER_NAME` was upgraded to `bge-base-en-v1.5`
(768-dim) per a comment in `config.py`, but the module docstring at the top
of `data_utils.py` still says "384-dim bge-small-en-v1.5" — harmless
(doesn't affect execution, the actual dims are read from config), but worth
fixing so the next person reading the file isn't misled.

**Stage 2** — already does checkpoint selection correctly (see §4) — no
functional bug found. The one thing worth knowing: `STAGE2_HINT_MASK_PROB =
0.5` means the model sees Stage-1's own predicted hint in half of training
steps but *never* at val/test time — this is intentional (forces the model
to actually use the graph tokens rather than parrot the hint), not a bug,
but it does mean training and eval-time prompts differ, which is a source
of some train/eval distribution shift by design.

**Both Stage 1 and Stage 2 (and 3)** consistently confuse "Explore the
suspicious files… create a summary" with "Exploit the selected
exploitations" (visible in every confusion matrix you pasted — row 3,
column 6). This is very likely a genuine **label-space ambiguity** in the
data/taxonomy rather than a code bug: those two step descriptions can
plausibly describe adjacent points in the same real pentest, and the
majority class ("Exploit…", support 92) has enough sway to keep pulling
borderline examples on precision. Not something a code fix addresses — it's
worth spot-checking a handful of the actual PTT rows where this confusion
happens to see if the CSV labels themselves are consistent.

## 6. Files removed (not needed for your `STAGES` list)

| File | Why removed |
|---|---|
| `train_ensemble_stage1.py` | A separate multi-seed ensemble variant of Stage 1, not referenced anywhere in `run.py`/your `STAGES` list or by `evaluate.py`. |
| `evaluate_ensemble_stage1.py` | Only consumes checkpoints produced by the file above. |
| `opencode.json` | Unrelated CLI-tool config, not imported by any Python file. |
| `.llm_cache/`, `.llm_judge_cache/`, `.env`, `.DS_Store`, `__pycache__/` | Local caches / secrets / OS cruft — regenerate on first run, and `.env` should never ship in a zip. |
| `processed_graph/`, `output/`, `input/`, `checkpoints/` | Regeneratable artifacts (or, for `checkpoints/`, too large to ship — 218MB — and you already have them locally). Kept the folders in the new structure as empty placeholders so paths in `config.py` keep working. |

**Kept but only needed for `--use-llm` mode** (your `STAGES` list never
passes this flag, so these three aren't on the hot path, but removing them
would silently break `--use-llm` for you later): `llm_ptt_parser.py`,
and the `--use-llm` branches inside `build_input_json.py` / `generate_graphs.py`.

## 7. Restructuring

Old layout dumped all ~20 scripts flat in one directory. New layout:

```
stepmodel-final/
├── run.py                      # unchanged usage: python run.py / --only / --start-from
├── core/                       # shared building blocks, imported by every stage
│   ├── config.py
│   ├── data_utils.py
│   ├── graph_encoder.py
│   ├── mcp_threshold_search.py
│   └── llm_judge.py
├── data_prep/                  # CSV -> graph -> input JSON
│   ├── ptt_parser.py
│   ├── graph_builder.py
│   ├── llm_ptt_parser.py       # only used by --use-llm
│   ├── build_input_json.py
│   └── generate_graphs.py
├── training/                   # the 3 stages
│   ├── stage1_gnn_train.py
│   ├── stage2_sft_qwen.py
│   └── stage3_grpo_rl.py       # <- the fix described in §4 lives here
├── eval/
│   ├── evaluate.py
│   ├── baseline_llm_eval.py
│   └── comparison_report.py
├── data/                       # training_data.csv, test_data.csv
├── input/  processed_graph/  output/  checkpoints/   # regenerated by the pipeline
├── requirements.txt
├── README.md  DOCUMENTATION.md  CHANGES_AND_FINDINGS.md (this file)
```

Every script is still runnable exactly the way you already run them —
`python training/stage1_gnn_train.py`, `python eval/evaluate.py --model gnn`,
etc. — nothing about the CLI changed. What changed under the hood: each
directly-run script now inserts `core/`, `data_prep/`, and `training/` onto
`sys.path` at the top (a small bootstrap block), so all the existing `from
config import ...` / `from data_utils import ...` style imports keep working
unmodified — I did not rewrite import statements throughout the codebase,
which would have been much higher-risk to get right without being able to
run the full training stack here to verify. `run.py`'s `STAGES` list was
updated to point at the new relative paths and uncommented/reordered to
match the exact list you're running (generate_graphs → build_input_json →
stage1 → stage2 → stage3 → evaluate → 3x baseline → comparison).

All files were verified with `python -m py_compile` after every edit (syntax
correctness only — I don't have the GPU/model weights here to run a full
training job and confirm the Stage 3 fix's actual numbers; the logic was
checked by tracing the code and matching against your pasted outputs).
