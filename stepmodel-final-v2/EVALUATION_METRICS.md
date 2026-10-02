# Evaluation Metrics — Definitions and Formulas

Source of truth: `eval/evaluate.py`, `core/llm_judge.py`, `training/stage3_grpo_rl.py`,
`core/config.py`. Every formula below was checked against the code, not
assumed — two claims that circulated earlier turned out to be stale and are
corrected inline (see the boxed notes in §3 and §4).

---

## 1. Step Classification Metrics

**Primary metrics**
- **Accuracy** — percentage of correct step predictions
- **Macro F1** — F1 averaged unweighted across all 10 step classes
- **Weighted F1** — F1 averaged, weighted by each class's support
- **Jaccard (step)** — exact-match ratio; identical to accuracy because step is single-label

**Diagnostics**
- Per-class precision / recall / F1 for each of the 10 step labels
- Confusion matrix (rows = gold, cols = predicted)

### Formulas

```
Accuracy = (# correct predictions) / (total samples)

For each class i:
  Precision_i = TP_i / (TP_i + FP_i)
  Recall_i    = TP_i / (TP_i + FN_i)
  F1_i        = 2 · Precision_i · Recall_i / (Precision_i + Recall_i)

Macro F1    = (1/10) · Σ F1_i                         (10 step classes)
Weighted F1 = Σ (support_i · F1_i) / Σ support_i

Jaccard (step, single-label) = 1.0 if pred == gold else 0.0
Mean Jaccard = (1/N) · Σ Jaccard_i
```

All predictions and gold labels pass through the same fuzzy-string
normalizer before comparison (exact match → regex rules → BGE embedding
similarity ≥ 0.55), so raw punctuation/truncation differences in the label
text never count as errors.

---

## 2. MCP Tool Classification Metrics

MCP is multi-label: 11 tools, any number can apply per row.

**Primary metrics**
- **Subset accuracy** — exact match of the *entire* predicted tool set (strictest)
- **Samples F1** — F1 computed per row, then averaged — **this is the paper's reported "Micro F1" (0.64)**
- **Micro F1** — sklearn's true micro-F1: TP/FP/FN pooled over the whole prediction matrix first, then one F1. Distinct from samples F1; not the paper-comparable number.
- **Macro F1** — F1 averaged per tool label, unweighted — surfaces failure on rare tools (e.g. `hydra`, `John-the-ripper`)

**Error analysis — over- and under-prediction**
- **Extra tools** (predicted but not in gold) — `avg_extra_tools`, `extra_tool_rate = mean(|pred−gold| / |pred|)`, `total_extra_tools`. Low **micro precision** signals this failure mode.
- **Missing tools** (in gold but not predicted) — `avg_missing_tools`, `missing_tool_rate = mean(|gold−pred| / |gold|)`, `total_missing_tools`. Low **micro recall** signals this failure mode.

**Other diagnostics**
- **Hamming loss** — fraction of individual tool slots wrong, out of N×11
- **Jaccard mean** — intersection/union per row, plus a pass rate at Jaccard ≥ 0.5
- Per-label precision/recall/F1 for each of the 11 tools

### Formulas

```
For each row i:
  pred_set_i = {tools predicted}
  gold_set_i = {tools in gold}

Subset accuracy = (1/N) · Σ [pred_set_i == gold_set_i]

extra_i          = pred_set_i − gold_set_i
missing_i        = gold_set_i − pred_set_i

Avg extra tools    = (1/N) · Σ |extra_i|
Avg missing tools  = (1/N) · Σ |missing_i|
Extra tool rate    = (1/N) · Σ (|extra_i| / |pred_set_i|)     [only rows with |pred_set_i| > 0]
Missing tool rate  = (1/N) · Σ (|missing_i| / |gold_set_i|)   [only rows with |gold_set_i| > 0]

Micro F1 (sklearn, pooled over the full N×11 matrix):
  TP = Σ_i Σ_j [pred_ij=1 ∧ gold_ij=1]
  FP = Σ_i Σ_j [pred_ij=1 ∧ gold_ij=0]
  FN = Σ_i Σ_j [pred_ij=0 ∧ gold_ij=1]
  Micro Precision = TP / (TP+FP)
  Micro Recall    = TP / (TP+FN)
  Micro F1        = 2·P·R / (P+R)

Samples F1 (per row, then averaged — paper-comparable):
  For row i: Precision_i = |pred_i∩gold_i| / |pred_i|,  Recall_i = |pred_i∩gold_i| / |gold_i|
             F1_i = 2·Precision_i·Recall_i / (Precision_i+Recall_i)
  Samples F1 = (1/N) · Σ F1_i

Macro F1 (per tool label, then averaged):
  For label j, pooled over all rows:
    Precision_j, Recall_j, F1_j as above but indexed by column j
  Macro F1 = (1/11) · Σ F1_j

Hamming Loss = (1/(N·11)) · Σ_i Σ_j [pred_ij ≠ gold_ij]

Jaccard_i    = |pred_i ∩ gold_i| / |pred_i ∪ gold_i|,  defined as 1.0 if both sets are empty
Mean Jaccard = (1/N) · Σ Jaccard_i
Jaccard pass rate = (# rows with Jaccard_i ≥ 0.5) / N
```

---

## 3. Step Explanation Metrics — LLM Judge

A **separate model** (`Qwen/Qwen2.5-7B-Instruct`, loaded independently of
whichever model is under test) scores each explanation on four dimensions,
each an integer 0–3:

| Dimension | Question | Weight (diagnostic score only) |
|---|---|---|
| Relevance | Does it justify the *predicted* step? | 1.0 |
| Technical accuracy | Are the technical claims correct? | 1.5 |
| Completeness | Does it cover the key reasoning? | 1.0 |
| Clarity | Is it well-structured and unambiguous? | 0.5 |

> **Correction to an earlier draft of this doc:** correctness is **not**
> "average the four dimensions and cut at 0.6." That was the original design
> and it was deliberately replaced — the code comment in `core/llm_judge.py`
> gives the concrete failure case: an explanation scoring relevance 3,
> technical_accuracy 0, completeness 3, clarity 3 weighted-averages to
> ≈0.69 (above 0.6) while being **technically wrong**. A blended score lets
> dimensions compensate for each other; the fix removes that compensation.

**What actually decides correctness** is a **gate on the raw integers**,
not a threshold on any composite:

```
is_correct = (relevance ≥ 2) AND (technical_accuracy ≥ 2) AND (completeness ≥ 1)
```

Clarity never blocks correctness — a correct-but-clunky explanation still
passes. The cutoffs (2 = "mostly/clearly right" on the rubric's own 0–3
scale, 1 = "at least some real reasoning") are read directly off the rubric,
introducing no new arbitrary constant.

`correctness_score` is still computed as a continuous 0–1 number — it is
kept purely as a **diagnostic trend indicator** across runs, and plays no
role in the accuracy figure:

```
correctness_score = Σ (weight_d · score_d) / (Σ weight_d · 3)     [diagnostic only]

Overall accuracy = (1/N) · Σ is_correct_i
```

**Reference-based metrics** (secondary, automatic — kept separate from the
judge because embedding similarity rewards lexical overlap, not factual
correctness):
- **BERTScore F1** — cosine similarity of contextual embeddings between prediction and gold
- **BLEURT** — a learned semantic-similarity score, if enabled

---

## 4. Stage 3 Reward Function

```
r = w_fmt · format_r + w_step · step_r + w_mcp · mcp_r + w_exp · explanation_r
```

| Weight | Value | Role |
|---|---|---|
| `w_fmt` | 0.01 | precondition, not a research objective |
| `w_step` | 0.33 | step correctness |
| `w_mcp` | 0.33 | tool correctness |
| `w_exp` | 0.33 | explanation quality |

All three research objectives (step / mcp / explanation) are weighted
**equally**. Per-objective advantages are z-scored independently before the
policy update (GDPO-style decoupled normalization), so a numerically noisy
objective cannot dominate the other two regardless of its raw scale.

**Format reward**
```
format_r = 1.0  if the completion parses as JSON with all 3 required keys
                 ("New step", "Step explanation", "MCP_tasks")
```
If parsing fails outright, every other term is forced to 0 and the reward
collapses to `0.01 × (keys present / 3)` — a malformed answer earns
essentially nothing no matter how good its content is.

**Step reward**
```
step_r = 1.0  if normalize(pred_step) == gold_step
       = 0.0  otherwise
```
Uses the same fuzzy normalizer as evaluation, so reward and eval measure
identically.

**MCP reward**

> **Correction to an earlier draft of this doc:** this is **not**
> rarity-weighted F1. `compute_reward()` in `training/stage3_grpo_rl.py`
> computes a plain, unweighted Jaccard over the predicted and gold tool
> sets — no `sqrt(1/frequency)` term or any other per-tool weighting exists
> in the reward path.

```
pred_set = {tools predicted}
gold_set = {tools in gold}
mcp_r = |pred_set ∩ gold_set| / |pred_set ∪ gold_set|      (1.0 if both sets are empty)
```

**Explanation reward** (deterministic proxy — *not* the LLM judge)
```
semantic     = cosine_similarity(BGE(pred_explanation), BGE(gold_explanation)),  rescaled to [0,1]
lexical      = SequenceMatcher ratio between pred and gold text (character-level)
step_support = 1.0 if the predicted step's own words appear in the explanation, else 0.5
tool_support = |pred_mcp ∩ gold_mcp| / |gold_mcp|   (or 1.0 if gold has no tools and none predicted, else 0.5)

explanation_r = 0.60·semantic + 0.20·lexical + 0.10·step_support + 0.10·tool_support
explanation_r = clip(explanation_r, 0.0, 1.0)
```

**Why a proxy and not the judge itself:** the judge is a separately-loaded
7B model — far too expensive to call thousands of times inside the RL
sampling loop, and using it as the optimization target would let the policy
learn to exploit a noisy evaluator, destroying the judge's value as an
independent test-time measure. The judge scores the *final* model only; it
never trains it.

---

## 5. Summary table

| Metric | Formula | Range | Reported for |
|---|---|---|---|
| Step accuracy | correct / total | [0, 1] | Stages 1–3 |
| Step macro-F1 | mean of per-class F1 | [0, 1] | Stage 1 |
| MCP subset accuracy | exact set matches / total | [0, 1] | all stages |
| MCP samples-F1 | mean of per-row F1 | [0, 1] | **paper-comparable** |
| MCP micro-F1 | pooled TP/(TP+FP), TP/(TP+FN) | [0, 1] | all stages |
| Extra-tool rate | mean(&#124;pred−gold&#124; / &#124;pred&#124;) | [0, 1] | all stages |
| Missing-tool rate | mean(&#124;gold−pred&#124; / &#124;gold&#124;) | [0, 1] | all stages |
| Hamming loss | errors / (N·11) | [0, 1] | all stages |
| MCP Jaccard | intersection / union, per row, meaned | [0, 1] | all stages |
| LLM-judge accuracy | gate: relevance≥2 ∧ tech_acc≥2 ∧ completeness≥1 | [0, 1] | Stages 2–3 |
| Stage 3 reward | 0.01·fmt + 0.33·step + 0.33·mcp + 0.33·exp | [0, 1] | Stage 3 training only |

---

*stepmodel-final-v2 · branch `restructured_code`. Baseline comparison:
Pen-Strategist, arXiv 2605.04499 — Step Model 82.87%, MCP Micro F1 0.64,
MCP subset accuracy 48.88%.*
