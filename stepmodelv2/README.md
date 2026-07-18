# Next-Step / MCP-Tool Prediction Pipeline
GNN state encoder → Qwen SFT → Qwen GRPO, evaluated on `test_data.csv`

## 0. What the raw data actually looks like (and why preprocessing matters)

Inspecting `training_data.csv` / `test_data.csv` directly:

- **`New step`** is free text with 115 unique strings for what should be your
  10 `STEP_LABELS`. Variants like missing trailing periods, `"...commands
  and..."` fragments, and typos (`"seleced"`) all refer to the same label.
  → `data_utils.StepLabelNormalizer` resolves ~93% of rows via exact-match +
  regex rules, and the remaining ambiguous/typo'd ~7% via sentence-embedding
  cosine similarity against the canonical label set.
- **`MCP_tasks`** is *not* reliably a Python dict. ~80% of cells are free
  text (`"Nmap: Perform comprehensive TCP port scan..."`), and it's
  inherently **multi-label** — many steps invoke 2+ tools
  (`{'Dirbuster': ..., 'Google search': ...}`). → `extract_mcp_labels()`
  tries `ast.literal_eval` first (highest precision when it's a real dict),
  then falls back to keyword/regex matching against `MCP_LABELS`, which
  covers 99.9% of rows in this dataset.
- **`PTT`** (Penetration Testing Tree) is the running, indentation-structured
  state of the engagement — this is the natural source for your GNN graph
  if `stepmodelv2/processed_data/{train,test}` doesn't already have an
  exported graph for a given row. `build_graph_from_ptt()` parses the
  indentation into a parent→child tree, embeds each node's text with a
  frozen sentence encoder, and adds temporal "next sibling" edges.

## 1. Architecture

```
                 ┌────────────────────────────────────────────┐
                 │              STAGE 1 (supervised)           │
                 │                                              │
   PTT text ──►  │  GraphEncoder (GATv2, 3 layers)              │
   or prebuilt    │      mean+max pool ──► graph_emb (256d)     │──┐
   .pt graph ──►  │                                              │  │
                 │  Context fields ──► frozen bge-small ──►     │  │
   (5 text        │      concat ──► MLP ──► context_emb (256d)  │  │
   fields)   ──►  │                                              │  │
                 │  fusion = MLP([graph_emb ; context_emb])     │  │
                 │      ├─ step_head  → softmax(10)  Step Type  │  │
                 │      └─ mcp_head   → sigmoid(11)  MCP Type   │  │
                 └────────────────────────────────────────────┘  │
                                                                   │ graph_emb
                                                                   ▼
                 ┌────────────────────────────────────────────┐
                 │         STAGE 2: SFT of Qwen2.5 (LoRA)       │
                 │                                              │
                 │  graph_emb ──► GraphPrefixAdapter ──►        │
                 │      8 soft-prompt token embeddings          │
                 │      prepended to Qwen's input embeddings    │
                 │                                              │
                 │  + text prompt: Previous strategy/step/      │
                 │    result, New strategy, Strategy expl.      │
                 │                                              │
                 │  target (loss-masked to assistant span only):│
                 │    {"New step": ..., "Step explanation": ...,│
                 │     "MCP_tasks": {...}}                      │
                 └────────────────────────────────────────────┘
                                     │ policy init
                                     ▼
                 ┌────────────────────────────────────────────┐
                 │         STAGE 3: GRPO RL on Stage-2 policy   │
                 │                                              │
                 │  sample G completions/prompt (group)         │
                 │  reward = w1*format_ok + w2*step_exact_match │
                 │         + w3*mcp_set_F1 + w4*explanation_    │
                 │           quality(G-Eval-style judge)        │
                 │  group-relative advantage, no critic network │
                 │  KL-regularized toward the Stage-2 policy    │
                 └────────────────────────────────────────────┘
```

**Why this split.** Stage 1 gives you a cheap, deterministic, fully
supervised classifier you can evaluate directly with sklearn metrics — no
generation/parsing noise. Its graph embedding is *also* reused as the
LLM's grounding signal in Stage 2, so the GNN's structural understanding of
"where are we in the engagement" transfers into the LLM rather than being
re-learned from scratch. Stage 2 (SFT) teaches the LLM the label taxonomy
and the JSON output contract via teacher forcing. Stage 3 (GRPO) is where
`Step explanation` quality is actually optimized — cross-entropy against
one gold explanation string over-penalizes equally-valid phrasings, so
free-form quality is better handled by a reward-based method once the
label taxonomy is already locked in from Stage 2. This mirrors the paper's
framing: CoT-style self-explanation improves the model's grounded reasoning,
and explanation quality is evaluated the way the paper's G-Eval pass does
(readability / coherence / informativeness), just moved from a *post-hoc
eval* into an *RL reward*.

## 2. File map

| File | Purpose |
|---|---|
| `config.py` | `STEP_LABELS`, `MCP_LABELS`, paths, hyperparameters |
| `data_utils.py` | CSV parsing, step-label normalizer, MCP multi-label extractor, PTT→graph fallback builder |
| `graph_encoder.py` | `GraphEncoder` (GATv2), `Stage1Classifier` (two-head model) |
| `stage1_gnn_train.py` | Trains the Stage-1 classifier; prints val Accuracy/F1 each epoch |
| `stage2_sft_qwen.py` | LoRA SFT of Qwen with graph soft-prompt adapter |
| `stage3_grpo_rl.py` | GRPO RL refinement of `Step explanation` (+ correctness rewards) |
| `evaluate.py` | **Runs on `test_data.csv`**, reports Accuracy/F1 for both Step Type and MCP Type, in `--model gnn` (fast, deterministic) or `--model llm` (end-to-end generative) mode |

## 3. Running it

```bash
pip install torch torch_geometric sentence-transformers transformers peft trl scikit-learn pandas

# Stage 1 — GNN classifier (do this first; Stage 2/3 reuse its graph embeddings)
python stage1_gnn_train.py

# Stage 2 — SFT Qwen with graph-conditioned soft prompt
python stage2_sft_qwen.py

# Stage 3 — GRPO RL for explanation quality
python stage3_grpo_rl.py

# Evaluate on test_data.csv
python evaluate.py --model gnn                                   # Stage-1 only, fast
python evaluate.py --model llm --adapter-dir checkpoints/stage3_qwen_grpo   # full pipeline
```

`evaluate.py` reports, separately for Step Type and MCP Type:
- **Step Type** (single-label, 10-way): Accuracy, Macro-F1, Weighted-F1,
  per-class precision/recall/F1, confusion matrix.
- **MCP Type** (multi-label, 11-way): Subset/exact-match Accuracy, Micro-F1,
  Macro-F1, Samples-F1, per-tool precision/recall/F1/support.

Both metrics matter for MCP because it's multi-label: subset accuracy is
strict (the *entire* predicted tool set must match), while micro/macro-F1
give credit for partially-correct tool sets (e.g. predicting `{Nmap}` when
gold is `{Nmap, Dirbuster}`).

## 4. Things you'll need to fill in for your environment

1. **`GRAPH_DIR_TRAIN` / `GRAPH_DIR_TEST`** in `config.py` — point these at
   your actual `stepmodelv2/processed_data/{train,test}` export and confirm
   the file naming convention matches `_find_prebuilt_graph()` in
   `data_utils.py` (adjust the glob pattern if your exporter names files
   differently, e.g. by a UUID instead of `machine__row_id`).
2. **Qwen model size** — `QWEN_MODEL_NAME` defaults to
   `Qwen/Qwen2.5-7B-Instruct`; swap for whatever Qwen checkpoint you have
   local/licensed access to. LoRA target modules assume the standard
   Qwen2 attention/MLP module names.
3. **GRPO + graph conditioning** — `trl`'s `GRPOTrainer` generates from
   plain text prompts, so Stage 3 distills the graph embedding into a short
   text "Graph summary" line rather than injecting soft-prompt tokens
   directly into the RL rollout (documented in `stage3_grpo_rl.py`). If you
   need the raw graph embedding inside the RL generation path itself,
   you'll need a custom generation loop (reuse `GraphPrefixAdapter` from
   Stage 2 and write your own group-sampling/advantage/update loop instead
   of `GRPOTrainer`) — flag this if you want that version too.
4. **Judge model for explanation reward** — `stage3_grpo_rl.py` reuses the
   base Qwen weights as a stand-in G-Eval-style judge. Swap in a stronger
   frozen judge (e.g. a larger model, or the real G-Eval log-prob-weighted
   scoring from the paper) once you're ready to score explanations
   properly — you mentioned that's a later step.

## 5. Explanation evaluation (deferred, per your note)

When you're ready, the natural next step (matching the paper's Section
4.3.2) is a G-Eval pass: prompt a judge LLM to produce its own
chain-of-thought evaluation steps for *readability*, *coherence*, and
*informativeness*, each scored 1–5 as a probability-weighted sum over
token logits (not just the argmax digit, which `stage3_grpo_rl.py` uses
as a cheap proxy). `evaluate.py` doesn't touch `Step explanation` at all
right now, by design — add a `--score-explanations` mode there once you
want it.
