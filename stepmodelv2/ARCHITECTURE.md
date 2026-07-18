# StepModel v2 — Architecture & Design Document

## Overview

StepModel v2 is a three-stage pipeline for autonomous penetration-testing step planning. Given the current state of a pentest (what has been done, what was found, what strategy is being followed), the system predicts:

1. **The next step type** — one of 10 canonical actions (e.g. "Exploit the selected exploitations", "Enumerate further on the X service…")
2. **The MCP tools to use** — a multi-label subset of 11 tools (Nmap, Metasploit, Dirbuster, etc.)
3. **A natural language explanation** of why this step was chosen

The pipeline moves from structured classification (Stage 1) to language model fine-tuning (Stage 2) to reinforcement learning for explanation quality (Stage 3).

---

## Data Flow: CSV → Graph → Input JSON → Training

### 1. Raw CSV Data

The raw data lives in `data/training_data.csv` and `data/test_data.csv`. Each row represents one decision point in a pentest session. The columns are:

| Column | Description |
|--------|-------------|
| `Machine` | Name of the target machine (e.g. `bashed`, `Kioptrix_Level_1`) |
| `PTT` | Penetration Testing Tree — a structured text log of all steps attempted so far, with status markers `(completed)`, `(to-do)`, etc. This is the cumulative state of the pentest at this decision point |
| `Previous strategy` | The high-level strategy phase active before this step |
| `Previous step` | The last concrete action taken |
| `Previous step result` | What that action found or produced |
| `New strategy` | The strategy phase for the upcoming step |
| `Strategy explanation` | Why the strategy is shifting |
| `New step` | **Gold label** — the correct next action (maps to `STEP_LABELS`) |
| `Step explanation` | **Gold label** — free-text justification for the chosen step |
| `MCP_tasks` | **Gold label** — dict of tools to invoke (maps to `MCP_LABELS`) |

Multiple rows can share the same `Machine` — each row is a different decision point during the pentest of that machine.

---

### 2. Graph Generation (`generate_graphs.py`)

For each machine, all its rows are read together to reconstruct how the Penetration Testing Tree evolved over time. Each step that completed adds a new node to the tree.

**What gets built:**

- **Agent nodes** (blue) — represent the cumulative PTT state after a step completes. These are the "what we know now" snapshots.
- **Search nodes** (orange) — represent the specific PTT item being worked at each decision point, along with the MCP tool calls made.
- **Track nodes** (green) — represent the findings payload of each completed item (what was actually discovered).

**How edges connect them:**

```
Agent(state N) ──StateTransition──► Agent(state N+1)
Agent(state N) ──SearchUpdate──────► Search(item being worked)
Search(item)   ──TrackUpdate───────► Track(findings)
Track(findings)──Prediction────────► Agent(state N+1)
```

This forms a directed graph that encodes not just what happened, but the causal chain: state → search → discovery → new state.

The graph is saved as both a `.json` file (nodes/edges with metadata) and an interactive `.html` visualisation under `processed_data/train/<machine>/` and `processed_data/test/<machine>/`.

**Why a graph and not just text?**  
The PTT text is a growing tree. A graph encoder can propagate information across the tree structure — e.g. a completed recon step downstream informs what exploitation steps make sense upstream. A flat text encoder treats the PTT as a sequence and loses the tree structure.

---

### 3. Input JSON Generation (`build_input_json.py`)

`input/train.json` and `input/test.json` are produced by joining each CSV row with the graph JSON for its machine. Each record contains:

```json
{
  "Machine": "bashed",
  "Graph": { ...full graph JSON from processed_data/train/bashed/... },
  "Previous strategy": "...",
  "Previous step": "...",
  "Previous step result": "...",
  "New strategy": "...",
  "Strategy explanation": "...",
  "Gold New step": "Exploit the selected exploitations",
  "Gold Step explanation": "...",
  "Gold MCP_tasks": "..."
}
```

These files are a human-readable record of the full dataset and are used for inspection and downstream tooling. The GNN and LLM training stages read directly from the CSV + graph files rather than these JSONs (the JSONs serve as a joined reference view).

---

## Stage 1 — GNN Classifier (`stage1_gnn_train.py`, `graph_encoder.py`)

### Goal
Train a graph neural network that takes the current pentest graph state + context text and predicts the next step type and required tools. This produces a **learned graph embedding** that is later used to condition the LLM in Stages 2 and 3.

### Data Preparation (`data_utils.py`)

**Step label normalisation:**  
The `New step` column in the CSV is noisy free text. `StepLabelNormalizer` maps it onto the fixed 10-label taxonomy using a 3-tier strategy:
1. Exact match after whitespace normalisation
2. Regex/keyword rules (fast, covers most cases)
3. Sentence embedding cosine similarity fallback (for anything the regexes miss, threshold 0.55)

Rows that cannot be mapped are dropped (~0.7% of data).

**MCP extraction:**  
The `MCP_tasks` column is sometimes a Python dict string, sometimes free text. `extract_mcp_labels` parses it and matches against 11 regex patterns to produce a multi-hot vector.

**Graph construction:**  
For each row, `load_graph` looks for a pre-built `.pt` file. If not found, it falls back to `build_graph_from_ptt`, which:
1. Parses the PTT text into `(depth, text, status)` tuples
2. Embeds each node's text with `BAAI/bge-small-en-v1.5` (frozen, 384-dim)
3. Appends a 5-dim one-hot status vector (completed/in-progress/pending/failed/unknown)
4. Constructs parent→child edges (tree structure) + next-sibling temporal edges, both directions

**Context embeddings:**  
The 5 context columns (`Previous strategy`, `Previous step`, `Previous step result`, `New strategy`, `Strategy explanation`) are each embedded with the same frozen sentence encoder and cached at dataset init time.

### Model Architecture (`graph_encoder.py`)

```
PTT Graph
  └─ Node features: [sentence_emb(384) | status_onehot(5)] = 389-dim
       │
  GraphEncoder (GATv2, 3 layers, 4 heads, hidden=256)
       │  residual connections + LayerNorm + Dropout
       │
  global_mean_pool + global_max_pool → concat → Linear(512→256)
       │
       │   GNN embedding (256-dim)  ←── reused in Stage 2
       │
Context Fields (5 × 384-dim frozen embeddings)
  └─ ContextTextProjector: Linear(1920→512→256)
       │
       │   Context embedding (256-dim)
       │
  Fusion: concat [GNN(256) | Context(256)] → Linear(512→512→512)
       │
  ┌────┴────┐
step_head  mcp_head
(512→10)  (512→11)
softmax   sigmoid (multi-label)
```

**Loss:**
```
total_loss = 1.0 × CrossEntropy(step_logits, step_label)
           + 1.0 × BinaryCrossEntropy(mcp_logits, mcp_multihot)
```

**Training:**
- AdamW, lr=2e-4, weight_decay=1e-4
- CosineAnnealingLR over 30 epochs
- Gradient clipping at 1.0
- 90/10 train/val split (random permutation, seeded)
- Best checkpoint saved on `step_macro_f1 + mcp_micro_f1`

**What gets saved:** `checkpoints/stage1_gnn_classifier.pt` — the full `Stage1Classifier` state dict. The `graph_encoder` sub-module is extracted and frozen for use in Stages 2 and 3.

---

## Stage 2 — Supervised Fine-Tuning of Qwen (`stage2_sft_qwen.py`)

### Goal
Fine-tune `Qwen/Qwen2.5-7B-Instruct` to generate structured JSON outputs (step type + explanation + tool dict) conditioned on both the text context and the graph state from Stage 1.

### Graph Conditioning via Soft-Prompt Prefix

The frozen Stage 1 graph encoder produces a 256-dim embedding for each training example. A trainable `GraphPrefixAdapter` projects this into 8 soft-prompt token embeddings in the LLM's hidden space:

```
graph_emb (256-dim)
    └─ Linear(256→hidden*2) → GELU → Linear(hidden*2→hidden*8)
         └─ reshape → (8, hidden)  ← 8 soft-prompt tokens
```

These 8 tokens are prepended to the token embeddings of the text prompt before the forward pass. The LLM sees them as part of its input context but they carry structured graph information, not text.

### Prompt and Target Format

**Input prompt (masked from loss):**
```
<|system|>
You are an autonomous penetration-testing planning assistant...
<|user|>
Machine: bashed
Previous strategy: ...
Previous step: Do an nmap scan...
Previous step result: Port 80 open, Apache 2.4.18...
New strategy: Enumerate HTTP service further...
Strategy explanation: ...
<|assistant|>
```

**Target (loss is computed only here):**
```json
{
  "New step": "Enumerate further on the X service to find software versions, hidden directories and file.",
  "Step explanation": "Apache 2.4.18 is running on port 80. We should enumerate hidden directories...",
  "MCP_tasks": {
    "Dirbuster": "Use Dirbuster as part of: Enumerate further...",
    "Interactive CLI": "Use Interactive CLI as part of: Enumerate further..."
  }
}
```

### Training Setup
- LoRA on all projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), r=16, alpha=32
- Only LoRA weights + GraphPrefixAdapter are trainable; base Qwen weights are frozen
- AdamW lr=1e-4, cosine schedule with warmup
- Batch size 2, gradient accumulation 8 (effective batch 16)
- 3 epochs
- Max sequence length 2048 tokens

**What gets saved:** `checkpoints/stage2_qwen_lora/` — LoRA adapter weights, `graph_adapter.pt`, tokenizer.

---

## Stage 3 — GRPO Reinforcement Learning (`stage3_grpo_rl.py`)

### Motivation

SFT (Stage 2) trains the model to copy gold outputs token-by-token. This works well for structured fields (`New step`, `MCP_tasks`) where there is one correct answer. It works poorly for `Step explanation` because:
- There are many equally valid ways to explain a reasoning step
- Cross-entropy punishes valid paraphrases as harshly as wrong answers
- The model learns to copy gold explanations rather than reason

GRPO (Group Relative Policy Optimization) fixes this by sampling multiple completions per prompt, scoring each with a reward function, and reinforcing completions that score above the group average — without needing a separate critic/value network.

### How GRPO Works Here

For each training prompt:
1. The model generates `num_generations=4` independent completions
2. Each completion is scored by the reward function
3. The gradient update reinforces completions with above-average reward and penalises below-average ones
4. A KL penalty (`beta=0.02`) keeps the policy close to the Stage 2 SFT checkpoint, preventing reward hacking

### Graph Conditioning in Stage 3

TRL's `GRPOTrainer` operates on plain text prompts and does not support soft-prompt prefixes during rollout. The graph signal is therefore linearised into a text line appended to the prompt:

```
Graph summary (from GNN state encoder): predicted step-type leaning='Exploit the selected exploitations', detected services/tools context='Nmap, Interactive CLI'
```

This is a distillation of Stage 1's predictions into text — the GNN embedding informs the summary, but the RL loop itself operates purely on language.

### Reward Function

The reward function scores each generated completion against the gold labels on four components:

```
r = 0.2 × format_reward
  + 0.3 × step_reward
  + 0.3 × mcp_reward
  + 0.2 × explanation_reward
```

**format_reward (0 or 1):**  
1.0 if the completion is valid JSON containing all three required keys (`New step`, `Step explanation`, `MCP_tasks`). 0.0 otherwise. This prevents the model from drifting into free-text responses that ignore the output schema.

**step_reward (0 or 1):**  
1.0 if `completion["New step"].strip() == gold["step_label"]` exactly. 0.0 otherwise. This is a hard exact-match signal that keeps the model on the fixed label taxonomy. Weight 0.3 makes it the dominant correctness signal.

**mcp_reward (0.0–1.0):**  
F1 score between the predicted MCP tool set and the gold tool set:
```
pred_mcp = set(completion["MCP_tasks"].keys()) ∩ MCP_LABELS
gold_mcp = set(gold["mcp_labels"])

precision = |pred ∩ gold| / |pred|
recall    = |pred ∩ gold| / |gold|
mcp_r     = 2 × precision × recall / (precision + recall)
```
Special case: if both pred and gold are empty, `mcp_r = 1.0` (correct prediction of "no tools").

**explanation_reward (0.0–1.0):**  
A G-Eval-style judge scores the generated explanation on readability, coherence, and informativeness. The judge model (a frozen copy of the same base Qwen weights) is prompted:

```
Rate the following pentest step explanation from 1 (poor) to 5 (excellent)
on readability, coherence, and informativeness combined.
Respond with only the integer.

Reference explanation: <gold_step_explanation>
Candidate explanation: <generated_explanation>

Score:
```

The integer response is normalised to [0, 1]: `(score - 1) / 4.0`.

**Why these weights?**  
- `step_reward` and `mcp_reward` together carry 0.6 of the total signal — correctness is primary
- `explanation_reward` at 0.2 provides a soft gradient for text quality without overriding the classification signal
- `format_reward` at 0.2 acts as a gate — a malformed output scoring 1.0 on format still can't exceed 0.2 total if everything else is wrong, but format failure zeroes out the whole reward by returning 0.0 early

---

## Evaluation (`evaluate.py`)

### GNN Mode (`--model gnn`)

Runs the Stage 1 classifier directly on the test set. Fast and deterministic.

For each test example:
1. Build/load the PTT graph
2. Embed the 5 context fields
3. Forward pass through `Stage1Classifier`
4. `argmax` on step logits → predicted step class
5. `sigmoid ≥ 0.5` on MCP logits → predicted tool set

**Step Type metrics:**
- Accuracy, Macro-F1, Weighted-F1
- Per-class precision/recall/F1
- Confusion matrix (10×10)

**MCP Type metrics (multi-label):**
- Subset accuracy (exact set match)
- Micro-F1, Macro-F1, Samples-F1
- Per-label precision/recall/F1/support

### LLM Mode (`--model llm`)

Runs the Stage 2 or Stage 3 adapter on the test set. Slower (full generation per example).

For each test example:
1. Build the text prompt (same format as Stage 2 training)
2. Generate with `do_sample=False` (greedy)
3. Parse the JSON block from the generated text
4. Map `"New step"` back onto `STEP_LABELS` using `StepLabelNormalizer` (same normaliser as training)
5. Extract MCP tool names from `"MCP_tasks"` keys
6. Score with the same metrics as GNN mode

Unparseable or unmappable outputs are scored as incorrect (predicted class = -1, which never matches any gold label).

---

## Results (Stage 1 GNN, Test Set)

| Metric | Value |
|--------|-------|
| Step Accuracy | 76.1% |
| Step Weighted F1 | 0.759 |
| Step Macro F1 | 0.576 |
| MCP Micro F1 | 0.672 |
| MCP Subset Accuracy | 0.493 |
| MCP Samples F1 | 0.660 |

**Strong classes:** "Exploit" (F1=0.85), "Enumerate further" (F1=0.83), "End task" (F1=0.91)  
**Weak classes:** "Enumerate the domain" (F1=0.00, 7 samples), "Analyze outcomes" (F1=0.00, 3 samples) — insufficient training data, not model failure

---

## File Reference

| File | Role |
|------|------|
| `config.py` | All hyperparameters, paths, label taxonomies |
| `data_utils.py` | Label normalisation, MCP extraction, graph loading, dataset |
| `generate_graphs.py` | CSV → per-machine graph JSON + HTML |
| `build_input_json.py` | CSV + graphs → `input/train.json`, `input/test.json` |
| `graph_encoder.py` | GATv2 GraphEncoder, ContextTextProjector, Stage1Classifier |
| `stage1_gnn_train.py` | Stage 1 training loop |
| `stage2_sft_qwen.py` | Stage 2 SFT with LoRA + graph prefix adapter |
| `stage3_grpo_rl.py` | Stage 3 GRPO RL training with composite reward |
| `evaluate.py` | GNN-mode and LLM-mode evaluation with full metrics |
| `run.py` | Pipeline runner (all stages in sequence) |
