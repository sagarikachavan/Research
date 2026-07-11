# StepModel v2 Implementation Runbook

This document explains how the StepModel v2 implementation is organized under
`Research/stepmodel-new`, what each file does, and the order in which to run the
pipeline.

## What Changed

StepModel v2 keeps the original supervised warmup followed by GRPO fine-tuning,
but adds a required Phase 0 before label training:

1. Dynamic per-step graphs are generated for each step pair.
2. The GNN and graph-token projector are pretrained without Step/MCP labels.
3. Qwen3-14B is frozen for supervised training and GRPO.
4. Checkpoints store only the trainable policy modules, not the full LLM.
5. GRPO skips zero-variance rollout groups to avoid noisy updates.
6. Evaluation continues to report the same Step/MCP metrics for comparison.

## File Map

### `config.json`

Central configuration for the v2 pipeline.

Important fields:

- `model.llm_name`: frozen LLM backbone, currently `Qwen/Qwen3-14B`.
- `model.text_embedding_model`: Sentence-BERT model for node/text embeddings.
- `model.gnn_out_dim`: GNN graph embedding size.
- `model.graph_token_count`: number of injected graph tokens.
- `training.phase0_checkpoint`: checkpoint produced by `pretrain_gnn.py`.
- `training.gnn_learning_rate`: reduced LR for the pretrained GNN.
- `training.projector_learning_rate`: reduced LR for the graph-token projector.
- `training.drift_guard_weight`: optional penalty against drifting far from Phase 0 GNN weights.
- `training.grpo_generate_temperature`: rollout-only temperature for GRPO action diversity.
- `training.grpo_reward_std_epsilon`: minimum reward std required for a GRPO update.

### `generate_graphs.py`

Builds machine-level PTT evolution graphs from:

- `data/training_data.csv`
- `data/test_data.csv`

Outputs:

- `processed_data/train/<machine>/<machine>_graph.json`
- `processed_data/test/<machine>/<machine>_graph.json`
- matching `.html` graph visualizations

This file creates the full graph per machine. The per-step dynamic graph slicing
happens later in `graph_to_embeddings.py`.

### `graph_to_embeddings.py`

Converts processed graph JSON files into model-ready embedding JSON.

Implemented v2 behavior:

- Drops rows whose `Machine` field is corrupted with PTT tree text.
- Creates one graph per `step_pair`, using only the graph state visible at that point.
- Stores `nodes` and `edges` directly inside each `step_pair`.
- Keeps machine-level `nodes` and `edges` for backward compatibility.
- Asserts train/test machines do not overlap after regeneration.

Outputs:

- `embeddings_data/train/all_processed.json`
- `embeddings_data/test/all_processed.json`
- per-machine processed JSON files under each split

### `pretrain_gnn.py`

Runs Phase 0: self-supervised GNN pretraining plus frozen-LLM token alignment.

Implemented losses:

- Graph contrastive InfoNCE over two augmented graph views.
- Token-embedding alignment against frozen Qwen embedding anchors.
- Optional link prediction loss.

Gate metrics printed before exit:

- Held-out link prediction accuracy.
- Random-edge baseline accuracy.
- Linear probe accuracy from pretrained GNN embeddings.
- Linear probe accuracy from random-init GNN embeddings.

Output:

- `checkpoints/phase0_gnn_projector.pt`

The script exits with failure if the Phase 0 gate does not pass.

### `train_gnn_rl.py`

Runs Phase 1 supervised warmup and Phase 2 GRPO.

Implemented v2 behavior:

- Loads `checkpoints/phase0_gnn_projector.pt` when present.
- Freezes all Qwen3-14B parameters.
- Removes LoRA and 4-bit training branches from the training path.
- Uses differential learning rates:
  - lower LR for GNN
  - lower LR for graph-token projector
  - full LR for fuse/text projection/readout/heads
- Adds optional GNN drift guard loss.
- Tunes MCP threshold on validation with `find_best_mcp_threshold`.
- Skips GRPO updates when rollout reward variance is below epsilon.
- Saves lightweight checkpoints containing only policy state.

Important checkpoints:

- `checkpoints/best_supervised_checkpoint.pt`
- `checkpoints/best_checkpoint.pt`
- `checkpoints/final_checkpoint.pt`

### `evaluate.py`

Evaluates a trained checkpoint on the test split.

Implemented v2 behavior:

- Loads frozen Qwen3-14B from `model.llm_name`.
- Loads only the trainable policy state from checkpoint.
- Does not require saved LoRA or full LLM weights.
- Reports:
  - `step_acc`
  - `step_micro_f1`
  - `mcp_acc`
  - `mcp_micro_f1`

### `run_ablation_ladder.py`

Orchestrates the ablation ladder across seeds.

Implemented configurations:

- Dynamic graph only.
- Frozen LLM without Phase 0.
- Phase 0 pretraining plus supervised warmup.
- Phase 0 pretraining plus fixed GRPO.

The two reference rows are required as external commands/metrics:

- previous static-graph LoRA baseline
- paper CNN/GPT-2 reference baseline

This script writes:

- `ablation_runs/ablation_results_by_seed.csv`
- `ablation_runs/ablation_summary.csv`

## Run Order

Run all commands from:

```bash
cd /Users/sagarikachavan/Documents/Research/stepmodel-new
```

### 1. Build PTT Graphs

```bash
python3 generate_graphs.py
```

This reads raw CSV files and writes machine-level graph JSON/HTML into
`processed_data`.

### 2. Generate Dynamic Per-Step Embeddings

```bash
python3 graph_to_embeddings.py
```

This reads `processed_data`, embeds graph node/edge text with Sentence-BERT, fixes
corrupted `Machine` rows by dropping them explicitly, creates dynamic per-step
graphs, and writes `embeddings_data`.

Run this step any time the CSVs or processed graphs change.

### 3. Run Phase 0 Pretraining

```bash
python3 pretrain_gnn.py --config config.json
```

Optional shorter smoke-test form:

```bash
python3 pretrain_gnn.py --config config.json --epochs 1 --batch-size 4
```

Expected output:

```text
checkpoints/phase0_gnn_projector.pt
```

Do not continue to Phase 1 if the Phase 0 gate fails.

### 4. Run Phase 1 and Phase 2 Training

```bash
python3 train_gnn_rl.py --config config.json
```

This runs:

1. supervised warmup
2. GRPO fine-tuning, if `training.num_grpo_epochs > 0`
3. final test evaluation
4. final checkpoint save

To run supervised-only, set this in `config.json`:

```json
"num_grpo_epochs": 0
```

### 5. Evaluate a Saved Checkpoint

```bash
python3 evaluate.py
```

This looks in `checkpoints` for a compatible checkpoint and evaluates it on
`embeddings_data/test/all_processed.json`.

## Ablation Ladder

The full ladder needs external reference metrics for:

- old static-graph LoRA baseline
- paper CNN/GPT-2 reference baseline

Example:

```bash
python3 run_ablation_ladder.py \
  --config config.json \
  --seeds 42,43,44 \
  --baseline-command "python3 /path/to/old/baseline_runner.py" \
  --baseline-metrics /path/to/baseline_metrics.json \
  --paper-command "python3 /path/to/paper_cnn_runner.py" \
  --paper-metrics /path/to/paper_metrics.json
```

Each metrics JSON should contain:

```json
{
  "step_acc": 0.0,
  "step_micro_f1": 0.0,
  "mcp_acc": 0.0,
  "mcp_micro_f1": 0.0,
  "mcp_threshold": 0.5
}
```

For a dry run:

```bash
python3 run_ablation_ladder.py --config config.json --dry-run
```

## Recommended Development Checks

Run syntax checks after code edits:

```bash
python3 -m py_compile \
  graph_to_embeddings.py \
  pretrain_gnn.py \
  train_gnn_rl.py \
  evaluate.py \
  run_ablation_ladder.py
```

Check generated data exists before training:

```bash
ls embeddings_data/train/all_processed.json
ls embeddings_data/test/all_processed.json
```

Check the Phase 0 checkpoint exists before full training:

```bash
ls checkpoints/phase0_gnn_projector.pt
```

## Notes and Caveats

- Qwen3-14B is loaded from Hugging Face by `pretrain_gnn.py`, `train_gnn_rl.py`,
  and `evaluate.py`. The model must be available locally or downloadable in the
  runtime environment.
- Phase 0 can be expensive because it loads Qwen embeddings to compute PCA
  directions for token alignment.
- The current dynamic graph cutoff is conservative: each step pair receives only
  graph state available before the label step it is predicting.
- `graph_to_embeddings.py` drops corrupted `Machine` rows instead of trying to
  silently assign them to a machine. The cleanup report is stored in processed
  output JSON under `csv_cleanup`.
- Checkpoints are intentionally smaller than the previous implementation because
  they do not save the full frozen LLM.
