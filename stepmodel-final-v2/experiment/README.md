# Text-Only Experiment (No Graph)

This experiment is a simplified version of the main StepModel pipeline that **does not use graph information**. It only uses text inputs (`new_strategy` and `strategy_explanation`) to predict:
- `gold_new_step` (step classification)
- `gold_step_explanation` (step explanation generation)
- `gold_mcp_tasks` (MCP tool prediction)

This serves as a baseline to measure the contribution of graph conditioning in the main pipeline.

**Note**: Stage 1 (GNN) is not needed for this text-only baseline since there's no graph to encode. The experiment directly uses Stage 2 (LLM SFT) and Stage 3 (GRPO RL) on text-only inputs.

## Directory Structure

```
experiment/
├── config.py                    # Configuration file
├── data_prep/
│   └── build_input_json.py     # Data preparation (text-only)
├── training/
│   ├── stage2_sft_qwen.py      # Stage 2: LLM SFT (no graph)
│   └── stage3_grpo_rl.py       # Stage 3: GRPO RL (no graph)
├── eval/
│   └── evaluate.py             # Evaluation script
├── input/                       # Generated input JSON files
│   ├── train.json
│   └── test.json
├── output/                      # Evaluation results
├── checkpoints/                 # Model checkpoints
└── README.md                    # This file
```

## Key Differences from Main Pipeline

| Aspect | Main Pipeline | Text-Only Experiment |
|--------|--------------|----------------------|
| **Input** | Graph + context text | Strategy text only |
| **Stage 1** | GNN (GraphEncoder) | N/A (not needed) |
| **Stage 2** | LLM with graph conditioning | LLM without graph conditioning |
| **Stage 3** | GRPO with graph conditioning | GRPO without graph conditioning |
| **Hint masking** | 50% (to force graph learning) | N/A (no hints) |

## Installation

Same dependencies as the main pipeline:
```bash
pip install torch transformers peft scikit-learn tqdm pandas
```

## Usage

### 1. Data Preparation

Build input JSON files from CSV data (text-only, no graph):

```bash
cd experiment/data_prep
python build_input_json.py
```

This creates:
- `experiment/input/train.json`
- `experiment/input/test.json`

Each record contains:
```json
{
  "machine": "...",
  "new_strategy": "...",
  "strategy_explanation": "...",
  "gold_new_step": "...",
  "gold_step_explanation": "...",
  "gold_mcp_tasks": "..."
}
```

### 2. Stage 2: LLM SFT (No Graph Conditioning)

Fine-tune Qwen LLM without graph conditioning:

```bash
cd experiment/training
python stage2_sft_qwen.py
```

**Key Features:**
- No graph prefix tokens
- No Stage 1 hints
- Direct text-to-text generation
- LoRA fine-tuning (r=32, alpha=64)

**Output:** `experiment/checkpoints/stage2_qwen_lora/`

### 3. Stage 3: GRPO RL (No Graph Conditioning)

Apply GRPO RL to improve explanation quality:

```bash
cd experiment/training
python stage3_grpo_rl.py
```

**Key Features:**
- No graph conditioning
- Group-relative policy optimization
- Reward function: step match + MCP F1 + explanation length
- G=20 completions per example

**Output:** `experiment/checkpoints/stage3_qwen_grpo/`

### 4. Evaluation

Evaluate Stage 2 and Stage 3 on test set:

```bash
cd experiment/eval
python evaluate.py --stage all  # Evaluate both stages
python evaluate.py --stage 2    # Evaluate Stage 2 only
python evaluate.py --stage 3    # Evaluate Stage 3 only
```

**Output:** `experiment/output/stage{N}_results.json`

**Metrics:**
- Stage 2/3: step accuracy, step macro F1

## Configuration

Edit `experiment/config.py` to modify:
- Model hyperparameters
- Training hyperparameters
- Paths
- Label spaces

## Comparison with Main Pipeline

After running both pipelines, compare results:

```bash
# Main pipeline results
cat output/stage2.csv
cat output/stage3.csv

# Experiment results
cat experiment/output/stage2_results.json
cat experiment/output/stage3_results.json
```

This comparison will show the contribution of graph conditioning to:
- Step classification accuracy
- MCP tool prediction
- Explanation quality

## Notes

- **Machine-level split**: Same 15% validation split as main pipeline for fair comparison
- **Random seed**: Same seed (42) for reproducibility
- **Model versions**: Same Qwen/Qwen3-14B as main pipeline
- **No graph data**: This experiment completely bypasses PTT parsing and graph construction
- **Stage 1 removed**: Not needed since there's no graph to encode

## Expected Results

The text-only experiment is expected to perform worse than the main pipeline because:
- No structural information from attack graphs
- No context from pentest state/action/finding nodes
- No graph-conditioning in LLM stages

The performance gap quantifies the value added by graph conditioning.
