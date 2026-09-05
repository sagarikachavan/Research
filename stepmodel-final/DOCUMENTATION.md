# StepModel: Multi-Stage Penetration Testing Step Prediction Pipeline

## Overview

StepModel is a three-stage machine learning pipeline that predicts the next step in penetration testing scenarios using graph neural networks and large language models. The system analyzes penetration testing trees (PTT) and context information to predict both the next step type and the required MCP (Model Context Protocol) tools.

### Key Features

- **Graph-based representation**: Converts PTT text into structured attack graphs using GNNs
- **Multi-task learning**: Simultaneously predicts step types and MCP tool requirements
- **Three-stage training pipeline**:
  - Stage 1: GNN-based classifier for step and MCP prediction
  - Stage 2: Supervised fine-tuning of LLM for step explanation generation
  - Stage 3: Reinforcement learning optimization using GRPO
- **Deterministic graph construction**: Rule-based PTT parsing (no API key required)
- **Data leakage prevention**: Machine-level train/validation/test splitting

## Project Structure

```
stepmodel-final/
├── config.py                 # Central configuration and hyperparameters
├── data_utils.py             # Data loading, preprocessing, and graph construction
├── graph_encoder.py          # GNN architecture and Stage 1 classifier
├── stage1_gnn_train.py       # Stage 1: GNN training for step/MCP classification
├── stage2_sft_qwen.py        # Stage 2: LLM fine-tuning for step explanations
├── stage3_grpo_rl.py         # Stage 3: RL optimization with GRPO
├── build_input_json.py       # Build training/test JSON inputs from CSV
├── generate_graphs.py        # Generate attack graphs from PTT text
├── ptt_parser.py             # Deterministic PTT parsing engine
├── llm_ptt_parser.py         # LLM-based PTT parsing (optional)
├── graph_builder.py          # Graph construction from parsed items
├── evaluate.py              # Comprehensive evaluation pipeline
├── baseline_llm_eval.py      # Baseline zero-shot/few-shot LLM evaluation
├── comparison_report.py      # Generate comparison reports
├── mcp_threshold_search.py   # Optimize MCP classification thresholds
├── llm_judge.py              # LLM-based judge for evaluation
├── run.py                    # Unified pipeline execution script
├── requirements.txt          # Python dependencies
├── data/                     # Input CSV files
│   ├── training_data.csv
│   └── test_data.csv
├── input/                    # Preprocessed JSON inputs
│   ├── train.json
│   └── test.json
├── checkpoints/              # Model checkpoints
├── output/                   # Evaluation outputs
└── processed_graph/          # Generated attack graphs
```

## Model Architecture

### Stage 1: Graph Neural Network Classifier

The Stage 1 model consists of three main components:

#### 1. Graph Encoder (GNN)
- **Architecture**: GATv2 (Graph Attention Network v2) with 4 layers
- **Hidden dimension**: 384
- **Attention heads**: 6
- **Node features**: 388-dimensional
  - 384-dim: BAAI/bge-small-en-v1.5 sentence embedding of node title
  - 3-dim: One-hot node type encoding (State/Action/Finding)
  - 1-dim: Normalized node degree
- **Pooling**: Multi-scale aggregation (mean + max + layer-wise mean)
- **Output dimension**: 384

#### 2. Context Text Projector
- **Input**: Concatenated frozen sentence embeddings of context fields
- **Architecture**: 5-layer MLP with LayerNorm and GELU activations
- **Output dimension**: 384 (matches graph encoder output)

#### 3. Gated Fusion Network
- **Mechanism**: Learnable gating for graph and context representations
- **Architecture**: 6-layer MLP with dropout
- **Output dimension**: 384

#### 4. Classification Heads
- **Step head**: Single-label classification over 10 step types (softmax)
- **MCP head**: Multi-label classification over 11 MCP tools (sigmoid)

### Stage 2: LLM Fine-Tuning
- **Base model**: Qwen/Qwen3-14B
- **Method**: LoRA (Low-Rank Adaptation) fine-tuning
- **LoRA parameters**: R=32, alpha=64, dropout=0.05
- **Task**: Generate step explanations given graph embeddings as soft prompts
- **Graph prefix**: 16 soft-prompt tokens from graph embedding

### Stage 3: RL Optimization
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Group size**: 16 samples per prompt
- **KL coefficient**: 0.015
- **PPO clip**: 0.18
- **Task**: Optimize step explanations using LLM judge feedback

## Training Pipeline

### Stage 1: GNN Training

**Input**: `input/train.json`, `input/test.json`

**Process**:
1. Load and preprocess data with machine-level train/val split
2. Calculate class weights for imbalanced data (MCP and STEP)
3. Train GNN classifier with focal loss for MCP, cross-entropy for STEP
4. Optimize per-class MCP thresholds on validation set
5. Evaluate on test set and save predictions

**Key hyperparameters**:
- Learning rate: 3e-4 with cosine annealing and warmup
- Batch size: 16
- Epochs: 45 (early stopping patience: 8)
- Label smoothing: 0.08
- Gradient clipping: 1.5
- Loss weights: STEP=1.0, MCP=1.2

**Metrics**:
- Step Accuracy, Macro F1, Weighted F1
- MCP Micro F1, Macro F1, Subset Accuracy, Samples F1
- Combined Score: 0.5 * step_accuracy + 0.5 * mcp_micro_f1

### Stage 2: LLM Fine-Tuning

**Input**: Stage 1 checkpoint + training data

**Process**:
1. Load Stage 1 model to extract graph embeddings
2. Fine-tune Qwen LLM with LoRA using graph embeddings as soft prompts
3. Train on step explanation generation task
4. Validate on held-out machine set

**Key hyperparameters**:
- Learning rate: 5e-5
- Batch size: 2 with gradient accumulation (8 steps)
- Epochs: 15 (early stopping patience: 4)
- Warmup ratio: 0.08
- Gradient clipping: 1.0

### Stage 3: RL Optimization

**Input**: Stage 2 checkpoint + validation data

**Process**:
1. Generate multiple step explanations per input
2. Use LLM judge to score explanations
3. Apply GRPO to optimize policy
4. Update model with RL gradients

**Key hyperparameters**:
- Learning rate: 5e-7
- Group size: 16
- Steps: 2500
- KL coefficient: 0.015
- PPO clip: 0.18
- Gradient accumulation: 4

## Label Spaces

### Step Labels (10 classes)
1. "Do a google search for more information"
2. "Enumerate further on the X service to find software versions, hidden directories and file."
3. "Explore the suspicious files, commands and create a summary of the findings."
4. "Further Enumerate the website. - hidden directories, links and software"
5. "Enumerate the domain"
6. "Exploit the selected exploitations"
7. "Analyze the outcomes of the previous step and find an attack path"
8. "Ask for human assistant"
9. "Explore the source code for vulnerabilities."
10. "End task and ask permission to generate the report"

### MCP Labels (11 classes)
1. Nmap
2. Metasploit
3. Netcat
4. Dirbuster
5. SQLmap
6. Smb client
7. hydra
8. John-the-ripper
9. Google search
10. Interactive CLI
11. Web page interaction

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone repository
cd stepmodel-final

# Install dependencies
pip install -r requirements.txt

# Prepare data
# Place training_data.csv and test_data.csv in data/ directory
```

### Environment Variables (Optional)
```bash
# For LLM-based parsing (optional)
export OPENAI_API_KEY=sk-...

# For NVIDIA API (if using opencode.json)
export NVIDIA_API_KEY=your_api_key
```

## Usage

### Quick Start

```bash
# Run complete pipeline
python run.py

# Run specific stages
python run.py --stage stage1
python run.py --stage stage2
python run.py --stage stage3
```

### Individual Stage Execution

#### Data Preparation
```bash
# Build input JSON files (deterministic, no API key needed)
python build_input_json.py

# Generate attack graphs
python generate_graphs.py

# Quick test with limited data
python build_input_json.py --limit 20
```

#### Stage 1 Training
```bash
python stage1_gnn_train.py
```

#### Stage 2 Fine-Tuning
```bash
python stage2_sft_qwen.py
```

#### Stage 3 RL Optimization
```bash
python stage3_grpo_rl.py
```

#### Evaluation
```bash
# Full evaluation pipeline
python evaluate.py

# Baseline LLM evaluation
python baseline_llm_eval.py --num_shots 0  # zero-shot
python baseline_llm_eval.py --num_shots 3  # 3-shot
python baseline_llm_eval.py --num_shots 5  # 5-shot

# Generate comparison report
python comparison_report.py
```

### Configuration

Edit `config.py` to modify:
- Model hyperparameters (GNN layers, hidden dimensions, dropout)
- Training hyperparameters (learning rates, batch sizes, epochs)
- Loss weights and label smoothing
- File paths and directories

## Data Format

### Input CSV Format

The training and test CSV files should contain the following columns:
- `Machine`: Machine identifier
- `PTT`: Penetration Testing Tree (indented text structure)
- `New strategy`: Current strategy description
- `Strategy explanation`: Strategy explanation
- `New step`: Target step label (to be predicted)
- `Step explanation`: Step explanation
- `MCP_tasks`: Required MCP tools (can be dict string or free text)

### Input JSON Format

After preprocessing with `build_input_json.py`, each record contains:
```json
{
  "machine": "machine_name",
  "graph": {
    "nodes": [
      {"id": "node_id", "title": "node_title", "type": "State|Action|Finding"}
    ],
    "edges": [
      {"from": "source_id", "to": "target_id"}
    ]
  },
  "new_strategy": "...",
  "strategy_explanation": "...",
  "gold_new_step": "...",
  "gold_step_explanation": "...",
  "gold_mcp_tasks": "..."
}
```

## Deterministic PTT Parsing

The system uses a deterministic rule engine for PTT parsing by default:

### Classification Rules

1. **No finding/data payload → State**: Even if title reads like a command, nothing has been produced yet
2. **Contextual/informational payload → State**: Machine names, IPs, status fields kept on State node
3. **Concrete action with payload → Action**: "Perform port scan", "Enumerate HTTP", etc. → separate Finding node

### Node Creation Rules
- Every numbered item gets its own node (including Target IP items with their own numbers)
- Only truly bare `{...}` blocks without labels are folded into parent
- Machine names preserved exactly as in CSV (case-sensitive)

### LLM Mode (Optional)
For ambiguous cases, use `--use-llm` flag:
```bash
python build_input_json.py --use-llm
python generate_graphs.py --use-llm
```

## Performance

### Current Results (Stage 1)
- Step Accuracy: 74.25%
- Step Macro F1: 0.6255
- Step Weighted F1: 0.7529
- MCP Micro F1: 0.5423
- MCP Macro F1: 0.4876
- Combined Score: 0.6424

### Target Performance
- Combined Score: ≥0.80 (approximately 80% overall performance)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `config.py`
2. **Step accuracy 0%**: Check STEP class weights calculation (fixed in current version)
3. **Data leakage**: Ensure machine-level splitting is working correctly
4. **Missing input files**: Run `build_input_json.py` before training

### Debug Mode

Add debugging prints by modifying the logging level in individual scripts or use Python's built-in logging.

## Citation

If you use this code in your research, please cite appropriately.

## License

Please refer to the project license file for usage terms.

## Contact

For questions or issues, please refer to the project repository or contact the maintainers.
