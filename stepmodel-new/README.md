# StepModel-NEW: LLM-based Penetration Testing Step Prediction

## Overview

StepModel-NEW is a machine learning system that predicts the next step in penetration testing scenarios using Large Language Models (LLMs) with reinforcement learning. The system takes graph information (network topology), new strategy, and strategy explanation as text input and predicts the next step, step explanation, and MCP (Model Context Protocol) tools to use.

## Key Changes from Original Architecture

- **Removed GNN components**: No longer uses Graph Neural Networks for graph embeddings
- **Text-based graph representation**: Graph structure is converted to text and fed directly to the LLM
- **Simplified architecture**: Uses only LLM with classification heads for predictions
- **Removed embedding dependencies**: No longer requires SentenceTransformer or graph embedding models

## Architecture

### Input Format
The model takes three types of input as text:

1. **Graph Information**: Network topology converted to text format
   - Nodes: Descriptions of machines, services, vulnerabilities
   - Edges: Connections between nodes

2. **New Strategy**: The proposed penetration testing strategy
3. **Strategy Explanation**: Detailed explanation of the strategy

### Output Format
The model predicts:
1. **Next Step**: Classification over fixed step ontology
2. **Step Explanation**: Text explanation of the next step
3. **MCP Tools**: Multi-label classification over available MCP tools

### Model Components

#### LLMPolicy
A simple policy network that:
- Takes LLM hidden states as input
- Applies pooling strategy (mean/max/hybrid)
- Produces logits for step classification and MCP tool classification

#### Components
- **Step Head**: Classifies next step from fixed ontology
- **MCP Head**: Multi-label classification for MCP tools

## Training Pipeline

### Phase 1: Supervised Warmup Training
- Trains using labeled data with cross-entropy loss
- Uses gradient accumulation for memory efficiency
- Validates after each epoch

### Phase 2: GRPO (Group Relative Policy Optimization) Reinforcement Learning
- Fine-tunes using reward-based learning
- Generates multiple rollouts per sample
- Uses advantage estimation for policy updates
- Includes auxiliary supervised loss for stability

## Data Format

### CSV Files
Training and test data are stored in CSV format with columns:
- `Machine`: Machine name (used to load graph data)
- `PTT`: Previous penetration testing timeline
- `Previous strategy`: Strategy used in previous step
- `Previous step`: Previous step taken
- `Previous step result`: Result of previous step
- `New strategy`: New strategy to execute
- `Strategy explanation`: Explanation of the new strategy
- `New step`: Next step to take (target label)
- `Step explanation`: Explanation of the next step
- `MCP_tasks`: MCP tools to use (multi-label)

### Graph JSON Files
Graph data is stored in JSON files in `embeddings_data/train/` and `embeddings_data/test/`:
- Each machine has a `{machine_name}_processed.json` file
- Contains `nodes` and `edges` with text attributes
- Embeddings are extracted and only text attributes are used

## Configuration

### Model Configuration
```json
{
  "model": {
    "llm_name": "Qwen/Qwen2.5-7B-Instruct",
    "trust_remote_code": true,
    "load_in_4bit": true,
    "use_lora": false,
    "torch_dtype": "float16",
    "pooling_strategy": "mean"
  }
}
```

### Training Configuration
```json
{
  "training": {
    "num_supervised_epochs": 5,
    "num_grpo_epochs": 5,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 3e-5,
    "max_seq_length": 512,
    "step_loss_weight": 4.0,
    "mcp_loss_weight": 1.0,
    "explanation_loss_weight": 0.5
  }
}
```

## Usage

### Training
```bash
python train_gnn_rl.py
```

The script will:
1. Load configuration from `config.json`
2. Load training/test data from CSV files
3. Load graph data from JSON files
4. Initialize LLM and policy network
5. Run supervised training phase
6. Run GRPO reinforcement learning phase
7. Evaluate on test set
8. Save checkpoints

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- (Optional) TensorBoard for logging
- (Optional) PEFT for LoRA

### Memory Optimization
- Uses 4-bit quantization for LLM
- Gradient accumulation to reduce memory usage
- Sequence length limited to 512 tokens
- Frequent GPU cache clearing

## Key Functions

### Data Loading
- `load_processed_data()`: Loads CSV data and merges with graph JSON files
- `graph_to_text()`: Converts graph structure to text representation
- `build_prompt_text()`: Constructs the full prompt for LLM

### Training Functions
- `compute_supervised_loss_for_sample()`: Computes supervised loss
- `compute_grpo_loss()`: Computes GRPO reinforcement learning loss
- `classify_sample()`: Runs inference on a single sample

### Evaluation Functions
- `find_best_mcp_threshold()`: Finds optimal MCP classification threshold
- `evaluate_on_dataset()`: Evaluates model on a dataset
- `compute_reward()`: Computes reward for GRPO training

## File Structure
```
stepmodel-new/
├── train_gnn_rl.py          # Main training script
├── config.json              # Configuration file
├── data/
│   ├── training_data.csv    # Training data
│   └── test_data.csv       # Test data
├── embeddings_data/
│   ├── train/              # Graph data for training machines
│   └── test/               # Graph data for test machines
└── checkpoints/            # Saved model checkpoints
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_seq_length` in config
- Increase `gradient_accumulation_steps`
- Use smaller LLM (e.g., 7B instead of 14B)
- Ensure `load_in_4bit` is true

### Data Loading Errors
- Ensure CSV files exist in `data/` directory
- Ensure graph JSON files exist in `embeddings_data/` directory
- Check that machine names match between CSV and JSON files

### Training Not Converging
- Adjust learning rate
- Check loss weights in config
- Ensure data is properly shuffled
- Monitor TensorBoard logs

## Evaluation Metrics

The system tracks:
- **Step Accuracy**: Accuracy of step classification
- **MCP F1**: F1 score for MCP tool classification
- **MCP Exact**: Exact match rate for MCP tools
- **Combined Score**: Weighted combination of metrics
- **Selection Score**: Metric for model selection

## Future Improvements

- Add support for more LLM architectures
- Implement more sophisticated graph-to-text conversion
- Add data augmentation for graph representations
- Implement curriculum learning
- Add multi-task learning for related tasks
