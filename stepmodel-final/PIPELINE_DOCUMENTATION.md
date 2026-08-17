# StepModel-Final Pipeline Documentation

## Overview
Multi-stage ML pipeline for penetration testing step prediction, MCP tool classification, and explanation generation using GNNs, LLM fine-tuning, and RL.

## Stage 1: GNN Classifier Training

### Purpose
Train graph neural network for step and MCP tool classification on attack path graphs.

### Architecture
- **Graph Encoder**: Frozen pre-trained encoder processes graph structure
- **Classification Heads**: Step (10 classes) + MCP (11 classes, multi-label)
- **Node Features**: 387-dim embeddings (384-dim text + 3-dim node type)

### Class Imbalance Handling
- **Inverse Frequency Weighting**: `weight = 1.0 / (frequency + epsilon)`
- **Rare Class Boost**: Double weight for classes with <15 samples
- **Normalization**: Mean weight = 1.0 to prevent extreme values

### Training
- **Input**: `input/train.json` with graph embeddings
- **Validation**: 10% split for early stopping
- **Loss**: Cross-entropy with class weights for MCP
- **Output**: `checkpoints/stage1_gnn_classifier.pt`

### Performance
- Step Accuracy: 100%
- MCP Micro F1: ~76%
- MCP Macro F1: ~67%

## Stage 2: SFT Fine-Tuning

### Purpose
Fine-tune Qwen LLM with graph conditioning for step prediction, MCP classification, and explanation generation.

### Architecture
- **GraphPrefixAdapter**: Trainable soft-prompt tokens encoding graph info
- **Frozen GNN**: Stage 1 encoder (frozen)
- **LoRA**: Parameter-efficient fine-tuning of Qwen

### Graph Conditioning
1. Process graph through frozen GNN
2. Generate graph embeddings
3. Apply trainable GraphPrefixAdapter
4. Prepend soft-prompt to input embeddings
5. Generate with graph-conditioned inputs

### Output Format
```json
{
  "New step": "predicted step",
  "Step explanation": "natural language explanation",
  "MCP_tasks": {"tool_name": {}}
}
```

### Step Label Extraction
- **Primary**: Normalization using StepLabelNormalizer
- **Fallback 1**: Direct string match
- **Fallback 2**: Fuzzy matching (60% similarity)
- **Fallback 3**: Mark as UNPARSEABLE

### Training
- **Input**: `input/train.json`
- **Batch Size**: 4-8 (memory constraints)
- **Epochs**: 1-2
- **Output**: `checkpoints/stage2_qwen_lora/`

### Performance (Test Set: 268 samples)
- **Step Accuracy**: 99.25%
- **Step Macro F1**: 80.73%
- **Step Weighted F1**: 99.45%
- **MCP Micro F1**: 72.18%
- **MCP Macro F1**: 65.14%
- **MCP Subset Accuracy**: 56.72%
- **Unparseable Predictions**: 0.4% (1/268)

## Stage 3: GRPO RL Fine-Tuning

### Purpose
Improve explanation quality through RL while maintaining classification performance.

### Why Custom Implementation?
Standard libraries don't support graph conditioning during generation, causing distribution shift.

### Custom GRPO Loop
1. Build graph prefix (frozen GNN + trainable adapter)
2. Generate G=8 completions with inputs_embeds
3. Score each completion with reward function
4. Compute group-relative advantages: `A_i = (r_i - mean) / (std + ε)`
5. Forward pass for generated tokens, compute log-probs
6. Apply clipped policy-gradient loss + KL penalty
7. Update LoRA + adapter weights

### Reward Function
```python
reward = 0.10 * format + 
         0.30 * step_similarity + 
         0.30 * mcp_f1 + 
         0.30 * llm_judge
```

### Components
- **Format (10%)**: Valid JSON with required keys
- **Step Similarity (30%)**: Embedding cosine similarity
- **MCP F1 (30%)**: Set F1 between predicted/gold tools
- **LLM Judge (30%)**: GPT-4o explanation evaluation

### Training
- **Group Size**: 8 completions
- **KL Penalty**: β=0.05
- **Gradient Accumulation**: 8 steps
- **Output**: `checkpoints/stage3_qwen_grpo/`

### Performance (Test Set: 268 samples)
- **Step Accuracy**: 98.88%
- **Step Macro F1**: 79.99%
- **Step Weighted F1**: 99.12%
- **MCP Micro F1**: 72.35%
- **MCP Macro F1**: 65.20%
- **MCP Subset Accuracy**: 56.34%
- **Unparseable Predictions**: 0.4% (1/268)

## Baseline Evaluation

### Purpose
Compare fine-tuned models against zero-shot and few-shot LLM baselines using Qwen/Qwen2.5-7B-Instruct.

### Methods
- **Zero-shot**: Direct prompting without examples
- **3-shot**: 3 diverse, size-capped examples in prompt
- **5-shot**: 5 diverse, size-capped examples in prompt

### Performance Comparison (Test Set: 268 samples)

| Model | Step Accuracy | Step Macro F1 | Step Weighted F1 | MCP Micro F1 | MCP Macro F1 | MCP Subset Accuracy |
|-------|--------------|---------------|------------------|--------------|--------------|---------------------|
| **Zero-shot** | 48.51% | 35.55% | 51.34% | 39.19% | 42.37% | 21.27% |
| **3-shot** | 62.69% | 42.97% | 64.19% | 51.73% | 46.11% | 35.45% |
| **5-shot** | 67.16% | 48.46% | 68.24% | 59.05% | 49.92% | 33.21% |
| **Stage 2** | **99.25%** | **80.73%** | **99.45%** | 72.18% | 65.14% | **56.72%** |
| **Stage 3** | 98.88% | 79.99% | 99.12% | **72.35%** | **65.20%** | 56.34% |

### Key Improvements (Stage 2 vs Zero-shot Baseline)
- **Step Accuracy**: +50.75 percentage points (+104.6%)
- **Step Macro F1**: +45.18 percentage points (+127.1%)
- **Step Weighted F1**: +48.11 percentage points (+93.7%)
- **MCP Micro F1**: +32.99 percentage points (+84.2%)
- **MCP Macro F1**: +22.77 percentage points (+53.7%)
- **MCP Subset Accuracy**: +35.45 percentage points (+166.7%)

### Best Model Per Metric
- **mcp_macro_f1**: Stage 3 (0.6520)
- **mcp_micro_f1**: Stage 3 (0.7235)
- **mcp_subset_accuracy**: Stage 2 (0.5672)
- **step_accuracy**: Stage 2 (0.9925)
- **step_macro_f1**: Stage 2 (0.8073)
- **step_weighted_f1**: Stage 2 (0.9945)

### Output Files
- `output/baseline_zeroshot.csv`
- `output/baseline_3shot.csv`
- `output/baseline_5shot.csv`

## Evaluation System

### LLM Judge
- **Model**: GPT-4o
- **Criteria**: Same meaning, equivalent concepts, classroom acceptable
- **Scoring**: 0.0-1.0, binary at 0.6 threshold
- **Performance**: 90.3% accuracy for Stage 2/3

### Metrics
- **Step Classification**: Accuracy, precision, recall, F1 (macro, weighted)
- **MCP Classification**: Subset accuracy, micro/macro/samples F1
- **Explanation Quality**: LLM judge correctness score

## Comparison Report

### Purpose
Consolidate metrics from all models and baselines for comprehensive analysis.

### Outputs
- `output/metrics_comparison.csv`: Performance comparison table
- `output/consolidated_predictions.csv`: All predictions combined
- `output/comparison_report.txt`: Summary report with key improvements
- `output/mcp_comparison.png`: MCP metrics visualization
- `output/step_comparison.png`: Step metrics visualization
- `output/radar_comparison.png`: Radar chart comparison

## Data Flow

### Train/Test Split
- **Training**: `input/train.json` (1,728 usable rows)
- **Test**: `input/test.json` (268 usable rows)
- **No Leakage**: Proper separation maintained throughout

### Pipeline
1. Data preparation → train/test JSON files
2. Stage 1 training → GNN classifier
3. Stage 2 training → Graph-conditioned LLM
4. Stage 3 training → RL-optimized LLM
5. Evaluation → All models on test set
6. Baseline evaluation → Zero/few-shot comparisons
7. Comparison report → Consolidated analysis

## Key Technical Innovations

1. **Graph-Structured Learning**: Leverages attack path topology
2. **Cost-Sensitive Learning**: Addresses MCP class imbalance
3. **Graph-Conditioned Generation**: Maintains graph awareness in LLM
4. **Custom GRPO**: Graph conditioning throughout RL
5. **LLM Judge Integration**: Semantic explanation evaluation
6. **Robust Parsing**: Multiple fallback strategies for label extraction

## Current Limitations

- Stage 3 provides marginal improvements over Stage 2 on MCP metrics but slightly lower step accuracy
- KL penalty may be too restrictive for exploration
- LLM judge weight might need adjustment in reward function
- Baseline evaluations show significant unparseable prediction rates (0.4-0.7%)

## Future Improvements

- Increase LLM judge reward weight to 40-50% to improve explanation quality
- Reduce KL penalty for more exploration in Stage 3
- Add explanation-specific bonus rewards
- Increase Stage 3 training steps for better convergence
- Investigate why Stage 3 slightly underperforms Stage 2 on step accuracy
- Improve baseline step extraction to reduce unparseable predictions
