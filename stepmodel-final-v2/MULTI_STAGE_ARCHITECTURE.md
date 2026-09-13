# Multi-Stage Training Pipeline Architecture

This document describes the three-stage training pipeline for the step prediction model with graph conditioning.

## Overview

The pipeline consists of three sequential training stages:

1. **Stage 1**: Graph + Context Understanding (CNN + GNN)
2. **Stage 2**: Supervised Fine-Tuning with Graph Conditioning
3. **Stage 3**: GRPO RL with Comprehensive Reward

## Stage 1: Graph + Context Understanding

### Architecture

- **Input**: New strategy + strategy explanation + graph structure
- **Components**:
  - **CNN Encoder**: Processes strategy and explanation through paper-inspired CNN
  - **GNN Encoder (GINE)**: Processes graph structure with typed edges
  - **Context Projector**: Projects BGE field embeddings
  - **Fusion Module**: Combines CNN, GNN, and context representations
  - **Prediction Heads**: Separate heads for Step and MCP prediction

### Output

- 512-d graph representation
- Step logits (10 classes)
- MCP logits (11 tools, multi-label)

### Training

- Loss: Weighted cross-entropy with focal loss
- Checkpoint saved to: `checkpoints/stage1_gnn_classifier.pt`

## Stage 2: Supervised Fine-Tuning with Graph Conditioning

### Architecture

- **Input**: New strategy + strategy explanation + graph structure
- **Components**:
  - **Stage 1 Checkpoint** (frozen): Provides graph encoding
  - **Prefix Adapter**: Converts 512-d GNN representation → 8 soft prompt tokens
  - **Qwen + LoRA**: Base LLM with trainable LoRA adapters
  - **Prompt Construction**: Strategy + explanation + soft prompt tokens

### Data Flow

```
Graph → Stage 1 (frozen) → 512-d representation → Prefix Adapter → 8 soft tokens
                                                          ↓
Strategy + Explanation → Tokenizer → Prompt tokens → Concatenation
                                                          ↓
                                    Qwen + LoRA → Step + MCP + Explanation
```

### Training

- Objective: Supervised fine-tuning on training data
- Checkpoint saved to: `checkpoints/stage2_qwen_lora/`
- Prefix adapter saved to: `checkpoints/stage2_qwen_lora/prefix_adapter.pt`

## Stage 3: GRPO RL with Comprehensive Reward

### Architecture

- **Input**: New strategy + strategy explanation + graph structure
- **Components**:
  - **Stage 2 Checkpoint** (trainable): Qwen + LoRA adapter
  - **Frozen Prefix Adapter** (from Stage 2): Provides graph conditioning
  - **Reference Model**: Frozen copy of Stage 2 for KL penalty
  - **Comprehensive Reward Function**: Multi-dimensional reward

### Reward Function

The reward function uses **equal weighting** for three components:

1. **Step Reward** (weight=1.0): Exact match with ground truth step
2. **MCP Reward** (weight=1.0): F1 score for tool prediction
3. **Explanation Reward** (weight=1.0): Multi-dimensional quality score

#### Explanation Evaluation

Based on research findings, explanation quality is evaluated using multiple dimensions:

- **Relevance**: Does explanation justify the predicted step?
- **Technical Accuracy**: Are technical claims correct?
- **Completeness**: Does it cover key justification points?
- **Clarity**: Is it well-structured and unambiguous?

These dimensions are evaluated using the LLM judge (`llm_judge.py`) with a rubric-based approach (0-3 scale per dimension).

### Training

- Algorithm: Group Relative Policy Optimization (GRPO)
- KL Penalty: Prevents policy drift from Stage 2
- Dual-Clip PPO: Handles negative advantages
- Checkpoint saved to: `checkpoints/stage3_qwen_grpo/`

## Evaluation Metrics

### Step Prediction

- **Accuracy**: Exact match rate
- **Micro-F1**: F1 score across all classes

### MCP Tool Prediction

- **Accuracy**: Per-tool accuracy
- **Micro-F1**: F1 score across all tools
- **Macro-F1**: Average per-tool F1
- **Missing Tools**: Average number of ground truth tools not predicted
- **Extra Tools**: Average number of predicted tools not in ground truth
- **Exact Match Rate**: Percentage of perfect tool set matches
- **Per-Tool Metrics**: Precision, Recall, F1 for each tool

### Explanation Quality

- **Structural Metrics**:
  - Length similarity with ground truth
  - Completeness (non-empty)
  
- **LLM Judge Metrics** (when available):
  - Relevance (0-3 scale)
  - Technical Accuracy (0-3 scale)
  - Completeness (0-3 scale)
  - Clarity (0-3 scale)
  - Faithfulness (mapped from technical accuracy)
  - Plausibility (mapped from relevance)
  - Correctness (gate-based accuracy)
  - Overall Quality (weighted average)

## File Structure

```
stepmodel-final-v2/
├── core/
│   ├── graph_encoder.py              # Stage 1 model
│   ├── graph_prefix_adapter.py       # GNN → soft prompt tokens
│   ├── comprehensive_evaluator.py   # Evaluation metrics
│   └── llm_judge.py                 # LLM-based explanation evaluation
├── training/
│   ├── stage2_sft_with_graph.py      # Stage 2 training
│   └── stage3_grpo_with_graph.py    # Stage 3 training
├── eval/
│   └── comprehensive_evaluation.py  # Final evaluation script
└── config.py                        # Configuration
```

## Usage

### Training Pipeline

```bash
# Stage 1: Train graph encoder (if not already done)
python training/stage1_gnn_train.py

# Stage 2: Supervised fine-tuning with graph conditioning
python training/stage2_sft_with_graph.py

# Stage 3: GRPO RL with comprehensive reward
python training/stage3_grpo_with_graph.py
```

### Evaluation

```bash
# Evaluate final model
python eval/comprehensive_evaluation.py \
    --checkpoint checkpoints/stage3_qwen_grpo \
    --test-data input/test.json \
    --output evaluation_results.json \
    --use-llm-judge
```

## Key Design Decisions

### Equal Weighting in Reward

The reward function uses equal weights (1.0 each) for step, MCP, and explanation components. This ensures:

- No single component dominates
- Balanced improvement across all tasks
- Flexibility to adjust weights based on empirical results

### Multi-Dimensional Explanation Evaluation

Based on research findings (Hase et al. 2020, ICE framework, etc.):

- Single-score metrics (BLEU, F1) have poor correlation with human judgment
- Multi-dimensional evaluation is more informative
- Rubric-based LLM evaluation provides reproducible scores
- Gate-based correctness (not thresholded composite) prevents compensation

### Graph Conditioning Strategy

- **Stage 2**: Prefix adapter is trainable, learns to map GNN → soft prompts
- **Stage 3**: Prefix adapter is frozen, only LLM is trained
- This prevents catastrophic forgetting of graph conditioning during RL

### Stability Features

- **Dual-Clip PPO**: Handles negative advantages to prevent loss explosions
- **KL Hard Cap**: Per-micro-batch KL limit (not window average)
- **Early Stopping**: Patience-based stopping on held-out validation
- **Gradient Clipping**: Prevents gradient explosions

## References

### Explanation Evaluation Research

1. **Hase et al. (2020)** - "ERASER: A Benchmark to Evaluate Rationalized NLP Models"
   - Introduced sufficiency and comprehensiveness metrics
   - Limitations: single intervention, no statistical testing

2. **ICE Framework (2025)** - "Intervention-Consistent Explanation Evaluation"
   - Randomized baselines with statistical testing
   - Operator-dependent faithfulness
   - Win rates with confidence intervals

3. **Walk the Talk (2025)** - "Measuring the Faithfulness of LLM Explanations"
   - Causal concept faithfulness definition
   - Counterfactual intervention with auxiliary LLM
   - Bayesian hierarchical modeling

4. **Causal Faithfulness (2025)** - "Towards Faithful NLEs: Activation Patching"
   - Uses activation patching instead of SHAP
   - Avoids out-of-distribution concerns
   - Token and layer-level faithfulness

### Training Algorithms

1. **GRPO (Group Relative Policy Optimization)**
   - Group-relative advantage normalization
   - More stable than standard PPO for small batch sizes

2. **Dual-Clip PPO (Ye et al. 2020)**
   - Additional clipping for negative advantages
   - Prevents loss explosions in negative-advantage regime

## Future Improvements

1. **Adaptive Reward Weights**: Learn optimal weights during training
2. **Curriculum Learning**: Start with step/MCP focus, gradually add explanation
3. **Ensemble Evaluation**: Combine multiple LLM judges for robustness
4. **Causal Faithfulness**: Implement activation patching for explanation faithfulness
5. **Tool-Specific Rewards**: Different weights for different MCP tools based on importance
