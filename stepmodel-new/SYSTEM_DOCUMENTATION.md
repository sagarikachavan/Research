# stepmodel-new System Documentation

## Overview

stepmodel-new is an advanced penetration testing step prediction system that combines Graph Neural Networks (GNNs), Large Language Models (LLMs), and Reinforcement Learning (RL) to predict penetration testing steps and identify relevant tools (MCP - Machine Control Protocol). This system represents a significant improvement over the baseline approach with enhanced data quality filtering, curriculum learning, and optimized training strategies.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     stepmodel-new System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Raw Data    │───▶│  Data        │───▶│  Filtered    │      │
│  │  (Graphs +   │    │  Quality     │    │  Dataset     │      │
│  │  Text)       │    │  Filtering   │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                            │                   │                 │
│                            ▼                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Text        │    │  Graph       │    │  Label       │      │
│  │  Embeddings  │    │  Embeddings  │    │  Processing  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                            │                   │                 │
│                            └────────┬──────────┘                 │
│                                     ▼                            │
│  ┌──────────────────────────────────────────────────┐           │
│  │           GNN-LLM Policy Network                 │           │
│  │  ┌─────────────┐    ┌─────────────┐            │           │
│  │  │  GNN Layer  │───▶│  Projector  │───▶ [GRAPH] │           │
│  │  │  (GCN/GAT)  │    │  (MLP)      │    Token     │           │
│  │  └─────────────┘    └─────────────┘    │         │           │
│  │                                          │         │           │
│  │  ┌──────────────────────────────────────┘         │           │
│  │  │                                                 │           │
│  │  ▼                                                 │           │
│  │  ┌─────────────┐    ┌─────────────┐              │           │
│  │  │  LLM        │    │  Output     │              │           │
│  │  │  (Qwen3-14B)│───▶│  Layer      │───▶ Step     │           │
│  │  │  + LoRA     │    │             │    + MCP     │           │
│  │  └─────────────┘    └─────────────┘    Prediction│           │
│  └──────────────────────────────────────────────────┘           │
│                                     │                            │
│                                     ▼                            │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Training Pipeline                   │           │
│  │  ┌──────────────┐    ┌──────────────┐           │           │
│  │  │  Phase 0:    │    │  Phase 1:    │           │           │
│  │  │  Pretraining │───▶│  Supervised  │───▶        │           │
│  │  │  (Link Pred)│    │  Training    │           │           │
│  │  └──────────────┘    └──────────────┘           │           │
│  │                            │                      │           │
│  │                            ▼                      │           │
│  │  ┌──────────────┐    ┌──────────────┐           │           │
│  │  │  Phase 2:    │    │  Curriculum  │           │           │
│  │  │  GRPO RL     │───▶│  Learning    │           │           │
│  │  │  Fine-tuning │    │  Stages     │           │           │
│  │  └──────────────┘    └──────────────┘           │           │
│  └──────────────────────────────────────────────────┘           │
│                                     │                            │
│                                     ▼                            │
│  ┌──────────────────────────────────────────────────┐           │
│  │              Evaluation & Metrics                 │           │
│  │  • Step Accuracy  • Step Micro F1                 │           │
│  │  • MCP Accuracy   • MCP Micro F1                  │           │
│  │  • Average Reward • Combined Score               │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Processing Pipeline

#### 1.1 Raw Data Input
- **Graph Data**: Network topology from penetration testing scenarios
- **Text Data**: Descriptions of penetration testing steps and contexts
- **Labels**: Step classifications and MCP tool selections

#### 1.2 Data Quality Filtering
The system implements sophisticated data quality filtering to handle the expanded dataset (8x larger than baseline):

**Quality Assessment Metrics:**
- **Text Quality**: Length validation, required field presence, structural integrity
- **Label Quality**: Consistency checks, validity of step and MCP labels
- **Graph Quality**: Node/edge validation, structural sanity checks
- **Embedding Quality**: Dimension validation, NaN/infinite value detection

**Filtering Process:**
```python
# Quality scoring (0-1 scale)
overall_score = (
    0.3 * text_quality +
    0.3 * label_quality +
    0.2 * graph_quality +
    0.2 * embedding_quality
)

# Filter based on threshold (default: 0.7)
filtered_samples = [s for s in samples if s.quality_score >= threshold]
```

#### 1.3 Embedding Generation
- **Text Embeddings**: Generated using SentenceTransformer (all-MiniLM-L6-v2)
- **Graph Embeddings**: Node and edge embeddings processed separately
- **Token Alignment**: Special [GRAPH] token for graph-LLM integration

### 2. Neural Network Architecture

#### 2.1 GNN Component
- **Architecture**: Graph Convolutional Network (GCN) with configurable GAT support
- **Hidden Dimensions**: 256 hidden units, 128 output dimensions
- **Pooling Strategy**: Hybrid pooling (mean + attention)
- **Graph Tokens**: 4 special [GRAPH] tokens for graph representation

#### 2.2 LLM Component
- **Base Model**: Qwen/Qwen3-14B (14 billion parameters)
- **Quantization**: 4-bit quantization for memory efficiency
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with rank 16, alpha 32
- **Target Modules**: Attention and feed-forward layers
- **Precision**: bfloat16 for training efficiency

#### 2.3 Integration Strategy
```
Graph Data → GNN → Projector → [GRAPH] Tokens → LLM → Predictions
```

The GNN processes graph structure and produces graph tokens that are inserted into the LLM's input sequence, allowing the language model to condition its predictions on graph topology.

### 3. Training Pipeline

#### 3.1 Phase 0: Self-Supervised Pretraining
**Objective**: Learn graph representations through link prediction
- **Task**: Predict missing edges in the graph
- **Loss**: Binary cross-entropy for edge existence
- **Output**: Pretrained GNN and projector weights
- **Status**: Optional (can be skipped if quality is insufficient)

#### 3.2 Phase 1: Supervised Training
**Objective**: Train the full model with labeled data
- **Loss Function**: Weighted combination of step and MCP losses
- **Step Loss**: Cross-entropy with class weighting (power 0.5, max weight 5.0)
- **MCP Loss**: Binary cross-entropy for multi-label classification
- **Optimizer**: AdamW with learning rate 5e-5
- **Scheduler**: Linear warmup (500 steps) with cosine decay
- **Epochs**: 16 (increased from 8 for better convergence)

**Key Improvements:**
- **Higher Learning Rate**: 5e-5 (vs 3e-5) for faster convergence
- **Increased Regularization**: weight_decay 0.02, drift_guard_weight 0.02
- **Step Loss Weight**: 2.0 (vs 1.5) to emphasize step prediction
- **Extended Training**: 16 epochs (vs 8) for larger dataset
- **Enhanced Patience**: 8 epochs (vs 5) for early stopping

#### 3.3 Phase 2: GRPO Reinforcement Learning
**Objective**: Fine-tune policy using reward-based optimization
- **Algorithm**: Group Relative Policy Optimization (GRPO)
- **Rollouts**: 4 generations per sample
- **Clip Epsilon**: 0.2 for policy update stability
- **Auxiliary Supervised Loss**: 0.15 weight to maintain supervised performance
- **Epochs**: 15 (increased from 10 for better RL convergence)

**GRPO Loss Computation:**
```python
# Compute advantages
rewards = [r['reward'] for r in rollouts]
advantages = (rewards - mean(rewards)) / (std(rewards) + epsilon)

# Compute policy ratio
ratio = exp(new_log_prob - old_log_prob)
clipped_ratio = clamp(ratio, 1 - clip_eps, 1 + clip_eps)

# GRPO objective
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

**Memory Optimizations:**
- `torch.no_grad()` for inference during rollouts
- Explicit tensor deletion after each iteration
- `torch.cuda.empty_cache()` for memory management
- Mixed precision training with `torch.amp.autocast`

### 4. Curriculum Learning

#### 4.1 Multi-Stage Training
The system implements curriculum learning with 3 progressive stages:

**Stage 0** (High-Quality Focus):
- Quality threshold: 0.9
- Only highest quality samples
- Establish strong foundation

**Stage 1** (Progressive Expansion):
- Quality threshold: 0.8
- Include medium-quality samples
- Expand knowledge base

**Stage 2** (Full Dataset):
- Quality threshold: 0.7
- Include all filtered samples
- Final fine-tuning

#### 4.2 Dynamic Dataset Selection
```python
def create_curriculum_dataset(dataset, stage, total_stages, threshold):
    progress = (stage + 1) / total_stages
    current_threshold = threshold + (0.2 * (1 - progress))
    return filter_by_quality(dataset, current_threshold)
```

### 5. Evaluation Metrics

#### 5.1 Primary Metrics
- **Step Accuracy**: Percentage of correct step predictions
- **Step Micro F1**: Harmonic mean of precision and recall for steps
- **MCP Accuracy**: Percentage of correct MCP tool predictions
- **MCP Micro F1**: Global F1 score for MCP predictions
- **Average Reward**: Combined reward signal from both tasks

#### 5.2 Selection Score
Weighted combination of step and MCP performance:
```python
selection_score = (
    0.85 * step_accuracy +
    0.15 * mcp_f1
)
```

#### 5.3 Baseline Comparison
Target performance from PenStrategist.pdf:
- **Step Accuracy**: 82.87%
- **Step Micro F1**: 0.80
- **MCP Accuracy**: 48.88%
- **MCP Micro F1**: 0.64

## Key Improvements Over Baseline

### 1. Data Quality Management
- **Comprehensive Filtering**: Multi-dimensional quality assessment
- **Noise Reduction**: Removes low-quality samples from expanded dataset
- **Quality Scoring**: Transparent quality metrics for each sample

### 2. Enhanced Training Strategy
- **Curriculum Learning**: Progressive training from high to full quality
- **Extended Training**: More epochs for better convergence
- **Optimized Hyperparameters**: Higher learning rates, better regularization

### 3. Memory Efficiency
- **4-bit Quantization**: Reduces memory footprint by 4x
- **LoRA Fine-tuning**: Efficient parameter-efficient training
- **Memory Management**: Explicit cleanup and gradient checkpointing

### 4. Robustness Features
- **Class Weighting**: Handles imbalanced step distributions
- **Weighted Sampling**: Balances training data distribution
- **Drift Guard**: Prevents catastrophic forgetting during RL

## Configuration Parameters

### Model Configuration
```json
{
  "model": {
    "llm_name": "Qwen/Qwen3-14B",
    "load_in_4bit": true,
    "use_lora": true,
    "lora_r": 16,
    "lora_alpha": 32,
    "gnn_type": "gcn",
    "gnn_hidden_dim": 256,
    "gnn_out_dim": 128,
    "graph_token_count": 4,
    "pooling_strategy": "hybrid"
  }
}
```

### Training Configuration
```json
{
  "training": {
    "num_supervised_epochs": 16,
    "num_grpo_epochs": 15,
    "learning_rate": 5e-5,
    "weight_decay": 0.02,
    "max_seq_length": 1024,
    "step_loss_weight": 2.0,
    "mcp_loss_weight": 1.5,
    "use_curriculum_learning": true,
    "curriculum_stages": 3,
    "data_quality_threshold": 0.7,
    "use_data_filtering": true
  }
}
```

## Usage Instructions

### 1. Data Preparation
```bash
# Filter dataset for quality
python3 filter_data_quality.py \
    --input embeddings_data/train/all_processed.json \
    --output embeddings_data/train/all_processed_filtered.json \
    --threshold 0.7 \
    --max_samples 5000000
```

### 2. Training
```bash
# Set environment for memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
python3 train_gnn_rl.py --config config.json
```

### 3. Evaluation
```bash
# Evaluate trained model
python3 evaluate.py --checkpoint checkpoints/best_checkpoint.pt
```

## Performance Optimization

### Memory Optimization
- Use 4-bit quantization for LLM
- Implement gradient checkpointing for large models
- Use mixed precision training (bfloat16)
- Explicit memory cleanup during training

### Training Speed
- Use weighted sampling for balanced batches
- Implement curriculum learning for faster convergence
- Optimize data loading with proper batching
- Use GPU acceleration for all components

### Quality Assurance
- Monitor validation metrics during training
- Implement early stopping based on selection score
- Use data quality filtering to reduce noise
- Apply curriculum learning for stable training

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size (currently 1)
- Decrease max_seq_length
- Reduce num_generations_per_sample
- Ensure 4-bit quantization is enabled

**Poor Step Accuracy:**
- Check data quality filtering threshold
- Verify curriculum learning stages
- Increase training epochs
- Adjust step_loss_weight

**Overfitting:**
- Increase weight_decay
- Add more regularization
- Use data augmentation
- Implement early stopping

## Future Enhancements

### Planned Improvements
1. **Advanced Graph Architectures**: Explore Transformer-based GNNs
2. **Multi-Task Learning**: Add auxiliary tasks for better representations
3. **Ensemble Methods**: Combine multiple models for robustness
4. **Active Learning**: Iteratively improve dataset quality
5. **Transfer Learning**: Pretrain on larger graph datasets

### Research Directions
1. **Graph-LLM Integration**: Explore more sophisticated fusion methods
2. **Reward Engineering**: Improve reward signals for RL
3. **Uncertainty Estimation**: Add confidence measures to predictions
4. **Explainability**: Improve model interpretability

## References

1. **PenStrategist Paper**: Baseline methodology and evaluation metrics
2. **GRPO Algorithm**: Group Relative Policy Optimization for RL
3. **LoRA Paper**: Low-Rank Adaptation for efficient fine-tuning
4. **GCN/GAT**: Graph Convolutional and Graph Attention Networks
5. **Curriculum Learning**: Progressive training strategies

## Contact and Support

For questions or issues related to stepmodel-new, please refer to:
- Implementation Runbook: `IMPLEMENTATION_RUNBOOK.md`
- Configuration File: `config.json`
- Training Script: `train_gnn_rl.py`
- Evaluation Script: `evaluate.py`

---

**Document Version**: 1.0  
**Last Updated**: 2025-07-12  
**System**: stepmodel-new  
**Status**: Production Ready
