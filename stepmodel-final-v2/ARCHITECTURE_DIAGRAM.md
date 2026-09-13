# Multi-Stage Graph-Conditioned Pipeline Architecture Diagram

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Graph + Context Understanding (Already Trained)

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  New Strategy    │     │ Strategy Expl.   │     │   Graph Data     │
│  (text)          │     │  (text)          │     │  (nodes, edges)  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┴────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Stage 1 Model     │
                    │  (graph_encoder.py) │
                    └─────────┬───────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ CNN Encoder  │   │ GNN (GINE)   │   │ Context      │
  │ (paper-style)│   │ 3 layers     │   │ Projector    │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Fusion Module     │
                    │ (concat + MLP)      │
                    └─────────┬───────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Step Head    │   │ MCP Head     │   │ 512-d Graph  │
  │ (10 classes) │   │ (11 tools)   │   │ Representation│
  └──────────────┘   └──────────────┘   └──────┬───────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Stage 1 Checkpt │
                                    │ (saved to disk) │
                                    └─────────────────┘

Output: Step logits, MCP logits, 512-d graph representation
```

## Stage 2: Supervised Fine-Tuning with Graph Conditioning

```
┌──────────────────┐     ┌──────────────────┐
│  New Strategy    │     │ Strategy Expl.   │
│  (text)          │     │  (text)          │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └────────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Tokenizer     │
          │   (Qwen)        │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Prompt Tokens  │
          └────────┬─────────┘
                   │
                   │
┌──────────────────┴──────────────────┐
│                                      │
│  ┌──────────────────────────────┐  │
│  │   Stage 1 Checkpt (FROZEN)    │  │
│  │   - GNN Encoder               │  │
│  │   - CNN Encoder               │  │
│  │   - Fusion Module             │  │
│  └──────────────┬───────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Graph Data → GNN → 512-d    │  │
│  │  Representation              │  │
│  └──────────────┬───────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Prefix Adapter (TRAINABLE)  │  │
│  │  512-d → 8 soft prompt      │  │
│  │  tokens (LLM hidden dim)    │  │
│  └──────────────┬───────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  8 Soft Prompt Tokens        │  │
│  └──────────────┬───────────────┘  │
└────────────────┼───────────────────┘
                 │
                 ▼
          ┌─────────────────┐
          │  Concatenate    │
          │  [Prompt + 8    │
          │   soft tokens]  │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Qwen + LoRA    │
          │  (TRAINABLE)    │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Output:        │
          │  - Step         │
          │  - MCP tools    │
          │  - Explanation  │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Stage 2 Checkpt│
          │  (Qwen + LoRA)  │
          │  + Prefix Adptr │
          └─────────────────┘

Training: Supervised learning on training data
Loss: Cross-entropy on generated tokens
```

## Stage 3: GRPO RL with Comprehensive Reward

```
┌──────────────────┐     ┌──────────────────┐
│  New Strategy    │     │ Strategy Expl.   │
│  (text)          │     │  (text)          │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └────────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   Tokenizer     │
          │   (Qwen)        │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Prompt Tokens  │
          └────────┬─────────┘
                   │
                   │
┌──────────────────┴──────────────────┐
│                                      │
│  ┌──────────────────────────────┐  │
│  │   Stage 1 Checkpt (FROZEN)    │  │
│  │   - GNN Encoder               │  │
│  └──────────────┬───────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Graph Data → GNN → 512-d    │  │
│  └──────────────┬───────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Prefix Adapter (FROZEN)     │  │
│  │  Loaded from Stage 2         │  │
│  │  512-d → 8 soft prompt      │  │
│  └──────────────┬───────────────┘  │
│                 │                   │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  8 Soft Prompt Tokens        │  │
│  └──────────────┬───────────────┘  │
└────────────────┼───────────────────┘
                 │
                 ▼
          ┌─────────────────┐
          │  Concatenate    │
          │  [Prompt + 8    │
          │   soft tokens]  │
          └────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│ Policy Model    │  │ Reference Model │
│ (Stage 2 ckpt)  │  │ (Stage 2 ckpt)  │
│ Qwen + LoRA     │  │ Qwen + LoRA     │
│ (TRAINABLE)     │  │ (FROZEN)        │
└────────┬────────┘  └────────┬────────┘
         │                    │
         │                    │
         └────────┬───────────┘
                  │
                  ▼
          ┌─────────────────┐
          │  Generate N=8   │
          │  Responses      │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Comprehensive  │
          │  Reward Function│
          │  (Equal Weights)│
          └────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌────────┐   ┌────────┐
│ Step   │   │ MCP    │   │ Expl   │
│ Reward │   │ Reward │   │ Reward │
│(match) │   │  (F1)  │   │(LLM    │
└────────┘   └────────┘   │Judge)  │
                          └────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Compute       │
          │  Advantages    │
          │  (Group Norm)  │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  GRPO Update   │
          │  - Policy Grad │
          │  - KL Penalty  │
          │  - Dual-Clip   │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Stage 3 Checkpt│
          │  (Qwen + LoRA)  │
          └─────────────────┘

Training: GRPO RL with comprehensive reward
Reward = (Step + MCP + Explanation) / 3 (equal weights)
```

## Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FINAL EVALUATION                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ Stage 3 Checkpt  │
│ (Qwen + LoRA)    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Test Data      │
│  (strategies,   │
│   explanations, │
│   graphs)       │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │
│  Predictions    │
└────────┬─────────┘
         │
         └──────────────────┬──────────────────┬──────────────────┐
                            │                  │                  │
                            ▼                  ▼                  ▼
                  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
                  │ Step Eval    │   │ MCP Eval     │   │ Expl Eval    │
                  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                         │                  │                  │
                         ▼                  ▼                  ▼
                  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
                  │ - Accuracy   │   │ - Accuracy   │   │ - LLM Judge  │
                  │ - Micro-F1   │   │ - F1         │   │   (multi-    │
                  │              │   │ - Micro-F1   │   │    dim)      │
                  └──────────────┘   │ - Macro-F1   │   │ - Relevance  │
                                     │ - Missing    │   │ - Tech Acc   │
                                     │   Tools      │   │ - Completeness│
                                     │ - Extra      │   │ - Clarity    │
                                     │   Tools      │   │ - Faithful-  │
                                     │ - Exact Match│   │   ness       │
                                     │ - Per-tool   │   │ - Plausibility│
                                     │   breakdown  │   │ - Correctness│
                                     └──────────────┘   └──────────────┘
```

## run_updated.py Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        run_updated.py                                       │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   Start         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Parse Args      │
                    │ --start-from   │
                    │ --only         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Select Stages   │
                    │ to Run          │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │                              │
              ▼                              ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Stage 2          │           │ Stage 3          │
    │ (SFT + Graph)    │           │ (GRPO RL)        │
    │ - Load Stage 1   │           │ - Load Stage 2   │
    │   checkpoint     │           │   checkpoint     │
    │ - Train prefix   │           │ - Freeze prefix  │
    │   adapter        │           │   adapter        │
    │ - Train Qwen     │           │ - GRPO training  │
    │   + LoRA         │           │   with equal     │
    │ - Save ckpt      │           │   weights        │
    └──────────────────┘           │ - Save ckpt      │
              │                     └──────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Evaluate       │
                    │ - Load Stage 3 │
                    │   checkpoint   │
                    │ - Generate     │
                    │   predictions  │
                    │ - Compute all  │
                    │   metrics      │
                    │ - Save results │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Baselines       │
                    │ (Optional)      │
                    │ - Zero-shot     │
                    │ - 3-shot        │
                    │ - 5-shot        │
                    │ - Paper CNN     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Comparison      │
                    │ Report          │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Summary         │
                    │ - All stages   │
                    │ - Total time    │
                    └─────────────────┘
```

## Key Implementation Details

### Stage 1 (Existing)
- **CNN Encoder**: Paper-inspired multi-kernel CNN for strategy/explanation
- **GNN Encoder**: GINE with 3 layers, typed edges, attention pooling
- **Fusion**: Concatenation of CNN, GNN, and context representations
- **Output**: 512-d graph representation + step/MCP predictions

### Stage 2 (New: stage2_sft_with_graph.py)
- **Stage 1 Checkpoint**: Loaded frozen, provides graph encoding
- **Prefix Adapter**: Trainable, converts 512-d → 8 soft prompt tokens
- **Qwen + LoRA**: Trainable, learns from strategy + explanation + graph tokens
- **Training**: Supervised fine-tuning on training data

### Stage 3 (New: stage3_grpo_with_graph.py)
- **Stage 2 Checkpoint**: Loaded as trainable policy and frozen reference
- **Prefix Adapter**: Frozen (from Stage 2), provides graph conditioning
- **Reward Function**: Equal weights (1.0 each) for step, MCP, explanation
- **Training**: GRPO RL with dual-clip PPO, KL penalty, early stopping

### Evaluation (New: comprehensive_evaluation.py)
- **Step Metrics**: Accuracy, Micro-F1
- **MCP Metrics**: Accuracy, F1, Micro-F1, Macro-F1, Missing Tools, Extra Tools, Exact Match, Per-tool breakdown
- **Explanation Metrics**: LLM Judge (relevance, technical accuracy, completeness, clarity, faithfulness, plausibility, correctness)

### Research-Based Explanation Evaluation
Based on research findings (Hase et al. 2020, ICE Framework, etc.):
- Multi-dimensional rubric (0-3 scale per dimension)
- Gate-based correctness (not thresholded composite)
- Integration with existing llm_judge.py
- Superior to single-score metrics (BLEU, F1)
