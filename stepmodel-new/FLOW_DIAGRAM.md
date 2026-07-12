# stepmodel-new System Flow Diagram

## Complete Training and Inference Flow

```mermaid
graph TD
    A[Raw Input Data] --> B[Data Quality Filtering]
    B --> C[Filtered Dataset]
    C --> D[Text Embedding Generation]
    C --> E[Graph Embedding Generation]
    C --> F[Label Processing]
    
    D --> G[Text Embeddings]
    E --> H[Graph Embeddings]
    F --> I[Processed Labels]
    
    G --> J[GNN-LLM Policy Network]
    H --> J
    I --> J
    
    J --> K[Phase 0: Pretraining]
    K --> L[Phase 1: Supervised Training]
    L --> M[Curriculum Learning Stages]
    M --> N[Phase 2: GRPO RL Fine-tuning]
    N --> O[Trained Model]
    
    O --> P[Evaluation]
    P --> Q[Performance Metrics]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
    style O fill:#fbf,stroke:#333,stroke-width:2px
    style Q fill:#f90,stroke:#333,stroke-width:2px
```

## Detailed Component Flow

### 1. Data Processing Pipeline

```mermaid
graph LR
    A[Raw Graphs] --> B[Graph Parser]
    C[Raw Text] --> D[Text Processor]
    E[Raw Labels] --> F[Label Validator]
    
    B --> G[Graph Structure]
    D --> H[Text Content]
    F --> I[Validated Labels]
    
    G --> J[Quality Assessment]
    H --> J
    I --> J
    
    J --> K{Quality Score ≥ 0.7?}
    K -->|Yes| L[Filtered Dataset]
    K -->|No| M[Discard]
    
    style J fill:#ff9,stroke:#333,stroke-width:2px
    style K fill:#9f9,stroke:#333,stroke-width:2px
    style L fill:#9ff,stroke:#333,stroke-width:2px
    style M fill:#f99,stroke:#333,stroke-width:2px
```

### 2. Neural Network Architecture Flow

```mermaid
graph TD
    A[Input Graph] --> B[GNN Layer GCN]
    B --> C[Graph Embeddings]
    C --> D[Projector MLP]
    D --> E[GRAPH Tokens]
    
    F[Input Text] --> G[Text Embeddings]
    G --> H[LLM Input Sequence]
    E --> H
    
    H --> I[Qwen3-14B LLM]
    I --> J[LoRA Adaptation]
    J --> K[Output Layer]
    
    K --> L[Step Logits]
    K --> M[MCP Logits]
    
    L --> N[Step Prediction]
    M --> O[MCP Prediction]
    
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bfb,stroke:#333,stroke-width:2px
    style J fill:#fbf,stroke:#333,stroke-width:2px
    style N fill:#f90,stroke:#333,stroke-width:2px
    style O fill:#f90,stroke:#333,stroke-width:2px
```

### 3. Training Pipeline Flow

```mermaid
graph TD
    A[Start Training] --> B{Phase 0 Checkpoint Exists?}
    B -->|Yes| C[Load Phase 0 Weights]
    B -->|No| D[Skip Phase 0]
    
    C --> E[Phase 1: Supervised Training]
    D --> E
    
    E --> F[Initialize Optimizer]
    F --> G[Curriculum Stage 0]
    G --> H[High Quality Samples Only]
    H --> I[Train Epoch]
    I --> J[Validate Performance]
    J --> K{Stage Complete?}
    K -->|No| I
    K -->|Yes| L[Curriculum Stage 1]
    
    L --> M[Medium Quality Samples]
    M --> N[Train Epoch]
    N --> O[Validate Performance]
    O --> P{Stage Complete?}
    P -->|No| N
    P -->|Yes| Q[Curriculum Stage 2]
    
    Q --> R[All Filtered Samples]
    R --> S[Train Epoch]
    S --> T[Validate Performance]
    T --> U{Stage Complete?}
    U -->|No| S
    U -->|Yes| V[Phase 2: GRPO RL]
    
    V --> W[Generate Rollouts]
    W --> X[Compute Rewards]
    X --> Y[Calculate Advantages]
    Y --> Z[Compute GRPO Loss]
    Z --> AA[Update Policy]
    AA --> AB{Epochs Complete?}
    AB -->|No| W
    AB -->|Yes| AC[Save Best Model]
    
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style V fill:#bfb,stroke:#333,stroke-width:2px
    style AC fill:#fbf,stroke:#333,stroke-width:2px
```

### 4. Curriculum Learning Flow

```mermaid
graph TD
    A[Full Dataset] --> B[Quality Scoring]
    B --> C[Stage 0: Threshold 0.9]
    C --> D[High Quality Subset]
    D --> E[Train Model]
    E --> F[Evaluate Performance]
    
    F --> G[Stage 1: Threshold 0.8]
    G --> H[Medium Quality Subset]
    H --> I[Continue Training]
    I --> J[Evaluate Performance]
    
    J --> K[Stage 2: Threshold 0.7]
    K --> L[Full Filtered Dataset]
    L --> M[Final Training]
    M --> N[Final Evaluation]
    
    style C fill:#f90,stroke:#333,stroke-width:2px
    style G fill:#9f0,stroke:#333,stroke-width:2px
    style K fill:#09f,stroke:#333,stroke-width:2px
    style N fill:#f0f,stroke:#333,stroke-width:2px
```

### 5. GRPO Reinforcement Learning Flow

```mermaid
graph TD
    A[Current Policy] --> B[Generate N Rollouts]
    B --> C[Sample 1]
    B --> D[Sample 2]
    B --> E[Sample N]
    
    C --> F[Compute Reward]
    D --> G[Compute Reward]
    E --> H[Compute Reward]
    
    F --> I[Reward Vector]
    G --> I
    H --> I
    
    I --> J[Compute Advantages]
    J --> K[Compute New Log Probs]
    K --> L[Compute Ratio]
    L --> M[Apply Clipping]
    M --> N[Compute GRPO Loss]
    
    N --> O[Add Auxiliary Supervised Loss]
    O --> P[Backpropagate]
    P --> Q[Update Policy]
    
    Q --> R[New Policy]
    R --> B
    
    style I fill:#ff9,stroke:#333,stroke-width:2px
    style N fill:#9f9,stroke:#333,stroke-width:2px
    style Q fill:#9ff,stroke:#333,stroke-width:2px
```

### 6. Evaluation and Metrics Flow

```mermaid
graph TD
    A[Trained Model] --> B[Load Test Dataset]
    B --> C[Process Each Sample]
    
    C --> D[Generate Predictions]
    D --> E[Step Predictions]
    D --> F[MCP Predictions]
    
    E --> G[Compare with Ground Truth]
    F --> H[Compare with Ground Truth]
    
    G --> I[Compute Step Accuracy]
    G --> J[Compute Step F1]
    
    H --> K[Compute MCP Accuracy]
    H --> L[Compute MCP F1]
    
    I --> M[Combined Metrics]
    J --> M
    K --> M
    L --> M
    
    M --> N[Selection Score]
    N --> O[Final Performance Report]
    
    style M fill:#f90,stroke:#333,stroke-width:2px
    style N fill:#f0f,stroke:#333,stroke-width:2px
    style O fill:#0f0,stroke:#333,stroke-width:2px
```

### 7. Memory Management Flow

```mermaid
graph TD
    A[Training Iteration] --> B[Load Batch]
    B --> C[Forward Pass]
    C --> D[Compute Loss]
    D --> E[Backward Pass]
    
    E --> F{Memory Check}
    F -->|High Memory| G[Enable Gradient Checkpointing]
    F -->|Normal Memory| H[Standard Training]
    
    G --> I[Clear Intermediate Tensors]
    H --> I
    
    I --> J[torch.cuda.empty_cache]
    J --> K[Next Iteration]
    
    style F fill:#ff9,stroke:#333,stroke-width:2px
    style G fill:#f99,stroke:#333,stroke-width:2px
    style J fill:#9f9,stroke:#333,stroke-width:2px
```

## Data Flow Summary

### Input → Output Transformation

1. **Raw Data**: Graph topology + text descriptions + labels
2. **Quality Filter**: Multi-dimensional quality assessment → filtered dataset
3. **Embeddings**: Text → sentence embeddings, Graph → node/edge embeddings
4. **Model Processing**: GNN → graph tokens → LLM → predictions
5. **Training**: Supervised → curriculum → GRPO → trained model
6. **Evaluation**: Predictions → metrics → performance report

### Key Decision Points

1. **Quality Threshold**: Sample inclusion based on quality score ≥ 0.7
2. **Curriculum Stages**: Progressive quality thresholds (0.9 → 0.8 → 0.7)
3. **Training Phases**: Pretraining → Supervised → GRPO
4. **Memory Management**: Dynamic gradient checkpointing based on memory usage
5. **Early Stopping**: Patience-based stopping on validation metrics

## Performance Optimization Flow

```mermaid
graph TD
    A[Start Training] --> B[Monitor GPU Memory]
    B --> C{Memory > 80%?}
    C -->|Yes| D[Enable Gradient Checkpointing]
    C -->|No| E[Standard Training]
    
    D --> F[Reduce Batch Size]
    F --> G[Clear Cache]
    G --> H[Continue Training]
    
    E --> I{Validation Improving?}
    I -->|Yes| H
    I -->|No| J[Adjust Learning Rate]
    
    J --> K{Still Not Improving?}
    K -->|Yes| L[Early Stopping]
    K -->|No| H
    
    H --> M[Save Checkpoint]
    M --> N{Best Model?}
    N -->|Yes| O[Update Best Model]
    N -->|No| P[Continue]
    
    O --> P
    P --> B
    
    style C fill:#ff9,stroke:#333,stroke-width:2px
    style D fill:#f99,stroke:#333,stroke-width:2px
    style L fill:#f90,stroke:#333,stroke-width:2px
    style O fill:#9f9,stroke:#333,stroke-width:2px
```

## System Integration Flow

```mermaid
graph TD
    A[User Input] --> B[Config File]
    B --> C[Training Script]
    C --> D[Data Pipeline]
    
    D --> E[Quality Filter]
    E --> F[Neural Network]
    F --> G[Training Pipeline]
    
    G --> H[Curriculum Learning]
    H --> I[GRPO RL]
    I --> J[Model Checkpoints]
    
    J --> K[Evaluation Script]
    K --> L[Performance Metrics]
    L --> M[Documentation]
    
    M --> N[System Improvements]
    N --> B
    
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bfb,stroke:#333,stroke-width:2px
    style I fill:#fbf,stroke:#333,stroke-width:2px
    style L fill:#f90,stroke:#333,stroke-width:2px
```

## File Structure and Data Flow

```
stepmodel-new/
├── config.json                 # Configuration parameters
├── filter_data_quality.py      # Data quality filtering
├── train_gnn_rl.py            # Main training script
├── evaluate.py                # Evaluation script
├── SYSTEM_DOCUMENTATION.md    # System documentation
├── FLOW_DIAGRAM.md           # This file
├── data/                     # Raw data directory
├── embeddings_data/          # Processed embeddings
│   ├── train/               # Training data
│   ├── val/                 # Validation data
│   └── test/                # Test data
└── checkpoints/             # Model checkpoints
```

## Execution Flow

### Training Execution

1. **Load Configuration**: Read `config.json` parameters
2. **Data Preparation**: Apply quality filtering to dataset
3. **Model Initialization**: Load LLM, initialize GNN and projector
4. **Phase 0** (Optional): Self-supervised pretraining
5. **Phase 1**: Supervised training with curriculum learning
6. **Phase 2**: GRPO reinforcement learning fine-tuning
7. **Checkpoint Saving**: Save best model based on validation metrics
8. **Final Evaluation**: Test on held-out test set

### Inference Execution

1. **Load Model**: Load trained checkpoint
2. **Process Input**: Generate embeddings for input graph and text
3. **Model Forward Pass**: Generate step and MCP predictions
4. **Post-processing**: Apply thresholding and formatting
5. **Output**: Return predicted steps and MCP tools

---

**Document Version**: 1.0  
**Last Updated**: 2025-07-12  
**System**: stepmodel-new  
**Purpose**: System Flow Visualization
