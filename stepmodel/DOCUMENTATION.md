# GNN + LLM + GRPO Penetration Testing Step Prediction Model - Full Documentation

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
   2.1 [Raw CSV Input](#21-raw-csv-input)
   2.2 [Step 1: Generate Graphs (`generate_graphs.py`)](#22-step-1-generate-graphs-generategraphspy)
   2.3 [Step 2: Generate Embeddings (`graph_to_embeddings.py`)](#23-step-2-generate-embeddings-graphtoembeddingspy)
3. [Model Architecture](#3-model-architecture)
   3.1 [GNN Model (`GNNModel`)](#31-gnn-model-gnnmodel)
   3.2 [GNN + LLM Policy (`GNNLLMPolicy`)](#32-gnn--llm-policy-gnnllmpolicy)
   3.3 [Reward Function (`compute_reward`)](#33-reward-function-computereward)
4. [Training Pipeline](#4-training-pipeline)
   4.1 [Phase 1: Supervised Warmup](#41-phase-1-supervised-warmup)
   4.2 [Phase 2: GRPO Reinforcement Learning Fine‑tuning](#42-phase-2-grpo-reinforcement-learning-fine-tuning)
5. [Evaluation](#5-evaluation)
6. [Configuration (`config.json`)](#6-configuration-configjson)
7. [Usage Instructions](#7-usage-instructions)

---

## 1. System Overview
This system trains a model to predict the next step in a penetration testing (pentest) workflow, given:
- A graph representation of the target environment (evolving PTT = Penetration Testing Tree)
- Previous step context (strategy, step taken, result, MCP tool calls, etc.)

The full pipeline goes from raw CSV data → structured graphs → text embeddings → supervised pre-training → RL fine-tuning (GRPO) → evaluation!

---

## 2. Data Pipeline

### 2.1 Raw CSV Input
First, we start with raw CSV files in [stepmodel/data/](file:///Users/sagarikachavan/Documents/Research/stepmodel/data):
- [training_data.csv](file:///Users/sagarikachavan/Documents/Research/stepmodel/data/training_data.csv): Training set pentest logs
- [test_data.csv](file:///Users/sagarikachavan/Documents/Research/stepmodel/data/test_data.csv): Test set pentest logs

Each row has these fields (for a single step of a pentest on a single machine):
- `Machine`: Name of the target machine (for grouping steps per environment)
- `Previous strategy`, `New strategy`: Strategies before/after this step
- `Strategy explanation`, `Step explanation`: Explanations
- `New step`, `Previous step result`: What step was taken and what happened
- `MCP_tasks`: MCP (Model Context Protocol) tool calls used
- `PTT`: Penetration Testing Tree (hierarchical task tree for the pentest so far)

---

### 2.2 Step 1: Generate Graphs (`generate_graphs.py`)
[generate_graphs.py](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py) converts raw CSV data into structured graphs with visualizations!

#### Key Functions in `generate_graphs.py`:
| Function | Purpose |
|----------|---------|
| [load_valid_machines()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L106-L110) | Loads CSV and filters valid machine names (removes fragments/noise) |
| [parse_ptt()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L49-L70) | Parses `PTT` text field into hierarchical task items (number, title, status, payload) |
| [detect_runs()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L113-L125) | Groups consecutive CSV rows into "runs" (full playthroughs of a pentest) by checking if `Previous strategy` matches prior `New strategy` |
| [build_machine_graph()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L137-L247) | **Core function**: Builds the graph for a single machine |
| [to_dict_json()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L250-L278) | Converts graph to structured JSON |
| [to_html()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L281-L336) | Generates interactive vis.js HTML visualization |
| [process_csv()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L343-L358) | Processes entire CSV into graphs |
| [main()](file:///Users/sagarikachavan/Documents/Research/stepmodel/generate_graphs.py#L361-L382) | Entry point: processes both training and test CSVs |

#### Graph Structure (from `build_machine_graph()`):
The graph has **3 node types** and **4 edge types**:

| Node Type | Color | Purpose |
|-----------|-------|---------|
| Agent (Blue/Pink Goal) | `#3A86FF` / `#FF006E` | Cumulative PTT (pentest state) after a new tree item is added |
| Search (Orange) | `#FB5607` | PTT item being worked on, plus MCP tool calls used |
| Track (Green) | `#06D6A0` | Findings payload of that PTT item (what was discovered) |

| Edge Type | Color | Purpose |
|-----------|-------|---------|
| StateTransition (Black) | `#000000` | Agent → Agent (label = PTT item number that was completed) |
| SearchUpdate (Green) | `#06D6A0` | Agent → Search (PTT item being worked on from that state) |
| TrackUpdate (Blue) | `#3A86FF` | Search → Track (what was found by executing that item) |
| Prediction (Purple) | `#8338EC` | Track → Agent (leads to next cumulative state) |

#### Output of `generate_graphs.py`:
Creates files in [stepmodel/processed_data/](file:///Users/sagarikachavan/Documents/Research/stepmodel/processed_data):
- [processed_data/train/](file:///Users/sagarikachavan/Documents/Research/stepmodel/processed_data/train/): Subdirectories per training machine
  - `{machine_name}_graph.json`: Structured graph data
  - `{machine_name}_graph.html`: Interactive vis.js graph visualization
- [processed_data/test/](file:///Users/sagarikachavan/Documents/Research/stepmodel/processed_data/test/): Same for test machines

---

### 2.3 Step 2: Generate Embeddings (`graph_to_embeddings.py`)
[graph_to_embeddings.py](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py) converts graph JSON files and raw CSV data into text embeddings (for model input) plus structured step pairs!

#### Key Functions in `graph_to_embeddings.py`:
| Function | Purpose |
|----------|---------|
| [parse_ptt()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L16-L37) | Same as in generate_graphs.py: parses PTT text |
| [load_graph_json()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L40-L43) | Loads graph from JSON file |
| [get_node_text()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L46-L48) | Combines node type/label/title into single text string for embedding |
| [get_edge_text()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L51-L55) | Combines edge type/label + source/target node info for embedding |
| [embed_texts()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L58-L62) | Uses Sentence‑BERT (`all-MiniLM-L6-v2`) to get text embeddings |
| [process_machine_graph()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L65-L124) | **Core function**: Processes one machine graph into embeddings + step pairs |
| [process_directory()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L127-L171) | Processes all machine graphs in a directory |
| [main()](file:///Users/sagarikachavan/Documents/Research/stepmodel/graph_to_embeddings.py#L173-L203) | Entry point: processes both train and test directories |

#### What `process_machine_graph()` Does:
1. **Loads Step Pairs from CSV**: Takes consecutive CSV rows (i, i+1) and creates `step_pairs` containing previous and next step fields
2. **Embeds Nodes**: For each graph node, creates text (type + label + title), embeds with Sentence‑BERT
3. **Embeds Edges**: For each graph edge, creates text (type + label + src/tgt node titles), embeds with Sentence‑BERT
4. **Saves Structured Data**: Combines nodes (with embeddings), edges (with embeddings), and step pairs into one dict

#### Output of `graph_to_embeddings.py`:
Creates files in [stepmodel/embeddings_data/](file:///Users/sagarikachavan/Documents/Research/stepmodel/embeddings_data):
- [embeddings_data/train/](file:///Users/sagarikachavan/Documents/Research/stepmodel/embeddings_data/train/):
  - `{machine_name}_processed.json`: Individual machine's data
  - `all_processed.json`: All training machines in one file (for model training)
- [embeddings_data/test/](file:///Users/sagarikachavan/Documents/Research/stepmodel/embeddings_data/test/): Same for test machines

---

## 3. Model Architecture
The core model code is in [train_gnn_rl.py](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py)!

---

### 3.1 GNN Model (`GNNModel`)
[GNNModel](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py#L45-L64) is a small Graph Neural Network that encodes the environment graph into a fixed-size vector!

#### Architecture:
- Supports **GCN** (default, from PyTorch Geometric `GCNConv`) OR **GAT** (set `config.model.use_gat = true`)
- Layer breakdown:
  1. GNN Conv Layer 1: Takes node embeddings → hidden dimension
  2. ReLU activation
  3. GNN Conv Layer 2: Hidden dimension → hidden dimension (if GAT, uses multi-head attention with 4 heads)
  4. ReLU activation
  5. Global Mean Pooling: Aggregates all node embeddings into one graph-level embedding (using `global_mean_pool` from PyTorch Geometric)
  6. Linear Layer: Projects to `gnn_out_dim` (default 128)

---

### 3.2 GNN + LLM Policy (`GNNLLMPolicy`)
[GNNLLMPolicy](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py#L67-L151) combines graph embedding and previous step embedding, then injects into the LLM!

#### Architecture Breakdown:
1. **`get_graph_embedding()`**: Converts raw nodes/edges into graph embedding (handles empty nodes gracefully with dummy data)
2. **`project_step_text()`**: Linear MLP (text_emb_dim → 256 → ReLU → gnn_out_dim) that projects previous step's text embedding to same dimension as graph embedding
3. **`forward()`**:
   a. Gets graph embedding via `get_graph_embedding()`
   b. Gets projected step text embedding
   c. Concatenates them → linear layer (2*gnn_out_dim → llm_hidden_size) + ReLU
   d. Returns combined embedding, which replaces the `[GRAPH]` token's embedding in the LLM's input!

#### How LLM Injection Works:
- The prompt starts with `[GRAPH] ` (special token added to tokenizer)
- Tokenize the prompt, get input embeddings from LLM's token embedding layer
- Replace the embedding of the `[GRAPH]` token with the output from `GNNLLMPolicy`
- Pass modified embeddings to LLM for generation/training!

---

### 3.3 Reward Function (`compute_reward`)
[compute_reward](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py#L161-L203) is the custom reward function for GRPO RL training! It scores generated text against the ground truth next step!

#### Reward Breakdown (Total Max 1.0):
| Component | Weight | Purpose |
|-----------|--------|---------|
| Step Token Overlap | 0.2 | Jaccard similarity between tokens of generated step and true step |
| MCP Tasks Token Overlap | 0.2 | Jaccard similarity between tokens of generated MCP tasks and true MCP tasks |
| Step Semantic Similarity | 0.3 | Cosine similarity between Sentence‑BERT embeddings of generated step and true step |
| MCP Tasks Semantic Similarity | 0.3 | Cosine similarity between Sentence‑BERT embeddings of generated MCP tasks and true MCP tasks |

#### `parse_prediction()` Helper:
[parse_prediction](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py#L206-L223) extracts structured components (strategy, step, MCP tasks) from generated text using regex (matches the format used in training data)!

---

## 4. Training Pipeline
Full training is done via [train_gnn_rl.py:main()](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py#L426-L762)! It's a **two‑phase pipeline**:

---

### 4.1 Phase 1: Supervised Warmup
First, we pre-train the model using **teacher forcing** on the expert step pairs! This gives the model a strong baseline before RL!

#### Steps:
1. **Data Loading**: Load `embeddings_data/train/all_processed.json` → split 90%/10% into train/val
2. **Tokenizer/LLM Setup**: Load tokenizer (from `config.model.llm_name`, default `distilgpt2`), add `[GRAPH]` special token, resize LLM's embedding layer
3. **Policy Setup**: Initialize `GNNLLMPolicy`
4. **Training Loop**:
   - For each sample, construct `full_text = prompt_text + target_text`
   - Mask all prompt tokens (set labels to `‑100` so they don't contribute to loss)
   - Inject `GNNLLMPolicy` output into `[GRAPH]` token's embedding
   - Compute cross‑entropy loss on target tokens
   - Backpropagate, optimize both policy and LLM parameters
   - Use linear scheduler with warmup
   - TensorBoard logging (supervised loss, val reward)
5. **Checkpointing**: Saves best checkpoint (best val reward) and per‑epoch checkpoints
6. **Early Stopping**: If val reward doesn't improve for `patience` epochs (default 5), move on to Phase 2

---

### 4.2 Phase 2: GRPO Reinforcement Learning Fine‑tuning
Next, we fine‑tune using **GRPO (Group Relative Policy Optimization)**, a variant of PPO that uses group‑relative advantages (lower variance)!

#### Key GRPO Functions:
- [generate_samples_with_policy()](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py#L226-L301): Generates `num_generations_per_sample` (default 4) rollouts per sample, computes log‑probs of generated sequences
- [compute_grpo_loss()](file:///Users/sagarikachavan/Documents/Research/stepmodel/train_gnn_rl.py#L304-L368): Computes GRPO loss with clipped policy ratio (clips ratio between 1‑clip_eps and 1+clip_eps, default clip_eps 0.2)

#### GRPO Training Steps:
1. **For each batch**:
   a. For each sample in batch, generate N rollouts
   b. Compute reward for each rollout using `compute_reward()`
   c. Compute **group advantages**: `(reward_i - mean(rewards)) / (std(rewards) + eps)` (normalizes rewards relative to group of rollouts)
   d. For each rollout, compute new log‑prob (with current policy), ratio (`exp(new_log_prob - old_log_prob)`), clipped ratio
   e. Compute policy loss: `-torch.min(ratio * advantage, clipped_ratio * advantage)`
   f. Backpropagate, optimize, step scheduler
   g. TensorBoard logging (GRPO loss, avg reward, val reward)
2. **Checkpointing/Early Stopping**: Same as Phase 1
3. **Final Test Evaluation**: Load best checkpoint, evaluate on test set!

---

## 5. Evaluation
Standalone evaluation is done via [evaluate.py](file:///Users/sagarikachavan/Documents/Research/stepmodel/evaluate.py)!

#### Evaluation Steps:
1. Load config, Sentence‑BERT model, test data from `embeddings_data/test/all_processed.json`
2. Load tokenizer and base LLM from `config.model.llm_name` (default `distilgpt2`)
3. Add the `[GRAPH]` special token and resize the LLM embedding layer to match the tokenizer
4. Load policy and LLM weights from `best_checkpoint.pt`
5. For each test sample:
   a. Compute policy output from graph + previous step
   b. Inject the policy output into the `[GRAPH]` token's embedding
   c. Tokenize and truncate the prompt to the model's maximum context length
   d. Generate text from the LLM
   e. Compute reward with `compute_reward()`
6. Log average test reward!

---

## 6. Configuration (`config.json`)
All hyperparameters and paths are in [config.json](file:///Users/sagarikachavan/Documents/Research/stepmodel/config.json)!

| Section | Key | Default Value | Purpose |
|---------|-----|---------------|---------|
| `model` | `llm_name` | `distilgpt2` | Name of LLM (from Hugging Face Transformers) |
| `model` | `text_embedding_model` | `all-MiniLM-L6-v2` | Name of Sentence‑BERT embedding model |
| `model` | `gnn_hidden_dim` | `256` | Hidden dimension for GNN layers |
| `model` | `gnn_out_dim` | `128` | Output dimension of GNN (and step projection) |
| `model` | `use_gat` | `false` | If true, use GATConv instead of GCNConv for GNN |
| `training` | `num_supervised_epochs` | `3` | Number of Phase 1 epochs |
| `training` | `num_grpo_epochs` | `7` | Number of Phase 2 epochs |
| `training` | `batch_size` | `2` | Batch size (for both phases, though Phase 1 processes samples individually) |
| `training` | `learning_rate` | `5e-5` | Learning rate for AdamW optimizer |
| `training` | `weight_decay` | `0.01` | Weight decay for AdamW |
| `training` | `max_grad_norm` | `1.0` | Max gradient norm for clipping |
| `training` | `num_warmup_steps` | `100` | Number of warmup steps for linear scheduler |
| `training` | `num_generations_per_sample` | `4` | Rollouts per sample for GRPO |
| `training` | `clip_eps` | `0.2` | Clip epsilon for PPO/GRPO |
| `training` | `generate_max_new_tokens` | `256` | Max new tokens to generate per rollout |
| `training` | `generate_temperature` | `0.9` | Sampling temperature for generation |
| `training` | `generate_top_p` | `0.95` | Top‑p (nucleus) sampling threshold |
| `training` | `validation_split` | `0.1` | Fraction of training data to use for validation |
| `training` | `patience` | `5` | Early stopping patience (epochs without improvement) |
| `paths` | `data_dir` | `data` | Path to raw CSV data |
| `paths` | `embeddings_dir` | `embeddings_data` | Path to processed embeddings data |
| `paths` | `output_dir` | `checkpoints` | Path to save model checkpoints |
| `paths` | `log_dir` | `logs` | Path to save TensorBoard logs |

---

## 7. Usage Instructions
Follow these steps in order!

### Step 0: Install Dependencies
First, install required packages (run from `stepmodel/` directory):
```bash
pip install torch torch-geometric sentence-transformers transformers pandas numpy tensorboard
```

### Step 1: Generate Graphs (if needed)
If you don't have `processed_data/` yet, run:
```bash
python generate_graphs.py
```
This reads from `data/` and writes to `processed_data/`.

### Step 2: Generate Embeddings (if needed)
If you don't have `embeddings_data/` yet, run:
```bash
python graph_to_embeddings.py
```
This reads from `processed_data/` and writes to `embeddings_data/`.

### Step 3: Train the Model
Run full training pipeline (Phase 1 + Phase 2):
```bash
python train_gnn_rl.py
```
This saves checkpoints to `checkpoints/` and TensorBoard logs to `logs/`.

### Step 4: Evaluate the Model
Run standalone evaluation using best checkpoint:
```bash
python evaluate.py
```

### Step 5: View TensorBoard Logs (optional)
To see training curves:
```bash
tensorboard --logdir logs
```
Then open http://localhost:6006 in your browser!

---
