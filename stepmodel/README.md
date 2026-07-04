# Research Step Model - GNN + LLM (Teacher-Forcing Training

This directory contains the implementation for generating PTT (Penetration Testing Tree) graph datasets and training a GNN + LLM model to predict next steps and MCP tasks with explanations.

## Files
1. `generate_graphs.py`: Converts CSV data (training_data.csv and test_data.csv) into graph files (JSON + HTML) using the same structure as `stepmodel/graph_dataset/pentest-dataset`.
2. `graph_to_embeddings.py`: Takes generated graphs, converts them into node/edge embeddings using sentence-transformers, and extracts step pairs from the CSV data.
3. `train_gnn_rl.py`: Training script using PyTorch Geometric for GNN, with teacher-forcing training for LLM, combining graph and text embeddings.
4. `evaluate.py`: Evaluation script for trained model, computes rewards based on prediction similarity.
5. `requirements.txt`: List of dependencies.

## Step-by-Step Usage Guide

### 1. Navigate to the stepmodel directory
```bash
cd /path/to/Research/stepmodel
```

### 2. Install Dependencies
```bash
# It's recommended to use the same Python as your environment (e.g., conda)
python -m pip install -r requirements.txt
```

### 3. Generate Graphs from CSV Data
This step converts the CSV files in `data/` into structured graph files (JSON and HTML) in `processed_data/`.
```bash
python generate_graphs.py
```

### 4. Generate Embeddings and Step Pairs
This step converts the graph files into numerical embeddings using sentence-transformers, and extracts step pairs for training in `embeddings_data/`.
```bash
python graph_to_embeddings.py
```

### 5. Train the Model
This step runs the training script and saves final checkpoint to `checkpoints/`.
```bash
python train_gnn_rl.py
```

### 6. Evaluate the Model (Optional)
Runs the evaluation script on test data, shows sample predictions and computes average reward.
```bash
python evaluate.py
```

## Important Notes
- **Large Files**: Files like `all_processed.json` in `embeddings_data/` and model checkpoints are excluded from git via `.gitignore` to avoid hitting GitHub file size limits. You'll need to regenerate them using `graph_to_embeddings.py` and `train_gnn_rl.py` respectively.
- **Conda/Env Compatibility**: If using a specific environment (e.g., conda), make sure to use that environment's `python` and `pip` commands (e.g., `conda activate your_env` first).
- **Detailed Documentation**: See `DOCUMENTATION.md` for detailed explanations, code walkthroughs, and examples.

