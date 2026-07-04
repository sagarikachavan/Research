# Research Step Model - GNN + RL + LLM

This directory contains the implementation for generating PTT (Penetration Testing Tree) graph datasets and training a GNN + RL (GRPO) + LLM model.

## Files
1. `generate_graphs.py`: Converts CSV data (training_data.csv and test_data.csv) into graph files (JSON + HTML) using the same structure as `stepmodel/graph_dataset/pentest-dataset`.
2. `graph_to_embeddings.py`: Takes the generated graphs and converts them into node/edge embeddings using sentence-transformers. Also extracts step pairs from the CSV data.
3. `train_gnn_rl.py`: Training script skeleton for GNN + RL (GRPO) + LLM.
4. `evaluate.py`: Evaluation script skeleton.
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
This step runs the training script and saves an initial checkpoint to `checkpoints/`.
```bash
python train_gnn_rl.py
```

### 6. Evaluate the Model (Optional)
Runs the evaluation script on test data.
```bash
python evaluate.py
```

## Important Notes
- **Large Files**: Files like `all_processed.json` in `embeddings_data/` and model checkpoints are excluded from git via `.gitignore` to avoid hitting GitHub file size limits. You'll need to regenerate them using `graph_to_embeddings.py` and `train_gnn_rl.py` respectively.
- **Conda/Env Compatibility**: If using a specific environment (e.g., conda), make sure to use that environment's `python` and `pip` commands (e.g., `conda activate your_env` first).
- **Detailed Documentation**: See `DOCUMENTATION.md` for detailed explanations, code walkthroughs, and examples.

