# Research Step Model - GNN + RL + LLM

This directory contains the implementation for generating PTT (Penetration Testing Tree) graph datasets and training a GNN + RL (GRPO) + LLM model.

## Files
1. `generate_graphs.py`: Converts CSV data (training_data.csv and test_data.csv) into graph files (JSON + HTML) using the same structure as `stepmodel/graph_dataset/pentest-dataset`.
2. `graph_to_embeddings.py`: Takes the generated graphs and converts them into node/edge embeddings using sentence-transformers. Also extracts step pairs from the CSV data.
3. `train_gnn_rl.py`: Training script skeleton for GNN + RL (GRPO) + LLM.
4. `evaluate.py`: Evaluation script skeleton.
5. `requirements.txt`: List of dependencies.

## Usage

### 1. Generate Graphs
```bash
python generate_graphs.py
```

### 2. Generate Embeddings
```bash
python graph_to_embeddings.py
```

### 3. (Optional) Install Dependencies
```bash
pip install -r requirements.txt
```
