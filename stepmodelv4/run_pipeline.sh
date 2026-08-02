#!/bin/bash
# Complete pipeline script for stepmodelv4
# Runs: graph generation -> input JSON building -> GNN training -> SFT training -> GRPO training -> evaluation

set -e  # Exit on error

echo "=========================================="
echo "StepModel v4 Complete Pipeline"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 0: Install requirements
echo ""
echo "[Step 0/7] Installing Python requirements..."
echo "-------------------------------------------"
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Requirements installed successfully"
else
    echo "✗ Requirements installation failed"
    exit 1
fi

# Step 1: Generate graphs from CSV data
echo ""
echo "[Step 1/7] Generating graphs from CSV data..."
echo "-------------------------------------------"
python ptt_parser_finalv.py
if [ $? -eq 0 ]; then
    echo "✓ Graph generation completed successfully"
else
    echo "✗ Graph generation failed"
    exit 1
fi

# Step 2: Build input JSON from graphs
echo ""
echo "[Step 2/7] Building input JSON from graphs..."
echo "-------------------------------------------"
python build_input_json_finalv.py
if [ $? -eq 0 ]; then
    echo "✓ Input JSON building completed successfully"
else
    echo "✗ Input JSON building failed"
    exit 1
fi

# Step 3: Train GNN encoder
echo ""
echo "[Step 3/7] Training GNN encoder..."
echo "-------------------------------------------"
python stage1_gnn_train.py
if [ $? -eq 0 ]; then
    echo "✓ GNN encoder training completed successfully"
else
    echo "✗ GNN encoder training failed"
    exit 1
fi

# Step 4: Train supervised model with graph embeddings
echo ""
echo "[Step 4/7] Training supervised model with graph embeddings..."
echo "-------------------------------------------"
python train_SFT.py
if [ $? -eq 0 ]; then
    echo "✓ Supervised training completed successfully"
else
    echo "✗ Supervised training failed"
    exit 1
fi

# Step 5: Train GRPO model with graph embeddings
echo ""
echo "[Step 5/7] Training GRPO model with graph embeddings..."
echo "-------------------------------------------"
python train_grpo.py
if [ $? -eq 0 ]; then
    echo "✓ GRPO training completed successfully"
else
    echo "✗ GRPO training failed"
    exit 1
fi

# Step 6: Evaluate supervised model
echo ""
echo "[Step 6/7] Evaluating supervised model..."
echo "-------------------------------------------"
python evaluate.py --model-type supervised --save-explanations out_supervised.csv
if [ $? -eq 0 ]; then
    echo "✓ Supervised model evaluation completed successfully"
else
    echo "✗ Supervised model evaluation failed"
    exit 1
fi

# Step 7: Evaluate GRPO model
echo ""
echo "[Step 7/7] Evaluating GRPO model..."
echo "-------------------------------------------"
python evaluate.py --model-type grpo --save-explanations out_grpo.csv
if [ $? -eq 0 ]; then
    echo "✓ GRPO model evaluation completed successfully"
else
    echo "✗ GRPO model evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Complete pipeline finished successfully"
echo "=========================================="
echo ""
echo "Generated outputs:"
echo "  - Graph data: processed_data/"
echo "  - Input JSON: input/"
echo "  - GNN encoder: /tmp/stage1_gnn_encoder.pt"
echo "  - Supervised model: /tmp/stage1_supervised/"
echo "  - GRPO model: /tmp/stage2_grpo_rl/"
echo "  - Supervised evaluation: out_supervised.csv"
echo "  - GRPO evaluation: out_grpo.csv"
