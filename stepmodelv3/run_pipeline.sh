#!/bin/bash
# Complete pipeline script for stepmodelv3
# Runs: graph generation -> input JSON building -> GRPO RL training -> evaluation

set -e  # Exit on error

echo "=========================================="
echo "StepModel v3 Complete Pipeline"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 0: Install requirements
echo ""
echo "[Step 0/6] Installing Python requirements..."
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
echo "[Step 1/6] Generating graphs from CSV data..."
echo "-------------------------------------------"
python generate_graphs_finalv.py
if [ $? -eq 0 ]; then
    echo "✓ Graph generation completed successfully"
else
    echo "✗ Graph generation failed"
    exit 1
fi

# Step 2: Build input JSON from generated graphs
echo ""
echo "[Step 2/6] Building input JSON from graphs..."
echo "-------------------------------------------"
python build_input_json_finalv.py
if [ $? -eq 0 ]; then
    echo "✓ Input JSON building completed successfully"
else
    echo "✗ Input JSON building failed"
    exit 1
fi

# Step 3: Run supervised training (required before GRPO)
echo ""
echo "[Step 3/6] Running supervised training..."
echo "-------------------------------------------"
python train_SFT.py
if [ $? -eq 0 ]; then
    echo "✓ Supervised training completed successfully"
else
    echo "✗ Supervised training failed"
    exit 1
fi

# Step 4: Run GRPO RL training
echo ""
echo "[Step 4/6] Running GRPO RL training..."
echo "-------------------------------------------"
python train_grpo.py
if [ $? -eq 0 ]; then
    echo "✓ GRPO RL training completed successfully"
else
    echo "✗ GRPO RL training failed"
    exit 1
fi

# Step 5: Evaluate supervised model
echo ""
echo "[Step 5/6] Evaluating supervised model..."
echo "-------------------------------------------"
python evaluate.py --model-type supervised --save-explanations out_supervised.csv
if [ $? -eq 0 ]; then
    echo "✓ Supervised model evaluation completed successfully"
else
    echo "✗ Supervised model evaluation failed"
    exit 1
fi

# Step 6: Evaluate GRPO model
echo ""
echo "[Step 6/6] Evaluating GRPO model..."
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
echo "  - Supervised model: /tmp/stage1_supervised/"
echo "  - GRPO model: /tmp/stage2_grpo_rl/"
echo "  - Supervised evaluation: out_supervised.csv"
echo "  - GRPO evaluation: out_grpo.csv"
