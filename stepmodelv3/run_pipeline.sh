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

# Step 1: Generate graphs from CSV data
echo ""
echo "[Step 1/4] Generating graphs from CSV data..."
echo "-------------------------------------------"
python generate_graphs.py
if [ $? -eq 0 ]; then
    echo "✓ Graph generation completed successfully"
else
    echo "✗ Graph generation failed"
    exit 1
fi

# Step 2: Build input JSON from generated graphs
echo ""
echo "[Step 2/4] Building input JSON from graphs..."
echo "-------------------------------------------"
python build_input_json.py
if [ $? -eq 0 ]; then
    echo "✓ Input JSON building completed successfully"
else
    echo "✗ Input JSON building failed"
    exit 1
fi

# Step 3: Run GRPO RL training
echo ""
echo "[Step 3/4] Running GRPO RL training..."
echo "-------------------------------------------"
python stage1_grpo_rl.py
if [ $? -eq 0 ]; then
    echo "✓ GRPO RL training completed successfully"
else
    echo "✗ GRPO RL training failed"
    exit 1
fi

# Step 4: Evaluate trained model
echo ""
echo "[Step 4/4] Evaluating trained GRPO model..."
echo "-------------------------------------------"
python evaluate.py
if [ $? -eq 0 ]; then
    echo "✓ Model evaluation completed successfully"
else
    echo "✗ Model evaluation failed"
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
echo "  - Trained model: checkpoints/stage1_grpo_rl/"
echo "  - Evaluation results: printed to console"
