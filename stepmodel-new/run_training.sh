#!/bin/bash

# Training script for stepmodel-new with proper logging
# Usage: ./run_training.sh [config_file]

set -e  # Exit on error

CONFIG_FILE="${1:-config.json}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "StepModel Training Script"
echo "=========================================="
echo "Config: ${CONFIG_FILE}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="
echo ""

# Check if config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file '${CONFIG_FILE}' not found!"
    exit 1
fi

# Check if Python script exists
if [ ! -f "train_gnn_rl.py" ]; then
    echo "ERROR: train_gnn_rl.py not found in current directory!"
    exit 1
fi

# Run training with both console output and logging
echo "Starting training..."
echo "Press Ctrl+C to interrupt"
echo ""

python train_gnn_rl.py --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code ${TRAINING_EXIT_CODE}"
fi
echo "Log saved to: ${LOG_FILE}"
echo "=========================================="

exit ${TRAINING_EXIT_CODE}
