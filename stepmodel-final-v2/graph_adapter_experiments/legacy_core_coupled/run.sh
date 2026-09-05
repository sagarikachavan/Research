#!/bin/bash

# Script to run all graph adapter experiments in order
# Based on the workflow in README.md

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Graph Adapter Experiments - Full Run${NC}"
echo -e "${BLUE}========================================${NC}"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Use local checkpoints directory for self-containment
LOCAL_CHECKPOINTS="$SCRIPT_DIR/checkpoints"
mkdir -p "$LOCAL_CHECKPOINTS"

# Export environment variables to override config paths
export CKPT_DIR="$LOCAL_CHECKPOINTS"
export STAGE1_CKPT="$LOCAL_CHECKPOINTS/stage1_gnn_classifier.pt"
export STAGE2_ADAPTER_DIR="$LOCAL_CHECKPOINTS/stage2_qwen_lora"
export STAGE3_ADAPTER_DIR="$LOCAL_CHECKPOINTS/stage3_qwen_grpo"

# Default parameters
CHECKPOINT="${CHECKPOINT:-$LOCAL_CHECKPOINTS/stage2_qwen_lora}"
MAX_ITEMS="${MAX_ITEMS:-150}"
STEPS_STRUCTURE="${STEPS_STRUCTURE:-1500}"
STEPS_MULTITASK="${STEPS_MULTITASK:-2000}"
STRUCTURE_FRAC="${STRUCTURE_FRAC:-0.3}"

# Step 0: Generate input JSON files from CSV data
echo -e "${GREEN}[0/11] Generating input JSON files from CSV data...${NC}"
if [ ! -f "input/train.json" ] || [ ! -f "input/test.json" ]; then
    python data_prep/build_input_json.py --mode rule
    echo -e "${GREEN}✓ Input JSON files generated${NC}"
else
    echo -e "${YELLOW}Input JSON files already exist, skipping generation${NC}"
fi

# Step 0.5: Train Stage 1 GNN if checkpoint doesn't exist
echo -e "${GREEN}[0.5/11] Checking Stage 1 checkpoint...${NC}"
if [ ! -f "$LOCAL_CHECKPOINTS/stage1_gnn_classifier.pt" ]; then
    echo -e "${YELLOW}Stage 1 checkpoint not found, training Stage 1 GNN...${NC}"
    python training/stage1_gnn_train.py
    echo -e "${GREEN}✓ Stage 1 training completed${NC}"
else
    echo -e "${YELLOW}Stage 1 checkpoint exists, skipping training${NC}"
fi

# Step 1: Build structure task datasets
echo -e "${GREEN}[1/11] Building structure task datasets...${NC}"
python graph_adapter_experiments/build_structure_tasks.py
echo -e "${GREEN}✓ Structure tasks built${NC}"

# Step 1.5: Train Stage 2 if checkpoint doesn't exist
echo -e "${GREEN}[1.5/11] Checking Stage 2 checkpoint...${NC}"
if [ ! -d "$LOCAL_CHECKPOINTS/stage2_qwen_lora" ] || [ ! -f "$LOCAL_CHECKPOINTS/stage2_qwen_lora/adapter_config.json" ]; then
    echo -e "${YELLOW}Stage 2 checkpoint not found, training Stage 2...${NC}"
    python training/stage2_sft_qwen.py
    echo -e "${GREEN}✓ Stage 2 training completed${NC}"
else
    echo -e "${YELLOW}Stage 2 checkpoint exists, skipping training${NC}"
fi

# Step 2: Run reliability suite on existing Stage-2 checkpoint
echo -e "${GREEN}[2/11] Running reliability suite on baseline checkpoint...${NC}"
python graph_adapter_experiments/run_reliability_suite.py \
    --checkpoint "$CHECKPOINT" \
    --split held_out \
    --max_items "$MAX_ITEMS" \
    --output stage2_qwen_lora
echo -e "${GREEN}✓ Reliability suite completed${NC}"

# Step 3: Analyze baseline results
echo -e "${GREEN}[3/11] Analyzing baseline results...${NC}"
python graph_adapter_experiments/analyze_results.py \
    --results graph_adapter_experiments/results/raw_results_stage2_qwen_lora.jsonl
echo -e "${GREEN}✓ Baseline analysis completed${NC}"

# Step 4: Train structure-focused adapter
echo -e "${GREEN}[4/11] Training structure-focused adapter...${NC}"
python graph_adapter_experiments/train_structure_adapter.py \
    --init_from "$CHECKPOINT" \
    --out_dir "$LOCAL_CHECKPOINTS/graph_structure" \
    --steps "$STEPS_STRUCTURE"
echo -e "${GREEN}✓ Structure adapter training completed${NC}"

# Step 5: Run reliability suite on structure adapter
echo -e "${GREEN}[5/11] Running reliability suite on structure adapter...${NC}"
python graph_adapter_experiments/run_reliability_suite.py \
    --checkpoint "$LOCAL_CHECKPOINTS/graph_structure/best" \
    --split held_out \
    --max_items "$MAX_ITEMS" \
    --output graph_structure_best
echo -e "${GREEN}✓ Reliability suite on structure adapter completed${NC}"

# Step 6: Analyze structure adapter results (compare with baseline)
echo -e "${GREEN}[6/11] Analyzing structure adapter results (comparison)...${NC}"
python graph_adapter_experiments/analyze_results.py \
    --results graph_adapter_experiments/results/raw_results_graph_structure_best.jsonl \
              graph_adapter_experiments/results/raw_results_stage2_qwen_lora.jsonl
echo -e "${GREEN}✓ Comparative analysis completed${NC}"

# Step 7: Evaluate structure impact on step task
echo -e "${GREEN}[7/11] Evaluating structure impact on step task...${NC}"
python graph_adapter_experiments/evaluate_structure_impact_on_step_task.py
echo -e "${GREEN}✓ Step task impact evaluation completed${NC}"

# Step 8: Train multi-task adapter (optional - can be commented out)
echo -e "${GREEN}[8/11] Training multi-task adapter...${NC}"
python graph_adapter_experiments/train_multitask_adapter.py \
    --init_from "$CHECKPOINT" \
    --out_dir "$LOCAL_CHECKPOINTS/multitask" \
    --steps "$STEPS_MULTITASK" \
    --structure_frac "$STRUCTURE_FRAC"
echo -e "${GREEN}✓ Multi-task adapter training completed${NC}"

# Step 9: Run reliability suite on multi-task adapter
echo -e "${GREEN}[9/11] Running reliability suite on multi-task adapter...${NC}"
python graph_adapter_experiments/run_reliability_suite.py \
    --checkpoint "$LOCAL_CHECKPOINTS/multitask/best" \
    --split held_out \
    --max_items "$MAX_ITEMS" \
    --output multitask_best
echo -e "${GREEN}✓ Reliability suite on multi-task adapter completed${NC}"

# Step 10: Analyze multi-task results
echo -e "${GREEN}[10/11] Analyzing multi-task adapter results...${NC}"
python graph_adapter_experiments/analyze_results.py \
    --results graph_adapter_experiments/results/raw_results_multitask_best.jsonl \
              graph_adapter_experiments/results/raw_results_stage2_qwen_lora.jsonl
echo -e "${GREEN}✓ Multi-task analysis completed${NC}"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Check results in: graph_adapter_experiments/results/"
echo -e "View the main report: graph_adapter_experiments/results/REPORT.md"
