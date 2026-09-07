#!/bin/bash
#SBATCH --job-name=graph_adapter
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
 
export STANDALONE_INPUT_TRAIN_JSON=$HOME/Research/stepmodel-final-v2/input/train.json
export STANDALONE_INPUT_TEST_JSON=$HOME/Research/stepmodel-final-v2/input/test.json
export STANDALONE_LLM_MODEL_NAME=${LLM_MODEL_NAME:-"Qwen/Qwen2.5-1.5B-Instruct"}
 
mkdir -p logs
cd $HOME/Research/stepmodel-final-v2/graph_adapter_experiments
 
echo "=== Starting improved training (5000 steps + LoRA + 0.7 consistency) ==="
echo "Date: $(date)"
 
if [ ! -f "standalone_tasks/train.jsonl" ]; then
    python build_probe_tasks.py
fi
 
python train_adapter.py \
    --steps 5000 \
    --eval_every 500 \
    --lr 2e-4 \
    --grad_accum 4 \
    --consistency_frac 0.7 \
    --use_lora \
    --model_name "$STANDALONE_LLM_MODEL_NAME" \
    --out_dir standalone_checkpoints/cluster_run_improved
 
if [ -f "standalone_checkpoints/cluster_run_improved/best/meta.json" ]; then
    mkdir -p standalone_results
    python eval_right_vs_wrong_graph.py \
        --checkpoint standalone_checkpoints/cluster_run_improved/best \
        --split held_out \
        --max_items 200 \
        --max_consistency_pairs 100 \
        --model_name "$STANDALONE_LLM_MODEL_NAME" \
        --verbose > standalone_results/verbose_evaluation_report_improved.txt 2>&1
    echo "Results saved to standalone_results/verbose_evaluation_report_improved.txt"
fi
 
echo "=== Complete ==="
echo "Date: $(date)"