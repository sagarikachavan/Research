"""
Configuration for stepmodelv4
Combines stepmodelv3 graph building with stepmodelv2 GNN embeddings
"""
import os

# Graph building (from stepmodelv3)
INPUT_CSV = "data/ptt_data.csv"
PROCESSED_DATA_DIR = "processed_data"
INPUT_TRAIN_JSON = "input/train.json"
INPUT_TEST_JSON = "input/test.json"

# GNN training (adapted from stepmodelv2)
GNN_HIDDEN = 256
GNN_LAYERS = 3
GNN_OUT_DIM = 256
FUSION_HIDDEN = 512
TEXT_EMB_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2
GNN_CKPT = "/tmp/stage1_gnn_encoder.pt"
GNN_LR = 1e-3
GNN_EPOCHS = 10
GNN_BATCH_SIZE = 8

# LLM training (from stepmodelv3)
QWEN_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
SUPERVISED_ADAPTER_DIR = "/tmp/stage1_supervised"
GRPO_ADAPTER_DIR = "/tmp/stage2_grpo_rl"
MAX_PROMPT_TOKENS = 1200
MAX_NEW_TOKENS = 350

# Graph Prefix Adapter
PREFIX_TOKENS = 8  # Number of soft prompt tokens for graph embedding
ADAPTER_HIDDEN = 512
ADAPTER_CKPT = "/tmp/graph_adapter.pt"

# Training hyperparameters
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 3e-6
NUM_EPOCHS = 5
LORA_R = 64
LORA_ALPHA = 128
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GRPO hyperparameters
GROUP_SIZE = 8
GRPO_LR = 3e-6
GRPO_STEPS = 5000
KL_COEF = 0.01
GRPO_GRAD_ACCUM = 16

RANDOM_SEED = 42
