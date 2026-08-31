"""
Central configuration: label spaces, paths, and hyperparameters.
"""
import os

# ----------------------------------------------------------------------------
# Label spaces
# ----------------------------------------------------------------------------
STEP_LABELS = [
    "Do a google search for more information",
    "Enumerate further on the X service to find software versions, hidden directories and file.",
    "Explore the suspicious files, commands and create a summary of the findings.",
    "Further Enumerate the website. - hidden directories, links and software",
    "Enumerate the domain",
    "Exploit the selected exploitations",
    "Analyze the outcomes of the previous step and find an attack path",
    "Ask for human assistant",
    "Explore the source code for vulnerabilities.",
    "End task and ask permission to generate the report",
]

MCP_LABELS = [
    "Nmap",
    "Metasploit",
    "Netcat",
    "Dirbuster",
    "SQLmap",
    "Smb client",
    "hydra",
    "John-the-ripper",
    "Google search",
    "Interactive CLI",
    "Web page interaction",
]

STEP2IDX = {l: i for i, l in enumerate(STEP_LABELS)}
IDX2STEP = {i: l for i, l in enumerate(STEP_LABELS)}
MCP2IDX = {l: i for i, l in enumerate(MCP_LABELS)}
IDX2MCP = {i: l for i, l in enumerate(MCP_LABELS)}

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.environ.get("PIPELINE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_CSV = os.environ.get("TRAIN_CSV", os.path.join(ROOT, "data", "training_data.csv"))
TEST_CSV = os.environ.get("TEST_CSV", os.path.join(ROOT, "data", "test_data.csv"))

# Primary input files: joined CSV + graph JSON produced by build_input_json.py.
# All three training stages (Stage 1, 2, 3) and evaluation read from these.
INPUT_TRAIN_JSON = os.environ.get(
    "INPUT_TRAIN_JSON", os.path.join(ROOT, "input", "train.json")
)
INPUT_TEST_JSON = os.environ.get(
    "INPUT_TEST_JSON", os.path.join(ROOT, "input", "test.json")
)

# Directory holding per-machine / per-row pre-built graph objects.
# These are only used as a fallback when input/train.json is not available.
GRAPH_DIR_TRAIN = os.environ.get(
    "GRAPH_DIR_TRAIN", os.path.join(ROOT, "processed_data", "train")
)
GRAPH_DIR_TEST = os.environ.get(
    "GRAPH_DIR_TEST", os.path.join(ROOT, "processed_data", "test")
)

CKPT_DIR = os.environ.get(
    "CKPT_DIR", os.path.join(ROOT, "checkpoints")
)
os.makedirs(CKPT_DIR, exist_ok=True)

STAGE1_CKPT = os.environ.get(
    "STAGE1_CKPT", os.path.join(CKPT_DIR, "stage1_gnn_classifier.pt")
)
STAGE2_ADAPTER_DIR = os.environ.get(
    "STAGE2_ADAPTER_DIR", os.path.join(CKPT_DIR, "stage2_qwen_lora")
)
STAGE3_ADAPTER_DIR = os.environ.get(
    "STAGE3_ADAPTER_DIR", os.path.join(CKPT_DIR, "stage3_qwen_grpo")
)

# ----------------------------------------------------------------------------
# Model / training hyperparameters
# ----------------------------------------------------------------------------
TEXT_ENCODER_NAME = "BAAI/bge-base-en-v1.5"   # upgraded for better semantic understanding
TEXT_EMB_DIM = 768

# Enhanced GNN architecture based on research from "Classic GNNs are Strong Baselines"
# and "Non-convolutional Graph Neural Networks" for better graph understanding
GNN_HIDDEN = 512                # Increased to match text encoder capacity
GNN_LAYERS = 4                  # Increased for better structural understanding
GNN_OUT_DIM = 512                # Match text encoder dimension for better fusion
FUSION_HIDDEN = 1024             # Increased for better fusion capacity
GNN_HEADS = 8                    # Increased for better attention
GNN_DROPOUT = 0.15               # Balanced for regularization

# 5-dim edge attr: one-hot over the 4 semantic PTT edge types
# (StateTransition, SearchUpdate, TrackUpdate, Prediction) + a self-loop
# indicator. Shared constant so data_utils.py (graph building) and
# graph_encoder.py (GATv2Conv edge_dim) can never drift out of sync.
EDGE_ATTR_DIM = 5

MCP_LOSS_WEIGHT = 2.0           # Further increased for MCP performance target
STEP_LOSS_WEIGHT = 2.0           # Further increased for step classification target
MCP_DECISION_THRESHOLD = 0.5
STEP_LABEL_SMOOTHING = 0.03       # Reduced for better discrimination

STAGE1_LR = 1.5e-4               # Slightly reduced for stability
STAGE1_EPOCHS = 80               # Increased for better convergence
STAGE1_BATCH_SIZE = 16
STAGE1_WARMUP_EPOCHS = 8          # Increased warmup
STAGE1_GRAD_CLIP = 1.0
STAGE1_WEIGHT_DECAY = 1e-2

QWEN_MODEL_NAME = "Qwen/Qwen3-14B"
LLM_JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # Separate model for LLM judge evaluation
GRAPH_PREFIX_TOKENS = 8           # Further reduced to save memory
LORA_R = 32                      # Reduced to save memory
LORA_ALPHA = 64                  # Reduced proportionally
LORA_DROPOUT = 0.12              # Slightly increased for regularization
STAGE2_LR = 1e-5                # Reduced for FP16 stability
STAGE2_EPOCHS = 20               # Increased for better convergence
STAGE2_BATCH_SIZE = 1
STAGE2_GRAD_ACCUM = 16
STAGE2_VAL_SPLIT = 0.15          # 15% held-out for validation
STAGE2_EARLY_STOP_PATIENCE = 8   # Increased patience for better training
STAGE2_GRAD_CLIP = 1.0
STAGE2_HINT_MASK_PROB = 0.5      # Probability of masking Stage 1 hint during training (forces learning from graph tokens)
STAGE2_WARMUP_RATIO = 0.10       # Increased warmup
STAGE2_WEIGHT_DECAY = 1e-4

STAGE3_GROUP_SIZE = 4            # Reduced to avoid CUDA OOM
STAGE3_LR = 2e-6                # Optimized for GRPO with enhanced reward
STAGE3_STEPS = 3000              # Increased for better convergence
STAGE3_KL_COEF = 0.02            # Increased for better stability
STAGE3_PPO_CLIP = 0.2            # Standard PPO clipping
STAGE3_GRAD_ACCUM = 4
STAGE3_GRAD_CLIP = 1.0

RANDOM_SEED = 42