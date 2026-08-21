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
ROOT = os.environ.get("PIPELINE_ROOT", os.path.dirname(os.path.abspath(__file__)))

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
TEXT_ENCODER_NAME = "BAAI/bge-small-en-v1.5"   # frozen sentence embedder for context text
TEXT_EMB_DIM = 384

# NOTE: previous values here (512/5-layer/1024-fusion, ~9.7M trainable params)
# were oversized for the actual dataset (~1.5k train rows / 149 machines) and
# were the likely cause of the large epoch-to-epoch validation swings seen in
# training logs (val_step_acc jumping 0.10 -> 0.80 -> 0.62 -> 0.83 ...).
# Right-sized down; raise these again only if validation curves are smooth
# AND still improving with the smaller model.
GNN_HIDDEN = 256                # was 512
GNN_LAYERS = 3                  # was 5 -- PTT graphs are shallow trees, 5 hops risks oversmoothing
GNN_OUT_DIM = 256                # was 512
FUSION_HIDDEN = 512              # was 1024
GNN_HEADS = 4                    # was 8
GNN_DROPOUT = 0.2                # was 0.15 -- slightly stronger given smaller-data overfit risk

# 5-dim edge attr: one-hot over the 4 semantic PTT edge types
# (StateTransition, SearchUpdate, TrackUpdate, Prediction) + a self-loop
# indicator. Shared constant so data_utils.py (graph building) and
# graph_encoder.py (GATv2Conv edge_dim) can never drift out of sync.
EDGE_ATTR_DIM = 5

MCP_LOSS_WEIGHT = 1.5           # Increased to prioritize MCP performance
STEP_LOSS_WEIGHT = 1.5           # Increased to prioritize step classification
MCP_DECISION_THRESHOLD = 0.5
STEP_LABEL_SMOOTHING = 0.05       # Reduced for better discrimination

STAGE1_LR = 2e-4                 # was 4e-4 -- too high for this batch size/model, contributed to instability
STAGE1_EPOCHS = 60
STAGE1_BATCH_SIZE = 16
STAGE1_WARMUP_EPOCHS = 5
STAGE1_GRAD_CLIP = 1.5
STAGE1_WEIGHT_DECAY = 1e-2       # was hardcoded 1e-4 in the training script -- stronger reg for small data

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_JUDGE_MODEL_NAME = "Qwen/Qwen3-14B"  # Separate model for LLM judge evaluation
GRAPH_PREFIX_TOKENS = 16          # number of soft-prompt tokens the graph embedding is expanded into
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1             # Increased from 0.05 to reduce overfitting
STAGE2_LR = 3e-5                # Reduced from 5e-5 for more stable training
STAGE2_EPOCHS = 10              # Reduced from 15 to prevent overfitting
STAGE2_BATCH_SIZE = 2
STAGE2_GRAD_ACCUM = 8
STAGE2_VAL_SPLIT = 0.15          # 15% held-out for validation
STAGE2_EARLY_STOP_PATIENCE = 3  # Reduced from 4 to stop earlier when overfitting starts
STAGE2_GRAD_CLIP = 1.0
STAGE2_WARMUP_RATIO = 0.08
STAGE2_WEIGHT_DECAY = 1e-4       # Added weight decay for regularization

STAGE3_GROUP_SIZE = 16           # number of samples per prompt for GRPO (increased from 2 for better gradient estimation)
STAGE3_LR = 5e-7                 # Slightly reduced for stability with better Stage 2 init
STAGE3_STEPS = 2500              # Increased for better convergence
STAGE3_KL_COEF = 0.015           # Slightly increased KL for better stability
STAGE3_PPO_CLIP = 0.18           # Tighter clipping for more stable updates
STAGE3_GRAD_ACCUM = 4
STAGE3_GRAD_CLIP = 1.0

RANDOM_SEED = 42