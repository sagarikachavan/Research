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

GNN_HIDDEN = 256
GNN_LAYERS = 3
GNN_OUT_DIM = 256
FUSION_HIDDEN = 512

MCP_LOSS_WEIGHT = 1.0
STEP_LOSS_WEIGHT = 1.0
MCP_DECISION_THRESHOLD = 0.5

STAGE1_LR = 2e-4
STAGE1_EPOCHS = 30
STAGE1_BATCH_SIZE = 16

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GRAPH_PREFIX_TOKENS = 8          # number of soft-prompt tokens the graph embedding is expanded into
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
STAGE2_LR = 1e-4
STAGE2_EPOCHS = 10               # increased from 3 → allows model to converge on JSON schema
STAGE2_BATCH_SIZE = 2
STAGE2_GRAD_ACCUM = 8
STAGE2_VAL_SPLIT = 0.1           # 10% held-out for validation
STAGE2_EARLY_STOP_PATIENCE = 3   # stop if val loss doesn't improve for 3 epochs

STAGE3_GROUP_SIZE = 16           # number of samples per prompt for GRPO (increased from 2 for better gradient estimation)
STAGE3_LR = 1e-6                 # Increased from 5e-7 for more meaningful updates
STAGE3_STEPS = 2000             # Increased from 1000 for better convergence
STAGE3_KL_COEF = 0.01           # Further reduced from 0.02 for more exploration

RANDOM_SEED = 42
