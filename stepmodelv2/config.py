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

# Directory holding per-machine / per-row pre-built graph objects.
# Expected file naming convention (adjust to match your actual export):
#   stepmodelv2/processed_data/train/<machine>__<row_id>.pt   (torch_geometric Data)
# If a graph file is not found, GraphDataset falls back to building a graph
# on the fly from the "PTT" (Penetration Testing Tree) text column.
GRAPH_DIR_TRAIN = os.environ.get(
    "GRAPH_DIR_TRAIN", "stepmodelv2/processed_data/train"
)
GRAPH_DIR_TEST = os.environ.get(
    "GRAPH_DIR_TEST", "stepmodelv2/processed_data/test"
)

CKPT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

STAGE1_CKPT = os.path.join(CKPT_DIR, "stage1_gnn_classifier.pt")
STAGE2_ADAPTER_DIR = os.path.join(CKPT_DIR, "stage2_qwen_lora")
STAGE3_ADAPTER_DIR = os.path.join(CKPT_DIR, "stage3_qwen_grpo")

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
STAGE2_EPOCHS = 3
STAGE2_BATCH_SIZE = 2
STAGE2_GRAD_ACCUM = 8

STAGE3_GROUP_SIZE = 4            # number of samples per prompt for GRPO
STAGE3_LR = 5e-6
STAGE3_STEPS = 1000
STAGE3_KL_COEF = 0.02

RANDOM_SEED = 42
