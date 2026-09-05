"""
Central configuration for text-only experiment (no graph).
This is a simplified version that only uses text inputs.
"""
import os

# ----------------------------------------------------------------------------
# Label spaces (same as main pipeline)
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
# Paths (experiment-specific)
# ----------------------------------------------------------------------------
ROOT = os.environ.get("PIPELINE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_CSV = os.environ.get("TRAIN_CSV", os.path.join(ROOT, "data", "training_data.csv"))
TEST_CSV = os.environ.get("TEST_CSV", os.path.join(ROOT, "data", "test_data.csv"))

# Input files for experiment (text-only, no graph)
INPUT_TRAIN_JSON = os.environ.get(
    "INPUT_TRAIN_JSON", os.path.join(ROOT, "experiment", "input", "train.json")
)
INPUT_TEST_JSON = os.environ.get(
    "INPUT_TEST_JSON", os.path.join(ROOT, "experiment", "input", "test.json")
)

# Output directories
CKPT_DIR = os.environ.get(
    "CKPT_DIR", os.path.join(ROOT, "experiment", "checkpoints")
)
os.makedirs(CKPT_DIR, exist_ok=True)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", os.path.join(ROOT, "experiment", "output")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGE1_CKPT = os.environ.get(
    "STAGE1_CKPT", os.path.join(CKPT_DIR, "stage1_text_classifier.pt")
)
STAGE2_ADAPTER_DIR = os.environ.get(
    "STAGE2_ADAPTER_DIR", os.path.join(CKPT_DIR, "stage2_qwen_lora")
)
STAGE3_ADAPTER_DIR = os.environ.get(
    "STAGE3_ADAPTER_DIR", os.path.join(CKPT_DIR, "stage3_qwen_grpo")
)

# ----------------------------------------------------------------------------
# Model / training hyperparameters (simplified for text-only)
# ----------------------------------------------------------------------------
QWEN_MODEL_NAME = "Qwen/Qwen3-14B"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.12
STAGE2_LR = 1e-5
STAGE2_EPOCHS = 20
STAGE2_BATCH_SIZE = 1
STAGE2_GRAD_ACCUM = 16
STAGE2_VAL_SPLIT = 0.15
STAGE2_EARLY_STOP_PATIENCE = 8
STAGE2_GRAD_CLIP = 1.0
STAGE2_WARMUP_RATIO = 0.10
STAGE2_WEIGHT_DECAY = 1e-4

STAGE3_GROUP_SIZE = 4            # Reduced to avoid CUDA OOM
STAGE3_LR = 2e-6
STAGE3_STEPS = 3000
STAGE3_KL_COEF = 0.02
STAGE3_PPO_CLIP = 0.2
STAGE3_GRAD_ACCUM = 4
STAGE3_GRAD_CLIP = 1.0

RANDOM_SEED = 42
