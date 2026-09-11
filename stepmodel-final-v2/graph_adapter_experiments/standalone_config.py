"""
standalone_config.py
=====================
Every setting the standalone Graph-Prefix-Adapter experiment needs, in one
place. This file imports NOTHING from `core/`, `data_prep/`, `training/`,
or `eval/` — that is the whole point of this experiment. It only reads the
raw input JSON *data* files your main pipeline already produced (the graph
field of each record), never the main pipeline's *code*.

If you want to point this at a different copy of the data, just override
the env vars below — nothing here assumes it's sitting next to `run.py`.
"""
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

# ---------------------------------------------------------------------------
# Data — the ONLY thing borrowed from the main pipeline: the raw JSON files
# it already exports (each record has "machine", "row_index"/"row_id" and a
# "graph" field with "nodes"/"edges"). We read the "graph" field only; the
# "new_strategy" / "strategy_explanation" / "gold_*" fields in that same
# file are IGNORED by everything in this folder — see graph_json.py.
# ---------------------------------------------------------------------------
INPUT_TRAIN_JSON = os.environ.get(
    "STANDALONE_INPUT_TRAIN_JSON", os.path.join(_REPO_ROOT, "input", "train.json")
)
INPUT_TEST_JSON = os.environ.get(
    "STANDALONE_INPUT_TEST_JSON", os.path.join(_REPO_ROOT, "input", "test.json")
)

# ---------------------------------------------------------------------------
# Where this experiment writes its own outputs. All under this folder —
# never back into the main pipeline's checkpoints/ or output/ directories.
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = _THIS_DIR
TASKS_DIR = os.path.join(EXPERIMENT_DIR, "standalone_tasks")
CKPT_DIR = os.path.join(EXPERIMENT_DIR, "standalone_checkpoints")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "standalone_results")
for _d in (TASKS_DIR, CKPT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Structural node/edge feature schema (graph_json.py)
#   Node type one-hot:   State, Action, Finding, Unknown        -> 4 dims
#   Node status one-hot: completed, in_progress, to_do, unknown -> 4 dims
#   Structural scalars:  in-deg, out-deg, total-deg (normalized),
#                        BFS depth from the row's start node (normalized),
#                        is-start flag                          -> 5 dims
# Deliberately NO text embedding of node titles anywhere in this file or in
# graph_json.py — this experiment is about whether the LLM can read pure
# graph STRUCTURE out of the soft-prompt tokens, so the node features must
# not leak semantic/text content the LLM could otherwise shortcut through.
# ---------------------------------------------------------------------------
NODE_TYPES = ["State", "Action", "Finding", "Unknown"]
NODE_STATUSES = ["completed", "in_progress", "to_do", "unknown"]
NODE_FEAT_DIM = len(NODE_TYPES) + len(NODE_STATUSES) + 5  # = 13

EDGE_TYPES = ["StateTransition", "SearchUpdate", "TrackUpdate", "Prediction"]
EDGE_FEAT_DIM = len(EDGE_TYPES) + 1  # +1 self-loop indicator = 5

# ---------------------------------------------------------------------------
# GNN (trained from scratch in this experiment; structure-only, no fused
# text/strategy input of any kind — contrast with core/graph_encoder.py's
# Stage1Classifier, which fuses the graph embedding with a text embedding of
# the "New strategy"/"Strategy explanation" columns before the LLM ever
# sees it. None of that fusion exists here.)
# ---------------------------------------------------------------------------
GNN_HIDDEN = 128
GNN_LAYERS = 3
GNN_HEADS = 4
GNN_OUT_DIM = 128
GNN_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Soft-prompt adapter + LLM
# ---------------------------------------------------------------------------
GRAPH_PREFIX_TOKENS = 8
# Small instruction model by default so this experiment is runnable on a
# single consumer GPU; override with --model_name on the CLI for anything
# else. This is intentionally NOT Qwen3-14B / not any main-pipeline
# checkpoint — a fresh base model, fresh LoRA (optional), fresh adapter.
LLM_MODEL_NAME = os.environ.get("STANDALONE_LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TRAIN_LR = 2e-4
TRAIN_BATCH_SIZE = 4
TRAIN_GRAD_ACCUM = 4
TRAIN_STEPS = 1500
TRAIN_EVAL_EVERY = 150
TRAIN_WARMUP_STEPS = 50
TRAIN_GRAD_CLIP = 1.0
MAX_NEW_TOKENS = 64

# Fraction of machines held out (machine-level split — never split by row,
# so a held-out graph can never share a machine with a training graph).
VAL_FRAC = 0.2
