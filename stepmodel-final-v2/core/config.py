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

# Stage-1 graph encoder: deliberately moderate capacity for the machine-held-out
# dataset.  Capacity is spent on graph/context interaction rather than a very
# deep graph stack, which was prone to overfitting the small training set.
GNN_HIDDEN = 384
GNN_LAYERS = 3
GNN_OUT_DIM = 512
FUSION_HIDDEN = 768
GNN_DROPOUT = 0.12  # was 0.10 -- small bump, see CHANGES_AND_FINDINGS.md refinement notes

# Training-time-only graph augmentation (no-op at eval). Cheap regularizer
# for a 7.8M-param model trained on ~1.5k examples: randomly drops edges /
# node feature channels each forward pass so the model can't over-rely on
# any single PTT edge or feature dimension. Set either to 0.0 to disable.
STAGE1_EDGE_DROPOUT = 0.10
STAGE1_NODE_FEAT_DROPOUT = 0.05

# One shallow global graph-token Transformer after local GINE message passing.
GLOBAL_ATTN_LAYERS = 1
GLOBAL_ATTN_HEADS = 6
GLOBAL_ATTN_DROPOUT = 0.08

# BGE title + type(3) + status(4) + structural features:
# total degree, in-degree, out-degree, hierarchy depth, ordered position,
# leaf flag, root flag, branch-count ratio.
NODE_AUX_DIM = 3 + 4 + 8

# Paper-inspired semantic CNN branch: frozen GPT-2 token embeddings +
# multiple temporal convolution kernels + global max pooling.
SEMANTIC_LM_NAME = "gpt2"
SEMANTIC_LM_DIM = 768
SEMANTIC_MAX_TOKENS = 384
SEMANTIC_PROTOTYPE_TOKENS = 64
SEMANTIC_CNN_DIM = 128
SEMANTIC_CNN_KERNELS = (2, 3, 4, 5)
SEMANTIC_CNN_DROPOUT = 0.10

# 5-dim edge attr: one-hot over the 4 semantic PTT edge types
# (StateTransition, ActionUpdate, FindingUpdate, Prediction) + a self-loop
# indicator. Shared constant so data_utils.py (graph building) and
# graph_encoder.py (typed edge-aware convolution) cannot drift out of sync.
EDGE_ATTR_DIM = 5

MCP_LOSS_WEIGHT = 1.50
STEP_LOSS_WEIGHT = 1.00
MCP_DECISION_THRESHOLD = 0.5
STEP_LABEL_SMOOTHING = 0.01

STAGE1_LR = 2.0e-4
STAGE1_EPOCHS = 60
STAGE1_BATCH_SIZE = 16
STAGE1_WARMUP_EPOCHS = 4
STAGE1_GRAD_CLIP = 1.0
STAGE1_WEIGHT_DECAY = 8e-3  # was 5e-3 -- a bit more L2 given train loss << val plateau gap
STAGE1_MAX_CLASS_WEIGHT = 2.0  # was 3.0 -- capped lower now that step weighting is enabled below,
                                # so rare-class upweighting can't dominate the majority "Exploit" class
STAGE1_MAX_MCP_WEIGHT = 4.0
STAGE1_HARD_NEGATIVE_WEIGHT = 0.15
STAGE1_SUPCON_WEIGHT = 0.00
STAGE1_HARD_NEGATIVE_MARGIN = 0.20
# Optional per-class boosts used by the Stage-1 hard-negative/class-aware loss.
# Keep these modest so rare/confusable classes get extra emphasis without
# distorting the overall class distribution.
#
# CHANGED: turned back on. The v2 run's step_macro_f1 (0.557 test) sitting
# far below step_accuracy (0.746 test) means the plain-CE ablation was
# leaving several minority Step classes (esp. idx 4 "Enumerate the domain",
# n=23, and idx 8 "Explore source code", n=19) under-fit. Capped at 2.0x
# (STAGE1_MAX_CLASS_WEIGHT above) plus a very mild focal term (gamma=1.0
# below) is enough to lift them without meaningfully hurting the majority
# "Exploit the selected exploitations" class (n=526) that accuracy leans on.
STAGE1_USE_STEP_CLASS_WEIGHTS = True
STAGE1_USE_STEP_FOCAL = True
STAGE1_STEP_FOCAL_GAMMA = 1.0  # mild -- 1.8 (the MCP value) was too aggressive for a 10-way softmax
STAGE1_STEP_HARD_CLASS_BOOSTS = {
    0: 1.10,
    1: 1.10,
    2: 1.10,
    3: 1.10,
    4: 1.15,
    6: 1.15,
    8: 1.20,
}

# Stochastic Weight Averaging: instead of keeping only the single best-val
# checkpoint (noisy signal on a 239-example val split -- see log epoch-to-
# epoch score oscillation between 0.65 and 0.76), average the weights of
# the top-K checkpoints by val score and keep whichever (single-best vs.
# SWA) scores higher on val. LayerNorm/GraphNorm have no running batch
# stats, so plain weight averaging is safe here without a recalibration pass.
STAGE1_SWA_TOP_K = 5

# Machine-level k-fold count for the optional ensemble trainer
# (training/stage1_gnn_train_kfold.py). Averaging predictions across
# folds trained on different train/val machine splits is the single
# highest-leverage change for a dataset this small (149 train machines /
# ~1.5k rows) -- it directly reduces the split-variance visible in the v2
# single-split log.
STAGE1_N_FOLDS = 5

QWEN_MODEL_NAME = "Qwen/Qwen3-14B"
LLM_JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # Separate model for LLM judge evaluation
GRAPH_PREFIX_TOKENS = 8           # Further reduced to save memory
LORA_R = 32                      # Reduced to save memory
LORA_ALPHA = 64                  # Reduced proportionally
LORA_DROPOUT = 0.12              # Slightly increased for regularization
STAGE2_LR = 1e-5                # Reduced for FP16 stability
STAGE2_EPOCHS = 8                # Short bridge stage before GRPO; avoid overtraining
STAGE2_BATCH_SIZE = 1
STAGE2_GRAD_ACCUM = 16
STAGE2_VAL_SPLIT = 0.15          # 15% held-out for validation
STAGE2_EARLY_STOP_PATIENCE = 3   # Stop quickly once validation stops improving
STAGE2_GRAD_CLIP = 1.0
STAGE2_HINT_MASK_PROB = 0.5      # Probability of masking Stage 1 hint during training (forces learning from graph tokens)
STAGE2_WARMUP_RATIO = 0.05       # Short warmup for the compact bridge stage
STAGE2_WEIGHT_DECAY = 1e-4

STAGE3_GROUP_SIZE = 8            # Reduced to avoid CUDA OOM
STAGE3_LR = 1e-7                # Optimized for GRPO with enhanced reward
STAGE3_STEPS = 600              # Increased for better convergence
STAGE3_KL_COEF = 0.08            # Increased for better stability
STAGE3_PPO_CLIP = 0.2            # Standard PPO clipping
STAGE3_GRAD_ACCUM = 4
STAGE3_GRAD_CLIP = 1.0

# --- Stability fixes for the pg_loss/kl explosions seen in real runs ---
# (e.g. pg_loss 127 -> 2897 -> 8255 -> 8175 at steps 350/800/850/1000,
# followed by held-out val reward collapsing from 0.454 at step 600 down to
# 0.29 by step 1000, and fmt compliance dropping to 1/4). Root cause: for a
# NEGATIVE-advantage sample whose importance ratio ratio=pi_new/pi_ref has
# drifted far above 1 (policy now assigns much higher probability than the
# reference to something that turned out low-reward), standard PPO-clip's
# min(unclipped, clipped) objective does NOT bound the loss in that specific
# quadrant — only the "good news" direction is capped. With log_ratio
# clamped at +-10, ratio can reach e^10 ~ 22000, and with advantage clamped
# to +-4.0 that's an unclipped |pg_loss| up to ~4 * 22000 ~ 88000 for a
# single one of only 4 completions in the group, which then dominates the
# averaged batch loss and produces exactly the kind of single-step gradient
# spike visible in the log.
STAGE3_DUAL_CLIP_COEF = 3.0      # Dual-clip PPO (Ye et al. 2020 / used in DAPO,
                                  # verl, TRL GRPO trainers): for advantage < 0,
                                  # additionally floor the objective at
                                  # DUAL_CLIP_COEF * advantage instead of letting
                                  # it run to -inf as ratio explodes. Must be > 1.
STAGE3_KL_HARD_CAP = 4.0         # Per-MICRO-BATCH mean-KL cap (not per-window
                                  # — see training/stage3_grpo_rl.py). Originally
                                  # set to 1.0 and applied to the whole 4-batch
                                  # window average, which caused ~28% of windows
                                  # to be discarded (including good micro-batches
                                  # riding along with one noisy one) for no real
                                  # safety benefit: real runs show dual-clip
                                  # alone already keeps pg_loss bounded even when
                                  # an individual micro-batch's kl spikes to 5-6
                                  # (kl=5.636 -> pg_loss only 0.787; kl=5.031 ->
                                  # pg_loss only 0.732). Raised to 4.0 and moved
                                  # to per-micro-batch granularity so this is now
                                  # a rare, genuinely-extreme-only safety net
                                  # rather than a routine filter — tune down if
                                  # you see pg_loss/kl explosions again despite
                                  # dual-clip, tune up (or disable by setting a
                                  # very large value) if it's still discarding a
                                  # meaningful fraction of micro-batches on your
                                  # data.
STAGE3_EARLY_STOP_PATIENCE = 2   # Stop the run after this many consecutive
                                  # held-out evals (every EVAL_EVERY=200 steps)
                                  # with no new best checkpoint. Deliberately
                                  # NOT implemented by shrinking STAGE3_STEPS —
                                  # STAGE3_STEPS also sets the CosineAnnealingLR
                                  # T_max, so cutting it reshapes/compresses the
                                  # whole LR schedule rather than just cutting
                                  # the unproductive tail. Patience-based
                                  # stopping keeps the schedule exactly as
                                  # tuned and just exits the loop once it's
                                  # no longer paying off — set to None to
                                  # disable and always run the full STEPS.

RANDOM_SEED = 42