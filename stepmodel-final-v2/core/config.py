"""
Central configuration: label spaces, paths, and hyperparameters.

=============================================================================
WHAT IS ACTUALLY ACTIVE
=============================================================================
This file carries substantial inline history -- each non-obvious value records
WHY it is what it is, usually because a measured run contradicted the obvious
choice. That history is deliberate: several values here look wrong until you
read the comment (STAGE1_USE_LOGIT_ADJUSTMENT=False, GNN_HIDDEN=384 rather
than 512, gamma_neg=2.0 rather than the paper's 4.0).

The cost is that "what architecture is running right now?" is hard to read off
a 900-line file. So:

    python core/config.py          <- prints every ACTIVE value, grouped

Sections, in order:
    1. LABEL SPACES          STEP_LABELS / MCP_LABELS and their index maps
    2. PATHS                 inputs, checkpoints, outputs
    3. MODEL                 encoder dims, GNN type, fusion, heads
    4. LOSS                  weights, imbalance handling, smoothing
    5. TRAINING              lr / epochs / batch / validation split
    6. ABLATION SWITCHES     experiment flags; defaults preserve behavior
    7. STAGE 2               SFT, LoRA, graph prefix adapter
    8. STAGE 3               GRPO, reward weights
    9. EVALUATION            judge model, thresholds

EVERY flag defaults to the currently-measured configuration, so an unset
environment reproduces the last reported numbers exactly.
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
# ── The project's ONE text encoder ──────────────────────────────────────────
TEXT_ENCODER_NAME = os.environ.get("TEXT_ENCODER_NAME", "Qwen/Qwen3-Embedding-0.6B")
TEXT_EMB_DIM = 768

# Stage-1 graph encoder capacity.
# ROUND 3 (see STAGE1_IMPROVEMENTS.md "Round 3"): GNN_HIDDEN/FUSION_HIDDEN had
GNN_HIDDEN = 384
GNN_LAYERS = 4  # kept -- graph depth, not width, and PTT graphs are small
GNN_OUT_DIM = 512  # contract dim consumed by Stage 2/3 -- do not change without retraining them
FUSION_HIDDEN = 768

# ---------------------------------------------------------------------------
# Stage-1 graph convolution type: "gatv2" (default) or "gine".
# ---------------------------------------------------------------------------
# CONVERTED TO GATv2 (user-directed, backed by the `main` branch's measured
STAGE1_GNN_TYPE = os.environ.get("STAGE1_GNN_TYPE", "gatv2").lower()

# Attention heads for the GATv2 blocks. Must divide GNN_HIDDEN evenly
# (384 / 8 = 48). main used 8 heads at hidden=512; 8 is kept here at 384.
GNN_HEADS = int(os.environ.get("GNN_HEADS", "8"))
GNN_DROPOUT = 0.15  # increased for better regularization

# Training-time-only graph augmentation (no-op at eval). Cheap regularizer
# for a 7.8M-param model trained on ~1.5k examples: randomly drops edges /
# node feature channels each forward pass so the model can't over-rely on
# any single PTT edge or feature dimension. Set either to 0.0 to disable.
STAGE1_EDGE_DROPOUT = 0.15  # increased for stronger regularization
STAGE1_NODE_FEAT_DROPOUT = 0.08  # increased for better generalization

# One shallow global graph-token Transformer after local message passing.
GLOBAL_ATTN_LAYERS = 1
GLOBAL_ATTN_HEADS = 6
GLOBAL_ATTN_DROPOUT = 0.08

# BGE title + type(3) + status(4) + structural features:
# total degree, in-degree, out-degree, hierarchy depth, ordered position,
# leaf flag, root flag, branch-count ratio.
NODE_AUX_DIM = 3 + 4 + 8

# REMOVED: the frozen-GPT-2 semantic CNN branch (SEMANTIC_LM_NAME,
# SEMANTIC_CNN_*, SEMANTIC_PROTOTYPE_TOKENS). Stage 1's text tower is now one
# Qwen3-Embedding vector per example, per the architecture diagram -- no
# second pretrained LM, no multi-kernel temporal convolution, no token-level
# cross-attention branch. The vector is projected to this width before fusion.
STAGE1_TEXT_PROJ_DIM = int(os.environ.get("STAGE1_TEXT_PROJ_DIM", "384"))
# Token-level text tower. The tower reads per-token Qwen3-Embedding hidden
# states and learns its own attention pooling, rather than being handed one
# pre-pooled sentence vector. TOKEN_DIM is the encoder's hidden size (1024 for
# Qwen3-Embedding-0.6B) and is asserted against the real data at load time.
STAGE1_TEXT_MAX_TOKENS = int(os.environ.get("STAGE1_TEXT_MAX_TOKENS", "256"))
STAGE1_TEXT_TOKEN_DIM = int(os.environ.get("STAGE1_TEXT_TOKEN_DIM", "1024"))
STAGE1_TEXT_ATTN_HEADS = int(os.environ.get("STAGE1_TEXT_ATTN_HEADS", "4"))
STAGE1_TEXT_DROPOUT = float(os.environ.get("STAGE1_TEXT_DROPOUT", "0.12"))

# 5-dim edge attr: one-hot over the 4 semantic PTT edge types
# (StateTransition, ActionUpdate, FindingUpdate, Prediction) + a self-loop
# indicator. Shared constant so data_utils.py (graph building) and
# graph_encoder.py (typed edge-aware convolution) cannot drift out of sync.
EDGE_ATTR_DIM = 5

MCP_LOSS_WEIGHT = 1.80  # increased from 1.50 to emphasize MCP learning
# REBALANCED (see STAGE1_IMPROVEMENTS.md): the first real post-fix training
# run landed at step_accuracy=0.75 (target 85-90%) vs mcp_micro_f1=0.6924
STEP_LOSS_WEIGHT = 1.50

# ---------------------------------------------------------------------------
# A3 — auxiliary coarse "phase" head (hierarchical supervision)
# ---------------------------------------------------------------------------
# The 10 step labels are not independent categories; they are positions in a
STEP_PHASE_OF = [0, 1, 2, 1, 1, 3, 2, 5, 2, 4]
N_STEP_PHASES = 6
STAGE1_PHASE_LOSS_WEIGHT = float(os.environ.get("STAGE1_PHASE_LOSS_WEIGHT", "0.30"))

# ---------------------------------------------------------------------------
# A4 — similarity-structured label smoothing
# ---------------------------------------------------------------------------
# Uniform label smoothing spreads target mass EQUALLY over all 10 classes,
STAGE1_USE_STRUCTURED_SMOOTHING = os.environ.get(
    "STAGE1_USE_STRUCTURED_SMOOTHING", "1") not in ("0", "false", "False")
STAGE1_SMOOTH_TEMP = 0.10        # softmax temperature over label-text similarity

# ---------------------------------------------------------------------------
# A5 — mask step classes that have no training support
# ---------------------------------------------------------------------------
# Class 7 ("Ask for human assistant") has 0 rows in train AND test. It cannot
STAGE1_MASK_UNSUPPORTED_CLASSES = os.environ.get(
    "STAGE1_MASK_UNSUPPORTED", "1") not in ("0", "false", "False")


# ---------------------------------------------------------------------------
# A6 — k-NN retrieval member for the step blend
# ---------------------------------------------------------------------------
# A third voice with a different inductive bias from both the GNN and the
# ===========================================================================
# ===========================================================================

# Manifold mixup interpolates two FUSED (graph+text) latents. In image
# classification the interpolant is still plausibly an image; here it is a
# blend of e.g. "HTTP enumeration" and "privilege escalation", which does not
# correspond to any real pentesting state. Suspect on this task specifically.
STAGE1_ABLATE_MIXUP = os.environ.get("STAGE1_ABLATE_MIXUP", "0") in ("1", "true", "True")

# SupCon needs positive pairs IN THE BATCH. With batch 20 and a ~31x head-to-
# tail ratio, the rare classes contribute 0-1 examples per batch, so they get
# almost no positive-pair signal. Supervised contrastive objectives are known
# to degrade under strong imbalance rather than fix it.
STAGE1_ABLATE_SUPCON = os.environ.get("STAGE1_ABLATE_SUPCON", "0") in ("1", "true", "True")


# Kang et al. (ICLR 2020) and LDAM-DRW both argue: learn the REPRESENTATION
# under natural sampling, rebalance the CLASSIFIER afterwards. This codebase
STAGE1_NATURAL_SAMPLING = os.environ.get("STAGE1_NATURAL_SAMPLING", "0") in ("1", "true", "True")

# Ablate the graph entirely (text-only Stage 1) or the text entirely
# (graph-only Stage 1). These two runs plus the default give the three rows
# that decide whether the graph earns its place:
#     text-only / graph-only / fusion
STAGE1_ABLATE_GRAPH = os.environ.get("STAGE1_ABLATE_GRAPH", "0") in ("1", "true", "True")
STAGE1_ABLATE_TEXT = os.environ.get("STAGE1_ABLATE_TEXT", "0") in ("1", "true", "True")



# ---------------------------------------------------------------------------
# Drop zero-support classes from the softmax entirely
# ---------------------------------------------------------------------------
# Masking a dead class's logit to -inf (STAGE1_MASK_UNSUPPORTED_CLASSES) stops
STAGE1_DROP_DEAD_CLASSES = os.environ.get("STAGE1_DROP_DEAD_CLASSES", "0") in ("1", "true", "True")


# ---------------------------------------------------------------------------
# Composite model-selection score
# ---------------------------------------------------------------------------
# Selecting on step accuracy alone rewards a model that collapses onto the
# 34%-prevalence "Exploit" class; selecting on MCP alone ignores the headline
# metric. This weighted combination is what checkpoint selection
# optimizes. Reported metrics stay separate and unweighted.
STAGE1_SEL_W_STEP_ACC = float(os.environ.get("STAGE1_SEL_W_STEP_ACC", "0.45"))
STAGE1_SEL_W_MCP_F1 = float(os.environ.get("STAGE1_SEL_W_MCP_F1", "0.35"))
STAGE1_SEL_W_STEP_MACRO = float(os.environ.get("STAGE1_SEL_W_STEP_MACRO", "0.20"))

# ---------------------------------------------------------------------------
# Tool-availability constraints for MCP
# ---------------------------------------------------------------------------
# Evidence-gated tool priors: if the PTT shows no SMB-related evidence, "Smb
STAGE1_USE_TOOL_CONSTRAINTS = os.environ.get("STAGE1_TOOL_CONSTRAINTS", "0") in ("1", "true", "True")
STAGE1_TOOL_CONSTRAINT_PENALTY = float(os.environ.get("STAGE1_TOOL_PENALTY", "1.0"))
TOOL_EVIDENCE_KEYWORDS = {
    "Nmap": ["port", "scan", "service", "nmap", "open"],
    "Metasploit": ["exploit", "cve", "vulnerab", "metasploit", "payload", "rce"],
    "Netcat": ["shell", "listener", "reverse", "netcat", "bind", "connect"],
    "Dirbuster": ["director", "web", "http", "url", "path", "brute"],
    "SQLmap": ["sql", "database", "inject", "query", "db"],
    "Smb client": ["smb", "share", "samba", "445", "netbios"],
    "hydra": ["password", "brute", "credential", "login", "hydra"],
    "John-the-ripper": ["hash", "crack", "password", "john", "shadow"],
    "Google search": ["research", "search", "version", "cve", "documentation"],
    "Interactive CLI": [],   # always plausible -- never penalized
    "Web page interaction": ["web", "http", "page", "browser", "form", "login"],
}





MCP_DECISION_THRESHOLD = 0.5
STEP_LABEL_SMOOTHING = 0.05  # increased from 0.01 for better generalization

STAGE1_LR = 3.0e-4  # increased from 2.0e-4 for faster convergence
STAGE1_EPOCHS = 80  # increased from 60 for longer training
STAGE1_BATCH_SIZE = 20  # increased from 16 for larger batch (if memory allows)
STAGE1_WARMUP_EPOCHS = 5  # increased from 4
STAGE1_GRAD_CLIP = 1.0
STAGE1_WEIGHT_DECAY = 1e-2  # increased from 8e-3 for stronger L2 regularization
# ----------------------------------------------------------------------------
# Stage-1 validation split (machine-grouped)
# ----------------------------------------------------------------------------
# Fraction of MACHINES held out for validation and for calibrating the MCP
# thresholds / step logit bias. Grouped by machine, never by row: rows from one
# machine share a PTT graph and adjacent steps, so a row-level split leaks.
# There is no cross-validation and no ensembling -- Stage 1 trains exactly one
# model, and that one model is what eval and Stage 2/3 consume.
STAGE1_VAL_SPLIT = float(os.environ.get("STAGE1_VAL_SPLIT", "0.2"))

# Step logit-bias calibration guardrails. The search fits one free parameter
# per class on the SAME ~380-row val split it is scored on, where the binomial
# SE is ~2pt -- so "beat plain argmax by anything" selects noise. Measured: a
# +1.25 bias on `End task` gained +0.5pt on val and cost 8 false positives on
# test (precision 0.73). Require a real margin, and cap the magnitude so no
# single class can be forced.
STAGE1_STEP_BIAS_MIN_GAIN = float(os.environ.get("STAGE1_STEP_BIAS_MIN_GAIN", "0.02"))
STAGE1_STEP_BIAS_MAX_ABS = float(os.environ.get("STAGE1_STEP_BIAS_MAX_ABS", "0.5"))

STAGE1_MAX_CLASS_WEIGHT = 2.5  # increased from 2.0 for better rare class handling
STAGE1_MAX_MCP_WEIGHT = 5.0  # increased from 4.0
STAGE1_HARD_NEGATIVE_WEIGHT = 0.20  # increased from 0.15
# ROUND 3 FIX: this weight was already wired into the training loop's loss
# sum (training/stage1_gnn_train.py) but the term it multiplied, `con`, was
STAGE1_SUPCON_WEIGHT = 0.15
STAGE1_SUPCON_TEMPERATURE = 0.10   # Khosla et al.'s recommended range (0.07-0.1)
STAGE1_HARD_NEGATIVE_MARGIN = 0.25  # increased from 0.20
# Optional per-class boosts used by the Stage-1 hard-negative/class-aware loss.
# Keep these modest so rare/confusable classes get extra emphasis without
STAGE1_USE_STEP_CLASS_WEIGHTS = True
STAGE1_USE_STEP_FOCAL = True
STAGE1_STEP_FOCAL_GAMMA = 1.5  # increased from 1.0 for better hard example focus
STAGE1_STEP_HARD_CLASS_BOOSTS = {
    0: 1.15,  # increased from 1.10
    1: 1.15,  # increased from 1.10
    2: 1.15,  # increased from 1.10
    3: 1.15,  # increased from 1.10
    4: 1.25,  # increased from 1.15 (rare class)
    6: 1.20,  # increased from 1.15
    8: 1.30,  # increased from 1.20 (rare class)
}

# ----------------------------------------------------------------------------
# Class-imbalance fixes added in the Stage-1 architecture/imbalance audit
# (see STAGE1_IMPROVEMENTS.md for full rationale + cited papers).
# ----------------------------------------------------------------------------
STAGE1_USE_LOGIT_ADJUSTMENT = False
STAGE1_LOGIT_ADJ_TAU = 1.0        # paper's default value. If combined with
                                   # STAGE1_USE_STEP_CLASS_WEIGHTS over-
                                   # corrects (majority-class recall drops a
                                   # lot), first try lowering this to ~0.5
                                   # before touching the weights -- see
                                   # STAGE1_IMPROVEMENTS.md.

# Asymmetric Loss for multi-label MCP prediction (Ridnik & Ben-Baruch et
# al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021):
STAGE1_MCP_LOSS_TYPE = "focal"    # "focal" (proven default, previous
                                   # behavior) or "asl" (opt-in, see above)
STAGE1_ASL_GAMMA_NEG = 2.0        # lowered from the paper's large-scale
                                   # default of 4.0 -- see fix note above
STAGE1_ASL_GAMMA_POS = 0.0        # paper default -- positives are already
                                   # the minority here, no need to also
                                   # down-weight hard positives
STAGE1_ASL_CLIP = 0.05            # paper default

# Manifold Mixup (Verma et al., "Manifold Mixup: Better Representations by
# Interpolating Hidden States", ICML 2019) applied to the fused Stage-1
STAGE1_USE_MANIFOLD_MIXUP = True
STAGE1_MIXUP_ALPHA = 0.2
STAGE1_MIXUP_WEIGHT = 0.3

# Decoupled classifier re-balancing (Kang et al., "Decoupling Representation
# and Classifier for Long-Tailed Recognition", ICLR 2020, arXiv:1910.09217).
STAGE1_USE_DECOUPLED_RETRAIN = True
STAGE1_DECOUPLED_EPOCHS = 15
STAGE1_DECOUPLED_LR = 5e-4

# ROUND 4 (see STAGE1_IMPROVEMENTS.md): the first real Round-3 run landed
# step accuracy at 0.7649 test -- identical to Round 2's number, so SupCon/
STAGE1_USE_GRAPH_GATE = True
STAGE1_GRAPH_GATE_INIT = 0.25   # legacy single-gate init; kept as the
                                  # fallback when per-head gating is disabled.

# ROUND 7 -- PER-HEAD GRAPH GATES (this is now empirically forced, not a
# hypothesis). The Round-6 run gave the decisive measurement: letting the
STAGE1_USE_PER_HEAD_GRAPH_GATE = True
STAGE1_GRAPH_GATE_INIT_STEP = 0.15   # Step: text-dominant, graph assists
STAGE1_GRAPH_GATE_INIT_MCP = 0.60    # MCP: graph-dominant, text assists


# ---------------------------------------------------------------------------
# Should the graph gate also attenuate the GRAPH->TEXT cross-attention term?
# ---------------------------------------------------------------------------
# The gate currently scales FOUR fusion terms: graph_proj, sem2graph_out,
STAGE1_GATE_GRAPH2SEM = os.environ.get("STAGE1_GATE_GRAPH2SEM", "1") not in ("0", "false", "False")
# ROUND 6 (see STAGE1_IMPROVEMENTS.md): the first real run with the gate
# showed it moving only 0.250 -> 0.232 (~7% relative) over the 44 epochs
STAGE1_GRAPH_GATE_LR_MULT = 12.0

# Stochastic Weight Averaging: instead of keeping only the single best-val
# checkpoint (noisy signal on a 239-example val split -- see log epoch-to-
STAGE1_SWA_TOP_K = 5


QWEN_MODEL_NAME = "Qwen/Qwen3-14B"
LLM_JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # Separate model for LLM judge evaluation
# Raised 8 -> 16: the adapter is now a learned-query resampler over per-node
# GINE states (see GraphPrefixAdapter in stage2_sft_qwen.py), so each token
# can carry distinct graph content instead of being a slice of one pooled
# vector -- and the rewrite made the adapter ~36x SMALLER in parameters, so
# doubling the token count is still a large net reduction.
GRAPH_PREFIX_TOKENS = 16
LORA_R = 32                      # Reduced to save memory
LORA_ALPHA = 64                  # Reduced proportionally
LORA_DROPOUT = 0.12              # Slightly increased for regularization
# CORRECTED (architecture re-audit): this was 1e-5 while
# training/stage2_sft_qwen.py's main() actually used an independently
STAGE2_LR = 2e-6
STAGE2_EPOCHS = 8                # Short bridge stage before GRPO; avoid overtraining
STAGE2_BATCH_SIZE = 1
STAGE2_GRAD_ACCUM = 16
# Upweight the loss on the "New step" label tokens (and MCP tokens) relative
# to the free-text explanation tokens. WHY: Stage 2's SFT loss is HuggingFace's
STAGE2_STEP_TOKEN_LOSS_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# B2 — Stage-3 reward rebalanced toward explanation
# ---------------------------------------------------------------------------
# Was 0.01 / 0.33 / 0.33 / 0.33 (format / step / mcp / explanation), i.e. GRPO
STAGE3_W_FMT = float(os.environ.get("STAGE3_W_FMT", "0.01"))
STAGE3_W_STEP = float(os.environ.get("STAGE3_W_STEP", "0.15"))
STAGE3_W_MCP = float(os.environ.get("STAGE3_W_MCP", "0.15"))
STAGE3_W_EXP = float(os.environ.get("STAGE3_W_EXP", "0.69"))


STAGE2_VAL_SPLIT = 0.15          # 15% held-out for validation
STAGE2_EARLY_STOP_PATIENCE = 3   # Stop quickly once validation stops improving
STAGE2_GRAD_CLIP = 1.0
STAGE2_WARMUP_RATIO = 0.05       # Short warmup for the compact bridge stage
STAGE2_WEIGHT_DECAY = 1e-4

# CORRECTED (Stage-2 regression post-mortem): the GraphPrefixAdapter is a
# ~14.6M-parameter RANDOMLY-INITIALIZED cross-attention resampler, while the
STAGE2_ADAPTER_LR_MULT = 50.0

# CORRECTED (user-directed, matching the main branch's working Stage-3
# regime): the previous 1e-7 / 600-step setting produced a DEAD run -- the
STAGE3_GROUP_SIZE = 8            # 8 completions per prompt for the GRPO group
STAGE3_LR = 2e-6                # matches main branch's working Stage-3 run;
                                  # 1e-7 was 20x too small and never moved the policy
STAGE3_STEPS = 1500             # user-requested; enough real updates to
                                  # actually shift the policy (150 updates at
                                  # 600 steps did nothing)
STAGE3_KL_COEF = 0.08            # Increased for better stability
STAGE3_PPO_CLIP = 0.2            # Standard (symmetric) PPO clipping lower bound
# DAPO "Clip-Higher" (Yu et al., "DAPO: An Open-Source LLM Reinforcement
# Learning System at Scale", 2025, arXiv:2503.14476): symmetric PPO clipping
STAGE3_USE_CLIP_HIGHER = True
STAGE3_CLIP_HIGH = 0.28          # DAPO paper's own reported high/low split
                                  # (low ~0.2, high ~0.28) for a comparable
                                  # PPO-clip setup; STAGE3_PPO_CLIP above is
                                  # used unchanged as the lower bound.
STAGE3_GRAD_ACCUM = 4
# CORRECTED (architecture re-audit): this was 1.0 while
# training/stage3_grpo_rl.py's actual grad-norm clip call had 0.5 hardcoded
STAGE3_GRAD_CLIP = 0.5

# --- Stability fixes for the pg_loss/kl explosions seen in real runs ---
# (e.g. pg_loss 127 -> 2897 -> 8255 -> 8175 at steps 350/800/850/1000,
STAGE3_DUAL_CLIP_COEF = 3.0
STAGE3_KL_HARD_CAP = 4.0         # Per-MICRO-BATCH mean-KL cap (not per-window
                                  # — see training/stage3_grpo_rl.py). Originally
                                  # set to 1.0 and applied to the whole 4-batch
STAGE3_EARLY_STOP_PATIENCE = 2   # Stop the run after this many consecutive
                                  # held-out evals (every EVAL_EVERY=200 steps)
                                  # with no new best checkpoint. Deliberately

# Env-overridable so a multi-seed variance check (see STAGE1_IMPROVEMENTS.md
# Round 6) doesn't require editing this file between runs:
#   RANDOM_SEED=1 python training/stage1_gnn_train.py
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# `python core/config.py` -- print the ACTIVE configuration
# ---------------------------------------------------------------------------
_SUMMARY_GROUPS = {
    "MODEL": [
        "TEXT_ENCODER_NAME", "TEXT_EMB_DIM", "GNN_TYPE_ACTIVE", "GNN_HIDDEN",
        "GNN_LAYERS", "GNN_OUT_DIM", "GNN_HEADS", "FUSION_HIDDEN",
        
        "NODE_AUX_DIM", "EDGE_ATTR_DIM",
        
        
        "N_STEP_PHASES",
    ],
    "LOSS / IMBALANCE": [
        "STEP_LOSS_WEIGHT", "MCP_LOSS_WEIGHT", "STEP_LABEL_SMOOTHING",
        "STAGE1_USE_STRUCTURED_SMOOTHING", "STAGE1_USE_STEP_CLASS_WEIGHTS",
        "STAGE1_MAX_CLASS_WEIGHT", "STAGE1_USE_STEP_FOCAL",
        "STAGE1_STEP_FOCAL_GAMMA", "STAGE1_USE_LOGIT_ADJUSTMENT",
        "STAGE1_MCP_LOSS_TYPE", "STAGE1_SUPCON_WEIGHT",
        "STAGE1_USE_MANIFOLD_MIXUP", "STAGE1_PHASE_LOSS_WEIGHT",
        "STAGE1_USE_DECOUPLED_RETRAIN",
    ],
    "TRAINING / ENSEMBLE": [
        "STAGE1_LR", "STAGE1_EPOCHS", "STAGE1_BATCH_SIZE", "RANDOM_SEED",
        "STAGE1_VAL_SPLIT", 
        
        "STAGE1_MASK_UNSUPPORTED_CLASSES", "STAGE1_DROP_DEAD_CLASSES",
        "STAGE1_SEL_W_STEP_ACC", "STAGE1_SEL_W_MCP_F1", "STAGE1_SEL_W_STEP_MACRO",
    ],
    "ABLATIONS (all default off)": [
        "STAGE1_ABLATE_MIXUP", "STAGE1_ABLATE_SUPCON", "STAGE1_ABLATE_GRAPH",
        "STAGE1_ABLATE_TEXT", "STAGE1_NATURAL_SAMPLING",
        "STAGE1_USE_TOOL_CONSTRAINTS",
    ],
    "STAGE 2": [
        "QWEN_MODEL_NAME", "LORA_R", "LORA_ALPHA", "STAGE2_LR",
        "STAGE2_ADAPTER_LR_MULT", "STAGE2_EPOCHS", "GRAPH_PREFIX_TOKENS",
        
    ],
    "STAGE 3": [
        "STAGE3_STEPS", "STAGE3_GROUP_SIZE", "STAGE3_LR",
        "STAGE3_W_FMT", "STAGE3_W_STEP", "STAGE3_W_MCP", "STAGE3_W_EXP",
        
    ],
    "EVALUATION": ["LLM_JUDGE_MODEL_NAME", "MCP_DECISION_THRESHOLD"],
}

GNN_TYPE_ACTIVE = STAGE1_GNN_TYPE  # alias so the summary reads naturally


def print_active_config():
    g = globals()
    print("=" * 74)
    print("ACTIVE STAGE-1/2/3 CONFIGURATION")
    print("=" * 74)
    for group, keys in _SUMMARY_GROUPS.items():
        print(f"\n[{group}]")
        for k in keys:
            if k in g:
                print(f"  {k:<36} {g[k]}")
    n_live = sum(1 for _ in STEP_LABELS)
    print(f"\n[LABEL SPACE]")
    print(f"  {'step classes':<36} {n_live}")
    print(f"  {'mcp tools':<36} {len(MCP_LABELS)}")
    print("\n" + "=" * 74)
    print("Every value above is what an unset environment produces.")
    print("Override any of them with the matching env var; see the inline")
    print("comment at each definition for why the default is what it is.")


if __name__ == "__main__":
    print_active_config()
