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
    5. TRAINING              lr / epochs / batch / K-fold / ensembling
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
TEXT_ENCODER_NAME = "BAAI/bge-base-en-v1.5"   # upgraded for better semantic understanding
TEXT_EMB_DIM = 768

# Stage-1 graph encoder capacity.
# ROUND 3 (see STAGE1_IMPROVEMENTS.md "Round 3"): GNN_HIDDEN/FUSION_HIDDEN had
# been raised to 512/1024 on the theory that more capacity would help -- with
# no A/B evidence either way at the time. The Round-2 real run gave that
# evidence and it points the other way: by epoch ~30/80 train loss had
# dropped near zero while val_step_acc still oscillated 10-25 points
# epoch-to-epoch, and Stage 1 carries ~22M trainable parameters against
# ~1.5k training rows (~15:1 params:rows) -- a textbook small-data overfit
# signature, not a capacity shortfall. Reverted both back to their
# pre-inflation values; the representation-quality work now comes from the
# supervised-contrastive term and decoupled classifier re-balancing below
# instead of from raw parameter count.
GNN_HIDDEN = 384
GNN_LAYERS = 4  # kept -- graph depth, not width, and PTT graphs are small
GNN_OUT_DIM = 512  # contract dim consumed by Stage 2/3 -- do not change without retraining them
FUSION_HIDDEN = 768

# ---------------------------------------------------------------------------
# Stage-1 graph convolution type: "gatv2" (default) or "gine".
# ---------------------------------------------------------------------------
# CONVERTED TO GATv2 (user-directed, backed by the `main` branch's measured
# results). main/output/eval_metrics_stage1_gnn.json -- a real GATv2 Stage-1
# run -- reports step_accuracy=0.8022, step_macro_f1=0.6765,
# mcp_samples_f1=0.7657, mcp_micro_f1=0.7457. The GINE 5-fold ensemble on this
# branch reports 0.7910 / 0.5984 / 0.7226 / 0.7096: GATv2 ahead on 4 of 5
# metrics, with the largest margin on MCP (+4.3pt samples-F1).
#
# That comparison is NOT fully controlled -- main also ran GNN_HIDDEN=512 with
# 8 heads on a single 85/15 split rather than this branch's 384-wide K-fold
# setup. Both branches skip the corrupted-Machine rows, so training-set size
# is comparable; width/heads/validation-scheme remain confounded.
#
# The stronger reason to expect the change to land on MCP specifically: the
# per-head graph gates show the STEP head learning to shut the graph off
# almost entirely (gate_step converges to ~0.04 from its 0.15 init, in all 5
# folds), while gate_mcp holds near 0.50. With ~96% of the graph signal
# suppressed into the step head, the choice of graph convolution can barely
# move step accuracy -- it can only really affect MCP, which is exactly where
# GATv2's measured margin is largest.
#
# Kept as a switch rather than a hard replacement so the controlled
# single-variable comparison (same 384 width, same K-fold pipeline, only the
# conv layer differs) can still be run with STAGE1_GNN_TYPE=gine.
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

# Paper-inspired semantic CNN branch: frozen GPT-2 token embeddings +
# multiple temporal convolution kernels + global max pooling.
SEMANTIC_LM_NAME = "gpt2"
SEMANTIC_LM_DIM = 768
SEMANTIC_MAX_TOKENS = 512  # increased from 384 for more context
SEMANTIC_PROTOTYPE_TOKENS = 64
SEMANTIC_CNN_DIM = 192  # increased from 128 for richer semantic features
SEMANTIC_CNN_KERNELS = (2, 3, 4, 5, 7)  # added larger kernel for wider context
SEMANTIC_CNN_DROPOUT = 0.12  # increased for better regularization

# 5-dim edge attr: one-hot over the 4 semantic PTT edge types
# (StateTransition, ActionUpdate, FindingUpdate, Prediction) + a self-loop
# indicator. Shared constant so data_utils.py (graph building) and
# graph_encoder.py (typed edge-aware convolution) cannot drift out of sync.
EDGE_ATTR_DIM = 5

MCP_LOSS_WEIGHT = 1.80  # increased from 1.50 to emphasize MCP learning
# REBALANCED (see STAGE1_IMPROVEMENTS.md): the first real post-fix training
# run landed at step_accuracy=0.75 (target 85-90%) vs mcp_micro_f1=0.6924
# (target 70-80%, i.e. already near/inside range) -- MCP is close to its
# target while Step has the larger gap. Both heads share the same fusion
# trunk and compete for gradient through the same total-loss sum, so
# shifting relative weight toward Step gives it more of that shared
# capacity without touching MCP's own loss function. Raised from 1.00 (the
# 78%-step-accuracy-era value; the audit's earlier 1.20 was applied without
# being able to check its effect against a real run) to 1.50; MCP_LOSS_
# WEIGHT left unchanged since MCP is already close to target. If a re-run
# shows MCP regressing, dial STEP_LOSS_WEIGHT back toward 1.20 rather than
# raising MCP_LOSS_WEIGHT further, to keep this a one-variable change.
STEP_LOSS_WEIGHT = 1.50

# ---------------------------------------------------------------------------
# A3 — auxiliary coarse "phase" head (hierarchical supervision)
# ---------------------------------------------------------------------------
# The 10 step labels are not independent categories; they are positions in a
# pentest workflow. Grouping them by phase:
#
#   RESEARCH  : 0            (google search)
#   ENUMERATE : 1, 3, 4      (enumerate service / website / domain)
#   ANALYZE   : 2, 6, 8      (explore files / analyze outcomes / read source)
#   EXPLOIT   : 5
#   END       : 9
#   OTHER     : 7            (dead class, zero rows anywhere)
#
# Implemented as an AUXILIARY head rather than a hard two-stage decision.
# Honest reason: the measured confusions are mostly CROSS-phase (1->2, 1->6,
# 3->2, 2->5), so a hard hierarchy would not fix them and could lock in a
# wrong coarse decision. What a coarse head does buy is multi-task
# regularization -- an easier, lower-variance target on the same shared trunk,
# which is the standard argument for auxiliary supervision on small data.
# Set STAGE1_PHASE_LOSS_WEIGHT = 0.0 to disable entirely.
STEP_PHASE_OF = [0, 1, 2, 1, 1, 3, 2, 5, 2, 4]
N_STEP_PHASES = 6
STAGE1_PHASE_LOSS_WEIGHT = float(os.environ.get("STAGE1_PHASE_LOSS_WEIGHT", "0.30"))

# ---------------------------------------------------------------------------
# A4 — similarity-structured label smoothing
# ---------------------------------------------------------------------------
# Uniform label smoothing spreads target mass EQUALLY over all 10 classes,
# including ones that are semantically unrelated -- and including class 7,
# which has zero rows and which is exactly how the logit-adjustment collapse
# got started. Structured smoothing instead spreads the smoothing mass in
# proportion to how similar the label TEXTS are (BGE cosine over STEP_LABELS),
# so "Further enumerate the website" leaks toward "Enumerate the X service"
# rather than toward "End task". Mueller et al., "When Does Label Smoothing
# Help?" (NeurIPS 2019) for why the target distribution's shape matters.
# Classes with zero training support receive zero smoothing mass.
STAGE1_USE_STRUCTURED_SMOOTHING = os.environ.get(
    "STAGE1_USE_STRUCTURED_SMOOTHING", "1") not in ("0", "false", "False")
STAGE1_SMOOTH_TEMP = 0.10        # softmax temperature over label-text similarity

# ---------------------------------------------------------------------------
# A5 — mask step classes that have no training support
# ---------------------------------------------------------------------------
# Class 7 ("Ask for human assistant") has 0 rows in train AND test. It cannot
# ever be predicted correctly, it contributes a guaranteed 0 to macro-F1
# (capping it at 0.90), and it is the class that triggered the
# label-smoothing x logit-adjustment collapse. STEP_LABELS is deliberately NOT
# reindexed -- that would break every saved checkpoint and the Stage-2/3 label
# contract. Instead its logit is masked to -inf at inference and it is
# excluded from macro-F1.
STAGE1_MASK_UNSUPPORTED_CLASSES = os.environ.get(
    "STAGE1_MASK_UNSUPPORTED", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# A2 — seed x fold ensembling
# ---------------------------------------------------------------------------
# Per-machine accuracy std is 0.115 across the 29 test machines: variance, not
# bias, is the dominant error source. Deep ensembles (Lakshminarayanan et al.,
# NeurIPS 2017) are the standard remedy. STAGE1_N_SEEDS=1 reproduces today's
# behavior exactly; 3 gives 5 folds x 3 seeds = 15 members at 3x wall-clock.
STAGE1_N_SEEDS = int(os.environ.get("STAGE1_N_SEEDS", "1"))

# ---------------------------------------------------------------------------
# A6 — k-NN retrieval member for the step blend
# ---------------------------------------------------------------------------
# A third voice with a different inductive bias from both the GNN and the
# linear text head (kNN-LM, Khandelwal et al., ICLR 2020). Cheap: reuses the
# BGE embeddings the text head already computes.
# ===========================================================================
# ABLATION SWITCHES — for establishing what actually carries the signal
# ===========================================================================
# Stage 1 accumulated ~15 mechanisms on ~1.9k training rows. None of the
# following has ever been ablated on this dataset; each is ON only because it
# was added, not because it was measured. These switches make the "which
# information source actually carries the prediction" experiment runnable.
#
# Defaults preserve TODAY's behavior exactly, so flipping one at a time is a
# clean single-variable experiment.

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

# Replace the bidirectional cross-attention + Hadamard fusion with the much
# simpler [text, graph, text*graph, |text-graph|] concatenation. The
# cross-attention block has many interaction parameters to learn from ~1.9k
# rows; simpler fusion is easier to optimize and is the honest baseline the
# complex version should have to beat.
STAGE1_SIMPLE_FUSION = os.environ.get("STAGE1_SIMPLE_FUSION", "0") in ("1", "true", "True")

# Kang et al. (ICLR 2020) and LDAM-DRW both argue: learn the REPRESENTATION
# under natural sampling, rebalance the CLASSIFIER afterwards. This codebase
# currently does both at once -- weighted sampler + class weights + focal
# during representation learning AND decoupled classifier retraining after --
# which partly defeats the reason for the decoupled stage. Turning this on
# disables the representation-phase rebalancing and leaves only the decoupled
# classifier stage doing the correction.
STAGE1_NATURAL_SAMPLING = os.environ.get("STAGE1_NATURAL_SAMPLING", "0") in ("1", "true", "True")

# Ablate the graph entirely (text-only Stage 1) or the text entirely
# (graph-only Stage 1). These two runs plus the default give the three rows
# that decide whether the graph earns its place:
#     text-only / graph-only / fusion
STAGE1_ABLATE_GRAPH = os.environ.get("STAGE1_ABLATE_GRAPH", "0") in ("1", "true", "True")
STAGE1_ABLATE_TEXT = os.environ.get("STAGE1_ABLATE_TEXT", "0") in ("1", "true", "True")

# ---------------------------------------------------------------------------
# Label prototypes in the DECISION function (not just the loss)
# ---------------------------------------------------------------------------
# The step labels are natural-language descriptions, not arbitrary IDs. We
# already use their pairwise similarity for structured smoothing; this uses
# the context-to-LABEL similarity directly as an additive logit. It is
# especially attractive for the tail: "Enumerate the domain" (24 rows) and
# "Explore the source code" (21 rows) have almost no training signal, but
# their label TEXT is fully informative and needs no training data at all.
STAGE1_USE_LABEL_PROTOTYPES = os.environ.get("STAGE1_USE_PROTOTYPES", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# Step-conditioned MCP head
# ---------------------------------------------------------------------------
# Tool choice depends heavily on WHICH ACTION is being taken: "enumerate the
# website" implies Dirbuster/web interaction, "exploit" implies Metasploit.
# Feeding the soft step distribution into the MCP head makes that dependency
# explicit instead of hoping the shared trunk encodes it:
#     context -> step ;  context + graph + step -> tools
# Uses the DETACHED step probabilities so MCP's gradient cannot corrupt the
# step head (one-directional conditioning, not joint optimization).
# ---------------------------------------------------------------------------
# Separate Step / MCP semantic towers  (Pen-Strategist section 4.2.2)
# ---------------------------------------------------------------------------
# The paper uses TWO INDEPENDENT convolutional encoders over the same frozen
# GPT-2 features -- one whose representation feeds step classification, one
# whose representation feeds MCP prediction. Its own words: "we apply two
# separate convolutional encoders... One representation is used for step
# classification, and the other for MCP server prediction."
#
# This codebase deliberately diverged, using ONE SHARED CNN, because the
# semantic vector then had to be fused with the graph representation before
# the heads split -- a single shared vector is what the fusion block needs.
# That was a defensible design decision, but it was never A/B tested, and it
# forces both tasks through one bottleneck. The measured per-head graph gates
# say the two heads want very different things from the representation
# (gate_step converges to ~0.04, gate_mcp holds ~0.49), which is evidence of
# exactly the negative transfer a shared tower invites.
#
# With this ON, each head gets its own SemanticCNNEncoder and its own set of
# fusion projections, sharing only the frozen GPT-2 features and the graph
# encoder. Cost is one extra CNN (~1.2M params).
#
# Default OFF so the current architecture is unchanged until the A/B is run.
# ---------------------------------------------------------------------------
# OOF stacking meta-learner (replaces the grid-searched blend weights)
# ---------------------------------------------------------------------------
# The blend currently picks ONE scalar weight per member by grid search over a
# simplex. That is a crude combiner: it can only apply a single global weight
# per model, so it cannot express "trust the k-NN on rare classes but the GNN
# on Exploit" -- which is exactly the structure the measured error overlap
# shows (GNN and text disagree on 43/268 rows, splitting 16/17).
#
# Stacking (Wolpert 1992) instead TRAINS a meta-classifier on the members'
# out-of-fold predictions, so it learns per-class trust. Inputs per row are
# the concatenated member logits; the meta-learner is a multinomial logistic
# regression (kept linear for the same reason the text head is: ~1.9k rows).
#
# The grid blend is retained as a fallback and the stacker is adopted only if
# it beats it on the same OOF pool.
# ---------------------------------------------------------------------------
# Final full-data encoder for Stage 2/3
# ---------------------------------------------------------------------------
# PROBLEM THIS FIXES: the Stage-1 headline result is a blended K-fold
# ensemble, but Stage 2 can load only ONE encoder -- so the model producing
# the headline number was never the model passed downstream. Worse, the fold
# handed over had trained on only 80% of the machines.
#
# With this ON, after the folds finish and the architecture is fixed, ONE more
# model is trained on ALL training machines and that is what Stage 2/3
# receive. The fold ensemble still produces the reported Stage-1 metrics.
#
# The final model has no held-out split of its own, so its checkpoint is taken
# at the MEDIAN best-epoch across folds rather than by early stopping -- using
# any validation data to stop it would contradict training on everything.
STAGE1_TRAIN_FINAL_ON_ALL = os.environ.get("STAGE1_FINAL_ON_ALL", "0") in ("1", "true", "True")

# ---------------------------------------------------------------------------
# Drop zero-support classes from the softmax entirely
# ---------------------------------------------------------------------------
# Masking a dead class's logit to -inf (STAGE1_MASK_UNSUPPORTED_CLASSES) stops
# it being predicted, but the class still occupies a softmax slot, still
# absorbs probability mass during training, and still counts as a guaranteed
# zero in a 10-class macro-F1.
#
# With this ON the classifier is built over only the classes that actually
# have training support (9 of 10 here -- "Ask for human assistant" has zero
# rows in train AND test), and macro-F1 is computed over those 9. STEP_LABELS
# is NOT reindexed: the model keeps a full-width output by scattering its
# 9 logits back into 10 slots, so every saved checkpoint and the Stage-2/3
# label contract stay valid. Escalation becomes a decision rule outside the
# classifier, which is what it always should have been.
STAGE1_DROP_DEAD_CLASSES = os.environ.get("STAGE1_DROP_DEAD_CLASSES", "0") in ("1", "true", "True")

STAGE1_USE_STACKING = os.environ.get("STAGE1_USE_STACKING", "1") not in ("0", "false", "False")
STAGE1_STACK_C = float(os.environ.get("STAGE1_STACK_C", "1.0"))

# ---------------------------------------------------------------------------
# Composite model-selection score
# ---------------------------------------------------------------------------
# Selecting on step accuracy alone rewards a model that collapses onto the
# 34%-prevalence "Exploit" class; selecting on MCP alone ignores the headline
# metric. This weighted combination is what checkpoint/fold selection
# optimizes. Reported metrics stay separate and unweighted.
STAGE1_SEL_W_STEP_ACC = float(os.environ.get("STAGE1_SEL_W_STEP_ACC", "0.45"))
STAGE1_SEL_W_MCP_F1 = float(os.environ.get("STAGE1_SEL_W_MCP_F1", "0.35"))
STAGE1_SEL_W_STEP_MACRO = float(os.environ.get("STAGE1_SEL_W_STEP_MACRO", "0.20"))

# ---------------------------------------------------------------------------
# Tool-availability constraints for MCP
# ---------------------------------------------------------------------------
# Evidence-gated tool priors: if the PTT shows no SMB-related evidence, "Smb
# client" should be unlikely. Implemented as a small additive logit penalty
# for tools whose evidence keywords are absent from the graph text, applied at
# INFERENCE only.
#
# CRITICAL: the evidence comes from the PTT text available at decision time --
# never from the gold MCP labels. Deriving a constraint from the target would
# be leakage dressed up as a prior.
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

STAGE1_SEPARATE_SEMANTIC_TOWERS = os.environ.get(
    "STAGE1_SEPARATE_TOWERS", "0") in ("1", "true", "True")

STAGE1_STEP_CONDITIONED_MCP = os.environ.get("STAGE1_STEP_COND_MCP", "1") not in ("0", "false", "False")

STAGE1_USE_KNN_MEMBER = os.environ.get("STAGE1_USE_KNN", "1") not in ("0", "false", "False")
STAGE1_KNN_K = int(os.environ.get("STAGE1_KNN_K", "15"))


MCP_DECISION_THRESHOLD = 0.5
STEP_LABEL_SMOOTHING = 0.05  # increased from 0.01 for better generalization

STAGE1_LR = 3.0e-4  # increased from 2.0e-4 for faster convergence
STAGE1_EPOCHS = 80  # increased from 60 for longer training
STAGE1_BATCH_SIZE = 20  # increased from 16 for larger batch (if memory allows)
STAGE1_WARMUP_EPOCHS = 5  # increased from 4
STAGE1_GRAD_CLIP = 1.0
STAGE1_WEIGHT_DECAY = 1e-2  # increased from 8e-3 for stronger L2 regularization
# ----------------------------------------------------------------------------
# Stage-1 K-fold ensembling (machine-grouped)
# ----------------------------------------------------------------------------
# WHY (evidence from output/stage1.csv, 268 test rows / 29 machines, acc 0.7948):
#
#  * There is NO label noise to blame: 262 distinct model inputs cover all 268
#    rows and ZERO inputs carry conflicting gold labels, so the achievable
#    ceiling on this test set is 100%. 85-90% is not blocked by the data.
#  * Errors are SPREAD, not clustered: per-machine accuracy std is 0.115 across
#    all 29 machines (min 0.57, max 1.00); the worst 5 machines hold only 33%
#    of errors while covering 18% of rows. That is a VARIANCE signature, not a
#    distribution-shift cliff -- and variance is exactly what ensembling fixes.
#  * 29 of 55 errors are rows pulled into two "attractor" classes: class 2
#    (predicted 39x, gold 30, precision 0.46 -- a catch-all sink) and class 5
#    (predicted 103x, gold 92). Correcting that needs a per-class logit bias
#    fit on data that actually EXHIBITS the pattern.
#
# The single 15%-machine val split could not deliver that second point: val
# scored 0.8410 vs test 0.7948 and simply does not show the attractor skew, so
# the bias search (which is itself sound -- verified against direct coordinate
# ascent on simulated data) found almost nothing to correct and transferred
# nothing. K-fold fixes BOTH problems at once:
#
#   1. K models averaged  -> attacks the variance that dominates the error.
#   2. Pooled OUT-OF-FOLD predictions over ALL training machines -> a
#      calibration set ~7x larger and far more representative than 48 val
#      machines, so the step-bias / MCP-threshold search is fit on data that
#      looks like the test distribution.
#
# K-fold is now the ONLY Stage-1 training path -- the previous single
# 15%-machine-split fallback has been removed. The best fold's weights (plus
# the pooled out-of-fold calibration) are written to STAGE1_CKPT, which is
# what eval/evaluate.py, stage2_sft_qwen.py and stage3_grpo_rl.py consume.
STAGE1_N_FOLDS = int(os.environ.get("STAGE1_N_FOLDS", "5"))

STAGE1_MAX_CLASS_WEIGHT = 2.5  # increased from 2.0 for better rare class handling
STAGE1_MAX_MCP_WEIGHT = 5.0  # increased from 4.0
STAGE1_HARD_NEGATIVE_WEIGHT = 0.20  # increased from 0.15
# ROUND 3 FIX: this weight was already wired into the training loop's loss
# sum (training/stage1_gnn_train.py) but the term it multiplied, `con`, was
# hard-coded to `fused_h.new_zeros(())` every batch -- the contrastive loss
# was never actually implemented, so this flag did nothing at any nonzero
# value. Now implemented as real Supervised Contrastive Loss (Khosla et al.,
# "Supervised Contrastive Learning", NeurIPS 2020, arXiv:2004.11362) on the
# fused representation, using the Step label to define positive pairs. This
# directly targets the accuracy gap: CE only tells the model "this row is
# class 5", SupCon additionally pulls every class-5 row's fused vector
# together and pushes other classes apart *within every batch*, which is a
# much denser training signal per row than CE alone on a ~1.5k-row dataset.
STAGE1_SUPCON_WEIGHT = 0.15
STAGE1_SUPCON_TEMPERATURE = 0.10   # Khosla et al.'s recommended range (0.07-0.1)
STAGE1_HARD_NEGATIVE_MARGIN = 0.25  # increased from 0.20
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
# Logit adjustment (Menon et al., "Long-tail learning via logit adjustment",
# ICLR 2021): shifts each class's TRAINING logit by tau * log(class_prior)
# so the model is explicitly pushed to leave a margin proportional to how
# rare a class is, on top of (not instead of) the existing capped inverse-
# frequency weights -- those lose information once STAGE1_MAX_CLASS_WEIGHT
# clips them, logit adjustment does not saturate the same way. Applied ONLY
# inside the training loss (graph_encoder.py -> Stage1Classifier.loss);
# argmax at eval/test time uses the model's raw, unadjusted logits exactly
# as prescribed by the paper's train-time variant, so evaluate.py needs no
# changes at all.
# DISABLED on measured evidence. A controlled ablation on this dataset's real
# class counts (104/301/183/114/16/550/65/0/14/154 train vs the real test
# counts), 8 seeds, difficulty tuned so the baseline lands near the real ~0.78:
#
#   config                              acc      macro-F1
#   baseline (nothing)                  0.5732   0.3890
#   class weights only                  0.5471   0.3900
#   focal only                          0.5648   0.3827
#   weights + focal                     0.5653   0.4081   <- best macro-F1
#   weights + focal + LA (was default)  0.4277   0.3650   <- -14.6pt accuracy
#   weights + LA                        0.2873   0.2891
#
# Logit adjustment costs accuracy on top of the class weights rather than
# adding to them -- which is what this file's own comment warned about ("If
# combined with STAGE1_USE_STEP_CLASS_WEIGHTS over-corrects"). Menon et al.
# present logit adjustment as an ALTERNATIVE to re-weighting, not a companion
# to it. weights + focal is the configuration to keep.
#
# Set True to re-enable (the zero-count-class hazard it used to carry is now
# neutralized in stage1_gnn_train.py regardless).
STAGE1_USE_LOGIT_ADJUSTMENT = False
STAGE1_LOGIT_ADJ_TAU = 1.0        # paper's default value. If combined with
                                   # STAGE1_USE_STEP_CLASS_WEIGHTS over-
                                   # corrects (majority-class recall drops a
                                   # lot), first try lowering this to ~0.5
                                   # before touching the weights -- see
                                   # STAGE1_IMPROVEMENTS.md.

# Asymmetric Loss for multi-label MCP prediction (Ridnik & Ben-Baruch et
# al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021):
# unlike symmetric focal BCE, ASL focuses/down-weights positives and
# negatives independently (gamma_pos != gamma_neg) and additionally hard-
# shifts very-easy negative probabilities to zero before their loss term
# is computed (asl_clip). Purpose-built for exactly the failure mode MCP
# prediction has here: 11 tools, most rows positive for only 1-2 of them,
# so "easy negatives" vastly outnumber positives for every class.
#
# REGRESSION, DIAGNOSED AND FIXED (see STAGE1_IMPROVEMENTS.md "Regression
# found and fixed" for the full account): this was shipped with
# STAGE1_MCP_LOSS_TYPE="asl" as the default and it collapsed MCP micro-F1
# from ~0.70 to ~0.22 in the very next real training run -- the model
# predicted all 11 tools positive on every single test row. Root cause was
# two-fold: (1) graph_encoder.py's asymmetric_loss() didn't detach its
# focusing term from autograd the way the official ASL implementation
# does, and (2) even after fixing that, a reproduction showed the paper's
# large-scale-dataset default gamma_neg=4.0 is simply too aggressive for
# a dataset this small trained this briefly -- it still runs away to a
# ~99% predicted-positive rate, just more slowly. Both are now fixed
# (asymmetric_loss() detaches correctly; gamma_neg default lowered to
# 2.0), but given the evidence that ASL is a worse fit than the
# previously-proven focal BCE for THIS codebase's data regime, the
# default has been reverted to "focal" -- the loss that was already
# achieving ~70% MCP micro-F1 before any of this audit's changes. ASL
# stays available, correctly implemented, for anyone who wants to
# deliberately A/B test it (set this to "asl" and watch predicted-
# positive-rate per epoch, not just the loss number, for the same
# collapse pattern).
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
# representation (post cross-modal fusion, pre classification heads): an
# AUXILIARY loss term computed on a convex combination of two random
# examples' fused vectors plus soft-mixed one-hot/multi-hot targets. Cheap
# (reuses the batch's already-computed fused_h; only re-runs the two small
# linear heads on the mixed vector) and directly targets the params:rows
# mismatch flagged elsewhere in this file (22M trainable params / ~1.5k
# train rows): smooths the decision boundary between classes instead of
# letting the model memorize point examples.
#
# TURNED ON: the first real post-loss-fix training run gave direct
# evidence of overfitting, not just the a-priori params:rows argument --
# by epoch 40/80 the printed per-epoch train loss (both step and mcp
# components) had dropped to ~0.01-0.03 while val/test metrics were still
# 10-30 points below target and val_step_acc oscillated epoch-to-epoch
# (0.55 -> 0.80 -> 0.69 -> 0.80 in consecutive epochs) well after train
# loss had already flattened near zero -- the classic signature of a model
# with far more capacity than the ~1.5k-row training set can constrain.
# Still worth A/B testing (compare one run with this False against one
# with it True on the same split) rather than assuming it strictly helps,
# but the earlier "wait and see" default is no longer the more
# conservative choice given this evidence -- see STAGE1_IMPROVEMENTS.md.
STAGE1_USE_MANIFOLD_MIXUP = True
STAGE1_MIXUP_ALPHA = 0.2
STAGE1_MIXUP_WEIGHT = 0.3

# Decoupled classifier re-balancing (Kang et al., "Decoupling Representation
# and Classifier for Long-Tailed Recognition", ICLR 2020, arXiv:1910.09217).
# Their central finding: jointly training representation + classifier under
# class-balancing (re-weighting/re-sampling) *hurts* the representation --
# imbalance itself is not what makes representations bad. The fix is to
# split training into two phases instead of fighting both problems with one
# loss: (1) learn the representation under the natural/instance-balanced
# distribution -- exactly what train_one_split's main loop above already
# does; (2) FREEZE that representation and re-train only the classifier
# (here: step_head + mcp_head) with class-balanced sampling, undoing the
# recency/majority bias without touching the feature space that phase 1
# spent its epochs learning. Directly targets this codebase's specific
# symptom: step_macro_f1 sitting far below step_accuracy (0.582 vs 0.765 in
# the Round-2 test run) -- exactly the "good representation, majority-biased
# classifier" pattern the paper describes, not a representation problem.
# Cheap: only two small Linear heads get gradients, backbone forward runs
# under no_grad. Never regresses silently -- kept only if it beats the
# pre-retrain val score (see retrain_classifier_heads() in
# training/stage1_gnn_train.py), same pattern as the SWA check below.
STAGE1_USE_DECOUPLED_RETRAIN = True
STAGE1_DECOUPLED_EPOCHS = 15
STAGE1_DECOUPLED_LR = 5e-4

# ROUND 4 (see STAGE1_IMPROVEMENTS.md): the first real Round-3 run landed
# step accuracy at 0.7649 test -- identical to Round 2's number, so SupCon/
# decoupled-retrain/capacity-cut didn't move it either way -- while MCP
# samples-F1 (0.7109) and subset accuracy (0.5336) both comfortably beat the
# paper (0.64 / 0.4888). The concrete, causally suggestive data point: the
# paper's OWN Step Model uses no graph at all (frozen GPT-2 + CNN on text
# only) and gets 82.87% -- 6.4 points ABOVE this graph-augmented model. That
# is a real signal the graph branch may currently be injecting noise into
# Step prediction specifically (MCP tool availability plausibly benefits
# more from graph/structural grounding than "which step type comes next,"
# which is largely determined by the strategy text itself).
#
# Fix: a single LEARNABLE gate (sigmoid-parameterized scalar) applied to
# every graph-derived term in the fusion concat (graph_proj, sem2graph_out,
# graph2sem_out, interaction) -- NOT to semantic_proj, which always enters
# fusion at full strength. Initialized low (graph starts as a genuine
# "add-on," per the user's own framing) but is a trainable parameter, not a
# fixed weight -- unlike a hand-picked constant, gradient descent can grow
# it back up if the graph does prove useful for a given prediction, so this
# doesn't foreclose graph-conditioning, it just removes the current
# hard-imposed assumption that graph and text contribute equally by
# default. One knob, easy to A/B against STAGE1_USE_GRAPH_GATE=False.
STAGE1_USE_GRAPH_GATE = True
STAGE1_GRAPH_GATE_INIT = 0.25   # legacy single-gate init; kept as the
                                  # fallback when per-head gating is disabled.

# ROUND 7 -- PER-HEAD GRAPH GATES (this is now empirically forced, not a
# hypothesis). The Round-6 run gave the decisive measurement: letting the
# SINGLE shared gate actually move (12x LR) pushed Step accuracy UP
# (0.7687 -> 0.7836) while pushing MCP micro-F1 DOWN (0.7036 -> 0.6539).
# One shared scalar cannot satisfy both heads at once -- Step wants LESS
# graph (its label is largely determined by the strategy wording), MCP wants
# MORE graph (which tools are usable depends on services/findings//state in
# the graph). The shared gate was therefore being pulled in two directions
# and settled on a compromise that is wrong for both.
#
# Fix: give each head its own gate over the graph-derived fusion terms, and
# run the (small) fusion MLP once per head. This is standard multi-task
# practice -- per-task gating over a shared bottom is exactly the Multi-gate
# Mixture-of-Experts formulation (Ma et al., "Modeling Task Relationships in
# Multi-task Learning with Multi-gate Mixture-of-Experts", KDD 2018), which
# exists precisely for tasks that conflict over a shared representation.
# Initialized asymmetrically in the direction the data already points:
# Step semantic-dominant, MCP graph-dominant. Both remain trainable, so
# gradient descent can still overrule these priors.
STAGE1_USE_PER_HEAD_GRAPH_GATE = True
STAGE1_GRAPH_GATE_INIT_STEP = 0.15   # Step: text-dominant, graph assists
STAGE1_GRAPH_GATE_INIT_MCP = 0.60    # MCP: graph-dominant, text assists


# ---------------------------------------------------------------------------
# Should the graph gate also attenuate the GRAPH->TEXT cross-attention term?
# ---------------------------------------------------------------------------
# The gate currently scales FOUR fusion terms: graph_proj, sem2graph_out,
# graph2sem_out and interaction. Three of those are genuinely graph-derived.
# `graph2sem_out` is not, quite: it is
#
#     cross_attn(query=graph_proj, key=value=semantic_token_kv)
#
# i.e. a weighted sum of TEXT values. The graph only chooses the attention
# WEIGHTS; the content that comes out is text. Attenuating it by the graph
# gate therefore throws away graph-*selected* text -- arguably the single most
# useful product of the fusion -- rather than throwing away graph content.
#
# Why this matters now: across all 5 folds the STEP gate converges to ~0.04
# from its 0.15 init. At that value four of the five concatenated blocks enter
# the fusion MLP at 4% of their natural scale, so the step head is reading
# essentially `semantic_proj` alone -- the entire cross-attention apparatus is
# switched off for step, including the text-side half of it. That is a
# plausible reason step accuracy is pinned near 0.79 regardless of graph
# encoder (it is also why swapping GINE->GATv2 is not expected to move step).
#
# Set to False to leave graph2sem_out UNGATED, so the step head keeps its
# graph-selected text signal even when the gate closes on graph content.
# Default True preserves today's measured behavior -- this is a one-variable
# experiment to run, not a change to make blind.
STAGE1_GATE_GRAPH2SEM = os.environ.get("STAGE1_GATE_GRAPH2SEM", "1") not in ("0", "false", "False")
# ROUND 6 (see STAGE1_IMPROVEMENTS.md): the first real run with the gate
# showed it moving only 0.250 -> 0.232 (~7% relative) over the 44 epochs
# before early stopping -- far too slow to have reached wherever its actual
# optimum is within the training budget, which makes that run's result
# (mixed: step macro-F1 +7.2pt, MCP subset accuracy -3.4pt) inconclusive
# rather than a real verdict on whether the graph helps or hurts. The gate
# is a single scalar with a short, cheap gradient path to the loss -- there
# is no reason for it to move at the same pace as the rest of a 14M-
# parameter model. Giving it its own, much higher learning rate (a separate
# optimizer param group, see train_one_split) lets it actually reach
# equilibrium in the same epoch budget, so the NEXT run's gate trajectory
# is a real signal instead of a truncated one.
STAGE1_GRAPH_GATE_LR_MULT = 12.0

# Stochastic Weight Averaging: instead of keeping only the single best-val
# checkpoint (noisy signal on a 239-example val split -- see log epoch-to-
# epoch score oscillation between 0.65 and 0.76), average the weights of
# the top-K checkpoints by val score and keep whichever (single-best vs.
# SWA) scores higher on val. LayerNorm/GraphNorm have no running batch
# stats, so plain weight averaging is safe here without a recalibration pass.
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
# hardcoded 2e-6 (env var STAGE2_SAFE_LR's default) -- the comment "Reduced
# for FP16 stability" describes a change that was made directly in the
# training script and never back-ported here, so editing this constant had
# zero effect on a real run. Set to the value actually exercised by every
# real Stage-2 run so far; stage2_sft_qwen.py now reads this as the
# STAGE2_SAFE_LR fallback default instead of a separate literal.
STAGE2_LR = 2e-6
STAGE2_EPOCHS = 8                # Short bridge stage before GRPO; avoid overtraining
STAGE2_BATCH_SIZE = 1
STAGE2_GRAD_ACCUM = 16
# Upweight the loss on the "New step" label tokens (and MCP tokens) relative
# to the free-text explanation tokens. WHY: Stage 2's SFT loss is HuggingFace's
# uniform-average cross-entropy over the ENTIRE JSON target, which is
# ~200-300 tokens dominated by the free-text "Step explanation". The "New
# step" canonical label is only ~10-20 of those tokens, so the classification
# signal is heavily diluted -- the model can minimize loss by writing fluent
# explanations and defaulting to the majority "Exploit" class, which is
# EXACTLY what the first real run's confusion matrix showed (predicts
# "Exploit" ~136x when gold=92; rare classes 4/6/8 -> 0 correct; step
# macro-F1 collapsed to 0.47). Upweighting the step-value span (already
# computed as `step_span` in SFTDataset, but previously unused for loss)
# forces the model to actually learn the classification, not just the prose.
# 1.0 = original uniform behavior; >1.0 weights the step tokens more. Applied
# via a manual weighted cross-entropy in forward_batch (train path only;
# validation still uses the plain loss for its own comparability).
STAGE2_STEP_TOKEN_LOSS_WEIGHT = 5.0

# ---------------------------------------------------------------------------
# B1 — feed Stage-1's prediction into the Stage-2 prompt
# ---------------------------------------------------------------------------
# Stage 2 currently RE-SOLVES classification from scratch: build_prompt gives
# it machine + strategy + graph prefix, and build_target asks for step +
# explanation + MCP. It never sees what Stage 1 already decided. That is the
# same problem solved twice, and it is why STAGE2_STEP_TOKEN_LOSS_WEIGHT has
# to be 5.0 -- the explanation tokens otherwise drown the label.
#
# With Stage 1 now at ~0.80 step / ~0.73 MCP samples-F1, its prediction is a
# strong prior worth conditioning on, letting Stage 2 specialize on the thing
# it is uniquely good at: the free-text explanation.
#
# ERROR PROPAGATION is the obvious risk -- a confidently wrong Stage-1 step
# would get an eloquent justification. Mitigated by SCHEDULED SAMPLING (Bengio
# et al., NeurIPS 2015): during training the prompt carries the GOLD step with
# probability STAGE2_GOLD_HINT_PROB and Stage 1's actual PREDICTION otherwise,
# so the model sees realistic wrong hints during training and learns it may
# override them. At inference only the prediction is available.
#
# NOTE: an earlier version of this repo had exactly this idea as a `mask_hint`
# parameter with a `stage1_hint` field -- but it was DEAD CODE (the function
# that produced the hint was never called anywhere, so every prompt was
# identical). The idea was right; it simply never ran.
STAGE2_USE_STAGE1_HINT = os.environ.get("STAGE2_USE_STAGE1_HINT", "1") not in ("0", "false", "False")
STAGE2_GOLD_HINT_PROB = float(os.environ.get("STAGE2_GOLD_HINT_PROB", "0.5"))

# ---------------------------------------------------------------------------
# B2 — Stage-3 reward rebalanced toward explanation
# ---------------------------------------------------------------------------
# Was 0.01 / 0.33 / 0.33 / 0.33 (format / step / mcp / explanation), i.e. GRPO
# spent two thirds of its signal re-optimizing classification that Stage 1
# already does better. Stage 1 beats the Pen-Strategist paper on MCP
# (0.7287 samples-F1 vs 0.64) and is within ~3pt on step, so Stage 3's job is
# the explanation. Step and MCP stay in the reward as GUARDRAILS -- non-zero
# so the policy cannot trade classification away, small enough that
# explanation dominates the gradient.
STAGE3_W_FMT = float(os.environ.get("STAGE3_W_FMT", "0.01"))
STAGE3_W_STEP = float(os.environ.get("STAGE3_W_STEP", "0.15"))
STAGE3_W_MCP = float(os.environ.get("STAGE3_W_MCP", "0.15"))
STAGE3_W_EXP = float(os.environ.get("STAGE3_W_EXP", "0.69"))

# ---------------------------------------------------------------------------
# B3 — entailment-based explanation reward
# ---------------------------------------------------------------------------
# The current explanation reward is 0.60*BGE-cosine + 0.20*lexical +
# 0.10*step-support + 0.10*tool-support. BGE cosine between any two on-topic
# pentest explanations is high, so it barely separates good reasoning from
# fluent-but-wrong reasoning -- GRPO is optimizing something close to noise on
# its largest-weighted term.
#
# An NLI model asks a sharper question: does the explanation ENTAIL the step
# that was chosen? That is much closer to what the LLM judge's "relevance" and
# "technical accuracy" dimensions actually measure, while staying cheap enough
# for the RL sampling loop and remaining a DIFFERENT model from the judge (so
# the judge stays an independent test-time measure and is not optimized
# against). Falls back to the existing proxy if the model cannot be loaded.
STAGE3_USE_NLI_REWARD = os.environ.get("STAGE3_USE_NLI_REWARD", "1") not in ("0", "false", "False")
STAGE3_NLI_MODEL = os.environ.get("STAGE3_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
STAGE3_NLI_WEIGHT = float(os.environ.get("STAGE3_NLI_WEIGHT", "0.45"))

STAGE2_VAL_SPLIT = 0.15          # 15% held-out for validation
STAGE2_EARLY_STOP_PATIENCE = 3   # Stop quickly once validation stops improving
STAGE2_GRAD_CLIP = 1.0
STAGE2_WARMUP_RATIO = 0.05       # Short warmup for the compact bridge stage
STAGE2_WEIGHT_DECAY = 1e-4

# CORRECTED (Stage-2 regression post-mortem): the GraphPrefixAdapter is a
# ~14.6M-parameter RANDOMLY-INITIALIZED cross-attention resampler, while the
# LoRA adapters it shares an optimizer with are low-rank deltas on an
# already-pretrained 14B model. Training both at STAGE2_LR=2e-6 is correct
# for LoRA and catastrophically too slow for the resampler: AdamW's update
# magnitude is ~lr per step (the gradient is normalized by sqrt(v)), so over
# a typical 744-step run a parameter can travel at most ~744*2e-6 = 1.5e-3.
# The learned queries are initialized at randn*0.02, so they move only ~7%
# of their init scale and stay effectively RANDOM. Random queries attend
# near-uniformly over the node set, which collapses all GRAPH_PREFIX_TOKENS
# outputs toward the same vector (the mean node state) -- strictly less
# informative than the previous adapter's 8 distinct fixed random
# projections. This is what regressed Stage 2 from 0.7950 -> 0.6151
# val_step_acc and 88.43% -> 66.42% test Step Exact Match.
#
# Fix (same shape as STAGE1_GRAPH_GATE_LR_MULT, which fixed the identical
# class of bug on Stage 1's graph gates): give the adapter its own AdamW
# param group at STAGE2_LR * STAGE2_ADAPTER_LR_MULT. 50x -> 1e-4, the
# standard LR for training a small projector/resampler from scratch on top
# of a frozen LLM (Flamingo/BLIP-2 train their resamplers at 1e-4). The LoRA
# params keep 2e-6 and are unaffected.
STAGE2_ADAPTER_LR_MULT = 50.0

# CORRECTED (user-directed, matching the main branch's working Stage-3
# regime): the previous 1e-7 / 600-step setting produced a DEAD run -- the
# training log showed kl=0.0000 at steps 1, 50, 100 (the policy literally
# never moved) because 1e-7 on a 128M-param LoRA over only 600 steps /
# grad_accum=4 = 150 real updates is far too little signal to change the
# model at all. The main branch (github.com/sagarikachavan/Research) ran
# Stage 3 at LR 2e-6 with 3000 steps and group size 4, which actually moves
# the policy. Restored those values. STAGE3_STEPS also drives the
# CosineAnnealingLR T_max in the training loop, so this is the schedule
# length, not just an iteration cap.
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
# bounds the upper side of the ratio just as tightly as the lower side, which
# can prematurely suppress the update for a token whose probability should
# increase a lot (a correct-but-currently-low-probability completion) --
# DAPO's own ablation found this causes entropy collapse (the policy gets
# stuck exploiting a narrow set of high-probability tokens instead of
# exploring). The fix widens only the UPPER clip bound (higher eps), leaving
# the lower bound unchanged so downweighting a bad completion is not
# affected. Cheap, additive, and orthogonal to the dual-clip fix above (dual-
# clip bounds the negative-advantage/exploding-ratio quadrant; clip-higher
# widens the positive-advantage upper bound) -- see
# training/stage3_grpo_rl.py's PPO loss for where both are applied together.
STAGE3_USE_CLIP_HIGHER = True
STAGE3_CLIP_HIGH = 0.28          # DAPO paper's own reported high/low split
                                  # (low ~0.2, high ~0.28) for a comparable
                                  # PPO-clip setup; STAGE3_PPO_CLIP above is
                                  # used unchanged as the lower bound.
STAGE3_GRAD_ACCUM = 4
# CORRECTED (architecture re-audit): this was 1.0 while
# training/stage3_grpo_rl.py's actual grad-norm clip call had 0.5 hardcoded
# directly in the loop, silently ignoring this constant entirely (along with
# every other STAGE3_* constant below -- see the wiring fix in
# stage3_grpo_rl.py's main(), which now reads all of them as fallback
# defaults for their STAGE3_SAFE_* env-var overrides). Set to the value that
# was actually exercised by every real Stage-3 run so far.
STAGE3_GRAD_CLIP = 0.5

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
# WIRED (architecture re-audit): this constant and STAGE3_KL_HARD_CAP below
# were imported by stage3_grpo_rl.py but never actually referenced again --
# main() re-declared its own hyperparameters from os.environ with
# independently hardcoded defaults, so editing these two had zero effect on
# a real run despite the detailed rationale below describing them as the fix
# for an observed incident. Both are now read as the fallback default for a
# STAGE3_SAFE_* env override (SAFE_DUAL_CLIP / SAFE_KL_HARD_CAP in main())
# and actually used in the PPO loss / KL-cap check. Not yet validated by a
# real training run -- see STAGE2_STAGE3_IMPROVEMENTS.md.
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
        "SEMANTIC_LM_NAME", "SEMANTIC_CNN_DIM", "SEMANTIC_CNN_KERNELS",
        "NODE_AUX_DIM", "EDGE_ATTR_DIM",
        "STAGE1_SEPARATE_SEMANTIC_TOWERS", "STAGE1_SIMPLE_FUSION",
        "STAGE1_USE_LABEL_PROTOTYPES", "STAGE1_STEP_CONDITIONED_MCP",
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
        "STAGE1_N_FOLDS", "STAGE1_N_SEEDS", "STAGE1_USE_KNN_MEMBER",
        "STAGE1_USE_STACKING", "STAGE1_TRAIN_FINAL_ON_ALL",
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
        "STAGE2_USE_STAGE1_HINT", "STAGE2_GOLD_HINT_PROB",
    ],
    "STAGE 3": [
        "STAGE3_STEPS", "STAGE3_GROUP_SIZE", "STAGE3_LR",
        "STAGE3_W_FMT", "STAGE3_W_STEP", "STAGE3_W_MCP", "STAGE3_W_EXP",
        "STAGE3_USE_NLI_REWARD", "STAGE3_NLI_WEIGHT",
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
