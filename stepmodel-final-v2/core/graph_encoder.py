"""Stage-1 hybrid graph + semantic CNN classifier.

Design:
- typed graph encoder (GATv2 by default, GINE selectable via
  STAGE1_GNN_TYPE) with GraphNorm and gated residual blocks;
- frozen-GPT-2-token-features + multi-kernel CNN is inspired by the
  Pen-Strategist paper's (arXiv:2605.04499) Step Model, which does the same
  frozen-GPT-2-plus-CNN feature extraction -- but the paper has no graph
  encoder at all and uses TWO SEPARATE convolutional encoders, one per head
  (its own Sec. 4.2.2: "we apply two separate convolutional encoders...
  One representation is used for step classification, and the other for
  MCP server prediction"). This project deliberately uses ONE SHARED CNN
  encoder instead, specifically because the semantic representation is then
  fused with the GNN's graph representation (see below) before the heads
  split -- a shared representation is what needs to exist for that fusion
  step to have a single semantic vector to fuse against. The graph encoder,
  the fusion, and the resulting shared-semantic-representation design are
  this project's own extension beyond the paper, not a reproduction of it;
- semantic/graph fusion is private to Stage 1 and feeds independent Step/MCP heads;
- the raw graph representation is exactly 512-d and is the sole graph input to Stage 2/3;
- Step gets a stronger private tower while MCP keeps an independent tower.

--------------------------------------------------------------------------
FUSION FIX (see STAGE1_IMPROVEMENTS.md for the full write-up): the previous
"cross-attention fusion" ran nn.MultiheadAttention with a query sequence of
length 1 (the pooled semantic vector) against a key/value sequence of
length 1 (the pooled graph vector). Softmax over a single key is
*identically* 1.0 regardless of the query's content, so that block was
mathematically guaranteed to return `out_proj(v_proj(graph_proj))` --
a fixed linear function of the graph vector alone, with **zero** gradient
ever reaching the query-side parameters. It looked like cross-modal fusion
but the two modalities never actually interacted through it (verified with
a standalone repro: two different semantic queries against the same
graph key/value produced byte-identical output, attention weight exactly
1.0 in both cases).

Replaced with real cross-attention: the graph branch now also exposes its
per-node hidden states (a genuine multi-token sequence per graph, via
GraphEncoder.forward_with_nodes) and the semantic branch exposes its
per-token GPT-2 embeddings (already computed upstream) as a genuine
multi-token sequence. Two-way cross-attention -- semantic-pooled-vector
queries the graph's node tokens, graph-pooled-vector queries the strategy
text's token embeddings -- lets each modality actually condition on the
fine-grained content of the other, plus a Hadamard/bilinear interaction
term (Kim et al., "Hadamard Product for Low-rank Bilinear Pooling", ICLR
2017) for a cheap explicit multiplicative interaction. See
Stage1Classifier.encode_and_predict below.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, GraphNorm, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

import math

from config import (
    GNN_HIDDEN, GNN_LAYERS, GNN_OUT_DIM, FUSION_HIDDEN,
    TEXT_EMB_DIM, STEP_LABELS, MCP_LABELS, GNN_DROPOUT,
    EDGE_ATTR_DIM, NODE_AUX_DIM, SEMANTIC_CNN_DIM, SEMANTIC_CNN_KERNELS,
    SEMANTIC_CNN_DROPOUT, STAGE1_EDGE_DROPOUT, STAGE1_NODE_FEAT_DROPOUT,
    STAGE1_USE_GRAPH_GATE, STAGE1_GRAPH_GATE_INIT,
    STAGE1_USE_PER_HEAD_GRAPH_GATE,
    STAGE1_GRAPH_GATE_INIT_STEP, STAGE1_GRAPH_GATE_INIT_MCP,
    STAGE1_GNN_TYPE, GNN_HEADS, STAGE1_GATE_GRAPH2SEM,
    N_STEP_PHASES, STAGE1_PHASE_LOSS_WEIGHT,
)


def _inv_sigmoid(p: float) -> float:
    """Logit of p, so sigmoid(logit) == p at initialization."""
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))
from data_utils import CONTEXT_COLUMNS

NODE_FEAT_DIM = TEXT_EMB_DIM + NODE_AUX_DIM


_STRUCT_SMOOTH_CACHE = {}


def build_structured_smoothing_targets(eps: float, support_mask, temp: float = 0.10):
    """(C, C) soft-target matrix for similarity-structured label smoothing.

    Row i is the training target for a true label i: (1-eps) on i, and eps
    distributed over the OTHER classes in proportion to how semantically
    similar their label TEXT is to label i (softmax over BGE cosine at
    temperature `temp`). Classes with no training support get zero mass.

    Why not uniform smoothing: uniform spreads eps equally onto every class,
    including semantically unrelated ones AND zero-support ones. The
    zero-support case is not hypothetical here -- uniform smoothing feeding
    mass to class 7 is precisely what let logit adjustment blow that class's
    logit up (see stage1_gnn_train.py's step_log_priors comment). Structured
    smoothing keeps the regularization benefit while putting the mass where a
    confusion would actually be plausible. Mueller et al., "When Does Label
    Smoothing Help?" (NeurIPS 2019).
    """
    import numpy as np
    key = (round(float(eps), 6), round(float(temp), 6), tuple(bool(b) for b in support_mask))
    if key in _STRUCT_SMOOTH_CACHE:
        return _STRUCT_SMOOTH_CACHE[key]
    from data_utils import _embed_texts
    C = len(STEP_LABELS)
    E = np.asarray(_embed_texts(list(STEP_LABELS)), dtype=np.float64)
    sim = E @ E.T
    sup = np.asarray(support_mask, dtype=bool)
    T = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        others = np.array([j for j in range(C) if j != i and sup[j]], dtype=int)
        T[i, i] = 1.0 - eps
        if len(others) == 0:
            T[i, i] = 1.0
            continue
        w = np.exp((sim[i, others] - sim[i, others].max()) / max(temp, 1e-6))
        w = w / w.sum()
        T[i, others] = eps * w
    out = torch.tensor(T, dtype=torch.float32)
    _STRUCT_SMOOTH_CACHE[key] = out
    return out


def mask_unsupported_logits(step_logits, support_mask):
    """Set logits of zero-training-support classes to -inf.

    STEP_LABELS is intentionally NOT reindexed -- that would invalidate every
    saved checkpoint and the Stage-2/3 label contract. Masking achieves the
    same effect at inference without touching the label space.
    """
    if support_mask is None:
        return step_logits
    m = torch.as_tensor(support_mask, dtype=torch.bool, device=step_logits.device)
    return step_logits.masked_fill(~m.view(1, -1), float("-inf"))


def asymmetric_loss(logits, targets, gamma_neg=2.0, gamma_pos=0.0,
                     clip=0.05, eps=1e-8, class_weights=None):
    """Asymmetric Loss for multi-label classification.

    Ridnik, Ben-Baruch et al., "Asymmetric Loss For Multi-Label
    Classification", ICCV 2021 (https://arxiv.org/abs/2009.14119).

    --------------------------------------------------------------------
    POST-DEPLOYMENT FIX (see STAGE1_IMPROVEMENTS.md "Regression found and
    fixed"): the first version of this function computed `focal_weight`
    directly from `logits` with no `torch.no_grad()` guard, so autograd
    back-propagated through the focusing term itself as well as through
    the raw loss term. The official ASL reference implementation always
    detaches this term (their `disable_torch_grad_focal_loss` option,
    on by default) specifically to prevent that -- without it, a
    confidently-WRONG prediction can start to suppress its own gradient
    (via the compounding derivative of `(1-pt)^gamma`) just as effectively
    as a confidently-correct one, which is a runway in the wrong
    direction, not a stability improvement. This was diagnosed from a
    real training run where MCP micro-F1 collapsed from ~0.70 to ~0.22:
    the saved predictions CSV showed the model predicting all 11 MCP
    tools positive on literally every single test row -- a full collapse
    to "always predict positive," which a non-detached, highly asymmetric
    (gamma_neg=4, gamma_pos=0) focusing term will drive a model toward
    once it drifts even slightly in that direction, because the negative
    term's gradient then shrinks faster than the positive term's does.

    Fixed here by wrapping the focusing-term computation in
    `torch.no_grad()` (matching the official implementation), AND by
    lowering the default `gamma_neg` from the paper's large-scale-dataset
    default of 4.0 to 2.0. Both changes were verified necessary: a toy
    reproduction (small dataset, few epochs, LR matching this codebase's
    STAGE1_LR) showed gamma_neg=4 still drifts toward ~99% predicted-
    positive rate even WITH detachment -- it converges more slowly toward
    the same bad place, not to a stable one. gamma_neg=2 recovers instead
    of collapsing in that same test. See config.py's STAGE1_MCP_LOSS_TYPE
    comment: given this, ASL is no longer the *default* MCP loss for this
    codebase (reverted to the previously-proven symmetric focal BCE) --
    this function is kept available and correctly implemented for anyone
    who wants to opt in and A/B test it, with a safer default gamma_neg.
    --------------------------------------------------------------------

    Unlike symmetric focal BCE (which uses one gamma for both the positive
    and negative term), ASL uses independent focusing for positives/
    negatives (gamma_pos, gamma_neg) and additionally shifts easy-negative
    probabilities down by `clip` before computing their loss term, so a
    confidently-correct negative (the overwhelmingly common case for an
    11-way multi-label head where most rows are positive for 1-2 labels)
    contributes ~zero loss/gradient instead of the small-but-nonzero
    contribution focal BCE still gives it. Returns a scalar (mean over all
    elements); apply `class_weights` (per-label) the same way the previous
    focal-BCE call site did.
    """
    p = torch.sigmoid(logits)
    p_pos = p.clamp(min=eps, max=1.0 - eps)
    p_neg = (p - clip).clamp(min=0.0) if clip and clip > 0 else p
    p_neg = p_neg.clamp(min=eps, max=1.0 - eps)

    loss_pos = targets * torch.log(p_pos)
    loss_neg = (1.0 - targets) * torch.log(1.0 - p_neg)

    # Asymmetric focusing: probability-of-truth pt, per-element gamma.
    # Computed with NO gradient -- this is a fixed re-weighting of the
    # loss surface at the current iterate, not something we want autograd
    # differentiating through a second time (see the fix note above).
    with torch.no_grad():
        pt = p_pos * targets + p_neg * (1.0 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
        focal_weight = torch.pow((1.0 - pt).clamp(min=0.0), gamma)

    loss = -(loss_pos + loss_neg) * focal_weight
    if class_weights is not None:
        loss = loss * class_weights.view(1, -1)
    return loss.mean()


class GINEBlock(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.conv = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
            ),
            edge_dim=hidden,
            train_eps=True,
        )
        self.norm = GraphNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h, edge_index, batch, edge_attr):
        e = self.edge_encoder(edge_attr)
        y = self.conv(h, edge_index, e)
        y = self.norm(y, batch)
        y = F.gelu(y)
        # Gated residual connection for better gradient flow
        gate = self.gate(torch.cat([h, y], dim=-1))
        y = self.dropout(y)
        return h + gate * y


class GATv2Block(nn.Module):
    """Attention-based graph block; the Stage-1 default (STAGE1_GNN_TYPE).

    Interface is IDENTICAL to GINEBlock -- forward(h, edge_index, batch,
    edge_attr) -> h of the same shape -- so everything else in GraphEncoder
    (input_proj, the global attention pass, mean/max/attention pooling,
    out_proj, hidden width, dropout, edge dropout) is shared between the two
    and the switch changes exactly one thing: how a node aggregates its
    neighbours.

    GATv2 (Brody et al., "How Attentive are Graph Attention Networks?",
    ICLR 2022) computes a *dynamic* attention weight per edge -- unlike the
    original GAT, whose attention ranking is static with respect to the query
    node. The edge feature enters through `edge_dim`, meaning the PTT relation
    type (StateTransition / ActionUpdate / FindingUpdate / Prediction /
    self-loop) modulates HOW MUCH a neighbour is attended to.

    This differs from GINEBlock, where the encoded edge feature is added into
    the message CONTENT before aggregation. Neither is strictly better a
    priori; the choice is settled empirically here -- see config.py's
    STAGE1_GNN_TYPE comment for the measured numbers behind defaulting to
    this block.

    The edge encoder, GraphNorm, gated residual and dropout are kept
    byte-identical to GINEBlock so the comparison stays single-variable.
    """

    def __init__(self, hidden: int, edge_dim: int, dropout: float,
                 heads: int = GNN_HEADS):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(
                f"GNN_HIDDEN={hidden} must be divisible by GNN_HEADS={heads}"
            )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # concat=True with out_channels = hidden // heads keeps the block's
        # output width exactly `hidden`, matching GINEBlock.
        self.conv = GATv2Conv(
            hidden, hidden // heads, heads=heads, concat=True,
            dropout=dropout, edge_dim=hidden, add_self_loops=True,
        )
        self.norm = GraphNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h, edge_index, batch, edge_attr):
        e = self.edge_encoder(edge_attr)
        y = self.conv(h, edge_index, edge_attr=e)
        y = self.norm(y, batch)
        y = F.gelu(y)
        # Gated residual connection for better gradient flow
        gate = self.gate(torch.cat([h, y], dim=-1))
        y = self.dropout(y)
        return h + gate * y


class GraphEncoder(nn.Module):
    """Encode a PTT graph to a 512-d representation."""

    def __init__(self, in_dim=NODE_FEAT_DIM, hidden=GNN_HIDDEN,
                 out_dim=GNN_OUT_DIM, num_layers=GNN_LAYERS,
                 dropout=GNN_DROPOUT, edge_dim=EDGE_ATTR_DIM,
                 edge_dropout=STAGE1_EDGE_DROPOUT,
                 node_feat_dropout=STAGE1_NODE_FEAT_DROPOUT,
                 gnn_type=STAGE1_GNN_TYPE):
        super().__init__()
        # Training-only graph augmentation (both are no-ops in eval mode via
        # self.training / nn.Dropout). Cheap regularizer for a small dataset:
        # random edge drop prevents the GINE stack from leaning on any one
        # PTT transition, random node-feature-channel drop does the same for
        # the 387-dim title+aux node features.
        self.edge_dropout_p = float(edge_dropout)
        self.node_feat_dropout = nn.Dropout(node_feat_dropout) if node_feat_dropout > 0 else None
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        gnn_type = str(gnn_type).lower()
        if gnn_type == "gatv2":
            block_cls = GATv2Block
        elif gnn_type == "gine":
            block_cls = GINEBlock
        else:
            raise ValueError(
                f"Unknown STAGE1_GNN_TYPE={gnn_type!r}; expected 'gatv2' or 'gine'"
            )
        self.gnn_type = gnn_type
        self.blocks = nn.ModuleList([
            block_cls(hidden, edge_dim, dropout) for _ in range(num_layers)
        ])
        self.node_norm = nn.LayerNorm(hidden)
        # Multi-head attention pooling for better graph-level representation
        self.attn_pool = nn.MultiheadAttention(hidden, num_heads=8, dropout=dropout, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward_nodes(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.shape[1], EDGE_ATTR_DIM))
        if self.training and self.edge_dropout_p > 0 and edge_index.shape[1] > 1:
            # Never drop every edge of a batch -- keep at least one.
            keep = torch.rand(edge_index.shape[1], device=edge_index.device) > self.edge_dropout_p
            if bool(keep.any()):
                edge_index = edge_index[:, keep]
                edge_attr = edge_attr[keep]
        if self.node_feat_dropout is not None:
            x = self.node_feat_dropout(x)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, edge_index, batch, edge_attr)
        return self.node_norm(h)

    @staticmethod
    def _pad_nodes(h, batch):
        """Pack a flat (total_nodes, D) node tensor into (B, max_nodes, D)
        plus a boolean validity mask (B, max_nodes), True where a node is
        real (not padding). Factored out so both the internal attention
        pool below and Stage1Classifier's cross-attention fusion (see
        forward_with_nodes) build this padded view identically instead of
        risking two implementations drifting apart.
        """
        batch_size = batch.max().item() + 1
        node_counts = torch.bincount(batch, minlength=batch_size)
        max_nodes = int(node_counts.max().item())

        h_padded = torch.zeros(batch_size, max_nodes, h.shape[-1], device=h.device, dtype=h.dtype)
        mask = torch.zeros(batch_size, max_nodes, device=h.device, dtype=torch.bool)
        for i in range(batch_size):
            mask_i = batch == i
            nodes_i = h[mask_i]
            h_padded[i, :len(nodes_i)] = nodes_i
            mask[i, :len(nodes_i)] = True
        return h_padded, mask

    def forward_with_nodes(self, x, edge_index, batch, edge_attr=None):
        """Same computation as forward(), but additionally returns the
        per-node hidden states (padded) and their validity mask so a
        caller can do real token-level cross-attention against individual
        graph nodes instead of only the single pooled vector. forward()
        below is kept as a thin wrapper so every existing caller that
        expects a single pooled tensor (Stage 2/3's GraphPrefixAdapter,
        eval/evaluate.py -- both call `stage1.graph_encoder(...)` directly)
        keeps working unmodified.
        """
        h = self.forward_nodes(x, edge_index, batch, edge_attr=edge_attr)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)

        h_padded, mask = self._pad_nodes(h, batch)

        # Multi-head self-attention pooling over the REAL node sequence
        # (this one always was legitimate multi-token attention -- unlike
        # the old Stage1Classifier "cross-attention", every graph here has
        # more than one node in the vast majority of cases, so softmax has
        # more than one key to weigh).
        attn_out, _ = self.attn_pool(h_padded, h_padded, h_padded, key_padding_mask=~mask)
        attn_out = attn_out.masked_fill(~mask.unsqueeze(-1), 0)
        attn_pool = attn_out.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        pooled = self.out_proj(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))
        return pooled, h_padded, mask

    def forward(self, x, edge_index, batch, edge_attr=None):
        pooled, _h_padded, _mask = self.forward_with_nodes(x, edge_index, batch, edge_attr=edge_attr)
        return pooled


class SemanticCNNEncoder(nn.Module):
    """Temporal CNN over frozen LM token embeddings, inspired by (but not a
    reproduction of) the Pen-Strategist paper's (arXiv:2605.04499) Step
    Model.

    The paper uses frozen GPT-2 token-level embeddings followed by multiple
    convolution kernels and global max pooling -- that part we reuse. But
    the paper feeds those embeddings into TWO SEPARATE convolutional
    encoders, one per head, and has no graph representation at all. This
    project uses ONE SHARED semantic encoder for both heads, because the
    shared representation is fused with the GINE graph encoder's output
    (see Stage1Classifier below) before the Step/MCP heads split -- that
    fusion step is this project's own extension, and it needs a single
    semantic vector to fuse against, which is why the CNN encoder is shared
    rather than duplicated per head as in the paper. Enhanced with residual
    connections and better normalization.
    """

    def __init__(self, input_dim: int, out_dim: int = SEMANTIC_CNN_DIM,
                 kernels=SEMANTIC_CNN_KERNELS, dropout=SEMANTIC_CNN_DROPOUT):
        super().__init__()
        self.kernels = kernels  # Store kernel sizes for forward method
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, out_dim, kernel_size=k, padding=0),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for k in kernels
        ])
        self.norm = nn.LayerNorm(out_dim * len(kernels))
        self.proj = nn.Sequential(
            nn.Linear(out_dim * len(kernels), out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, token_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # token_embs: (B,L,D), mask: (B,L)
        x = token_embs.transpose(1, 2)  # (B,D,L)
        mask_f = mask.to(dtype=x.dtype).unsqueeze(1)
        pooled = []
        for conv, k in zip(self.convs, self.kernels):
            # Pad short sequences so every configured kernel is valid.
            if x.shape[-1] < k:
                x_conv = F.pad(x, (0, k - x.shape[-1]))
                m_conv = F.pad(mask_f, (0, k - mask_f.shape[-1]))
            else:
                x_conv, m_conv = x, mask_f
            y = conv(x_conv)  # Conv1d is now inside Sequential with BatchNorm, GELU, Dropout
            # A window is valid only when all of its source tokens are real.
            valid = F.conv1d(m_conv, x.new_ones(1, 1, k), stride=1).squeeze(1) >= float(k) - 1e-6
            y = y.masked_fill(~valid.unsqueeze(1), torch.finfo(y.dtype).min)
            pooled_k = y.max(dim=-1).values
            # BUG FIX: when a sample has fewer valid tokens than this
            # kernel's width (e.g. a row whose "New strategy"/"Strategy
            # explanation" are both empty -> the "empty empty" placeholder
            # text, ~2-3 GPT-2 tokens, shorter than kernel=5 or kernel=7),
            # EVERY window for that sample is invalid, so max() above
            # returns the raw finfo.min fill value itself for every
            # channel. That huge-magnitude constant then blows up the
            # LayerNorm variance computation a few lines down into inf/NaN,
            # which poisons the whole batch's loss and gradient (verified
            # with a repro: a single such row makes step_logits/mcp_logits
            # NaN for the entire batch). Zero those channels instead --
            # equivalent to "this kernel contributes nothing for this
            # sample" rather than "this kernel is catastrophically
            # confident about a value that doesn't exist".
            no_valid_window = ~valid.any(dim=-1, keepdim=True)  # (B,1)
            pooled_k = pooled_k.masked_fill(no_valid_window, 0.0)
            pooled.append(pooled_k)
        z = torch.cat(pooled, dim=-1)
        z = self.norm(z)
        return self.proj(z)


class ContextTextProjector(nn.Module):
    """Project the two BGE field embeddings into the graph fusion space."""

    def __init__(self, n_fields=None, field_dim=TEXT_EMB_DIM, out_dim=GNN_OUT_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear((len(CONTEXT_COLUMNS) if n_fields is None else n_fields) * field_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.08),
        )

    def forward(self, field_embs):
        return self.proj(field_embs.reshape(field_embs.shape[0], -1))


class Stage1Classifier(nn.Module):
    """Stage-1 semantic-CNN + GINE fusion classifier.

    Contract:
      New strategy + strategy explanation -> frozen GPT-2 token embeddings
      -> shared multi-kernel semantic CNN -> semantic representation.

      PTT graph -> typed GINE -> raw 512-d graph representation.

      The semantic and graph representations are fused only inside Stage 1
      and feed independent Step and MCP supervised heads.  The fused vector
      is private to Stage 1: Stage 2/3 receive *only* the raw 512-d GINE
      representation through GraphPrefixAdapter.
    """

    def __init__(self, edge_dim: int = EDGE_ATTR_DIM, semantic_input_dim: int = 768):
        super().__init__()
        self.graph_encoder = GraphEncoder(edge_dim=edge_dim)
        self.semantic_cnn = SemanticCNNEncoder(semantic_input_dim)

        # One shared semantic representation is deliberately used for both
        # prediction heads, so there is a single semantic vector for the
        # graph-fusion step below to fuse against. NOTE: this is NOT what
        # the Pen-Strategist paper does -- its Step Model has no graph and
        # uses two separate CNN encoders, one per head (see SemanticCNNEncoder
        # docstring above). The graph encoder and the shared-then-fused
        # design are this project's own extension.
        self.semantic_dim = SEMANTIC_CNN_DIM
        self.graph_dim = GNN_OUT_DIM
        self.fused_dim = FUSION_HIDDEN
        fusion_half = FUSION_HIDDEN // 2

        # --- Real cross-modal fusion -----------------------------------
        # Two token-level projections so each modality has an actual
        # multi-token sequence to be queried against (not just its own
        # pooled vector -- see the module docstring for why the previous
        # version's "cross-attention" never fused anything).
        #   * graph_node_proj: projects GraphEncoder's per-node hidden
        #     states (GNN_HIDDEN-d) into fusion space -- one real token
        #     per PTT graph node.
        #   * semantic_token_proj: projects the per-token frozen GPT-2
        #     embeddings (already computed upstream for the CNN branch)
        #     into fusion space -- one real token per strategy/explanation
        #     word-piece.
        self.graph_node_proj = nn.Sequential(
            nn.Linear(GNN_HIDDEN, fusion_half),
            nn.LayerNorm(fusion_half),
            nn.GELU(),
        )
        self.semantic_token_proj = nn.Sequential(
            nn.Linear(semantic_input_dim, fusion_half),
            nn.LayerNorm(fusion_half),
            nn.GELU(),
        )

        # Bidirectional cross-attention: each modality's pooled vector
        # queries the OTHER modality's real token sequence, so the output
        # actually depends on both the query's content (which key/value
        # pairs get high softmax weight) and the key/value content -- the
        # property the old length-1/length-1 attention provably lacked.
        self.cross_attn_sem2graph = nn.MultiheadAttention(
            embed_dim=fusion_half, num_heads=8, dropout=0.10, batch_first=True
        )
        self.cross_attn_graph2sem = nn.MultiheadAttention(
            embed_dim=fusion_half, num_heads=8, dropout=0.10, batch_first=True
        )

        # Cheap explicit multiplicative interaction term (Hadamard/low-rank
        # bilinear pooling -- Kim et al., ICLR 2017): captures "this graph
        # state AND this strategy phrasing together" interactions that a
        # purely additive/concatenative fusion can under-represent.
        self.interaction_proj = nn.Sequential(
            nn.Linear(fusion_half, fusion_half),
            nn.LayerNorm(fusion_half),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        # ROUND 4 (see config.py's STAGE1_USE_GRAPH_GATE comment): learnable
        # gate on every graph-DERIVED fusion term (graph_proj, sem2graph_out,
        # graph2sem_out, interaction) -- semantic_proj always enters fusion
        # at full strength, unscaled. Motivated by a real run where this
        # graph-augmented model scored BELOW the paper's own text-only (no
        # graph) Step Model on step accuracy (76.49% vs 82.87%), while MCP
        # already comfortably beat the paper -- suggesting the graph branch
        # may be adding noise specifically to step prediction. Stored as a
        # raw logit and passed through sigmoid so it's always in (0,1);
        # initialized so sigmoid(graph_gate_raw) == STAGE1_GRAPH_GATE_INIT
        # (graph starts as a genuine minority contributor) but remains a
        # trainable nn.Parameter, so gradient descent -- not a hand-picked
        # constant -- decides whether to grow it back up.
        self.use_graph_gate = STAGE1_USE_GRAPH_GATE
        # ROUND 7: per-head gates (see config.py's
        # STAGE1_USE_PER_HEAD_GRAPH_GATE comment). `graph_gate_raw` stays as
        # the single shared gate for the legacy/disabled path so an A/B
        # against the old behavior is still one flag away.
        self.use_per_head_gate = STAGE1_USE_PER_HEAD_GRAPH_GATE
        self.graph_gate_raw = nn.Parameter(
            torch.tensor(_inv_sigmoid(STAGE1_GRAPH_GATE_INIT))
        )
        self.graph_gate_step_raw = nn.Parameter(
            torch.tensor(_inv_sigmoid(STAGE1_GRAPH_GATE_INIT_STEP))
        )
        self.graph_gate_mcp_raw = nn.Parameter(
            torch.tensor(_inv_sigmoid(STAGE1_GRAPH_GATE_INIT_MCP))
        )

        # Enhanced fusion with residual connections
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.semantic_dim, fusion_half),
            nn.LayerNorm(fusion_half),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_dim, fusion_half),
            nn.LayerNorm(fusion_half),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        # Fusion input: semantic_proj + graph_proj + sem2graph cross-attn
        # output + graph2sem cross-attn output + Hadamard interaction,
        # each fusion_half-wide -> 5 * D/2.
        fusion_input_dim = 5 * fusion_half
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.08),
        )

        self.step_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(FUSION_HIDDEN, len(STEP_LABELS)),
        )
        # A3: auxiliary coarse-phase head. Shares the step head's fused
        # representation, so it regularizes the same trunk the step head uses.
        # Costs ~0.4% extra parameters. See config.py's STEP_PHASE_OF.
        self.phase_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.LayerNorm(FUSION_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(FUSION_HIDDEN // 2, N_STEP_PHASES),
        )
        self.mcp_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.LayerNorm(FUSION_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(FUSION_HIDDEN // 2, len(MCP_LABELS)),
        )

    def encode_and_predict(self, x, edge_index, batch, field_embs=None,
                           semantic_tokens=None, semantic_mask=None, edge_attr=None):
        if semantic_tokens is None or semantic_mask is None:
            raise ValueError(
                "Stage 1 requires semantic_tokens and semantic_mask built from "
                "New strategy + Strategy explanation."
            )

        # Semantic path: strategy + explanation -> frozen GPT-2 features -> CNN.
        semantic_h = self.semantic_cnn(semantic_tokens, semantic_mask)

        # Graph path: PTT -> typed GINE -> exactly 512-d pooled vector, PLUS
        # the per-node hidden states + mask (needed below for real
        # cross-attention). The pooled `graph_h` remains the only graph
        # representation exposed downstream to Stage 2/3 (via
        # `self.graph_encoder(...)`'s plain forward(), called separately in
        # stage2_sft_qwen.py / evaluate.py) -- forward_with_nodes()'s extra
        # outputs never leave this function.
        graph_h, graph_nodes, graph_node_mask = self.graph_encoder.forward_with_nodes(
            x, edge_index, batch, edge_attr=edge_attr
        )
        if graph_h.shape[-1] != GNN_OUT_DIM:
            raise RuntimeError(
                f"Stage-1 GINE must produce {GNN_OUT_DIM} dims, got {graph_h.shape[-1]}"
            )

        # Project both modalities' POOLED vectors to fusion space.
        semantic_proj = self.semantic_proj(semantic_h)   # (B, D/2)
        graph_proj = self.graph_proj(graph_h)             # (B, D/2)

        # Project both modalities' TOKEN sequences to the same fusion
        # space, so each can serve as a real (length > 1, in the typical
        # case) key/value sequence for the other modality's pooled query.
        graph_node_kv = self.graph_node_proj(graph_nodes)        # (B, N_nodes, D/2)
        semantic_token_kv = self.semantic_token_proj(semantic_tokens)  # (B, L_tokens, D/2)

        # Semantic vector asks "which graph nodes matter for this
        # strategy?" -- key_padding_mask blocks attention to padded nodes.
        sem2graph_out, _ = self.cross_attn_sem2graph(
            semantic_proj.unsqueeze(1), graph_node_kv, graph_node_kv,
            key_padding_mask=~graph_node_mask,
        )
        sem2graph_out = sem2graph_out.squeeze(1)  # (B, D/2)

        # Graph vector asks "which words in the strategy/explanation
        # matter for this graph state?" -- key_padding_mask blocks
        # attention to padded token positions (semantic_mask: True=real).
        graph2sem_out, _ = self.cross_attn_graph2sem(
            graph_proj.unsqueeze(1), semantic_token_kv, semantic_token_kv,
            key_padding_mask=~semantic_mask,
        )
        graph2sem_out = graph2sem_out.squeeze(1)  # (B, D/2)

        # Cheap explicit multiplicative (Hadamard) interaction between the
        # two pooled vectors, on top of the attention-based interaction
        # above.
        interaction = self.interaction_proj(semantic_proj * graph_proj)  # (B, D/2)

        # ROUND 4/7: scale every graph-derived term by a learnable gate before
        # concatenation. semantic_proj is deliberately left unscaled -- it is
        # the one term with no graph involvement at all.
        #
        # ROUND 7: the gate is now PER HEAD. Step and MCP demonstrably want
        # different amounts of graph (see config.py's
        # STAGE1_USE_PER_HEAD_GRAPH_GATE comment: moving the shared gate
        # raised Step 0.7687->0.7836 while dropping MCP micro-F1
        # 0.7036->0.6539), so each head gets its own gate and its own pass
        # through the fusion MLP. The fusion MLP is small (5*D/2 -> D -> D)
        # relative to the GINE stack and the semantic CNN, so running it
        # twice is cheap; the graph/semantic encoders themselves still run
        # exactly once and are shared, which is what keeps this a
        # shared-bottom multi-task model rather than two separate models.
        def _fuse(gate_raw):
            if not self.use_graph_gate:
                return self.fusion(torch.cat(
                    [semantic_proj, graph_proj, sem2graph_out, graph2sem_out, interaction],
                    dim=-1,
                ))
            g = torch.sigmoid(gate_raw)
            # graph2sem_out is a weighted sum of TEXT values (the graph only
            # picks the attention weights), so gating it by the GRAPH gate
            # discards graph-selected TEXT rather than graph content. See
            # config.py's STAGE1_GATE_GRAPH2SEM comment.
            g2s = g * graph2sem_out if STAGE1_GATE_GRAPH2SEM else graph2sem_out
            return self.fusion(torch.cat(
                [semantic_proj,
                 g * graph_proj,
                 g * sem2graph_out,
                 g2s,
                 g * interaction],
                dim=-1,
            ))

        if self.use_graph_gate and self.use_per_head_gate:
            fused_step = _fuse(self.graph_gate_step_raw)
            fused_mcp = _fuse(self.graph_gate_mcp_raw)
        else:
            fused_step = _fuse(self.graph_gate_raw)
            fused_mcp = fused_step

        # Private Stage-1 fusion. Never pass these fused representations to
        # the Stage-2/3 prefix adapter.
        step_logits = self.step_head(fused_step)
        phase_logits = self.phase_head(fused_step)
        mcp_logits = self.mcp_head(fused_mcp)
        self._last_phase_logits = phase_logits
        return (fused_step, fused_mcp), step_logits, mcp_logits

    def forward(self, x, edge_index, batch, field_embs=None,
                semantic_tokens=None, semantic_mask=None, edge_attr=None):
        h, step_logits, mcp_logits = self.encode_and_predict(
            x, edge_index, batch, field_embs,
            semantic_tokens=semantic_tokens,
            semantic_mask=semantic_mask,
            edge_attr=edge_attr,
        )
        return step_logits, mcp_logits, h

    def predict_from_fused(self, fused_h, fused_mcp=None):
        """Run only the (cheap) classification heads on already-fused
        representation(s). Used by the optional Manifold Mixup auxiliary loss
        and by the decoupled classifier re-balancing phase
        (training/stage1_gnn_train.py) without re-running the graph/semantic
        encoders.

        ROUND 7: with per-head gates the two heads read DIFFERENT fused
        vectors, so callers should pass both. `fused_h` is the Step-side
        representation; `fused_mcp` defaults to it for the single-gate /
        gate-disabled path and for any caller that still has only one tensor.
        """
        if fused_mcp is None:
            fused_mcp = fused_h
        return self.step_head(fused_h), self.mcp_head(fused_mcp)

    def loss(self, step_logits, mcp_logits, step_labels, mcp_targets,
             step_w=1.0, mcp_w=1.0, mcp_class_weights=None,
             use_focal=True, focal_gamma=2.0, label_smoothing=0.0,
             step_class_weights=None, use_step_focal=False,
             step_focal_gamma=2.0, step_log_priors=None, logit_adj_tau=0.0,
             smoothing_targets=None, phase_logits=None, phase_labels=None,
             phase_loss_weight=0.0,
             use_asl=False, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05):
        """
        step_log_priors / logit_adj_tau: optional logit adjustment (Menon
        et al., ICLR 2021). When set, `logit_adj_tau * step_log_priors` is
        added to the step logits BEFORE the softmax cross-entropy (and
        before the focal re-weighting, so focal's pt is computed on the
        adjusted distribution too) -- this is the paper's train-time
        variant. It is intentionally NOT applied to the `step_logits`
        returned by forward()/encode_and_predict(), so evaluate.py's plain
        argmax(step_logits) at val/test time is exactly the paper's
        prescribed inference rule (no adjustment at test time).

        use_asl: if True, the MCP multi-label loss uses Asymmetric Loss
        (Ridnik/Ben-Baruch et al., ICCV 2021, see asymmetric_loss() above)
        instead of focal BCE. `use_focal`/`focal_gamma` are then ignored
        for the MCP term (they still gate the STEP-side focal loss only if
        this function is also called with use_step_focal=True, which is an
        independent flag).
        """
        adj_step_logits = step_logits
        if step_log_priors is not None and logit_adj_tau:
            adj_step_logits = step_logits + logit_adj_tau * step_log_priors.view(1, -1)

        if smoothing_targets is not None:
            # A4: similarity-structured soft targets replace uniform smoothing.
            tgt = smoothing_targets.to(adj_step_logits.device)[step_labels]
            logp = F.log_softmax(adj_step_logits, dim=-1)
            per_row = -(tgt * logp).sum(dim=-1)
            if step_class_weights is not None:
                per_row = per_row * step_class_weights.to(per_row.device)[step_labels]
            step_loss = per_row.mean()
        else:
            step_loss = F.cross_entropy(
                adj_step_logits, step_labels,
                weight=step_class_weights,
                label_smoothing=label_smoothing,
            )
        if use_step_focal:
            p = torch.softmax(adj_step_logits, dim=-1)
            pt = p.gather(1, step_labels.view(-1, 1)).squeeze(1).clamp_min(1e-7)
            if smoothing_targets is not None:
                tgt = smoothing_targets.to(adj_step_logits.device)[step_labels]
                ce = -(tgt * F.log_softmax(adj_step_logits, dim=-1)).sum(dim=-1)
                if step_class_weights is not None:
                    ce = ce * step_class_weights.to(ce.device)[step_labels]
            else:
                ce = F.cross_entropy(
                    adj_step_logits, step_labels,
                    weight=step_class_weights,
                    reduction="none",
                    label_smoothing=label_smoothing,
                )
            step_loss = ((1.0 - pt).pow(step_focal_gamma) * ce).mean()

        # A3: auxiliary coarse-phase loss on the same fused representation.
        if phase_logits is not None and phase_labels is not None and phase_loss_weight:
            step_loss = step_loss + phase_loss_weight * F.cross_entropy(phase_logits, phase_labels)

        if use_asl:
            mcp_loss = asymmetric_loss(
                mcp_logits, mcp_targets,
                gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
                class_weights=mcp_class_weights,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(mcp_logits, mcp_targets, reduction="none")
            if use_focal:
                p = torch.sigmoid(mcp_logits)
                pt = torch.where(mcp_targets > 0.5, p, 1.0 - p)
                bce = bce * (1.0 - pt).pow(focal_gamma)
            if mcp_class_weights is not None:
                bce = bce * mcp_class_weights.view(1, -1)
            mcp_loss = bce.mean()

        total = step_w * step_loss + mcp_w * mcp_loss
        return total, step_loss.detach(), mcp_loss.detach()

