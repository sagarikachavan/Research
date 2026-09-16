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
    EDGE_ATTR_DIM, NODE_AUX_DIM,
    STAGE1_TEXT_PROJ_DIM, STAGE1_TEXT_DROPOUT,
    STAGE1_TEXT_TOKEN_DIM, STAGE1_TEXT_ATTN_HEADS,
    STAGE1_EDGE_DROPOUT, STAGE1_NODE_FEAT_DROPOUT,
    STAGE1_GNN_TYPE, GNN_HEADS,
    N_STEP_PHASES,
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


class Stage1Classifier(nn.Module):
    """Stage-1 step + MCP classifier: a GATv2 graph tower and a Qwen text tower.

    ARCHITECTURE
    ------------
        Graph ─┬─ node feat 783-d ─┐
               └─ edge feat   5-d ─┴─> GATv2 ──> 512-d graph emb ─┐
                                                                   ├─> Fusion ─┬─> Step
        New strategy + explanation ─> Qwen3-Embedding ─> text emb ─┘           └─> MCP

    Node features are 768-d Qwen3-Embedding of the node title plus
    NODE_AUX_DIM(15) structural/type/status channels = 783-d. Edge features are
    the 5-d typed PTT edge encoding. The graph tower's 512-d output is the
    contract Stage 2/3 consume through the graph-prefix adapter -- do not
    change GNN_OUT_DIM without retraining them.

    WHAT WAS REMOVED, AND WHY
    -------------------------
    This class used to carry a second pretrained text model and four optional
    architectures layered on top of each other:

      * SemanticCNNEncoder -- a frozen-GPT-2 token tower feeding five parallel
        temporal convolutions (k=2,3,4,5,7) plus a token-level cross-attention
        branch. Gone: the text tower is ONE Qwen3-Embedding vector, so the
        project runs on a single encoder family rather than BGE + GPT-2.
      * Separate Step/MCP semantic towers, label prototypes, step-conditioned
        MCP, and a `simple_fusion` alternative path -- four mutually exclusive
        architectures selected by env var, none of which the diagram contains.

    WHAT WAS KEPT
    -------------
      * Per-head graph gates. These are not decoration: in training runs
        the step gate converges to ~0.04 while the MCP gate holds near ~0.50,
        i.e. the step head learns to ignore the graph while the MCP head
        relies on it. That asymmetry is a measured result worth preserving and
        reporting, and it costs two scalars.
      * The auxiliary coarse-phase head (an auxiliary LOSS on the fused step
        representation, see loss()), and every training-time term: focal /
        asymmetric loss, class weights, structured label smoothing, logit
        adjustment, SupCon and manifold mixup driven from the train script.

    FUSION. With two VECTORS rather than two sequences, cross-attention is
    degenerate (a length-1 query against a length-1 key), so the fusion is the
    standard vector pair interaction [t, g, t*g, |t-g|] -> MLP. That gives the
    heads both the raw modalities and their multiplicative and differential
    interactions, which is what the old cross-attention block was actually
    approximating once its sequence dimension collapsed to one.
    """

    def __init__(self, edge_dim: int = EDGE_ATTR_DIM,
                 text_input_dim: int = STAGE1_TEXT_TOKEN_DIM):
        super().__init__()
        self.graph_encoder = GraphEncoder(edge_dim=edge_dim)
        self.graph_dim = GNN_OUT_DIM
        self.text_dim = STAGE1_TEXT_PROJ_DIM
        self.fused_dim = FUSION_HIDDEN

        # ── Text tower: per-token Qwen3-Embedding states -> learned pooling ──
        # A single pre-pooled sentence vector measured 0.7388 step accuracy on
        # the 268-row test set; a token-level model on the same rows measured
        # 0.8396. The step head barely uses the graph (gate_step ~0.04), so the
        # text representation IS the step model -- pooling before the model
        # sees anything was the single largest accuracy cost.
        self.text_token_proj = nn.Linear(text_input_dim, self.text_dim)
        self.text_token_norm = nn.LayerNorm(self.text_dim)
        # One learned query attends over the token sequence, so the model picks
        # which tokens matter ("Research", "Exploit") instead of averaging them
        # into each other.
        self.text_query = nn.Parameter(torch.randn(1, 1, self.text_dim) * 0.02)
        self.text_attn = nn.MultiheadAttention(
            self.text_dim, STAGE1_TEXT_ATTN_HEADS,
            dropout=STAGE1_TEXT_DROPOUT, batch_first=True)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.GELU(),
            nn.Dropout(STAGE1_TEXT_DROPOUT),
        )
        # ── Graph tower projection, to the same width as the text tower ─────
        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.text_dim),
            nn.LayerNorm(self.text_dim),
            nn.GELU(),
            nn.Dropout(STAGE1_TEXT_DROPOUT),
        )

        # Per-head graph gates, initialised to ~0.15 so the graph starts as a
        # minority contributor and each head learns how much it actually wants.
        self.graph_gate_raw = nn.Parameter(torch.tensor(_inv_sigmoid(0.15)))
        self.graph_gate_step_raw = nn.Parameter(torch.tensor(_inv_sigmoid(0.15)))
        self.graph_gate_mcp_raw = nn.Parameter(torch.tensor(_inv_sigmoid(0.15)))

        # [t, g, t*g, |t-g|] -> fused representation
        self.fusion = nn.Sequential(
            nn.Linear(4 * self.text_dim, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(STAGE1_TEXT_DROPOUT),
        )

        self.step_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(STAGE1_TEXT_DROPOUT),
            nn.Linear(FUSION_HIDDEN // 2, len(STEP_LABELS)),
        )
        self.mcp_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(STAGE1_TEXT_DROPOUT),
            nn.Linear(FUSION_HIDDEN // 2, len(MCP_LABELS)),
        )
        # Auxiliary coarse-phase head (recon / enumerate / exploit / report ...).
        self.phase_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 4),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN // 4, N_STEP_PHASES),
        )
        self._last_phase_logits = None

    def _fuse(self, text_h, graph_h, gate_raw):
        g = torch.sigmoid(gate_raw) * graph_h
        return self.fusion(torch.cat(
            [text_h, g, text_h * g, torch.abs(text_h - g)], dim=-1))

    def encode_text(self, text_tokens, text_mask):
        """(B, L, H_enc) token states + (B, L) bool mask -> (B, text_dim)."""
        h = self.text_token_norm(self.text_token_proj(text_tokens))
        q = self.text_query.expand(h.shape[0], -1, -1).to(h.dtype)
        # A row with zero valid tokens would make every key masked, which makes
        # softmax produce NaN. Force at least the first position valid.
        km = ~text_mask
        all_masked = km.all(dim=1)
        if bool(all_masked.any()):
            km = km.clone()
            km[all_masked, 0] = False
        pooled, _ = self.text_attn(q, h, h, key_padding_mask=km, need_weights=False)
        return self.text_proj(pooled.squeeze(1))

    def encode_and_predict(self, x, edge_index, batch, text_tokens=None,
                           text_mask=None, edge_attr=None):
        """Returns ((fused_step, fused_mcp), step_logits, mcp_logits).

        `text_tokens` is (B, L, H_enc) frozen Qwen3-Embedding token states for
        "New strategy" + "Strategy explanation", with `text_mask` marking the
        valid positions (see data_utils.precompute_text_tokens). The encoder is
        frozen, so the caller precomputes these once per split.
        """
        if text_tokens is None or text_mask is None:
            raise ValueError(
                "Stage 1 requires text_tokens and text_mask from "
                "data_utils.precompute_text_tokens()."
            )
        graph_h = self.graph_encoder(x, edge_index, batch, edge_attr=edge_attr)
        text_h = self.encode_text(text_tokens.to(graph_h.dtype), text_mask)
        graph_p = self.graph_proj(graph_h)

        fused_step = self._fuse(text_h, graph_p, self.graph_gate_step_raw)
        fused_mcp = self._fuse(text_h, graph_p, self.graph_gate_mcp_raw)

        step_logits = self.step_head(fused_step)
        mcp_logits = self.mcp_head(fused_mcp)
        self._last_phase_logits = self.phase_head(fused_step)
        return (fused_step, fused_mcp), step_logits, mcp_logits

    def forward(self, x, edge_index, batch, text_tokens=None, text_mask=None,
                edge_attr=None):
        h, step_logits, mcp_logits = self.encode_and_predict(
            x, edge_index, batch, text_tokens=text_tokens, text_mask=text_mask,
            edge_attr=edge_attr)
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



def load_graph_encoder(ckpt_path: str, device: str = "cpu") -> GraphEncoder:
    """Load ONLY the GATv2 graph encoder from a Stage-1 checkpoint.

    STAGE BOUNDARY, ENFORCED IN CODE. Stage 1 is:

        graph ──> GATv2 ──> 512-d graph emb ─┐
                                              ├─> Fusion ─┬─> Step
        text  ──> Qwen3-Embedding ──────────┘             └─> MCP

    Only the part LEFT of the fusion crosses into Stage 2/3, as the 512-d
    vector the GraphPrefixAdapter turns into 16 soft-prompt tokens. Everything
    from the fusion rightward -- fusion MLP, step head, MCP head, phase head,
    the text tower and the graph gates -- is Stage 1's own classifier and must
    NOT reach the generator.

    Stage 2 used to build a full `Stage1Classifier` and `load_state_dict` the
    entire checkpoint, then call only `.graph_encoder`. The heads were frozen
    and never invoked, so no Stage-1 PREDICTION ever reached the LLM -- but the
    weights were resident, and nothing structurally prevented a later edit from
    reading them. Extracting the `graph_encoder.*` sub-tree makes the boundary
    explicit: the classifier weights are not loaded at all, and a checkpoint
    that somehow lacked a graph encoder fails loudly here rather than silently
    conditioning the LLM on random projections.
    """
    import torch as _torch
    ckpt = _torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    prefix = "graph_encoder."
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise KeyError(
            f"No 'graph_encoder.*' weights in {ckpt_path}. Stage 2/3 load only "
            f"the graph encoder; found top-level keys: "
            f"{sorted({k.split('.')[0] for k in state})}"
        )
    enc = GraphEncoder(edge_dim=EDGE_ATTR_DIM)
    missing, unexpected = enc.load_state_dict(sub, strict=True), None
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    dropped = sorted({k.split(".")[0] for k in state if not k.startswith(prefix)})
    print(f"[graph] Loaded graph encoder only ({len(sub)} tensors) from {ckpt_path}")
    if dropped:
        print(f"[graph] Stage-1 classifier weights deliberately NOT loaded: {dropped}")
    return enc
