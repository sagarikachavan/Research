"""Stage-1 hybrid graph + semantic CNN classifier.

Design:
- typed GINE graph encoder with GraphNorm and residual blocks;
- paper-inspired frozen language-model token features consumed by one shared
  multi-kernel CNN semantic encoder for New Strategy + Strategy Explanation;
- semantic/graph fusion is private to Stage 1 and feeds independent Step/MCP heads;
- the raw GINE graph representation is exactly 512-d and is the sole graph input to Stage 2/3;
- Step gets a stronger private tower while MCP keeps an independent tower.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

from config import (
    GNN_HIDDEN, GNN_LAYERS, GNN_OUT_DIM, FUSION_HIDDEN,
    TEXT_EMB_DIM, STEP_LABELS, MCP_LABELS, GNN_DROPOUT,
    EDGE_ATTR_DIM, NODE_AUX_DIM, SEMANTIC_CNN_DIM, SEMANTIC_CNN_KERNELS,
    SEMANTIC_CNN_DROPOUT,
)
from data_utils import CONTEXT_COLUMNS

NODE_FEAT_DIM = TEXT_EMB_DIM + NODE_AUX_DIM


class GINEBlock(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.conv = GINEConv(
            nn=nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
            ),
            edge_dim=hidden,
            train_eps=True,
        )
        self.norm = GraphNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, batch, edge_attr):
        e = self.edge_encoder(edge_attr)
        y = self.conv(h, edge_index, e)
        y = self.norm(y, batch)
        y = F.gelu(y)
        y = self.dropout(y)
        return h + y


class GraphEncoder(nn.Module):
    """Encode a PTT graph to a 512-d representation."""

    def __init__(self, in_dim=NODE_FEAT_DIM, hidden=GNN_HIDDEN,
                 out_dim=GNN_OUT_DIM, num_layers=GNN_LAYERS,
                 dropout=GNN_DROPOUT, edge_dim=EDGE_ATTR_DIM):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            GINEBlock(hidden, edge_dim, dropout) for _ in range(num_layers)
        ])
        self.node_norm = nn.LayerNorm(hidden)
        self.attn_pool = nn.Linear(hidden, 1)
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
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, edge_index, batch, edge_attr)
        return self.node_norm(h)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.forward_nodes(x, edge_index, batch, edge_attr=edge_attr)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        # Learned, context-independent structural attention pooling.  It keeps
        # Stage 1 free of label-conditioned graph shortcuts while allowing the
        # GINE encoder to emphasize structurally informative PTT nodes.
        scores = self.attn_pool(h).squeeze(-1)
        attn = pyg_softmax(scores, batch)
        attn_pool = torch.zeros_like(mean_pool)
        attn_pool.index_add_(0, batch, attn.unsqueeze(-1) * h)
        return self.out_proj(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))


class SemanticCNNEncoder(nn.Module):
    """Paper-inspired temporal CNN over frozen LM token embeddings.

    The reference paper uses frozen GPT-2 token-level embeddings followed by
    multiple convolution kernels and global max pooling. We reproduce that
    design with a single shared semantic encoder for the two supervised heads.
    """

    def __init__(self, input_dim: int, out_dim: int = SEMANTIC_CNN_DIM,
                 kernels=SEMANTIC_CNN_KERNELS, dropout=SEMANTIC_CNN_DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, out_dim, kernel_size=k, padding=0)
            for k in kernels
        ])
        self.norm = nn.LayerNorm(out_dim * len(kernels))
        self.proj = nn.Sequential(
            nn.Linear(out_dim * len(kernels), out_dim * 2),
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
        for conv in self.convs:
            k = conv.kernel_size[0]
            # Pad short sequences so every configured kernel is valid.
            if x.shape[-1] < k:
                x_conv = F.pad(x, (0, k - x.shape[-1]))
                m_conv = F.pad(mask_f, (0, k - mask_f.shape[-1]))
            else:
                x_conv, m_conv = x, mask_f
            y = F.gelu(conv(x_conv))
            # A window is valid only when all of its source tokens are real.
            valid = F.conv1d(m_conv, x.new_ones(1, 1, k), stride=1).squeeze(1) >= float(k) - 1e-6
            y = y.masked_fill(~valid.unsqueeze(1), torch.finfo(y.dtype).min)
            pooled.append(y.max(dim=-1).values)
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
        # prediction heads.  This matches the intended paper-inspired flow:
        # strategy + explanation are understood once, then fused with graph.
        self.semantic_dim = SEMANTIC_CNN_DIM
        self.graph_dim = GNN_OUT_DIM
        self.fused_dim = FUSION_HIDDEN
        self.fusion = nn.Sequential(
            nn.Linear(self.semantic_dim + self.graph_dim, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.06),
        )

        self.step_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(FUSION_HIDDEN, len(STEP_LABELS)),
        )
        self.mcp_head = nn.Sequential(
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.LayerNorm(FUSION_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.06),
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

        # Graph path: PTT -> typed GINE -> exactly 512-d.  This raw graph
        # embedding is the only graph representation exposed downstream.
        graph_h = self.graph_encoder(
            x, edge_index, batch, edge_attr=edge_attr
        )
        if graph_h.shape[-1] != GNN_OUT_DIM:
            raise RuntimeError(
                f"Stage-1 GINE must produce {GNN_OUT_DIM} dims, got {graph_h.shape[-1]}"
            )

        # Private Stage-1 fusion. Never pass this fused representation to the
        # Stage-2/3 prefix adapter.
        fused_h = self.fusion(torch.cat([semantic_h, graph_h], dim=-1))
        step_logits = self.step_head(fused_h)
        mcp_logits = self.mcp_head(fused_h)
        return fused_h, step_logits, mcp_logits

    def forward(self, x, edge_index, batch, field_embs=None,
                semantic_tokens=None, semantic_mask=None, edge_attr=None):
        h, step_logits, mcp_logits = self.encode_and_predict(
            x, edge_index, batch, field_embs,
            semantic_tokens=semantic_tokens,
            semantic_mask=semantic_mask,
            edge_attr=edge_attr,
        )
        return step_logits, mcp_logits, h

    def loss(self, step_logits, mcp_logits, step_labels, mcp_targets,
             step_w=1.0, mcp_w=1.0, mcp_class_weights=None,
             use_focal=True, focal_gamma=2.0, label_smoothing=0.0,
             step_class_weights=None, use_step_focal=False,
             step_focal_gamma=2.0):
        step_loss = F.cross_entropy(
            step_logits, step_labels,
            weight=step_class_weights,
            label_smoothing=label_smoothing,
        )
        if use_step_focal:
            p = torch.softmax(step_logits, dim=-1)
            pt = p.gather(1, step_labels.view(-1, 1)).squeeze(1).clamp_min(1e-7)
            ce = F.cross_entropy(
                step_logits, step_labels,
                weight=step_class_weights,
                reduction="none",
                label_smoothing=label_smoothing,
            )
            step_loss = ((1.0 - pt).pow(step_focal_gamma) * ce).mean()

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

