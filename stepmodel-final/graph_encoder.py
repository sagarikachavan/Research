"""
Stage-1 model: GraphEncoder (GNN over the PTT/recon-state graph) fused with
a frozen text encoder over the context fields, feeding two heads:
  - step_head:  single-label softmax over STEP_LABELS
  - mcp_head:   multi-label sigmoid over MCP_LABELS

This is the module that later hands its pooled graph embedding to the LLM
stage (as a short sequence of soft-prompt tokens), the same way the paper's
one-shot LLM framework hands the model structured context to reason over —
except here the "in-context example" is replaced by a learned graph vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
except ImportError:  # keep module importable for label-normalization-only usage
    GATv2Conv = None

from config import (
    GNN_HIDDEN, GNN_LAYERS, GNN_OUT_DIM, FUSION_HIDDEN,
    TEXT_EMB_DIM, STEP_LABELS, MCP_LABELS, GNN_HEADS, GNN_DROPOUT,
    EDGE_ATTR_DIM,
)
from data_utils import CONTEXT_COLUMNS

NODE_FEAT_DIM = TEXT_EMB_DIM + 4  # sentence-embedding + one-hot node type (Agent/Search/Track) + degree


class GraphEncoder(nn.Module):
    def __init__(self, in_dim=NODE_FEAT_DIM, hidden=GNN_HIDDEN,
                 out_dim=GNN_OUT_DIM, num_layers=GNN_LAYERS, heads=GNN_HEADS, dropout=GNN_DROPOUT,
                 edge_dim=None):
        super().__init__()
        assert GATv2Conv is not None, "torch_geometric is required for GraphEncoder"
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout,
                          edge_dim=edge_dim, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = self.input_proj(x)
        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            if edge_attr is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = norm(h + residual)
            h = self.dropout(F.gelu(h))
            layer_outputs.append(h)
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        # Mean of per-layer mean pools (dense multi-scale aggregation)
        device = h.device
        dtype = h.dtype
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        layer_mean_accum = torch.zeros(B, h.shape[-1], device=device, dtype=dtype)
        for lo in layer_outputs:
            layer_mean_accum = layer_mean_accum + global_mean_pool(lo, batch)
        layer_mean_pool = layer_mean_accum / max(1, len(layer_outputs))
        pooled = torch.cat([mean_pool, max_pool, layer_mean_pool], dim=-1)
        return self.out_proj(pooled)  # (batch, GNN_OUT_DIM)


class ContextTextProjector(nn.Module):
    """Projects concatenated frozen sentence-embeddings of the context
    fields into the same space as the graph embedding."""

    def __init__(self, n_fields=None, field_dim=TEXT_EMB_DIM, out_dim=GNN_OUT_DIM):
        if n_fields is None:
            n_fields = len(CONTEXT_COLUMNS)
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_fields * field_dim, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN // 2, out_dim),
        )

    def forward(self, field_embs):  # (batch, n_fields, field_dim)
        b = field_embs.shape[0]
        flat = field_embs.reshape(b, -1)
        return self.proj(flat)


class Stage1Classifier(nn.Module):
    def __init__(self, edge_dim: int = EDGE_ATTR_DIM):
        super().__init__()
        self.graph_encoder = GraphEncoder(edge_dim=edge_dim)
        self.context_encoder = ContextTextProjector()
        self.graph_gate = nn.Sequential(
            nn.Linear(GNN_OUT_DIM, GNN_OUT_DIM),
            nn.Sigmoid(),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(GNN_OUT_DIM, GNN_OUT_DIM),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(GNN_OUT_DIM * 2, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.step_head = nn.Linear(FUSION_HIDDEN // 2, len(STEP_LABELS))
        self.mcp_head = nn.Linear(FUSION_HIDDEN // 2, len(MCP_LABELS))

    def forward(self, x, edge_index, batch, field_embs, edge_attr=None):
        g = self.graph_encoder(x, edge_index, batch, edge_attr=edge_attr)           # (B, GNN_OUT_DIM)
        c = self.context_encoder(field_embs)                    # (B, GNN_OUT_DIM)
        g_gate = self.graph_gate(g)
        c_gate = self.context_gate(c)
        g_weighted = g * g_gate
        c_weighted = c * c_gate
        h = self.fusion(torch.cat([g_weighted, c_weighted], dim=-1))  # (B, FUSION_HIDDEN//2)
        step_logits = self.step_head(h)
        mcp_logits = self.mcp_head(h)
        return step_logits, mcp_logits, g  # g is reused as the LLM graph-prefix source

    def loss(self, step_logits, mcp_logits, step_labels, mcp_targets,
              step_w=1.0, mcp_w=1.0, mcp_class_weights=None, use_focal=True, focal_gamma=2.0,
              label_smoothing=0.0, step_class_weights=None):
        if label_smoothing > 0:
            step_loss = self._label_smooth_ce(step_logits, step_labels, label_smoothing, step_class_weights)
        else:
            if step_class_weights is not None:
                step_loss = F.cross_entropy(step_logits, step_labels, weight=step_class_weights)
            else:
                step_loss = F.cross_entropy(step_logits, step_labels)

        if use_focal:
            bce_loss = F.binary_cross_entropy_with_logits(mcp_logits, mcp_targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_weight = (1 - pt) ** focal_gamma
            if mcp_class_weights is not None:
                focal_weight = focal_weight * mcp_class_weights
            mcp_loss = (focal_weight * bce_loss).mean()
        else:
            if mcp_class_weights is not None:
                mcp_loss = F.binary_cross_entropy_with_logits(
                    mcp_logits, mcp_targets, weight=mcp_class_weights
                )
            else:
                mcp_loss = F.binary_cross_entropy_with_logits(mcp_logits, mcp_targets)

        return step_w * step_loss + mcp_w * mcp_loss, step_loss.detach(), mcp_loss.detach()

    @staticmethod
    def _label_smooth_ce(logits, labels, eps, class_weights=None):
        n_classes = logits.size(-1)
        one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
        smoothed = one_hot * (1 - eps) + eps / n_classes
        log_probs = F.log_softmax(logits, dim=-1)
        if class_weights is not None:
            loss = -(smoothed * log_probs) * class_weights.unsqueeze(0)
        else:
            loss = -(smoothed * log_probs)
        return loss.sum(dim=-1).mean()