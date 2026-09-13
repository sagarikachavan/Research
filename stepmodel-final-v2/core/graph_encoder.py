"""Stage-1 hybrid graph + semantic CNN classifier.

Design:
- typed GINE graph encoder with GraphNorm and residual blocks;
- paper-inspired frozen language-model token features consumed by two
  independent multi-kernel CNN heads (Step/MCP);
- graph-conditioned fusion keeps the 384-d representation expected by Stage 2/3;
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
        # Attention pooling is deliberately simple and context-independent;
        # context conditioning is applied after node encoding.
        scores = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        attn = pyg_softmax(scores, batch)
        attn_pool = torch.zeros_like(mean_pool)
        attn_pool.index_add_(0, batch, attn.unsqueeze(-1) * h)
        return self.out_proj(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))


class SemanticCNNEncoder(nn.Module):
    """Paper-inspired temporal CNN over frozen LM token embeddings.

    The reference paper uses frozen GPT-2 token-level embeddings followed by
    multiple convolution kernels and global max pooling. We reproduce that
    design while keeping separate Step and MCP encoders.
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
        pooled = []
        for conv in self.convs:
            y = F.gelu(conv(x))
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
    """Joint semantic + typed-graph Step/MCP classifier.

    Outputs a 384-d task-conditioned representation for Stage 2/3.
    Semantic prototypes are fixed-size buffers derived from canonical labels.
    """

    def __init__(self, edge_dim: int = EDGE_ATTR_DIM, semantic_input_dim: int = 768):
        super().__init__()
        self.graph_encoder = GraphEncoder(edge_dim=edge_dim)
        self.context_encoder = ContextTextProjector()

        self.step_text_cnn = SemanticCNNEncoder(semantic_input_dim)
        self.mcp_text_cnn = SemanticCNNEncoder(semantic_input_dim)

        self.node_key = nn.Linear(GNN_HIDDEN, GNN_HIDDEN)
        self.node_value = nn.Linear(GNN_HIDDEN, GNN_HIDDEN)
        self.ctx_query = nn.Linear(GNN_OUT_DIM, GNN_HIDDEN)
        self.ctx_gate = nn.Sequential(
            nn.Linear(GNN_HIDDEN * 2, GNN_HIDDEN),
            nn.GELU(),
            nn.Linear(GNN_HIDDEN, 1),
        )
        self.node_attn_norm = nn.LayerNorm(GNN_HIDDEN)

        # Per-task fusion. Step receives semantic tokens + conditioned graph +
        # global graph; MCP receives its own semantic branch + same conditioned
        # graph + context projection.
        step_in = SEMANTIC_CNN_DIM + GNN_HIDDEN + GNN_OUT_DIM
        mcp_in = SEMANTIC_CNN_DIM + GNN_HIDDEN + GNN_OUT_DIM
        self.step_fusion = nn.Sequential(
            nn.Linear(step_in, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.LayerNorm(FUSION_HIDDEN // 2), nn.GELU(), nn.Dropout(0.06),
        )
        self.mcp_fusion = nn.Sequential(
            nn.Linear(mcp_in, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN // 2),
            nn.LayerNorm(FUSION_HIDDEN // 2), nn.GELU(), nn.Dropout(0.06),
        )
        fused_dim = FUSION_HIDDEN // 2
        self.step_tower = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2), nn.GELU(), nn.Dropout(0.08),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(fused_dim, len(STEP_LABELS)),
        )
        self.mcp_tower = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim), nn.GELU(), nn.Dropout(0.06),
            nn.Linear(fused_dim, len(MCP_LABELS)),
        )
        self.fused_dim = fused_dim

        # Fixed-shape semantic prototype buffers. These are set after loading
        # canonical Step/MCP label token sequences in stage1_gnn_train.py.
        proto_tokens = 64
        proto_dim = semantic_input_dim
        self.register_buffer("step_proto_tokens", torch.zeros(len(STEP_LABELS), proto_tokens, proto_dim), persistent=True)
        self.register_buffer("step_proto_mask", torch.zeros(len(STEP_LABELS), proto_tokens, dtype=torch.bool), persistent=True)
        self.register_buffer("mcp_proto_tokens", torch.zeros(len(MCP_LABELS), proto_tokens, proto_dim), persistent=True)
        self.register_buffer("mcp_proto_mask", torch.zeros(len(MCP_LABELS), proto_tokens, dtype=torch.bool), persistent=True)
        self.prototype_scale_step = nn.Parameter(torch.tensor(2.0))
        self.prototype_scale_mcp = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def set_semantic_prototypes(self, step_tokens, step_mask, mcp_tokens, mcp_mask):
        """Copy fixed-size prototype token tensors into checkpoint buffers."""
        def _fit(src, rows, cols, dtype):
            out = torch.zeros(rows, cols, src.shape[-1], dtype=dtype, device=self.step_proto_tokens.device)
            mask = torch.zeros(rows, cols, dtype=torch.bool, device=self.step_proto_tokens.device)
            for i in range(min(rows, src.shape[0])):
                L = min(cols, src.shape[1])
                out[i, :L] = src[i, :L].to(out.dtype)
                mask[i, :L] = True
            return out, mask
        s, sm = _fit(step_tokens, len(STEP_LABELS), self.step_proto_tokens.shape[1], self.step_proto_tokens.dtype)
        m, mm = _fit(mcp_tokens, len(MCP_LABELS), self.mcp_proto_tokens.shape[1], self.mcp_proto_tokens.dtype)
        self.step_proto_tokens.copy_(s)
        self.step_proto_mask.copy_(sm)
        self.mcp_proto_tokens.copy_(m)
        self.mcp_proto_mask.copy_(mm)

    def _condition_graph_on_context(self, node_h, batch, context):
        q = self.ctx_query(context)
        k = self.node_key(node_h)
        v = self.node_value(node_h)
        qn = q[batch]
        scores = (k * qn).sum(dim=-1) / (k.shape[-1] ** 0.5)
        scores = scores + self.ctx_gate(torch.cat([k, qn], dim=-1)).squeeze(-1)
        weights = pyg_softmax(scores, batch)
        pooled = torch.zeros(q.shape[0], v.shape[-1], device=v.device, dtype=v.dtype)
        pooled.index_add_(0, batch, weights.unsqueeze(-1) * v)
        return self.node_attn_norm(pooled + q)

    def _prototype_logits(self, query, cnn, proto_tokens, proto_mask, scale):
        if proto_tokens.numel() == 0:
            return query.new_zeros((query.shape[0], 0))
        n = proto_tokens.shape[0]
        expanded = proto_tokens.to(query.device)
        pm = proto_mask.to(query.device)
        proto_repr = cnn(expanded, pm)
        qn = F.normalize(query, dim=-1)
        pn = F.normalize(proto_repr, dim=-1)
        return torch.clamp(scale, min=0.05, max=10.0) * (qn @ pn.transpose(0, 1))

    def encode_and_predict(self, x, edge_index, batch, field_embs,
                           semantic_tokens=None, semantic_mask=None, edge_attr=None):
        if semantic_tokens is None or semantic_mask is None:
            # Stage 2/3 compatibility fallback: construct a short semantic
            # sequence from the two BGE field embeddings.
            semantic_tokens = field_embs.new_zeros((field_embs.shape[0], 5, field_embs.shape[-1]))
            semantic_tokens[:, 0:2, :] = field_embs
            semantic_tokens[:, 2:, :] = field_embs[:, 1:2, :].expand(-1, 3, -1)
            semantic_mask = torch.ones(field_embs.shape[0], 5, dtype=torch.bool, device=field_embs.device)

        node_h = self.graph_encoder.forward_nodes(x, edge_index, batch, edge_attr=edge_attr)
        g = self.graph_encoder(x, edge_index, batch, edge_attr=edge_attr)
        c = self.context_encoder(field_embs)
        graph_ctx = self._condition_graph_on_context(node_h, batch, c)
        step_text = self.step_text_cnn(semantic_tokens, semantic_mask)
        mcp_text = self.mcp_text_cnn(semantic_tokens, semantic_mask)

        step_h = self.step_fusion(torch.cat([step_text, graph_ctx, g], dim=-1))
        mcp_h = self.mcp_fusion(torch.cat([mcp_text, graph_ctx, c], dim=-1))
        step_logits = self.step_tower(step_h)
        mcp_logits = self.mcp_tower(mcp_h)

        # Semantic prototype scores provide an explicit label-semantic signal.
        # They are added as a residual logit term rather than replacing the
        # supervised classifier, keeping the system robust to label wording.
        proto_step = self._prototype_logits(
            step_text, self.step_text_cnn, self.step_proto_tokens,
            self.step_proto_mask, self.prototype_scale_step,
        )
        proto_mcp = self._prototype_logits(
            mcp_text, self.mcp_text_cnn, self.mcp_proto_tokens,
            self.mcp_proto_mask, self.prototype_scale_mcp,
        )
        if proto_step.shape[-1] == step_logits.shape[-1]:
            step_logits = step_logits + proto_step
        if proto_mcp.shape[-1] == mcp_logits.shape[-1]:
            mcp_logits = mcp_logits + proto_mcp

        # Exactly 384 dims when FUSION_HIDDEN=768, preserving the Stage-2/3
        # graph-prefix interface.
        h = torch.cat([
            step_h[:, :self.fused_dim // 2],
            mcp_h[:, :self.fused_dim // 2],
        ], dim=-1)
        return h, step_logits, mcp_logits

    def forward(self, x, edge_index, batch, field_embs,
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
