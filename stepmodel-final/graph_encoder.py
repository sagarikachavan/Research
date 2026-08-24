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
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
    from torch_geometric.nn.aggr import AttentionalAggregation
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
        # ---------------------------------------------------------------
        # Attentional (learned) readout, added alongside mean/max/layer-mean.
        # WHY: on shallow PTT trees (3-8 nodes typically), the node that
        # actually decides the next step is usually the terminal
        # state/Prediction node, not "the average node". Mean/max pooling
        # treat every node as equally informative regardless of its role;
        # a learned gate lets the model down-weight boilerplate
        # Agent/Search scaffolding nodes and up-weight the
        # StateTransition/Prediction node(s) that carry the decision-
        # relevant signal. Cheap (one extra linear gate) and safe for
        # small data since it degrades gracefully to ~mean pooling if the
        # gate saturates uniform.
        # ---------------------------------------------------------------
        self.attn_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
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
        attn_pool = self.attn_pool(h, batch)
        # Mean of per-layer mean pools (dense multi-scale aggregation)
        device = h.device
        dtype = h.dtype
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        layer_mean_accum = torch.zeros(B, h.shape[-1], device=device, dtype=dtype)
        for lo in layer_outputs:
            layer_mean_accum = layer_mean_accum + global_mean_pool(lo, batch)
        layer_mean_pool = layer_mean_accum / max(1, len(layer_outputs))
        pooled = torch.cat([mean_pool, max_pool, attn_pool, layer_mean_pool], dim=-1)
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
        h, step_logits, mcp_logits = self.encode_and_predict(
            x, edge_index, batch, field_embs, edge_attr=edge_attr
        )
        # NOTE: 3rd return value changed from the raw pooled graph embedding
        # `g` to the fused representation `h` (see encode_and_predict's
        # docstring for why). Every existing call site (stage1_gnn_train.py,
        # evaluate.py) unpacks this as `step_logits, mcp_logits, _` and
        # discards it, so this is a safe change for Stage 1 training/eval --
        # it only matters to code that starts consuming it, which should now
        # get the better representation by default.
        return step_logits, mcp_logits, h

    def encode_and_predict(self, x, edge_index, batch, field_embs, edge_attr=None):
        """
        Full frozen-inference path meant for the LLM stages (Stage 2 SFT,
        Stage 3 GRPO, and evaluate.py's Stage-2/3 generation) to call
        directly, instead of each reimplementing their own recombination
        of graph_encoder + context_encoder.

        WHY THIS EXISTS: forward()'s 3rd return value used to be `g`, the
        RAW pooled graph embedding computed BEFORE this model's own
        graph_gate/context_gate/fusion stack -- on the theory that the LLM
        stages should learn their own fusion of graph+context via the
        GraphPrefixAdapter/LoRA rather than inherit a fusion tuned for a
        9-way/11-way classification head. In practice, Stage 2 and Stage 3
        each reimplemented their OWN, much weaker recombination instead:
        Stage 2 used a parameter-free sigmoid-gated average of g and c
        (`sigmoid((g*c).sum(-1))`, i.e. no learned weights at all); Stage 3
        used the graph embedding ALONE, dropping context/strategy-text
        entirely. Both discard exactly the representation `h` below --
        which is what step_head/mcp_head are actually calibrated against,
        and what gets this model to ~75% step accuracy / ~0.72 combined
        Jaccard -- and ask a LoRA adapter to reconstruct something
        equivalent from next-token language-modeling gradient alone on
        ~1.5k rows. That is a much weaker training signal than the direct
        classification loss this module was actually trained with.

        Returns:
            h            (B, FUSION_HIDDEN//2)  -- the fused, decision-ready
                          representation. Feed THIS into GraphPrefixAdapter,
                          not the raw graph embedding.
            step_logits  (B, len(STEP_LABELS))
            mcp_logits   (B, len(MCP_LABELS))
        """
        g = self.graph_encoder(x, edge_index, batch, edge_attr=edge_attr)   # (B, GNN_OUT_DIM)
        c = self.context_encoder(field_embs)                                # (B, GNN_OUT_DIM)
        g_gate = self.graph_gate(g)
        c_gate = self.context_gate(c)
        h = self.fusion(torch.cat([g * g_gate, c * c_gate], dim=-1))        # (B, FUSION_HIDDEN//2)
        step_logits = self.step_head(h)
        mcp_logits = self.mcp_head(h)
        return h, step_logits, mcp_logits

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