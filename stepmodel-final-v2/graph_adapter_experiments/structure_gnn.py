"""
structure_gnn.py
=================
A small GATv2-based graph encoder, trained FROM SCRATCH inside this
experiment (random init every run — no loading of the main pipeline's
`checkpoints/stage1_gnn_classifier.pt` or any other pretrained weights).

Contrast with core/graph_encoder.py's Stage1Classifier:
  - Stage1Classifier consumes (graph, text-field embeddings) and FUSES them
    via cross-attention + gating before anything reaches the LLM.
  - StructureGNN below consumes ONLY the graph (x, edge_index, edge_attr
    from graph_json.to_pyg_data). There is no text input anywhere in this
    class, no fusion step, and no dependence on the "New strategy"/
    "Strategy explanation" columns. Its only job is to produce a pooled
    embedding that is a function of graph topology and nothing else.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from standalone_config import (
    NODE_FEAT_DIM, EDGE_FEAT_DIM, GNN_HIDDEN, GNN_LAYERS, GNN_HEADS,
    GNN_OUT_DIM, GNN_DROPOUT,
)


class StructureGNN(nn.Module):
    def __init__(self, in_dim: int = NODE_FEAT_DIM, edge_dim: int = EDGE_FEAT_DIM,
                 hidden: int = GNN_HIDDEN, layers: int = GNN_LAYERS,
                 heads: int = GNN_HEADS, out_dim: int = GNN_OUT_DIM,
                 dropout: float = GNN_DROPOUT):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        assert hidden % heads == 0, "GNN_HIDDEN must be divisible by GNN_HEADS"
        self.dropout = dropout
        self.in_proj = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(
                GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout,
                          edge_dim=edge_dim, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(hidden))

        # Pool with mean + max concatenated (cheap, and gives the LLM-facing
        # projector both "typical node" and "most extreme node" signal to
        # work with) then project to the fixed output width.
        from torch_geometric.nn import global_mean_pool, global_max_pool
        self._mean_pool = global_mean_pool
        self._max_pool = global_max_pool
        self.out_proj = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = F.gelu(self.in_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual
        pooled = torch.cat([self._mean_pool(h, batch), self._max_pool(h, batch)], dim=-1)
        return self.out_proj(pooled)  # (batch, GNN_OUT_DIM)
