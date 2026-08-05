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

try:
    from config import (
        GNN_HIDDEN, GNN_LAYERS, GNN_OUT_DIM, FUSION_HIDDEN,
        TEXT_EMB_DIM, STEP_LABELS, MCP_LABELS,
    )
except ImportError:
    # Fallback defaults for when config is not available
    GNN_HIDDEN = 256
    GNN_LAYERS = 3
    GNN_OUT_DIM = 256
    FUSION_HIDDEN = 512
    TEXT_EMB_DIM = 384
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
        "Nmap", "Metasploit", "Netcat", "Dirbuster", "SQLmap",
        "Smb client", "hydra", "John-the-ripper", "Google search",
        "Interactive CLI", "Web page interaction",
    ]

NODE_FEAT_DIM = TEXT_EMB_DIM + 3  # sentence-embedding + one-hot node type (Agent/Search/Track)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim=NODE_FEAT_DIM, hidden=GNN_HIDDEN,
                 out_dim=GNN_OUT_DIM, num_layers=GNN_LAYERS, heads=4, dropout=0.1):
        super().__init__()
        assert GATv2Conv is not None, "torch_geometric is required for GraphEncoder"
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout)
            )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden * 2, out_dim)  # *2 for mean+max pool concat

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(h + residual)
            h = self.dropout(F.relu(h))
        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.out_proj(pooled)  # (batch, GNN_OUT_DIM)


class ContextTextProjector(nn.Module):
    """Projects concatenated frozen sentence-embeddings of the context
    fields into the same space as the graph embedding."""

    def __init__(self, n_fields=3, field_dim=TEXT_EMB_DIM, out_dim=GNN_OUT_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_fields * field_dim, FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(FUSION_HIDDEN, out_dim),
        )

    def forward(self, field_embs):  # (batch, n_fields, field_dim)
        b = field_embs.shape[0]
        flat = field_embs.reshape(b, -1)
        return self.proj(flat)


class Stage1Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_encoder = GraphEncoder()
        self.context_encoder = ContextTextProjector()
        self.fusion = nn.Sequential(
            nn.Linear(GNN_OUT_DIM * 2, FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(FUSION_HIDDEN, FUSION_HIDDEN),
            nn.ReLU(),
        )
        self.step_head = nn.Linear(FUSION_HIDDEN, len(STEP_LABELS))
        self.mcp_head = nn.Linear(FUSION_HIDDEN, len(MCP_LABELS))

    def forward(self, x, edge_index, batch, field_embs):
        g = self.graph_encoder(x, edge_index, batch)           # (B, GNN_OUT_DIM)
        c = self.context_encoder(field_embs)                    # (B, GNN_OUT_DIM)
        h = self.fusion(torch.cat([g, c], dim=-1))               # (B, FUSION_HIDDEN)
        step_logits = self.step_head(h)
        mcp_logits = self.mcp_head(h)
        return step_logits, mcp_logits, g  # g is reused as the LLM graph-prefix source

    def loss(self, step_logits, mcp_logits, step_labels, mcp_targets,
              step_w=1.0, mcp_w=1.0):
        step_loss = F.cross_entropy(step_logits, step_labels)
        mcp_loss = F.binary_cross_entropy_with_logits(mcp_logits, mcp_targets)
        return step_w * step_loss + mcp_w * mcp_loss, step_loss.detach(), mcp_loss.detach()
