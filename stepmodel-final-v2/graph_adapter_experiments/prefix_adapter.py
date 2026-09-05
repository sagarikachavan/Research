"""
prefix_adapter.py
==================
Projects a single StructureGNN embedding into GRAPH_PREFIX_TOKENS soft
prompt token embeddings, sized to whatever base LLM this experiment is
pointed at. Written fresh for this experiment (does not import
training/stage2_sft_qwen.py's GraphPrefixAdapter) so this folder has zero
import edges into the main pipeline's code, per the isolation requirement.
"""
import torch
import torch.nn as nn

from standalone_config import GRAPH_PREFIX_TOKENS


class GraphPrefixAdapter(nn.Module):
    """
    graph_emb (B, graph_dim)
        -> Linear -> LayerNorm -> GELU -> Dropout
        -> Linear -> LayerNorm -> GELU -> Dropout
        -> Linear(-> n_tokens * llm_hidden) -> reshape -> LayerNorm
        -> (B, n_tokens, llm_hidden)
    """

    def __init__(self, graph_dim: int, llm_hidden: int,
                 n_tokens: int = GRAPH_PREFIX_TOKENS, dropout: float = 0.1):
        super().__init__()
        self.n_tokens = n_tokens
        self.llm_hidden = llm_hidden
        hid = max(llm_hidden, graph_dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(graph_dim, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hid, llm_hidden * n_tokens),
        )
        self.output_norm = nn.LayerNorm(llm_hidden)

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        b = graph_emb.shape[0]
        raw = self.proj(graph_emb).view(b, self.n_tokens, self.llm_hidden)
        return self.output_norm(raw)
