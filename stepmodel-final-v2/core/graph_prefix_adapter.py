"""
Graph prefix adapter for converting GNN embeddings to soft prompt tokens.

This module implements the prefix adapter that converts the 512-d GNN graph
representation into 8 soft prompt tokens for the LLM, as specified in the
multi-stage training pipeline.
"""
import torch
import torch.nn as nn

from config import GRAPH_PREFIX_TOKENS, GNN_OUT_DIM


class GraphPrefixAdapter(nn.Module):
    """
    Converts GNN graph embeddings to soft prompt tokens for LLM prefix tuning.
    
    Architecture:
        graph_emb (B, 512) -> Linear -> LayerNorm -> GELU -> Dropout
                          -> Linear -> LayerNorm -> GELU -> Dropout  
                          -> Linear(-> 8 * llm_hidden) -> reshape -> LayerNorm
                          -> (B, 8, llm_hidden)
    
    Args:
        graph_dim: Input dimension from GNN (default: 512)
        llm_hidden: Hidden dimension of the LLM (e.g., 5120 for Qwen-14B)
        n_tokens: Number of soft prompt tokens (default: 8)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, graph_dim: int = GNN_OUT_DIM, 
                 llm_hidden: int = 5120,
                 n_tokens: int = GRAPH_PREFIX_TOKENS, 
                 dropout: float = 0.1):
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
        """
        Convert graph embeddings to soft prompt tokens.
        
        Args:
            graph_emb: (B, graph_dim) tensor from GNN
            
        Returns:
            (B, n_tokens, llm_hidden) tensor of soft prompt embeddings
        """
        b = graph_emb.shape[0]
        raw = self.proj(graph_emb).view(b, self.n_tokens, self.llm_hidden)
        return self.output_norm(raw)
    
    def get_token_embeddings(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        Get individual token embeddings for inspection/debugging.
        
        Args:
            graph_emb: (B, graph_dim) tensor from GNN
            
        Returns:
            (B, n_tokens, llm_hidden) tensor
        """
        return self.forward(graph_emb)
