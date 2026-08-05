"""
Graph Prefix Adapter for stepmodelv4
Projects graph embeddings into soft prompt tokens for LLM input
"""
import torch
import torch.nn as nn
from config import GNN_OUT_DIM, PREFIX_TOKENS, ADAPTER_HIDDEN


class GraphPrefixAdapter(nn.Module):
    """
    Projects graph embeddings into soft prompt tokens that can be prepended
    to LLM input. This enables the LLM to "read" graph structure.
    """
    def __init__(self, graph_dim=GNN_OUT_DIM, llm_dim=4096, num_tokens=PREFIX_TOKENS, hidden=ADAPTER_HIDDEN):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        
        # Project graph embedding to soft prompt tokens
        self.adapter = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_tokens * llm_dim),
        )
    
    def forward(self, graph_embedding):
        """
        Args:
            graph_embedding: (batch, graph_dim) - graph encoder output
        
        Returns:
            soft_prompt: (batch, num_tokens, llm_dim) - soft prompt tokens
        """
        batch_size = graph_embedding.shape[0]
        projected = self.adapter(graph_embedding)  # (batch, num_tokens * llm_dim)
        soft_prompt = projected.view(batch_size, self.num_tokens, self.llm_dim)
        return soft_prompt


def load_graph_encoder_and_adapter(gnn_ckpt_path, adapter_ckpt_path, device, train_adapter=True):
    """
    Load frozen graph encoder and graph prefix adapter.
    
    Args:
        gnn_ckpt_path: Path to GNN encoder checkpoint
        adapter_ckpt_path: Path to graph adapter checkpoint
        device: torch device
        train_adapter: If True, adapter parameters will be trainable
    
    Returns:
        graph_encoder: Frozen Stage1Classifier
        graph_adapter: GraphPrefixAdapter (trainable if train_adapter=True)
    """
    from graph_encoder import Stage1Classifier
    
    # Load GNN encoder
    graph_encoder = Stage1Classifier().to(device)
    graph_encoder.load_state_dict(torch.load(gnn_ckpt_path, map_location=device))
    graph_encoder.eval()
    for param in graph_encoder.parameters():
        param.requires_grad = False
    
    # Load graph adapter (initialize if checkpoint doesn't exist)
    graph_adapter = GraphPrefixAdapter().to(device)
    if adapter_ckpt_path:
        try:
            graph_adapter.load_state_dict(torch.load(adapter_ckpt_path, map_location=device))
            print(f"[adapter] Loaded adapter checkpoint from {adapter_ckpt_path}")
        except FileNotFoundError:
            print(f"[adapter] Adapter checkpoint not found at {adapter_ckpt_path}, initializing randomly")
    else:
        print(f"[adapter] Initializing adapter randomly (will be trained during SFT)")
    
    if train_adapter:
        graph_adapter.train()
        for param in graph_adapter.parameters():
            param.requires_grad = True
        print(f"[adapter] Adapter set to trainable mode")
    else:
        graph_adapter.eval()
        for param in graph_adapter.parameters():
            param.requires_grad = False
        print(f"[adapter] Adapter set to frozen mode")
    
    return graph_encoder, graph_adapter
