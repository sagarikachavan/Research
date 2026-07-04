#!/usr/bin/env python3
"""
Training script for GNN + RL (GRPO) + LLM to predict next step and MCP tasks with explanations.
"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
# from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# Constants
# -----------------------------
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
    "Nmap",
    "Metasploit",
    "Netcat",
    "Dirbuster",
    "SQLmap",
    "Smb client",
    "hydra",
    "John-the-ripper",
    "Google search",
    "Interactive CLI",
    "Web page interaction",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Simple GNN Model
# -----------------------------
class SimpleGNN(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor):
        # Initial node features
        x = self.relu(self.fc1(node_embeddings))

        # Simple message passing: average over neighbors
        if edge_index.numel() > 0:
            src_nodes = edge_index[0]
            x[edge_index[1]] = x[edge_index[1]] + x[src_nodes] * 0.5

        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Global mean pooling
        graph_embedding = x.mean(dim=0, keepdim=True)
        return graph_embedding


class GNNRLPolicy(nn.Module):
    def __init__(self, gnn_out_dim: int, text_emb_dim: int, llm_hidden_size: int):
        super().__init__()
        self.gnn = SimpleGNN(node_dim=text_emb_dim, hidden_dim=256, output_dim=gnn_out_dim)
        self.project_step_text = nn.Sequential(
            nn.Linear(text_emb_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, gnn_out_dim)
        )
        self.combine = nn.Sequential(
            nn.Linear(gnn_out_dim * 2, llm_hidden_size),
            nn.ReLU()
        )

    def get_graph_embedding(self, nodes, edges):
        node_embs = torch.tensor([n['embedding'] for n in nodes], dtype=torch.float32)
        node_id_map = {n['id']: i for i, n in enumerate(nodes)}
        edge_index_list = []
        for e in edges:
            if e['from'] in node_id_map and e['to'] in node_id_map:
                edge_index_list.append([node_id_map[e['from']], node_id_map[e['to']]])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2, 0), dtype=torch.long)
        return self.gnn(node_embs, edge_index)


def load_processed_data(embeddings_path: str):
    with open(embeddings_path, 'r') as f:
        return json.load(f)


# -----------------------------
# Reward Function (from PenStrategist)
# -----------------------------
def compute_reward(pred_step: str, true_step: str, pred_mcp: str, true_mcp: str) -> float:
    reward = 0.0
    # Step similarity (overlap)
    pred_tokens = set(pred_step.lower().split())
    true_tokens = set(true_step.lower().split())
    if len(pred_tokens) > 0 and len(true_tokens) > 0:
        step_overlap = len(pred_tokens & true_tokens) / max(len(pred_tokens), len(true_tokens))
        reward += step_overlap * 0.5
    # MCP similarity
    pred_mcp_tokens = set(pred_mcp.lower().split())
    true_mcp_tokens = set(true_mcp.lower().split())
    if len(pred_mcp_tokens) > 0 and len(true_mcp_tokens) > 0:
        mcp_overlap = len(pred_mcp_tokens & true_mcp_tokens) / max(len(pred_mcp_tokens), len(true_mcp_tokens))
        reward += mcp_overlap * 0.5
    return reward


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    embeddings_dir = os.path.join(base_dir, "embeddings_data")
    output_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sentence transformer
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_emb_dim = text_model.get_sentence_embedding_dimension()

    # Load processed data
    train_data = load_processed_data(os.path.join(embeddings_dir, "train", "all_processed.json"))
    test_data = load_processed_data(os.path.join(embeddings_dir, "test", "all_processed.json"))

    # Load LLM (we'll use distilgpt2 for lightweight training)
    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(llm_name)
    llm_hidden_size = llm.config.hidden_size

    # Initialize our policy model
    policy = GNNRLPolicy(
        gnn_out_dim=128,
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm_hidden_size
    ).to(device)

    # TODO: Full implementation would connect policy to LLM and use GRPO
    print("Model initialized! For full GRPO training, see TRL library documentation.")

    # Save initial checkpoints
    torch.save({
        "policy": policy.state_dict(),
        "llm": llm.state_dict()
    }, os.path.join(output_dir, "initial_checkpoint.pt"))
    llm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved initial checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
