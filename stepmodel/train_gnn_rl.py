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
from torch_geometric.nn import GCNConv, global_mean_pool

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
# GNN Model using PyTorch Geometric
# -----------------------------
class GNNModel(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class GNNRLPolicy(nn.Module):
    def __init__(self, gnn_out_dim: int, text_emb_dim: int, llm_hidden_size: int):
        super().__init__()
        self.gnn = GNNModel(node_dim=text_emb_dim, hidden_dim=256, output_dim=gnn_out_dim)
        self.project_step_text = nn.Sequential(
            nn.Linear(text_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, gnn_out_dim)
        )
        self.combine = nn.Sequential(
            nn.Linear(gnn_out_dim * 2, llm_hidden_size),
            nn.ReLU()
        )

    def get_graph_embedding(self, nodes, edges, device):
        node_embs = torch.tensor([n['embedding'] for n in nodes], dtype=torch.float32).to(device)
        node_id_map = {n['id']: i for i, n in enumerate(nodes)}
        edge_index_list = []
        for e in edges:
            if e['from'] in node_id_map and e['to'] in node_id_map:
                edge_index_list.append([node_id_map[e['from']], node_id_map[e['to']]])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device) if edge_index_list else torch.empty((2, 0), dtype=torch.long).to(device)
        batch = torch.zeros(node_embs.size(0), dtype=torch.long).to(device)
        return self.gnn(node_embs, edge_index, batch)

    def forward(self, nodes, edges, step_text_embeddings, device):
        graph_emb = self.get_graph_embedding(nodes, edges, device)
        step_proj = self.project_step_text(step_text_embeddings)
        combined = self.combine(torch.cat([graph_emb, step_proj], dim=-1))
        return combined


# -----------------------------
# Dataset Class
# -----------------------------
class PenTestDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], text_model: SentenceTransformer):
        self.data = data
        self.text_model = text_model
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for machine in self.data:
            nodes = machine['nodes']
            edges = machine['edges']
            for step_pair in machine['step_pairs']:
                self.samples.append({
                    'nodes': nodes,
                    'edges': edges,
                    'step_pair': step_pair
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        step_pair = sample['step_pair']
        # Use a clear prompt for the input part
        prompt_text = (
            f"Previous penetration testing context:\n"
            f"Strategy: {step_pair['previous_strategy']}\n"
            f"Step: {step_pair['previous_step']}\n"
            f"Result: {step_pair['previous_step_result']}\n\n"
            f"Next:\n"
        )
        # Target text is what we want the model to generate
        target_text = (
            "Strategy: " + step_pair['next_strategy'] + "\n" +
            "Strategy Explanation: " + step_pair['next_strategy_explanation'] + "\n" +
            "Step: " + step_pair['next_step'] + "\n" +
            "Step Explanation: " + step_pair['next_step_explanation'] + "\n" +
            "MCP Tasks: " + step_pair['next_mcp_tasks']
        )
        # Combine prompt and target for teacher-forcing training
        full_text = prompt_text + target_text
        return {
            'nodes': sample['nodes'],
            'edges': sample['edges'],
            'prompt_text': prompt_text,
            'target_text': target_text,
            'full_text': full_text,
            'step_pair': step_pair
        }


def load_processed_data(embeddings_path: str):
    with open(embeddings_path, 'r') as f:
        return json.load(f)


# -----------------------------
# Reward Function (from PenStrategist)
# -----------------------------
def compute_reward(pred_step: str, true_step: str, pred_mcp: str, true_mcp: str) -> float:
    reward = 0.0
    pred_tokens = set(pred_step.lower().split())
    true_tokens = set(true_step.lower().split())
    if len(pred_tokens) > 0 and len(true_tokens) > 0:
        step_overlap = len(pred_tokens & true_tokens) / max(len(pred_tokens), len(true_tokens))
        reward += step_overlap * 0.5
    pred_mcp_tokens = set(pred_mcp.lower().split())
    true_mcp_tokens = set(true_mcp.lower().split())
    if len(pred_mcp_tokens) > 0 and len(true_mcp_tokens) > 0:
        mcp_overlap = len(pred_mcp_tokens & true_mcp_tokens) / max(len(pred_mcp_tokens), len(true_mcp_tokens))
        reward += mcp_overlap * 0.5
    return reward


def collate_fn(batch, tokenizer, text_model, device):
    previous_texts = [item['previous_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    previous_embs = torch.tensor(text_model.encode(previous_texts, convert_to_numpy=True), dtype=torch.float32).to(device)
    tokenized_targets = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    return {
        'nodes_list': [item['nodes'] for item in batch],
        'edges_list': [item['edges'] for item in batch],
        'previous_embs': previous_embs,
        'tokenized_targets': tokenized_targets
    }


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
    text_emb_dim = text_model.get_embedding_dimension()

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
    llm.to(device)

    # Initialize our policy model
    policy = GNNRLPolicy(
        gnn_out_dim=128,
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm_hidden_size
    ).to(device)

    # Create datasets and dataloaders
    train_dataset = PenTestDataset(train_data, text_model)
    test_dataset = PenTestDataset(test_data, text_model)

    # Training setup
    num_epochs = 1
    batch_size = 2
    learning_rate = 1e-4

    # Optimizers
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(llm.parameters()), lr=learning_rate)

    # Training loop
    print("Starting training...")
    policy.train()
    llm.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # For simplicity, let's process each sample individually (since graphs can vary in size)
        for idx in range(min(100, len(train_dataset))):
            sample = train_dataset[idx]
            nodes = sample['nodes']
            edges = sample['edges']
            step_pair = sample['step_pair']
            prompt_text = sample['prompt_text']
            full_text = sample['full_text']

            # Create previous_text for the policy (same as before)
            previous_text = (
                step_pair['previous_strategy'] + " " +
                step_pair['previous_strategy_explanation'] + " " +
                step_pair['previous_step'] + " " +
                step_pair['previous_step_explanation'] + " " +
                step_pair['previous_step_result'] + " " +
                step_pair['previous_mcp_tasks']
            )
            
            # Get text embeddings for previous step
            previous_emb = torch.tensor(text_model.encode([previous_text], convert_to_numpy=True), dtype=torch.float32).to(device)

            # Forward pass through policy
            policy_out = policy(nodes, edges, previous_emb, device)

            # Tokenize full text (prompt + target)
            tokenized = tokenizer(
                [full_text],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)

            # Get input embeddings
            inputs_embeds = llm.get_input_embeddings()(tokenized['input_ids'])
            
            # Add policy embedding as a prefix to the input embeddings
            policy_emb_expanded = policy_out.unsqueeze(1)
            inputs_embeds = torch.cat([policy_emb_expanded, inputs_embeds], dim=1)
            
            # Extend attention mask to include the policy embedding
            attention_mask = torch.cat(
                [torch.ones(1, 1, dtype=torch.long, device=device), tokenized['attention_mask']], dim=1)
            
            # Create labels: -100 for the policy embedding (ignore loss there)
            # For simplicity, we'll compute loss on all other tokens (prompt + target) for now
            labels = tokenized['input_ids'].clone()
            # Extend labels to match inputs_embeds length (add -100 for policy embedding position)
            labels = torch.cat([torch.tensor([[-100]], dtype=torch.long, device=device), labels], dim=1)
            
            # Forward pass through LLM
            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(llm.parameters()), max_norm=1.0)
            optimizer.step()

            if (idx + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Sample {idx+1}/{len(train_dataset)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

    # Save checkpoints
    torch.save({
        "policy": policy.state_dict(),
        "llm": llm.state_dict(),
        "optimizer": optimizer.state_dict()
    }, os.path.join(output_dir, "final_checkpoint.pt"))
    llm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training complete! Saved checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
