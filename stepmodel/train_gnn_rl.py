#!/usr/bin/env python3
"""
Training script for GNN + LLM + GRPO (Group Relative Policy Optimization).
First trains with supervised teacher-forcing, then fine-tunes with GRPO RL!
Uses special [GRAPH] token approach to condition LLM on graph info.
"""

import os
import json
import random
import numpy as np
import time
import urllib.request
from typing import List, Dict, Any, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
try:
    # First try importing tensorboard directly to catch errors
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, Exception) as e:
    TENSORBOARD_AVAILABLE = False
    print(f"Warning: TensorBoard not available ({type(e).__name__}: {e}), skipping logging.")
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNNModel(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, output_dim: int, use_gat: bool = False):
        super().__init__()
        self.use_gat = use_gat
        if use_gat:
            self.conv1 = GATConv(node_dim, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
            self.fc = nn.Linear(hidden_dim * 4, output_dim)
        else:
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


class GNNLLMPolicy(nn.Module):
    def __init__(self, gnn_out_dim: int, text_emb_dim: int, llm_hidden_size: int, use_gat: bool = False):
        super().__init__()
        self.gnn = GNNModel(node_dim=text_emb_dim, hidden_dim=256, output_dim=gnn_out_dim, use_gat=use_gat)
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
        """
        Converts raw penetration testing environment graph data (nodes + edges)
        into a fixed-size embedding vector using the GNN model.
        
        This function:
        1. Validates input (handles empty node lists gracefully)
        2. Converts raw node data into PyTorch tensor embeddings
        3. Maps string node IDs from raw data to integer indices for PyG
        4. Converts raw edge data into PyTorch Geometric's edge_index format
        5. Creates batch tensor (single graph, all nodes in one batch)
        6. Passes everything to GNNModel to get final graph embedding
        
        Args:
            nodes (List[Dict]): List of nodes, each node is a dict with:
                - 'id': Unique string identifier for the node (e.g., machine name/service ID)
                - 'embedding': Pre-computed Sentence-BERT embedding vector for the node
            edges (List[Dict]): List of edges, each edge is a dict with:
                - 'from': Source node ID (string, matches node['id'])
                - 'to': Target node ID (string, matches node['id'])
            device (torch.device): Device to move tensors to (CPU or CUDA)
        
        Returns:
            torch.Tensor: Fixed-size graph embedding vector of shape (1, gnn_out_dim)
        """
        # --- Handle empty nodes case gracefully ---
        # If there are no nodes in the graph, we create a dummy graph with a single zero embedding
        # node and no edges to avoid errors from the GNN expecting non-empty input
        if not nodes:
            # Get the expected input dimension for the GNN's first convolutional layer
            node_dim = self.gnn.conv1.in_channels
            # Create a dummy node embedding: shape (1, node_dim), all zeros
            dummy_node_emb = torch.zeros((1, node_dim), dtype=torch.float32).to(device)
            # Create dummy edge index: shape (2, 0), which means "no edges" in PyG
            dummy_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            # Create dummy batch tensor: shape (1,), all zeros (single graph, one node)
            dummy_batch = torch.zeros(1, dtype=torch.long).to(device)
            # Pass dummy graph through GNN and return the result
            return self.gnn(dummy_node_emb, dummy_edge_index, dummy_batch)
        
        # --- Process non-empty graph ---
        # 1. Convert node embeddings from Python lists to PyTorch tensor
        # Shape: (num_nodes, text_emb_dim)
        node_embs = torch.tensor([n['embedding'] for n in nodes], dtype=torch.float32).to(device)
        
        # 2. Create a mapping from raw string node IDs to 0-based integer indices
        # (PyTorch Geometric requires integer node indices)
        node_id_map = {n['id']: i for i, n in enumerate(nodes)}
        
        # 3. Process edges and build edge_index
        edge_index_list = []
        for e in edges:
            # Only keep edges where both source and target nodes exist in our node list
            if e['from'] in node_id_map and e['to'] in node_id_map:
                # Convert string IDs to integer indices and add to list
                edge_index_list.append([node_id_map[e['from']], node_id_map[e['to']]])
        
        # 4. Convert edge list to PyTorch Geometric's edge_index format:
        # Shape (2, num_edges), where first row = source indices, second row = target indices
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
        else:
            # If no valid edges, use empty edge_index tensor (shape (2,0))
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        
        # 5. Create batch tensor: all nodes belong to same graph (all zeros)
        # Shape: (num_nodes,)
        batch = torch.zeros(node_embs.size(0), dtype=torch.long).to(device)
        
        # 6. Pass processed graph data to GNNModel to get final fixed-size graph embedding
        return self.gnn(node_embs, edge_index, batch)

    def forward(self, nodes, edges, step_text_embeddings, device):
        graph_emb = self.get_graph_embedding(nodes, edges, device)
        step_proj = self.project_step_text(step_text_embeddings)
        combined = self.combine(torch.cat([graph_emb, step_proj], dim=-1))
        return combined


class PenTestDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], text_model: SentenceTransformer, max_seq_length=1024):
        self.data = data
        self.text_model = text_model
        self.max_seq_length = max_seq_length
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
        prompt_text = (
            "[GRAPH]\n"
            "### Previous Penetration Testing Context ###\n"
            f"Strategy: {step_pair['previous_strategy']}\n"
            f"Step: {step_pair['previous_step']}\n"
            f"Result: {step_pair['previous_step_result']}\n\n"
            "### Generate Next Step ###\n"
            "Respond with exactly these five labeled lines and do not repeat the prompt:\n"
            "Strategy:\n"
            "Strategy Explanation:\n"
            "Step:\n"
            "Step Explanation:\n"
            "MCP Tasks:"
        )
        target_text = (
            "\nStrategy: " + step_pair['next_strategy'] + "\n"
            "Strategy Explanation: " + step_pair['next_strategy_explanation'] + "\n"
            "Step: " + step_pair['next_step'] + "\n"
            "Step Explanation: " + step_pair['next_step_explanation'] + "\n"
            "MCP Tasks: " + step_pair['next_mcp_tasks']
        )
        full_text = prompt_text + target_text
        return {
            'nodes': sample['nodes'],
            'edges': sample['edges'],
            'prompt_text': prompt_text,
            'target_text': target_text,
            'full_text': full_text,
            'step_pair': step_pair
        }


def collate_fn(batch):
    """
    Collate function for PenTestDataset (each sample is variable-length, keep as list)
    """
    return batch


def load_processed_data(embeddings_path: str):
    with open(embeddings_path, 'r') as f:
        return json.load(f)


def split_train_val(train_data: List[Dict], val_split: float = 0.1, seed: int = 42):
    random.seed(seed)
    random.shuffle(train_data)
    val_size = int(len(train_data) * val_split)
    return train_data[val_size:], train_data[:val_size]


def _debug_report(hypothesis_id: str, location: str, msg: str, data: Dict[str, Any], run_id: str = "pre-fix"):
    # #region debug-point shared:report
    env_path = '.dbg/avg-reward-zero.env'
    server_url = 'http://127.0.0.1:7777/event'
    session_id = 'avg-reward-zero'
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as env_file:
                for line in env_file:
                    if line.startswith('DEBUG_SERVER_URL='):
                        server_url = line.split('=', 1)[1].strip()
                    elif line.startswith('DEBUG_SESSION_ID='):
                        session_id = line.split('=', 1)[1].strip()
        except Exception:
            pass
    payload = {
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": data,
        "ts": int(time.time() * 1000),
    }
    try:
        req = urllib.request.Request(
            server_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=1).read()
    except Exception:
        pass
    # #endregion


def compute_reward(pred_text: str, true_step: str, true_mcp: str, text_model: SentenceTransformer, step_pair: Dict) -> float:
    """
    Improved reward function inspired by PenStrategist:
    Combines token overlap, semantic similarity, and structured output validity.
    """
    reward = 0.0
    
    # Parse predicted components
    _, pred_step, pred_mcp = parse_prediction(pred_text)
    if not pred_step:
        pred_step = pred_text
    if not pred_mcp:
        pred_mcp = pred_text
    # #region debug-point A:parsed-reward-inputs
    _debug_report(
        "A",
        "train_gnn_rl.py:278",
        "[DEBUG] Parsed reward inputs",
        {
            "pred_step_empty": not bool(pred_step.strip()),
            "pred_mcp_empty": not bool(pred_mcp.strip()),
            "pred_text_prefix": pred_text[:180],
            "pred_step_prefix": pred_step[:180],
            "pred_mcp_prefix": pred_mcp[:180],
            "true_step_prefix": true_step[:180],
            "true_mcp_prefix": true_mcp[:180],
        },
    )
    # #endregion

    # 1. Token overlap for step and MCP (weighted 0.2 each)
    pred_step_tokens = set(pred_step.lower().split())
    true_step_tokens = set(true_step.lower().split())
    if len(pred_step_tokens) > 0 and len(true_step_tokens) > 0:
        step_overlap = len(pred_step_tokens & true_step_tokens) / max(len(pred_step_tokens), len(true_step_tokens))
        reward += step_overlap * 0.2

    pred_mcp_tokens = set(pred_mcp.lower().split())
    true_mcp_tokens = set(true_mcp.lower().split())
    if len(pred_mcp_tokens) > 0 and len(true_mcp_tokens) > 0:
        mcp_overlap = len(pred_mcp_tokens & true_mcp_tokens) / max(len(pred_mcp_tokens), len(true_mcp_tokens))
        reward += mcp_overlap * 0.2

    # 2. Semantic similarity using Sentence-BERT (weighted 0.3 each)
    try:
        step_emb_pred = text_model.encode(pred_step, convert_to_tensor=False)
        step_emb_true = text_model.encode(true_step, convert_to_tensor=False)
        step_sem_sim = np.dot(step_emb_pred, step_emb_true) / (np.linalg.norm(step_emb_pred) * np.linalg.norm(step_emb_true) + 1e-8)
        reward += max(0, step_sem_sim) * 0.3

        mcp_emb_pred = text_model.encode(pred_mcp, convert_to_tensor=False)
        mcp_emb_true = text_model.encode(true_mcp, convert_to_tensor=False)
        mcp_sem_sim = np.dot(mcp_emb_pred, mcp_emb_true) / (np.linalg.norm(mcp_emb_pred) * np.linalg.norm(mcp_emb_true) + 1e-8)
        reward += max(0, mcp_sem_sim) * 0.3
    except Exception as exc:
        # #region debug-point C:semantic-similarity-failure
        _debug_report(
            "C",
            "train_gnn_rl.py:311",
            "[DEBUG] Semantic similarity failed",
            {"error": repr(exc), "pred_step_prefix": pred_step[:180], "pred_mcp_prefix": pred_mcp[:180]},
        )
        # #endregion
        pass

    # #region debug-point B:reward-summary
    _debug_report(
        "B",
        "train_gnn_rl.py:317",
        "[DEBUG] Reward computed",
        {
            "reward": float(reward),
            "step_token_overlap": len(pred_step_tokens & true_step_tokens) / max(len(pred_step_tokens), len(true_step_tokens)) if pred_step_tokens and true_step_tokens else 0.0,
            "mcp_token_overlap": len(pred_mcp_tokens & true_mcp_tokens) / max(len(pred_mcp_tokens), len(true_mcp_tokens)) if pred_mcp_tokens and true_mcp_tokens else 0.0,
            "pred_step_token_count": len(pred_step_tokens),
            "pred_mcp_token_count": len(pred_mcp_tokens),
        },
    )
    # #endregion
    return reward


def parse_prediction(pred_text: str):
    import re
    strategy = ""
    step = ""
    mcp_tasks = ""

    strategy_matches = re.findall(r"Strategy:\s*(.*?)(?=\n(?:Strategy Explanation:|Step:|Step Explanation:|MCP Tasks:)|$)", pred_text, re.DOTALL)
    if strategy_matches:
        strategy = " ".join(strategy_matches[-1].split())

    step_matches = re.findall(r"Step:\s*(.*?)(?=\n(?:Step Explanation:|MCP Tasks:)|$)", pred_text, re.DOTALL)
    if step_matches:
        step = " ".join(step_matches[-1].split())

    mcp_matches = re.findall(r"MCP Tasks:\s*(.*?)(?=\nMCP Tasks:|$)", pred_text, re.DOTALL)
    if mcp_matches:
        mcp_tasks = " ".join(mcp_matches[-1].split())

    placeholder_values = {"<your strategy>", "<why this strategy>", "<what to do>", "<why this step>", "<tools/tasks to use>"}
    if strategy.lower() in placeholder_values:
        strategy = ""
    if step.lower() in placeholder_values:
        step = ""
    if mcp_tasks.lower() in placeholder_values:
        mcp_tasks = ""

    return strategy, step, mcp_tasks


def generate_samples_with_policy(
    policy, llm, tokenizer, text_model, sample, device, num_generations_per_sample=4,
    max_new_tokens=1024, temperature=0.9, top_p=0.95
):
    """
    Generate multiple completions per sample using current policy, with consistent log prob calculation.
    """
    nodes = sample['nodes']
    edges = sample['edges']
    step_pair = sample['step_pair']
    prompt_text = sample['prompt_text']
    previous_text = (
        step_pair['previous_strategy'] + " " +
        step_pair['previous_strategy_explanation'] + " " +
        step_pair['previous_step'] + " " +
        step_pair['previous_step_explanation'] + " " +
        step_pair['previous_step_result'] + " " +
        step_pair['previous_mcp_tasks']
    )
    true_step = step_pair['next_step']
    true_mcp = step_pair['next_mcp_tasks']

    rollouts = []

    with torch.no_grad():
        previous_emb = torch.tensor(text_model.encode([previous_text], convert_to_numpy=True), dtype=torch.float32).to(device)
        policy_out = policy(nodes, edges, previous_emb, device)

        tokenized_prompt = tokenizer([prompt_text], return_tensors='pt', truncation=True, max_length=1024).to(device)
        inputs_embeds = llm.get_input_embeddings()(tokenized_prompt['input_ids'])
        inputs_embeds[:, 0, :] = policy_out
        attention_mask = tokenized_prompt['attention_mask']
        prompt_len = tokenized_prompt['input_ids'].shape[1]

        effective_max_new_tokens = min(max_new_tokens, 256)
        for _ in range(num_generations_per_sample):
            outputs = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=effective_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

            gen_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # When generate() is driven by inputs_embeds, sequences contains only
            # the generated continuation, so transition_scores already aligns to it.
            transition_scores = llm.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            gen_log_probs = transition_scores[0]
            log_prob = gen_log_probs.sum().item()

            reward = compute_reward(gen_text, true_step, true_mcp, text_model, step_pair)
            # #region debug-point E:generation-preview
            _debug_report(
                "E",
                "train_gnn_rl.py:407",
                "[DEBUG] Generated rollout sample",
                {
                    "prompt_len": int(prompt_len),
                    "generated_length": int(outputs.sequences.shape[1]),
                    "generated_text_prefix": gen_text[:200],
                    "reward": float(reward),
                },
            )
            # #endregion

            rollouts.append({
                'nodes': nodes,
                'edges': edges,
                'prompt_text': prompt_text,
                'prompt_input_ids': tokenized_prompt['input_ids'],
                'previous_text': previous_text,
                'generated_text': gen_text,
                'generated_ids': outputs.sequences,
                'prompt_len': prompt_len,
                'old_log_prob': log_prob,
                'reward': reward,
                'true_step': true_step,
                'true_mcp': true_mcp
            })

    return rollouts


def compute_grpo_loss(
    policy, llm, tokenizer, text_model, rollouts, device, clip_eps=0.2
):
    """
    Compute GRPO loss with consistent log probability calculation.
    """
    total_loss = 0.0
    num_valid = 0

    # Compute group statistics for all rollouts in batch
    rewards = torch.tensor([r['reward'] for r in rollouts], dtype=torch.float32)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r

    for i, rollout in enumerate(rollouts):
        nodes = rollout['nodes']
        edges = rollout['edges']
        previous_text = rollout['previous_text']
        prompt_input_ids = rollout['prompt_input_ids']
        generated_ids = rollout['generated_ids']
        gen_seq = torch.cat([prompt_input_ids, generated_ids], dim=1)
        prompt_len = rollout['prompt_len']
        old_log_prob = rollout['old_log_prob']
        advantage = advantages[i].item()

        previous_emb = torch.tensor(text_model.encode([previous_text], convert_to_numpy=True), dtype=torch.float32).to(device)
        policy_out = policy(nodes, edges, previous_emb, device)

        inputs_embeds = llm.get_input_embeddings()(gen_seq)
        inputs_embeds[:, 0, :] = policy_out

        # Forward pass to get logits
        with autocast():
            attention_mask = torch.ones_like(gen_seq, device=gen_seq.device)
            outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels (shift_logits[i] predicts shift_labels[i])
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = gen_seq[:, 1:].contiguous()

            # Calculate log probs ONLY for generated part (after prompt)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Mask everything before prompt_len - 1 (since shift_labels starts at pos 1)
            mask = torch.ones_like(token_log_probs, dtype=torch.bool)
            mask[:, :prompt_len - 1] = False
            
            # Compute new log prob as sum of masked log probs
            masked_log_probs = token_log_probs * mask.float()
            new_log_prob = masked_log_probs.sum()
        
        # Compute policy ratio
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surr1 = ratio * advantage
        surr2 = clipped_ratio * advantage
        policy_loss = -torch.min(surr1, surr2)

        total_loss += policy_loss
        num_valid += 1

    if num_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / num_valid


def evaluate_on_dataset(
    dataset, policy, llm, tokenizer, text_model, device, num_samples: Optional[int] = None
):
    """Evaluate model on given dataset, compute average reward."""
    policy.eval()
    llm.eval()
    total_reward = 0.0
    num_evals = 0

    eval_indices = list(range(len(dataset)))
    if num_samples is not None:
        eval_indices = eval_indices[:num_samples]

    with torch.no_grad():
        for idx in eval_indices:
            sample = dataset[idx]
            step_pair = sample['step_pair']
            previous_text = (
                step_pair['previous_strategy'] + " " +
                step_pair['previous_strategy_explanation'] + " " +
                step_pair['previous_step'] + " " +
                step_pair['previous_step_explanation'] + " " +
                step_pair['previous_step_result'] + " " +
                step_pair['previous_mcp_tasks']
            )
            true_step = step_pair['next_step']
            true_mcp = step_pair['next_mcp_tasks']

            previous_emb = torch.tensor(text_model.encode([previous_text], convert_to_numpy=True), dtype=torch.float32).to(device)
            policy_out = policy(sample['nodes'], sample['edges'], previous_emb, device)

            prompt_text = sample['prompt_text']
            tokenized_prompt = tokenizer([prompt_text], return_tensors='pt').to(device)
            inputs_embeds = llm.get_input_embeddings()(tokenized_prompt['input_ids'])
            inputs_embeds[:, 0, :] = policy_out

            output_ids = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=tokenized_prompt['attention_mask'],
                max_new_tokens=min(256, llm.config.n_positions - tokenized_prompt['input_ids'].shape[1]),
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            reward = compute_reward(pred_text, true_step, true_mcp, text_model, step_pair)
            total_reward += reward
            num_evals += 1

    avg_reward = total_reward / max(num_evals, 1)
    policy.train()
    llm.train()
    return avg_reward


def main():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, config['paths']['data_dir'])
    embeddings_dir = os.path.join(base_dir, config['paths']['embeddings_dir'])
    output_dir = os.path.join(base_dir, config['paths']['output_dir'])
    log_dir = os.path.join(base_dir, config['paths']['log_dir'])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        try:
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to:", log_dir)
        except Exception as e:
            print(f"Warning: Failed to initialize TensorBoard writer: {e}")

    # Load models
    text_model = SentenceTransformer(config['model']['text_embedding_model'])
    text_emb_dim = text_model.get_embedding_dimension()

    # Load data
    full_train_data = load_processed_data(os.path.join(embeddings_dir, "train", "all_processed.json"))
    test_data = load_processed_data(os.path.join(embeddings_dir, "test", "all_processed.json"))
    train_data, val_data = split_train_val(full_train_data, val_split=config['training']['validation_split'])

    # Load tokenizer and LLM
    llm_name = config['model']['llm_name']
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    special_tokens_dict = {'additional_special_tokens': ['[GRAPH]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")

    llm = AutoModelForCausalLM.from_pretrained(llm_name)
    llm.resize_token_embeddings(len(tokenizer))
    llm.gradient_checkpointing_enable()
    llm_hidden_size = llm.config.hidden_size
    llm.to(device)

    # Initialize policy
    policy = GNNLLMPolicy(
        gnn_out_dim=config['model']['gnn_out_dim'],
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm_hidden_size,
        use_gat=config['model']['use_gat']
    ).to(device)

    # Training setup first (define batch_size before creating loader)
    num_supervised_epochs = config['training']['num_supervised_epochs']
    num_grpo_epochs = config['training']['num_grpo_epochs']
    batch_size = config['training']['batch_size']
    
    # Create datasets and dataloaders
    max_seq_length = config.get('training', {}).get('max_seq_length', 1024)
    train_dataset = PenTestDataset(train_data, text_model, max_seq_length=max_seq_length)
    val_dataset = PenTestDataset(val_data, text_model, max_seq_length=max_seq_length)
    test_dataset = PenTestDataset(test_data, text_model, max_seq_length=max_seq_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    max_grad_norm = config['training']['max_grad_norm']
    num_warmup_steps = config['training']['num_warmup_steps']
    num_generations_per_sample = config['training']['num_generations_per_sample']
    clip_eps = config['training']['clip_eps']
    generate_max_new_tokens = config['training']['generate_max_new_tokens']
    generate_temperature = config['training']['generate_temperature']
    generate_top_p = config['training']['generate_top_p']
    patience = config['training']['patience']

    # Fix total steps calculation: both phases use batch steps (Phase1 uses batch_size=1 effectively for now)
    supervised_updates_per_epoch = len(train_loader)
    total_supervised_steps = num_supervised_epochs * supervised_updates_per_epoch
    total_grpo_steps = num_grpo_epochs * len(train_loader)
    total_steps = total_supervised_steps + total_grpo_steps

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(llm.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler()

    # Training
    global_step = 0
    best_val_reward = 0.0
    patience_counter = 0

    # --------------------------
    # Phase 1: Supervised Warmup
    # --------------------------
    print("\n" + "="*60)
    print("PHASE 1: SUPERVISED WARMUP TRAINING")
    print("="*60 + "\n")
    policy.train()
    llm.train()

    for epoch in range(num_supervised_epochs):
        total_loss = 0.0
        num_samples = 0

        for batch_idx, batch_samples in enumerate(train_loader):
            # Still process each sample individually since variable-length, but use loader for shuffle
            for sample in batch_samples:
                full_text = sample['full_text']
                prompt_text = sample['prompt_text']
                step_pair = sample['step_pair']
                previous_text = (
                    step_pair['previous_strategy'] + " " +
                    step_pair['previous_strategy_explanation'] + " " +
                    step_pair['previous_step'] + " " +
                    step_pair['previous_step_explanation'] + " " +
                    step_pair['previous_step_result'] + " " +
                    step_pair['previous_mcp_tasks']
                )

                tokenized_full = tokenizer(
                    [full_text],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(device)

                tokenized_prompt = tokenizer(
                    [prompt_text],
                    return_tensors='pt'
                ).to(device)
                prompt_token_len = tokenized_prompt['input_ids'].shape[1]

                previous_emb = torch.tensor(text_model.encode([previous_text], convert_to_numpy=True), dtype=torch.float32).to(device)
                policy_out = policy(sample['nodes'], sample['edges'], previous_emb, device)

                inputs_embeds = llm.get_input_embeddings()(tokenized_full['input_ids'])
                inputs_embeds[:, 0, :] = policy_out

                labels = tokenized_full['input_ids'].clone()
                # Mask ALL prompt tokens (including [GRAPH])
                labels[:, :prompt_token_len] = -100

                optimizer.zero_grad()
                
                with autocast():
                    outputs = llm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=tokenized_full['attention_mask'],
                        labels=labels
                    )
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(llm.parameters()), max_norm=max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                total_loss += loss.item()
                num_samples += 1

                if writer is not None:
                    writer.add_scalar("Supervised/loss", loss.item(), global_step)
                if num_samples % 100 == 0:
                    avg_loss = total_loss / num_samples
                    print(f"Epoch {epoch+1}/{num_supervised_epochs}, Sample {num_samples}/{len(train_dataset)}, Avg Loss: {avg_loss:.4f}")

        avg_epoch_loss = total_loss / num_samples
        val_reward = evaluate_on_dataset(val_dataset, policy, llm, tokenizer, text_model, device)

        if writer is not None:
            writer.add_scalar("Supervised/avg_loss", avg_epoch_loss, epoch+1)
            writer.add_scalar("Supervised/val_reward", val_reward, epoch+1)

        print(f"\nSupervised Epoch {epoch+1} Complete! Avg Loss: {avg_epoch_loss:.4f}, Val Reward: {val_reward:.4f}\n")

        # Early stopping check
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            patience_counter = 0
            torch.save({
                "policy": policy.state_dict(),
                "llm": llm.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "tokenizer": tokenizer,
                "llm_name": llm_name,
                "llm_hidden_size": llm_hidden_size,
                "epoch": epoch+1,
                "val_reward": val_reward,
                "phase": "supervised"
            }, os.path.join(output_dir, "best_supervised_checkpoint.pt"))
        else:
            patience_counter += 1

        torch.save({
            "policy": policy.state_dict(),
            "llm": llm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "tokenizer": tokenizer,
            "llm_name": llm_name,
            "llm_hidden_size": llm_hidden_size
        }, os.path.join(output_dir, f"supervised_checkpoint_epoch_{epoch+1}.pt"))

    # --------------------------
    # Phase 2: GRPO Fine-tuning
    # --------------------------
    print("\n" + "="*60)
    print("PHASE 2: GRPO REINFORCEMENT LEARNING FINE-TUNING")
    print("="*60 + "\n")
    policy.train()
    llm.train()
    patience_counter = 0  # Reset patience

    for epoch in range(num_grpo_epochs):
        total_loss = 0.0
        total_reward = 0.0
        num_updates = 0

        for batch_samples in train_loader:

            all_rollouts = []
            for sample in batch_samples:
                sample_rollouts = generate_samples_with_policy(
                    policy, llm, tokenizer, text_model, sample, device,
                    num_generations_per_sample=num_generations_per_sample,
                    max_new_tokens=generate_max_new_tokens,
                    temperature=generate_temperature,
                    top_p=generate_top_p
                )
                all_rollouts.extend(sample_rollouts)

            avg_reward = np.mean([r['reward'] for r in all_rollouts])
            total_reward += avg_reward
            # #region debug-point D:grpo-update-summary
            reward_values = [float(r['reward']) for r in all_rollouts]
            _debug_report(
                "D",
                "train_gnn_rl.py:822",
                "[DEBUG] GRPO update reward summary",
                {
                    "epoch": epoch + 1,
                    "update": num_updates + 1,
                    "num_rollouts": len(all_rollouts),
                    "avg_reward": float(avg_reward),
                    "min_reward": min(reward_values) if reward_values else None,
                    "max_reward": max(reward_values) if reward_values else None,
                    "sample_rewards": reward_values[:4],
                },
            )
            # #endregion

            print(f"GRPO Epoch {epoch+1}/{num_grpo_epochs}, Update {num_updates + 1}, Avg Reward: {avg_reward:.4f}")

            optimizer.zero_grad()
            
            with autocast():
                loss = compute_grpo_loss(policy, llm, tokenizer, text_model, all_rollouts, device, clip_eps=clip_eps)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(llm.parameters()), max_norm=max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if writer is not None:
                writer.add_scalar("GRPO/loss", loss.item(), global_step)
                writer.add_scalar("GRPO/avg_reward", avg_reward, global_step)

            total_loss += loss.item()
            num_updates += 1

        avg_epoch_loss = total_loss / max(num_updates, 1)
        avg_epoch_reward = total_reward / max(num_updates, 1)
        val_reward = evaluate_on_dataset(val_dataset, policy, llm, tokenizer, text_model, device)

        if writer is not None:
            writer.add_scalar("GRPO/avg_loss", avg_epoch_loss, epoch+1)
            writer.add_scalar("GRPO/val_reward", val_reward, epoch+1)

        print(f"\nGRPO Epoch {epoch+1} Complete!")
        print(f"Avg Loss: {avg_epoch_loss:.4f}, Avg Train Reward: {avg_epoch_reward:.4f}, Val Reward: {val_reward:.4f}\n")

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            patience_counter = 0
            torch.save({
                "policy": policy.state_dict(),
                "llm": llm.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "tokenizer": tokenizer,
                "llm_name": llm_name,
                "llm_hidden_size": llm_hidden_size,
                "epoch": epoch+1,
                "val_reward": val_reward,
                "phase": "grpo"
            }, os.path.join(output_dir, "best_checkpoint.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} GRPO epochs!")
                break

        torch.save({
            "policy": policy.state_dict(),
            "llm": llm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "tokenizer": tokenizer,
            "llm_name": llm_name,
            "llm_hidden_size": llm_hidden_size
        }, os.path.join(output_dir, f"grpo_checkpoint_epoch_{epoch+1}.pt"))

    # Final test evaluation with best checkpoint
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60 + "\n")

    checkpoint = torch.load(os.path.join(output_dir, "best_checkpoint.pt"), map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    llm.load_state_dict(checkpoint["llm"])

    test_reward = evaluate_on_dataset(test_dataset, policy, llm, tokenizer, text_model, device)
    print(f"Test Average Reward: {test_reward:.4f}\n")

    # Save final
    torch.save({
        "policy": policy.state_dict(),
        "llm": llm.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "tokenizer": tokenizer,
        "llm_name": llm_name,
        "llm_hidden_size": llm_hidden_size,
        "test_reward": test_reward
    }, os.path.join(output_dir, "final_checkpoint.pt"))
    llm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if writer is not None:
        writer.close()
    print("Training Complete!")


if __name__ == "__main__":
    main()
