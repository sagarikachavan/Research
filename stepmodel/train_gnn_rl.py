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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

from label_space import (
    STEP_LABELS,
    MCP_LABELS,
    step_label_to_id,
    raw_mcp_to_multihot,
    step_id_to_label,
    multihot_to_mcp_tools,
    set_f1,
    classification_reward,
)


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
        self.classifier_dropout = nn.Dropout(0.1)
        self.step_head = nn.Linear(llm_hidden_size, len(STEP_LABELS))
        self.mcp_head = nn.Linear(llm_hidden_size, len(MCP_LABELS))

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

    def classify(self, pooled_hidden: torch.Tensor):
        hidden = self.classifier_dropout(pooled_hidden)
        return self.step_head(hidden), self.mcp_head(hidden)


class PenTestDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], text_model: SentenceTransformer, max_seq_length=1024):
        self.data = data
        self.text_model = text_model
        self.max_seq_length = max_seq_length
        self.samples = []
        self.skipped_unknown_step = 0
        self._prepare_samples()

    def _prepare_samples(self):
        for machine in self.data:
            nodes = machine['nodes']
            edges = machine['edges']
            for step_pair in machine['step_pairs']:
                if step_label_to_id(step_pair.get('next_step')) is None:
                    self.skipped_unknown_step += 1
                    continue
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
        step_id = step_label_to_id(step_pair['next_step'])
        if step_id is None:
            raise ValueError(f"Filtered dataset still contains unknown next_step label: {step_pair['next_step']}")
        return {
            'nodes': sample['nodes'],
            'edges': sample['edges'],
            'prompt_text': build_prompt_text(step_pair),
            'previous_text': build_previous_text(step_pair),
            'step_pair': step_pair,
            'step_label': step_id,
            'mcp_multihot': raw_mcp_to_multihot(step_pair['next_mcp_tasks']),
        }


def collate_fn(batch):
    """
    Collate function for PenTestDataset (each sample is variable-length, keep as list)
    """
    return batch


def build_previous_text(step_pair: Dict[str, Any]) -> str:
    parts = [
        step_pair.get('previous_strategy', ''),
        step_pair.get('previous_strategy_explanation', ''),
        step_pair.get('previous_step', ''),
        step_pair.get('previous_step_explanation', ''),
        step_pair.get('previous_step_result', ''),
        step_pair.get('previous_mcp_tasks', ''),
    ]
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def build_prompt_text(step_pair: Dict[str, Any]) -> str:
    return (
        "[GRAPH]\n"
        "### Previous Penetration Testing Context ###\n"
        f"Strategy: {step_pair.get('previous_strategy', '')}\n"
        f"Strategy Explanation: {step_pair.get('previous_strategy_explanation', '')}\n"
        f"Step: {step_pair.get('previous_step', '')}\n"
        f"Step Explanation: {step_pair.get('previous_step_explanation', '')}\n"
        f"Result: {step_pair.get('previous_step_result', '')}\n"
        f"MCP Tasks: {step_pair.get('previous_mcp_tasks', '')}\n\n"
        "### Prediction Task ###\n"
        "Predict the next Step label and the MCP tool labels from the fixed ontology."
    )


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


def compute_reward(
    pred_step_id: int,
    true_step_id: int,
    pred_mcp_multihot,
    true_mcp_multihot,
    threshold: float = 0.5,
) -> float:
    return classification_reward(
        pred_step_id,
        true_step_id,
        pred_mcp_multihot,
        true_mcp_multihot,
        threshold=threshold,
    )


def predict_mcp_multihot(mcp_logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = torch.sigmoid(mcp_logits).detach().cpu().numpy()
    return (probs >= threshold).astype(np.float32)


def classify_sample(
    policy,
    llm,
    tokenizer,
    text_model,
    sample,
    device,
    max_seq_length: int = 1024,
):
    previous_emb = torch.tensor(
        text_model.encode([sample['previous_text']], convert_to_numpy=True),
        dtype=torch.float32,
        device=device,
    )
    policy_out = policy(sample['nodes'], sample['edges'], previous_emb, device)
    tokenized_prompt = tokenizer(
        [sample['prompt_text']],
        return_tensors='pt',
        truncation=True,
        max_length=max_seq_length,
    ).to(device)
    inputs_embeds = llm.get_input_embeddings()(tokenized_prompt['input_ids'])
    inputs_embeds[:, 0, :] = policy_out

    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=tokenized_prompt['attention_mask'],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    mask = tokenized_prompt['attention_mask'].unsqueeze(-1).float()
    pooled_hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    step_logits, mcp_logits = policy.classify(pooled_hidden)
    return step_logits, mcp_logits


def compute_supervised_loss_for_sample(
    policy,
    llm,
    tokenizer,
    text_model,
    sample,
    device,
):
    step_logits, mcp_logits = classify_sample(
        policy, llm, tokenizer, text_model, sample, device
    )
    step_target = torch.tensor([sample['step_label']], dtype=torch.long, device=device)
    mcp_target = torch.tensor(sample['mcp_multihot'], dtype=torch.float32, device=device).unsqueeze(0)
    step_loss = F.cross_entropy(step_logits, step_target)
    mcp_loss = F.binary_cross_entropy_with_logits(mcp_logits, mcp_target)
    total_loss = step_loss + mcp_loss
    return total_loss, step_loss.detach(), mcp_loss.detach()


def generate_samples_with_policy(
    policy, llm, tokenizer, text_model, sample, device, num_generations_per_sample=4,
    max_new_tokens=1024, temperature=0.9, top_p=0.95
):
    del max_new_tokens, top_p
    rollouts = []

    with torch.no_grad():
        step_logits, mcp_logits = classify_sample(
            policy, llm, tokenizer, text_model, sample, device
        )
        step_dist = torch.distributions.Categorical(logits=step_logits.squeeze(0) / max(temperature, 1e-6))
        mcp_probs = torch.sigmoid(mcp_logits.squeeze(0) / max(temperature, 1e-6)).clamp(1e-6, 1 - 1e-6)
        true_step_id = int(sample['step_label'])
        true_mcp_multihot = np.asarray(sample['mcp_multihot'], dtype=np.float32)

        for _ in range(num_generations_per_sample):
            step_action = step_dist.sample()
            mcp_action = torch.bernoulli(mcp_probs)
            bernoulli_log_prob = (
                mcp_action * torch.log(mcp_probs) + (1 - mcp_action) * torch.log(1 - mcp_probs)
            ).sum()
            old_log_prob = step_dist.log_prob(step_action) + bernoulli_log_prob
            reward = compute_reward(
                int(step_action.item()),
                true_step_id,
                mcp_action.cpu().numpy(),
                true_mcp_multihot,
            )

            _debug_report(
                "E",
                "train_gnn_rl.py:classification-rollout",
                "[DEBUG] Sampled classification rollout",
                {
                    "step_action": step_id_to_label(int(step_action.item())),
                    "mcp_action": sorted(multihot_to_mcp_tools(mcp_action.cpu().numpy())),
                    "reward": float(reward),
                },
            )

            rollouts.append({
                'sample': sample,
                'step_action': int(step_action.item()),
                'mcp_action': mcp_action.detach().cpu().numpy().astype(np.float32),
                'old_log_prob': float(old_log_prob.item()),
                'reward': float(reward),
            })

    return rollouts


def compute_grpo_loss(
    policy, llm, tokenizer, text_model, rollouts, device, clip_eps=0.2
):
    rewards = torch.tensor([r['reward'] for r in rollouts], dtype=torch.float32, device=device)
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - mean_r) / std_r

    total_loss = 0.0
    for i, rollout in enumerate(rollouts):
        sample = rollout['sample']
        step_logits, mcp_logits = classify_sample(
            policy, llm, tokenizer, text_model, sample, device
        )
        step_dist = torch.distributions.Categorical(logits=step_logits.squeeze(0))
        mcp_probs = torch.sigmoid(mcp_logits.squeeze(0)).clamp(1e-6, 1 - 1e-6)

        step_action = torch.tensor(rollout['step_action'], dtype=torch.long, device=device)
        mcp_action = torch.tensor(rollout['mcp_action'], dtype=torch.float32, device=device)
        bernoulli_log_prob = (
            mcp_action * torch.log(mcp_probs) + (1 - mcp_action) * torch.log(1 - mcp_probs)
        ).sum()
        new_log_prob = step_dist.log_prob(step_action) + bernoulli_log_prob

        ratio = torch.exp(new_log_prob - rollout['old_log_prob'])
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        advantage = advantages[i]
        total_loss = total_loss + (-torch.min(ratio * advantage, clipped_ratio * advantage))

    return total_loss / max(len(rollouts), 1)


def evaluate_on_dataset(
    dataset, policy, llm, tokenizer, text_model, device, num_samples: Optional[int] = None
):
    """Evaluate label-only Step/MCP reward on a dataset."""
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
            step_logits, mcp_logits = classify_sample(
                policy, llm, tokenizer, text_model, sample, device
            )
            pred_step_id = int(step_logits.argmax(dim=-1).item())
            pred_mcp = predict_mcp_multihot(mcp_logits.squeeze(0))
            reward = compute_reward(
                pred_step_id,
                int(sample['step_label']),
                pred_mcp,
                sample['mcp_multihot'],
            )
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
    if train_dataset.skipped_unknown_step or val_dataset.skipped_unknown_step or test_dataset.skipped_unknown_step:
        print(
            "Skipped samples with unknown Step labels: "
            f"train={train_dataset.skipped_unknown_step}, "
            f"val={val_dataset.skipped_unknown_step}, "
            f"test={test_dataset.skipped_unknown_step}"
        )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
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
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

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
        total_step_loss = 0.0
        total_mcp_loss = 0.0
        num_samples = 0

        for batch_samples in train_loader:
            for sample in batch_samples:
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    loss, step_loss, mcp_loss = compute_supervised_loss_for_sample(
                        policy, llm, tokenizer, text_model, sample, device
                    )

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(llm.parameters()), max_norm=max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                total_loss += loss.item()
                total_step_loss += step_loss.item()
                total_mcp_loss += mcp_loss.item()
                num_samples += 1

                if writer is not None:
                    writer.add_scalar("Supervised/loss", loss.item(), global_step)
                    writer.add_scalar("Supervised/step_loss", step_loss.item(), global_step)
                    writer.add_scalar("Supervised/mcp_loss", mcp_loss.item(), global_step)
                if num_samples % 100 == 0:
                    avg_loss = total_loss / num_samples
                    print(
                        f"Epoch {epoch+1}/{num_supervised_epochs}, "
                        f"Sample {num_samples}/{len(train_dataset)}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Step CE: {total_step_loss / num_samples:.4f}, "
                        f"MCP BCE: {total_mcp_loss / num_samples:.4f}"
                    )

        avg_epoch_loss = total_loss / num_samples
        avg_epoch_step_loss = total_step_loss / num_samples
        avg_epoch_mcp_loss = total_mcp_loss / num_samples
        val_reward = evaluate_on_dataset(val_dataset, policy, llm, tokenizer, text_model, device)

        if writer is not None:
            writer.add_scalar("Supervised/avg_loss", avg_epoch_loss, epoch+1)
            writer.add_scalar("Supervised/avg_step_loss", avg_epoch_step_loss, epoch+1)
            writer.add_scalar("Supervised/avg_mcp_loss", avg_epoch_mcp_loss, epoch+1)
            writer.add_scalar("Supervised/val_reward", val_reward, epoch+1)

        print(
            f"\nSupervised Epoch {epoch+1} Complete! "
            f"Avg Loss: {avg_epoch_loss:.4f}, "
            f"Step CE: {avg_epoch_step_loss:.4f}, "
            f"MCP BCE: {avg_epoch_mcp_loss:.4f}, "
            f"Val Reward: {val_reward:.4f}\n"
        )

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
                "step_labels": STEP_LABELS,
                "mcp_labels": MCP_LABELS,
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
            "llm_hidden_size": llm_hidden_size,
            "step_labels": STEP_LABELS,
            "mcp_labels": MCP_LABELS,
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

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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
                "step_labels": STEP_LABELS,
                "mcp_labels": MCP_LABELS,
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
            "llm_hidden_size": llm_hidden_size,
            "step_labels": STEP_LABELS,
            "mcp_labels": MCP_LABELS,
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
        "step_labels": STEP_LABELS,
        "mcp_labels": MCP_LABELS,
        "test_reward": test_reward
    }, os.path.join(output_dir, "final_checkpoint.pt"))
    llm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if writer is not None:
        writer.close()
    print("Training Complete!")


if __name__ == "__main__":
    main()
