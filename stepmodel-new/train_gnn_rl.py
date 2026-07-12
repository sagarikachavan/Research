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
import argparse
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_curriculum_dataset(dataset, stage: int, total_stages: int, quality_threshold: float = 0.7):
    """
    Create curriculum learning dataset based on training stage.
    
    Args:
        dataset: Full dataset
        stage: Current curriculum stage (0 to total_stages-1)
        total_stages: Total number of curriculum stages
        quality_threshold: Minimum quality score for samples
    
    Returns:
        Subset of dataset for current stage
    """
    if total_stages <= 1:
        return dataset
    
    # Calculate progress through curriculum
    progress = (stage + 1) / total_stages
    
    # For early stages, use only high-quality samples
    # For later stages, gradually include more samples
    if stage == 0:
        # Stage 0: Only highest quality samples
        threshold = quality_threshold + 0.2
    elif stage < total_stages - 1:
        # Middle stages: Progressive quality threshold
        threshold = quality_threshold + (0.2 * (1 - progress))
    else:
        # Final stage: All samples above base threshold
        threshold = quality_threshold
    
    # Filter dataset based on quality scores if available
    filtered_indices = []
    for idx, sample in enumerate(dataset):
        # If sample has quality score, use it
        if hasattr(sample, 'quality_score'):
            if sample.quality_score >= threshold:
                filtered_indices.append(idx)
        else:
            # If no quality score, include all samples in later stages
            if stage >= total_stages - 2:
                filtered_indices.append(idx)
    
    if len(filtered_indices) == 0:
        # Fallback: return random subset if no quality scores
        subset_size = int(len(dataset) * progress)
        filtered_indices = random.sample(range(len(dataset)), min(subset_size, len(dataset)))
    
    # Create subset
    from torch.utils.data import Subset
    return Subset(dataset, filtered_indices)


def freeze_module(module: nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def snapshot_trainable_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
        if torch.is_floating_point(tensor)
    }


def drift_guard_loss(module: nn.Module, reference_state: Dict[str, torch.Tensor], device) -> torch.Tensor:
    if not reference_state:
        return torch.zeros((), device=device)
    terms = []
    current_state = module.state_dict()
    for name, ref in reference_state.items():
        cur = current_state.get(name)
        if cur is not None and torch.is_floating_point(cur):
            terms.append(F.mse_loss(cur, ref.to(device)))
    if not terms:
        return torch.zeros((), device=device)
    return torch.stack(terms).mean()


def build_checkpoint_payload(
    policy,
    llm,
    llm_name: str,
    llm_hidden_size: int,
    optimizer=None,
    scheduler=None,
    extra: Optional[Dict[str, Any]] = None,
):
    del llm
    payload = {
        "policy": {name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()},
        "llm_checkpoint_mode": "frozen_external",
        "llm_name": llm_name,
        "llm_hidden_size": llm_hidden_size,
        "step_labels": STEP_LABELS,
        "mcp_labels": MCP_LABELS,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    return payload


def atomic_torch_save(obj, path: str):
    tmp_path = f"{path}.tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


class GNNModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        output_dim: int,
        gnn_type: str = "gcn",
        use_gat: bool = False,
    ):
        super().__init__()
        if use_gat:
            gnn_type = "gat"
        self.gnn_type = str(gnn_type or "gcn").lower()
        if self.gnn_type == "gat":
            self.conv1 = GATConv(node_dim, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
            self.fc = nn.Linear(hidden_dim * 4, output_dim)
        elif self.gnn_type == "gcn":
            self.conv1 = GCNConv(node_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, output_dim)
        elif self.gnn_type == "sage":
            self.conv1 = SAGEConv(node_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class GNNLLMPolicy(nn.Module):
    def __init__(
        self,
        gnn_out_dim: int,
        text_emb_dim: int,
        llm_hidden_size: int,
        gnn_type: str = "gcn",
        use_gat: bool = False,
        pooling_strategy: str = "mean",
        graph_token_count: int = 4,
    ):
        super().__init__()
        self.pooling_strategy = str(pooling_strategy or "mean").lower()
        self.graph_token_count = int(graph_token_count)
        self.gnn = GNNModel(
            node_dim=text_emb_dim,
            hidden_dim=256,
            output_dim=gnn_out_dim,
            gnn_type=gnn_type,
            use_gat=use_gat,
        )
        self.project_step_text = nn.Sequential(
            nn.Linear(text_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, gnn_out_dim)
        )
        self.combine = nn.Sequential(
            nn.Linear(gnn_out_dim * 2, gnn_out_dim),
            nn.ReLU()
        )
        self.graph_token_projector = nn.Linear(gnn_out_dim, llm_hidden_size * self.graph_token_count)
        self.classifier_dropout = nn.Dropout(0.1)
        if self.pooling_strategy == "hybrid":
            self.readout_projection = nn.Sequential(
                nn.LayerNorm(llm_hidden_size * 2),
                nn.Linear(llm_hidden_size * 2, llm_hidden_size),
                nn.ReLU(),
            )
        elif self.pooling_strategy in {"mean", "first"}:
            self.readout_projection = nn.Identity()
        else:
            raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")
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
        graph_tokens = self.graph_token_projector(combined)
        return graph_tokens.view(combined.size(0), self.graph_token_count, -1)

    def pool_hidden_states(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        mean_hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if self.pooling_strategy == "mean":
            return mean_hidden

        first_hidden = hidden[:, 0, :]
        if self.pooling_strategy == "first":
            return first_hidden
        if self.pooling_strategy == "hybrid":
            return self.readout_projection(torch.cat([first_hidden, mean_hidden], dim=-1))
        raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

    def classify(self, pooled_hidden: torch.Tensor):
        hidden = self.classifier_dropout(pooled_hidden)
        return self.step_head(hidden), self.mcp_head(hidden)


class PenTestDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        text_model: SentenceTransformer,
        max_seq_length=1024,
        prompt_style: str = "full",
        graph_token_count: int = 1,
    ):
        self.data = data
        self.text_model = text_model
        self.max_seq_length = max_seq_length
        self.prompt_style = str(prompt_style or "full").lower()
        self.graph_token_count = int(graph_token_count)
        self.samples = []
        self.skipped_unknown_step = 0
        self._prepare_samples()

    def _prepare_samples(self):
        for machine in self.data:
            machine_nodes = machine.get('nodes', [])
            machine_edges = machine.get('edges', [])
            for step_pair in machine['step_pairs']:
                if step_label_to_id(step_pair.get('next_step')) is None:
                    self.skipped_unknown_step += 1
                    continue
                self.samples.append({
                    'nodes': step_pair.get('nodes', machine_nodes),
                    'edges': step_pair.get('edges', machine_edges),
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
            'prompt_text': build_prompt_text(
                step_pair,
                prompt_style=self.prompt_style,
                graph_token_count=self.graph_token_count,
            ),
            'previous_text': build_previous_text(step_pair, prompt_style=self.prompt_style),
            'step_pair': step_pair,
            'step_label': step_id,
            'mcp_multihot': raw_mcp_to_multihot(step_pair['next_mcp_tasks']),
        }


def collate_fn(batch):
    """
    Collate function for PenTestDataset (each sample is variable-length, keep as list)
    """
    return batch


def _prompt_fields(step_pair: Dict[str, Any], prompt_style: str = "full"):
    prompt_style = str(prompt_style or "full").lower()
    if prompt_style == "compact":
        return [
            ("Strategy", step_pair.get('previous_strategy', '')),
            ("Step", step_pair.get('previous_step', '')),
            ("Result", step_pair.get('previous_step_result', '')),
            ("MCP Tasks", step_pair.get('previous_mcp_tasks', '')),
        ]

    return [
        ("Strategy", step_pair.get('previous_strategy', '')),
        ("Strategy Explanation", step_pair.get('previous_strategy_explanation', '')),
        ("Step", step_pair.get('previous_step', '')),
        ("Step Explanation", step_pair.get('previous_step_explanation', '')),
        ("Result", step_pair.get('previous_step_result', '')),
        ("MCP Tasks", step_pair.get('previous_mcp_tasks', '')),
    ]


def build_previous_text(step_pair: Dict[str, Any], prompt_style: str = "full") -> str:
    parts = [value for _, value in _prompt_fields(step_pair, prompt_style=prompt_style)]
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def build_prompt_text(
    step_pair: Dict[str, Any],
    prompt_style: str = "full",
    graph_token_count: int = 1,
) -> str:
    context_lines = "\n".join(
        f"{label}: {value}"
        for label, value in _prompt_fields(step_pair, prompt_style=prompt_style)
    )
    graph_prefix = " ".join(["[GRAPH]"] * max(int(graph_token_count), 1))
    return (
        f"{graph_prefix}\n"
        "### Previous Penetration Testing Context ###\n"
        f"{context_lines}\n\n"
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


def _ensure_finite_tensor(tensor: torch.Tensor, name: str, clip_value: float = 50.0) -> torch.Tensor:
    """Replace NaN/Inf values so training can continue instead of crashing."""
    if torch.isfinite(tensor).all():
        return tensor.clamp(min=-clip_value, max=clip_value)

    non_finite = (~torch.isfinite(tensor)).sum().item()
    _debug_report(
        "F",
        "train_gnn_rl.py:ensure-finite",
        "[DEBUG] Non-finite tensor sanitized",
        {
            "name": name,
            "shape": list(tensor.shape),
            "non_finite_count": int(non_finite),
        },
    )
    cleaned = torch.nan_to_num(tensor, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return cleaned.clamp(min=-clip_value, max=clip_value)


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


def compute_step_label_counts(dataset) -> torch.Tensor:
    counts = torch.zeros(len(STEP_LABELS), dtype=torch.float32)
    for sample in dataset.samples:
        step_id = step_label_to_id(sample['step_pair'].get('next_step'))
        if step_id is not None:
            counts[step_id] += 1.0
    return counts


def compute_step_class_weights(
    dataset,
    power: float = 0.5,
    max_weight: float = 4.0,
) -> torch.Tensor:
    counts = compute_step_label_counts(dataset)
    nonzero = counts > 0
    weights = torch.ones_like(counts)
    if nonzero.any():
        reference = counts[nonzero].mean()
        weights[nonzero] = torch.pow(reference / counts[nonzero], float(power))
        weights[nonzero] = torch.clamp(weights[nonzero], min=1.0 / max_weight, max=max_weight)
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean().clamp_min(1e-8)
    weights[~nonzero] = 0.0
    return weights


def build_step_weighted_sampler(
    dataset,
    power: float = 1.0,
):
    counts = compute_step_label_counts(dataset)
    sample_weights = []
    for sample in dataset.samples:
        step_id = step_label_to_id(sample['step_pair'].get('next_step'))
        if step_id is None or counts[step_id] <= 0:
            sample_weights.append(0.0)
        else:
            sample_weights.append(float(torch.pow(counts[step_id], -float(power)).item()))

    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, counts


def compute_selection_score(
    metrics: Dict[str, float],
    step_weight: float = 0.75,
    mcp_weight: float = 0.25,
    both_exact_weight: float = 0.0,
) -> float:
    return (
        float(step_weight) * float(metrics["step_acc"])
        + float(mcp_weight) * float(metrics["mcp_f1"])
        + float(both_exact_weight) * float(metrics["both_exact"])
    )


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
    previous_emb = _ensure_finite_tensor(previous_emb, "previous_emb")
    graph_tokens = policy(sample['nodes'], sample['edges'], previous_emb, device)
    graph_tokens = _ensure_finite_tensor(graph_tokens, "graph_tokens")
    tokenized_prompt = tokenizer(
        [sample['prompt_text']],
        return_tensors='pt',
        truncation=True,
        max_length=max_seq_length,
    ).to(device)
    inputs_embeds = llm.get_input_embeddings()(tokenized_prompt['input_ids']).clone()
    num_graph_tokens = min(policy.graph_token_count, inputs_embeds.size(1))
    inputs_embeds[:, :num_graph_tokens, :] = graph_tokens[:, :num_graph_tokens, :]

    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=tokenized_prompt['attention_mask'],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = _ensure_finite_tensor(outputs.hidden_states[-1], "llm_hidden")
    pooled_hidden = policy.pool_hidden_states(hidden, tokenized_prompt['attention_mask'])
    pooled_hidden = _ensure_finite_tensor(pooled_hidden, "pooled_hidden")
    step_logits, mcp_logits = policy.classify(pooled_hidden)
    return _ensure_finite_tensor(step_logits, "step_logits"), _ensure_finite_tensor(mcp_logits, "mcp_logits")


def compute_supervised_loss_for_sample(
    policy,
    llm,
    tokenizer,
    text_model,
    sample,
    device,
    step_loss_weight: float = 1.0,
    mcp_loss_weight: float = 1.5,
    step_class_weights: Optional[torch.Tensor] = None,
):
    step_logits, mcp_logits = classify_sample(
        policy, llm, tokenizer, text_model, sample, device
    )
    step_target = torch.tensor([sample['step_label']], dtype=torch.long, device=device)
    mcp_target = torch.tensor(sample['mcp_multihot'], dtype=torch.float32, device=device).unsqueeze(0)
    step_loss = F.cross_entropy(step_logits, step_target, weight=step_class_weights)
    mcp_loss = F.binary_cross_entropy_with_logits(mcp_logits, mcp_target)
    total_loss = step_loss_weight * step_loss + mcp_loss_weight * mcp_loss
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
        with torch.no_grad():
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
        
        # Clear intermediate tensors to free memory
        del step_logits, mcp_logits, step_dist, mcp_probs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

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
    llm.eval()
    return avg_reward


def evaluate_metrics_on_dataset(
    dataset,
    policy,
    llm,
    tokenizer,
    text_model,
    device,
    threshold: float = 0.5,
    num_samples: Optional[int] = None,
    selection_step_weight: float = 0.75,
    selection_mcp_weight: float = 0.25,
    selection_both_exact_weight: float = 0.0,
):
    """CNN-style exact Step accuracy + multi-label MCP metrics."""
    policy.eval()
    llm.eval()

    total_reward = 0.0
    total_step_correct = 0
    total_mcp_exact = 0
    total_mcp_f1 = 0.0
    total_mcp_prec = 0.0
    total_mcp_rec = 0.0
    total_both_exact = 0
    mcp_tp = 0
    mcp_fp = 0
    mcp_fn = 0
    total_step_predictions = 0
    total_mcp_label_correct = 0.0
    total_mcp_label_count = 0
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
            pred_mcp = predict_mcp_multihot(mcp_logits.squeeze(0), threshold=threshold)
            true_mcp = np.asarray(sample['mcp_multihot'], dtype=np.float32)

            reward = compute_reward(
                pred_step_id,
                int(sample['step_label']),
                pred_mcp,
                true_mcp,
                threshold=threshold,
            )

            pred_arr = pred_mcp.astype(np.float32)
            true_arr = true_mcp.astype(np.float32)
            tp = float((pred_arr * true_arr).sum())
            fp = float((pred_arr * (1.0 - true_arr)).sum())
            fn = float(((1.0 - pred_arr) * true_arr).sum())

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

            step_correct = int(pred_step_id == int(sample['step_label']))
            mcp_exact = int((pred_arr == true_arr).all())

            total_reward += reward
            total_step_correct += step_correct
            total_mcp_exact += mcp_exact
            total_mcp_f1 += f1
            total_mcp_prec += precision
            total_mcp_rec += recall
            total_both_exact += int(step_correct and mcp_exact)
            mcp_tp += int(tp)
            mcp_fp += int(fp)
            mcp_fn += int(fn)
            total_step_predictions += 1
            total_mcp_label_correct += float((pred_arr == true_arr).sum())
            total_mcp_label_count += int(true_arr.size)
            num_evals += 1

    avg_reward = total_reward / max(num_evals, 1)
    step_acc = total_step_correct / max(num_evals, 1)
    step_micro_precision = total_step_correct / max(total_step_predictions, 1)
    step_micro_recall = total_step_correct / max(total_step_predictions, 1)
    step_micro_denom = step_micro_precision + step_micro_recall
    step_micro_f1 = (
        0.0
        if step_micro_denom == 0
        else 2.0 * step_micro_precision * step_micro_recall / step_micro_denom
    )
    mcp_exact = total_mcp_exact / max(num_evals, 1)
    mcp_f1 = total_mcp_f1 / max(num_evals, 1)
    mcp_prec = total_mcp_prec / max(num_evals, 1)
    mcp_rec = total_mcp_rec / max(num_evals, 1)
    mcp_acc = total_mcp_label_correct / max(total_mcp_label_count, 1)
    both_exact = total_both_exact / max(num_evals, 1)
    micro_precision = mcp_tp / max(mcp_tp + mcp_fp, 1)
    micro_recall = mcp_tp / max(mcp_tp + mcp_fn, 1)
    micro_denom = micro_precision + micro_recall
    mcp_micro_f1 = 0.0 if micro_denom == 0 else 2.0 * micro_precision * micro_recall / micro_denom

    policy.train()
    llm.eval()
    return {
        "avg_reward": avg_reward,
        "step_acc": step_acc,
        "step_micro_f1": step_micro_f1,
        "mcp_acc": mcp_acc,
        "mcp_exact": mcp_exact,
        "mcp_f1": mcp_f1,
        "mcp_prec": mcp_prec,
        "mcp_rec": mcp_rec,
        "both_exact": both_exact,
        "mcp_micro_f1": mcp_micro_f1,
        "combined_score": step_acc + mcp_f1,
        "selection_score": compute_selection_score(
            {
                "step_acc": step_acc,
                "mcp_f1": mcp_f1,
                "both_exact": both_exact,
            },
            step_weight=selection_step_weight,
            mcp_weight=selection_mcp_weight,
            both_exact_weight=selection_both_exact_weight,
        ),
        "threshold": threshold,
    }


def find_best_mcp_threshold(
    dataset,
    policy,
    llm,
    tokenizer,
    text_model,
    device,
    selection_step_weight: float = 0.75,
    selection_mcp_weight: float = 0.25,
    selection_both_exact_weight: float = 0.0,
):
    candidate_thresholds = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
    best_metrics = None
    for threshold in candidate_thresholds:
        metrics = evaluate_metrics_on_dataset(
            dataset,
            policy,
            llm,
            tokenizer,
            text_model,
            device,
            threshold=threshold,
            selection_step_weight=selection_step_weight,
            selection_mcp_weight=selection_mcp_weight,
            selection_both_exact_weight=selection_both_exact_weight,
        )
        if (
            best_metrics is None
            or metrics["selection_score"] > best_metrics["selection_score"]
            or (
                metrics["selection_score"] == best_metrics["selection_score"]
                and metrics["step_acc"] > best_metrics["step_acc"]
            )
            or (
                metrics["selection_score"] == best_metrics["selection_score"]
                and metrics["step_acc"] == best_metrics["step_acc"]
                and metrics["mcp_exact"] > best_metrics["mcp_exact"]
            )
        ):
            best_metrics = metrics
    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, config['paths']['data_dir'])
    embeddings_dir = os.path.join(base_dir, config['paths']['embeddings_dir'])
    output_dir = os.path.join(base_dir, config['paths']['output_dir'])
    log_dir = os.path.join(base_dir, config['paths']['log_dir'])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(int(config.get('training', {}).get('seed', 42)))
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
    trust_remote_code = bool(config.get('model', {}).get('trust_remote_code', False))

    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    special_tokens_dict = {'additional_special_tokens': ['[GRAPH]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")

    torch_dtype_name = str(config.get('model', {}).get('torch_dtype', 'float16')).lower()
    if torch_dtype_name in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif torch_dtype_name in {"fp16", "float16"}:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.to(device)
    freeze_module(llm)

    llm_hidden_size = llm.config.hidden_size
    gnn_type = str(config.get('model', {}).get('gnn_type', 'gcn')).lower()
    pooling_strategy = str(config.get('model', {}).get('pooling_strategy', 'hybrid')).lower()
    graph_token_count = int(config.get('model', {}).get('graph_token_count', 4))
    prompt_style = str(config.get('training', {}).get('prompt_style', 'compact')).lower()

    # Initialize policy
    policy = GNNLLMPolicy(
        gnn_out_dim=config['model']['gnn_out_dim'],
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm_hidden_size,
        gnn_type=gnn_type,
        use_gat=config['model']['use_gat'],
        pooling_strategy=pooling_strategy,
        graph_token_count=graph_token_count,
    ).to(device)

    pretrain_checkpoint_path = config.get('training', {}).get(
        'phase0_checkpoint',
        os.path.join(output_dir, "phase0_gnn_projector.pt"),
    )
    pretrained_gnn_reference = None
    if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path):
        phase0 = torch.load(pretrain_checkpoint_path, map_location=device)
        if "gnn" in phase0:
            policy.gnn.load_state_dict(phase0["gnn"], strict=False)
        if "graph_token_projector" in phase0:
            policy.graph_token_projector.load_state_dict(phase0["graph_token_projector"], strict=False)
        pretrained_gnn_reference = snapshot_trainable_state(policy.gnn)
        print(f"Loaded Phase 0 GNN/projector checkpoint: {pretrain_checkpoint_path}")
    else:
        print(f"Phase 0 checkpoint not found at {pretrain_checkpoint_path}; training from random GNN/projector init.")

    # Training setup first (define batch_size before creating loader)
    num_supervised_epochs = config['training']['num_supervised_epochs']
    num_grpo_epochs = config['training']['num_grpo_epochs']
    batch_size = config['training']['batch_size']
    
    # Create datasets and dataloaders
    max_seq_length = config.get('training', {}).get('max_seq_length', 1024)
    train_dataset = PenTestDataset(
        train_data,
        text_model,
        max_seq_length=max_seq_length,
        prompt_style=prompt_style,
        graph_token_count=graph_token_count,
    )
    val_dataset = PenTestDataset(
        val_data,
        text_model,
        max_seq_length=max_seq_length,
        prompt_style=prompt_style,
        graph_token_count=graph_token_count,
    )
    test_dataset = PenTestDataset(
        test_data,
        text_model,
        max_seq_length=max_seq_length,
        prompt_style=prompt_style,
        graph_token_count=graph_token_count,
    )
    if train_dataset.skipped_unknown_step or val_dataset.skipped_unknown_step or test_dataset.skipped_unknown_step:
        print(
            "Skipped samples with unknown Step labels: "
            f"train={train_dataset.skipped_unknown_step}, "
            f"val={val_dataset.skipped_unknown_step}, "
            f"test={test_dataset.skipped_unknown_step}"
        )
    
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    max_grad_norm = config['training']['max_grad_norm']
    num_warmup_steps = config['training']['num_warmup_steps']
    num_generations_per_sample = config['training']['num_generations_per_sample']
    clip_eps = config['training']['clip_eps']
    generate_max_new_tokens = config['training']['generate_max_new_tokens']
    generate_temperature = config['training']['generate_temperature']
    grpo_generate_temperature = float(config['training'].get('grpo_generate_temperature', max(generate_temperature, 1.1)))
    generate_top_p = config['training']['generate_top_p']
    patience = config['training']['patience']
    step_loss_weight = config['training'].get('step_loss_weight', 1.5)
    mcp_loss_weight = config['training'].get('mcp_loss_weight', 1.5)
    rl_aux_supervised_weight = float(config['training'].get('rl_aux_supervised_weight', 0.1))
    step_class_weighting = bool(config['training'].get('step_class_weighting', True))
    step_class_weight_power = float(config['training'].get('step_class_weight_power', 0.5))
    max_step_class_weight = float(config['training'].get('max_step_class_weight', 4.0))
    use_weighted_sampler = bool(config['training'].get('use_weighted_sampler', True))
    sampler_power = float(config['training'].get('sampler_power', 1.0))
    selection_step_weight = float(config['training'].get('selection_step_weight', 0.75))
    selection_mcp_weight = float(config['training'].get('selection_mcp_weight', 0.25))
    selection_both_exact_weight = float(config['training'].get('selection_both_exact_weight', 0.0))
    gnn_learning_rate = float(config['training'].get('gnn_learning_rate', learning_rate / 10.0))
    projector_learning_rate = float(config['training'].get('projector_learning_rate', gnn_learning_rate))
    drift_guard_weight = float(config['training'].get('drift_guard_weight', 0.0))
    grpo_reward_std_epsilon = float(config['training'].get('grpo_reward_std_epsilon', 1e-6))
    step_class_weights = None
    step_label_counts = compute_step_label_counts(train_dataset)
    zero_step_labels = [STEP_LABELS[i] for i, count in enumerate(step_label_counts.tolist()) if count == 0]
    if zero_step_labels:
        print(f"Warning: zero-shot Step labels in training data: {zero_step_labels}")
    if step_class_weighting:
        step_class_weights = compute_step_class_weights(
            train_dataset,
            power=step_class_weight_power,
            max_weight=max_step_class_weight,
        ).to(device)
        print(f"Using Step class weights: {step_class_weights.detach().cpu().tolist()}")
    train_sampler = None
    if use_weighted_sampler:
        train_sampler, _ = build_step_weighted_sampler(
            train_dataset,
            power=sampler_power,
        )
        print(f"Using WeightedRandomSampler for Step balance (power={sampler_power:.2f}).")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Fix total steps calculation: both phases use batch steps (Phase1 uses batch_size=1 effectively for now)
    supervised_updates_per_epoch = len(train_loader)
    total_supervised_steps = num_supervised_epochs * supervised_updates_per_epoch
    total_grpo_steps = num_grpo_epochs * len(train_loader)
    total_steps = total_supervised_steps + total_grpo_steps

    # Optimizer and scheduler. Qwen is frozen; only policy components receive gradients.
    optimizer = torch.optim.AdamW(
        [
            {"params": policy.gnn.parameters(), "lr": gnn_learning_rate},
            {"params": policy.graph_token_projector.parameters(), "lr": projector_learning_rate},
            {
                "params": list(policy.project_step_text.parameters())
                + list(policy.combine.parameters())
                + list(policy.readout_projection.parameters())
                + list(policy.step_head.parameters())
                + list(policy.mcp_head.parameters()),
                "lr": learning_rate,
            },
        ],
        weight_decay=weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    # Training
    global_step = 0
    best_val_reward = float("-inf")
    best_val_combined = float("-inf")
    best_mcp_threshold = 0.5
    patience_counter = 0

    # --------------------------
    # Phase 1: Supervised Warmup
    # --------------------------
    print("\n" + "="*60)
    print("PHASE 1: SUPERVISED WARMUP TRAINING")
    print("="*60 + "\n")
    policy.train()
    freeze_module(llm)

    for epoch in range(num_supervised_epochs):
        total_loss = 0.0
        total_step_loss = 0.0
        total_mcp_loss = 0.0
        num_samples = 0

        for batch_samples in train_loader:
            for sample in batch_samples:
                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    loss, step_loss, mcp_loss = compute_supervised_loss_for_sample(
                        policy, llm, tokenizer, text_model, sample, device,
                        step_loss_weight=step_loss_weight,
                        mcp_loss_weight=mcp_loss_weight,
                        step_class_weights=step_class_weights,
                    )
                    if drift_guard_weight > 0.0 and pretrained_gnn_reference:
                        loss = loss + drift_guard_weight * drift_guard_loss(policy.gnn, pretrained_gnn_reference, device)
                if not torch.isfinite(loss):
                    _debug_report(
                        "F",
                        "train_gnn_rl.py:supervised-loss",
                        "[DEBUG] Skipping non-finite supervised loss",
                        {"epoch": epoch + 1, "sample_index": num_samples + 1},
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=max_grad_norm
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
        val_metrics = find_best_mcp_threshold(
            val_dataset,
            policy,
            llm,
            tokenizer,
            text_model,
            device,
            selection_step_weight=selection_step_weight,
            selection_mcp_weight=selection_mcp_weight,
            selection_both_exact_weight=selection_both_exact_weight,
        )
        val_reward = val_metrics["avg_reward"]

        if writer is not None:
            writer.add_scalar("Supervised/avg_loss", avg_epoch_loss, epoch+1)
            writer.add_scalar("Supervised/avg_step_loss", avg_epoch_step_loss, epoch+1)
            writer.add_scalar("Supervised/avg_mcp_loss", avg_epoch_mcp_loss, epoch+1)
            writer.add_scalar("Supervised/val_reward", val_reward, epoch+1)
            writer.add_scalar("Supervised/val_step_acc", val_metrics["step_acc"], epoch+1)
            writer.add_scalar("Supervised/val_mcp_f1", val_metrics["mcp_f1"], epoch+1)
            writer.add_scalar("Supervised/val_mcp_exact", val_metrics["mcp_exact"], epoch+1)
            writer.add_scalar("Supervised/val_combined_score", val_metrics["combined_score"], epoch+1)
            writer.add_scalar("Supervised/val_selection_score", val_metrics["selection_score"], epoch+1)

        print(
            f"\nSupervised Epoch {epoch+1} Complete! "
            f"Avg Loss: {avg_epoch_loss:.4f}, "
            f"Step CE: {avg_epoch_step_loss:.4f}, "
            f"MCP BCE: {avg_epoch_mcp_loss:.4f}, "
            f"Loss Weights (step={step_loss_weight:.2f}, mcp={mcp_loss_weight:.2f}), "
            f"Val Reward: {val_reward:.4f}, "
            f"Val Step Acc: {val_metrics['step_acc']:.4f}, "
            f"Val MCP F1: {val_metrics['mcp_f1']:.4f}, "
            f"Val MCP Exact: {val_metrics['mcp_exact']:.4f}, "
            f"Selection Score: {val_metrics['selection_score']:.4f}, "
            f"Best Thr: {val_metrics['threshold']:.2f}\n"
        )

        # Early stopping check
        if val_metrics["selection_score"] > best_val_combined:
            best_val_combined = val_metrics["selection_score"]
            best_val_reward = val_reward
            best_mcp_threshold = val_metrics["threshold"]
            patience_counter = 0
            atomic_torch_save(
                build_checkpoint_payload(
                    policy,
                    llm,
                    llm_name,
                    llm_hidden_size,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra={
                        "epoch": epoch + 1,
                        "val_reward": val_reward,
                        "val_step_acc": val_metrics["step_acc"],
                        "val_mcp_f1": val_metrics["mcp_f1"],
                        "val_mcp_exact": val_metrics["mcp_exact"],
                        "val_combined_score": val_metrics["combined_score"],
                        "val_selection_score": val_metrics["selection_score"],
                        "mcp_threshold": val_metrics["threshold"],
                        "phase": "supervised",
                        "gnn_type": gnn_type,
                        "graph_token_count": graph_token_count,
                        "pooling_strategy": pooling_strategy,
                        "prompt_style": prompt_style,
                    },
                ),
                os.path.join(output_dir, "best_supervised_checkpoint.pt"),
            )
        else:
            patience_counter += 1

        atomic_torch_save(
            build_checkpoint_payload(
                policy,
                llm,
                llm_name,
                llm_hidden_size,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={
                    "gnn_type": gnn_type,
                    "graph_token_count": graph_token_count,
                    "pooling_strategy": pooling_strategy,
                    "prompt_style": prompt_style,
                },
            ),
            os.path.join(output_dir, f"supervised_checkpoint_epoch_{epoch+1}.pt"),
        )

    # --------------------------
    # Phase 2: GRPO Fine-tuning
    # --------------------------
    print("\n" + "="*60)
    print("PHASE 2: GRPO REINFORCEMENT LEARNING FINE-TUNING")
    print("="*60 + "\n")
    print(f"RL auxiliary supervised weight: {rl_aux_supervised_weight:.3f}")
    policy.train()
    freeze_module(llm)
    patience_counter = 0  # Reset patience

    for epoch in range(num_grpo_epochs):
        total_loss = 0.0
        total_grpo_loss = 0.0
        total_aux_sup_loss = 0.0
        total_reward = 0.0
        num_updates = 0

        for batch_samples in train_loader:

            all_rollouts = []
            for sample in batch_samples:
                sample_rollouts = generate_samples_with_policy(
                    policy, llm, tokenizer, text_model, sample, device,
                    num_generations_per_sample=num_generations_per_sample,
                    max_new_tokens=generate_max_new_tokens,
                    temperature=grpo_generate_temperature,
                    top_p=generate_top_p
                )
                all_rollouts.extend(sample_rollouts)

            avg_reward = np.mean([r['reward'] for r in all_rollouts])
            reward_std = float(np.std([r['reward'] for r in all_rollouts])) if all_rollouts else 0.0
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
                    "reward_std": reward_std,
                    "sample_rewards": reward_values[:4],
                },
            )
            # #endregion

            print(f"GRPO Epoch {epoch+1}/{num_grpo_epochs}, Update {num_updates + 1}, Avg Reward: {avg_reward:.4f}")
            if reward_std < grpo_reward_std_epsilon:
                if writer is not None:
                    writer.add_scalar("GRPO/skipped_zero_variance_group", 1.0, global_step)
                print(
                    f"Skipping GRPO update {num_updates + 1}: reward std {reward_std:.6g} "
                    f"< epsilon {grpo_reward_std_epsilon:.6g}"
                )
                continue

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                grpo_loss = compute_grpo_loss(
                    policy, llm, tokenizer, text_model, all_rollouts, device, clip_eps=clip_eps
                )
                aux_sup_loss = torch.zeros((), device=device, dtype=grpo_loss.dtype)
                if rl_aux_supervised_weight > 0.0:
                    aux_loss_terms = []
                    for sample in batch_samples:
                        with torch.amp.autocast('cuda', enabled=amp_enabled):
                            sup_loss, _, _ = compute_supervised_loss_for_sample(
                                policy,
                                llm,
                                tokenizer,
                                text_model,
                                sample,
                                device,
                                step_loss_weight=step_loss_weight,
                                mcp_loss_weight=mcp_loss_weight,
                                step_class_weights=step_class_weights,
                            )
                        aux_loss_terms.append(sup_loss)
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    if aux_loss_terms:
                        aux_sup_loss = torch.stack(aux_loss_terms).mean()
                loss = grpo_loss + rl_aux_supervised_weight * aux_sup_loss
            if not torch.isfinite(loss):
                _debug_report(
                    "F",
                    "train_gnn_rl.py:grpo-loss",
                    "[DEBUG] Skipping non-finite GRPO loss",
                    {"epoch": epoch + 1, "update": num_updates + 1},
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if writer is not None:
                writer.add_scalar("GRPO/loss", loss.item(), global_step)
                writer.add_scalar("GRPO/grpo_loss", grpo_loss.item(), global_step)
                writer.add_scalar("GRPO/aux_supervised_loss", aux_sup_loss.item(), global_step)
                writer.add_scalar("GRPO/avg_reward", avg_reward, global_step)

            total_loss += loss.item()
            total_grpo_loss += grpo_loss.item()
            total_aux_sup_loss += aux_sup_loss.item()
            num_updates += 1

        avg_epoch_loss = total_loss / max(num_updates, 1)
        avg_epoch_grpo_loss = total_grpo_loss / max(num_updates, 1)
        avg_epoch_aux_sup_loss = total_aux_sup_loss / max(num_updates, 1)
        avg_epoch_reward = total_reward / max(num_updates, 1)
        val_metrics = find_best_mcp_threshold(
            val_dataset,
            policy,
            llm,
            tokenizer,
            text_model,
            device,
            selection_step_weight=selection_step_weight,
            selection_mcp_weight=selection_mcp_weight,
            selection_both_exact_weight=selection_both_exact_weight,
        )
        val_reward = val_metrics["avg_reward"]

        if writer is not None:
            writer.add_scalar("GRPO/avg_loss", avg_epoch_loss, epoch+1)
            writer.add_scalar("GRPO/avg_grpo_loss", avg_epoch_grpo_loss, epoch+1)
            writer.add_scalar("GRPO/avg_aux_supervised_loss", avg_epoch_aux_sup_loss, epoch+1)
            writer.add_scalar("GRPO/val_reward", val_reward, epoch+1)
            writer.add_scalar("GRPO/val_step_acc", val_metrics["step_acc"], epoch+1)
            writer.add_scalar("GRPO/val_mcp_f1", val_metrics["mcp_f1"], epoch+1)
            writer.add_scalar("GRPO/val_mcp_exact", val_metrics["mcp_exact"], epoch+1)
            writer.add_scalar("GRPO/val_combined_score", val_metrics["combined_score"], epoch+1)
            writer.add_scalar("GRPO/val_selection_score", val_metrics["selection_score"], epoch+1)

        print(f"\nGRPO Epoch {epoch+1} Complete!")
        print(
            f"Avg Loss: {avg_epoch_loss:.4f}, "
            f"Avg GRPO Loss: {avg_epoch_grpo_loss:.4f}, "
            f"Avg Aux Sup Loss: {avg_epoch_aux_sup_loss:.4f}, "
            f"Avg Train Reward: {avg_epoch_reward:.4f}, "
            f"Val Reward: {val_reward:.4f}, "
            f"Val Step Acc: {val_metrics['step_acc']:.4f}, "
            f"Val MCP F1: {val_metrics['mcp_f1']:.4f}, "
            f"Val MCP Exact: {val_metrics['mcp_exact']:.4f}, "
            f"Selection Score: {val_metrics['selection_score']:.4f}, "
            f"Best Thr: {val_metrics['threshold']:.2f}\n"
        )

        if val_metrics["selection_score"] > best_val_combined:
            best_val_combined = val_metrics["selection_score"]
            best_val_reward = val_reward
            best_mcp_threshold = val_metrics["threshold"]
            patience_counter = 0
            atomic_torch_save(
                build_checkpoint_payload(
                    policy,
                    llm,
                    llm_name,
                    llm_hidden_size,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra={
                        "epoch": epoch + 1,
                        "val_reward": val_reward,
                        "val_step_acc": val_metrics["step_acc"],
                        "val_mcp_f1": val_metrics["mcp_f1"],
                        "val_mcp_exact": val_metrics["mcp_exact"],
                        "val_combined_score": val_metrics["combined_score"],
                        "val_selection_score": val_metrics["selection_score"],
                        "mcp_threshold": val_metrics["threshold"],
                        "phase": "grpo",
                        "gnn_type": gnn_type,
                        "graph_token_count": graph_token_count,
                        "pooling_strategy": pooling_strategy,
                        "prompt_style": prompt_style,
                    },
                ),
                os.path.join(output_dir, "best_checkpoint.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} GRPO epochs!")
                break

        atomic_torch_save(
            build_checkpoint_payload(
                policy,
                llm,
                llm_name,
                llm_hidden_size,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={
                    "gnn_type": gnn_type,
                    "graph_token_count": graph_token_count,
                    "pooling_strategy": pooling_strategy,
                    "prompt_style": prompt_style,
                },
            ),
            os.path.join(output_dir, f"grpo_checkpoint_epoch_{epoch+1}.pt"),
        )

    # Final test evaluation with best checkpoint
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60 + "\n")

    best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")
    if not os.path.exists(best_checkpoint_path):
        best_checkpoint_path = os.path.join(output_dir, "best_supervised_checkpoint.pt")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    best_mcp_threshold = float(checkpoint.get("mcp_threshold", best_mcp_threshold))

    test_metrics = evaluate_metrics_on_dataset(
        test_dataset, policy, llm, tokenizer, text_model, device, threshold=best_mcp_threshold
    )
    print(f"Test Average Reward: {test_metrics['avg_reward']:.4f}")
    print(f"Test Step Accuracy: {test_metrics['step_acc']:.4f}")
    print(f"Test MCP F1: {test_metrics['mcp_f1']:.4f}")
    print(f"Test MCP Exact: {test_metrics['mcp_exact']:.4f}")
    print(f"Test Both Exact: {test_metrics['both_exact']:.4f}")
    print(f"Test MCP Threshold: {best_mcp_threshold:.2f}\n")

    # Save final
    atomic_torch_save(
        build_checkpoint_payload(
            policy,
            llm,
            llm_name,
            llm_hidden_size,
            optimizer=optimizer,
            scheduler=scheduler,
            extra={
                "test_reward": test_metrics["avg_reward"],
                "test_step_acc": test_metrics["step_acc"],
                "test_step_micro_f1": test_metrics["step_micro_f1"],
                "test_mcp_acc": test_metrics["mcp_acc"],
                "test_mcp_micro_f1": test_metrics["mcp_micro_f1"],
                "test_mcp_f1": test_metrics["mcp_f1"],
                "test_mcp_exact": test_metrics["mcp_exact"],
                "mcp_threshold": best_mcp_threshold,
                "gnn_type": gnn_type,
                "graph_token_count": graph_token_count,
                "pooling_strategy": pooling_strategy,
                "prompt_style": prompt_style,
            },
        ),
        os.path.join(output_dir, "final_checkpoint.pt"),
    )
    tokenizer.save_pretrained(output_dir)

    if writer is not None:
        writer.close()
    print("Training Complete!")


if __name__ == "__main__":
    main()
