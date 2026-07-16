#!/usr/bin/env python3
"""
Training script for LLM + GRPO (Group Relative Policy Optimization).
First trains with supervised teacher-forcing, then fine-tunes with GRPO RL!
Uses text-based graph representation to condition LLM on graph information.
"""

import os
import json
import random
import numpy as np
import time
import urllib.request
import argparse
from importlib import metadata as importlib_metadata
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
# from sentence_transformers import SentenceTransformer  # Not used anymore
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
# from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool  # Not used anymore

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PEFT_AVAILABLE = False

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


def _parse_version_tuple(version_str: str):
    parts = []
    for token in str(version_str).replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def get_bitsandbytes_4bit_status():
    try:
        version = importlib_metadata.version("bitsandbytes")
    except importlib_metadata.PackageNotFoundError:
        return False, (
            "4-bit quantization is enabled in config.json, but `bitsandbytes` is not installed. "
            "Falling back to non-4bit loading. Install it with `pip install -U bitsandbytes>=0.46.1` "
            "to re-enable 4-bit quantization."
        )

    if _parse_version_tuple(version) < (0, 46, 1):
        return False, (
            f"4-bit quantization requires `bitsandbytes>=0.46.1`, but found {version}. "
            "Falling back to non-4bit loading. Upgrade it with `pip install -U bitsandbytes>=0.46.1` "
            "to re-enable 4-bit quantization."
        )
    return True, f"bitsandbytes {version} detected; using 4-bit quantization."


def extract_llm_checkpoint_state(llm):
    has_peft = bool(getattr(llm, "peft_config", None))
    if has_peft:
        llm_state = {}
        for name, param in llm.named_parameters():
            if param.requires_grad or "lora_" in name or "modules_to_save" in name:
                llm_state[name] = param.detach().cpu()
        return llm_state, "trainable_only"

    llm_state = {name: tensor.detach().cpu() for name, tensor in llm.state_dict().items()}
    return llm_state, "full"


def build_checkpoint_payload(
    policy,
    llm,
    llm_name: str,
    llm_hidden_size: int,
    optimizer=None,
    scheduler=None,
    extra: Optional[Dict[str, Any]] = None,
):
    llm_state, llm_checkpoint_mode = extract_llm_checkpoint_state(llm)
    payload = {
        "policy": {name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()},
        "llm": llm_state,
        "llm_checkpoint_mode": llm_checkpoint_mode,
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
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"Warning: Failed to save checkpoint to {path}: {e}")
        print("Training will continue without saving this checkpoint")
        # Don't raise - allow training to continue


class LLMPolicy(nn.Module):
    def __init__(
        self,
        llm_hidden_size: int,
        pooling_strategy: str = "mean",
    ):
        super().__init__()
        self.pooling_strategy = str(pooling_strategy or "mean").lower()
        self.classifier_norm = nn.LayerNorm(llm_hidden_size)
        self.classifier_dropout = nn.Dropout(0.2)
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
        
        # Enhanced classifier heads with better regularization
        self.step_head = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size // 2),
            nn.LayerNorm(llm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(llm_hidden_size // 2, len(STEP_LABELS))
        )
        
        self.mcp_head = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size // 2),
            nn.LayerNorm(llm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(llm_hidden_size // 2, len(MCP_LABELS))
        )

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
        hidden = self.classifier_dropout(self.classifier_norm(pooled_hidden))
        step_logits = self.step_head(hidden)
        mcp_logits = self.mcp_head(hidden)
        return step_logits, mcp_logits


class PenTestDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        text_model=None,  # Not used anymore, kept for compatibility
        max_seq_length=1024,
        prompt_style: str = "full",
    ):
        self.data = data
        self.text_model = text_model  # Not used, kept for compatibility
        self.max_seq_length = max_seq_length
        self.prompt_style = str(prompt_style or "full").lower()
        self.samples = []
        self.skipped_unknown_step = 0
        self._prepare_samples()

    def _prepare_samples(self):
        for sample in self.data:
            step_pair = sample['step_pair']
            if step_label_to_id(step_pair.get('next_step')) is None:
                self.skipped_unknown_step += 1
                continue
            self.samples.append(sample)

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
            ),
            'previous_text': build_previous_text(step_pair, prompt_style=self.prompt_style),
            'step_pair': step_pair,
            'step_label': step_id,
            'step_explanation': step_pair.get('next_step_explanation', ''),
            'mcp_multihot': raw_mcp_to_multihot(step_pair['next_mcp_tasks']),
        }


def collate_fn(batch):
    """
    Collate function for PenTestDataset (each sample is variable-length, keep as list)
    """
    return batch


def graph_to_text(nodes, edges) -> str:
    """Convert graph structure to textual representation."""
    if not nodes:
        return "No graph information available."
    
    # Build node descriptions
    node_descriptions = []
    for node in nodes:
        node_id = node.get('id', 'unknown')
        node_type = node.get('type', 'unknown')
        node_label = node.get('label', 'unknown')
        node_title = node.get('title', '')
        desc = f"Node {node_id}: type={node_type}, label={node_label}"
        if node_title:
            desc += f", title={node_title}"
        node_descriptions.append(desc)
    
    # Build edge descriptions
    edge_descriptions = []
    for edge in edges:
        from_node = edge.get('from', 'unknown')
        to_node = edge.get('to', 'unknown')
        edge_descriptions.append(f"{from_node} -> {to_node}")
    
    # Combine into graph description
    graph_text = "Graph Structure:\n"
    graph_text += "Nodes:\n" + "\n".join(f"  - {desc}" for desc in node_descriptions) + "\n"
    if edge_descriptions:
        graph_text += "Edges:\n" + "\n".join(f"  - {edge}" for edge in edge_descriptions)
    else:
        graph_text += "Edges: None"
    
    return graph_text


def _prompt_fields(step_pair: Dict[str, Any], prompt_style: str = "full"):
    prompt_style = str(prompt_style or "full").lower()
    if prompt_style == "compact":
        return [
            ("New Strategy", step_pair.get('next_strategy', '')),
            ("Strategy Explanation", step_pair.get('next_strategy_explanation', '')),
        ]

    return [
        ("New Strategy", step_pair.get('next_strategy', '')),
        ("Strategy Explanation", step_pair.get('next_strategy_explanation', '')),
    ]


def build_previous_text(step_pair: Dict[str, Any], prompt_style: str = "full") -> str:
    parts = [value for _, value in _prompt_fields(step_pair, prompt_style=prompt_style)]
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def build_prompt_text(
    step_pair: Dict[str, Any],
    prompt_style: str = "full",
) -> str:
    context_lines = "\n".join(
        f"{label}: {value}"
        for label, value in _prompt_fields(step_pair, prompt_style=prompt_style)
    )
    graph_text = graph_to_text(step_pair.get('nodes', []), step_pair.get('edges', []))
    return (
        "### Graph Information ###\n"
        f"{graph_text}\n\n"
        "### New Strategy and Explanation ###\n"
        f"{context_lines}\n\n"
        "### Prediction Task ###\n"
        "Predict the next Step label, Step explanation, and MCP tool labels from the fixed ontology."
    )


def load_processed_data(csv_path: str, graph_data_dir: str = None):
    """Load data from CSV file and merge with graph data from JSON files."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    if graph_data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        graph_data_dir = os.path.join(base_dir, "embeddings_data")
    
    # Determine if this is train or test data
    if "training" in csv_path:
        graph_subdir = "train"
    else:
        graph_subdir = "test"
    
    graph_dir = os.path.join(graph_data_dir, graph_subdir)
    
    # Load graph data from JSON files
    graph_data = {}
    if os.path.exists(graph_dir):
        for json_file in os.listdir(graph_dir):
            if json_file.endswith("_processed.json"):
                machine_name = json_file.replace("_processed.json", "")
                json_path = os.path.join(graph_dir, json_file)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # Extract nodes and edges (text representation only)
                        nodes = []
                        for node in data.get('nodes', []):
                            nodes.append({
                                'id': node.get('id', ''),
                                'label': node.get('label', ''),
                                'type': node.get('type', ''),
                                'title': node.get('title', ''),
                            })
                        edges = []
                        for edge in data.get('edges', []):
                            edges.append({
                                'source': edge.get('source', ''),
                                'target': edge.get('target', ''),
                                'label': edge.get('label', ''),
                            })
                        graph_data[machine_name] = {
                            'nodes': nodes,
                            'edges': edges
                        }
                except Exception as e:
                    print(f"Warning: Failed to load graph data for {machine_name}: {e}")
    
    # Convert CSV to list of dicts matching expected format
    data = []
    for _, row in df.iterrows():
        machine_name = row.get('Machine', '')
        
        # Map CSV columns to expected field names
        step_pair = {
            'previous_strategy': row.get('Previous strategy', ''),
            'previous_step': row.get('Previous step', ''),
            'previous_step_result': row.get('Previous step result', ''),
            'next_strategy': row.get('New strategy', ''),
            'next_strategy_explanation': row.get('Strategy explanation', ''),
            'next_step': row.get('New step', ''),
            'next_step_explanation': row.get('Step explanation', ''),
            'next_mcp_tasks': row.get('MCP_tasks', ''),
        }
        
        # Get graph data for this machine
        graph_info = graph_data.get(machine_name, {'nodes': [], 'edges': []})
        
        # Create sample with graph data
        sample = {
            'step_pair': step_pair,
            'nodes': graph_info['nodes'],
            'edges': graph_info['edges'],
        }
        data.append(sample)
    
    print(f"Loaded {len(data)} samples from {csv_path}")
    print(f"Graph data available for {len(graph_data)} machines")
    return data


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
        "train_llm_rl.py:ensure-finite",
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
    thresholds = _threshold_array(threshold, probs.shape[-1])
    return (probs >= thresholds).astype(np.float32)


def _threshold_array(threshold, size: int) -> np.ndarray:
    if isinstance(threshold, (list, tuple, np.ndarray)):
        values = np.asarray(threshold, dtype=np.float32)
        if values.size != size:
            raise ValueError(f"Expected {size} MCP thresholds, got {values.size}")
        return values
    return np.full(size, float(threshold), dtype=np.float32)


def format_threshold(threshold) -> str:
    if isinstance(threshold, (list, tuple, np.ndarray)):
        return "[" + ", ".join(f"{float(value):.2f}" for value in threshold) + "]"
    return f"{float(threshold):.2f}"


def compute_step_label_counts(dataset) -> torch.Tensor:
    counts = torch.zeros(len(STEP_LABELS), dtype=torch.float32)
    for sample in dataset.samples:
        step_id = step_label_to_id(sample['step_pair'].get('next_step'))
        if step_id is not None:
            counts[step_id] += 1.0
    return counts


def compute_mcp_label_counts(dataset) -> torch.Tensor:
    counts = torch.zeros(len(MCP_LABELS), dtype=torch.float32)
    for idx in range(len(dataset)):
        counts += torch.tensor(dataset[idx]['mcp_multihot'], dtype=torch.float32)
    return counts


def compute_mcp_pos_weights(
    dataset,
    power: float = 0.5,
    max_weight: float = 8.0,
) -> torch.Tensor:
    positives = compute_mcp_label_counts(dataset)
    total = float(len(dataset))
    negatives = torch.clamp(torch.full_like(positives, total) - positives, min=0.0)
    weights = torch.ones_like(positives)
    nonzero = positives > 0
    weights[nonzero] = torch.pow(negatives[nonzero] / positives[nonzero].clamp_min(1.0), float(power))
    weights[nonzero] = torch.clamp(weights[nonzero], min=1.0 / max_weight, max=max_weight)
    weights[~nonzero] = 0.0
    return weights


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
    # Graph is now passed as text in the prompt, no graph tokens needed
    tokenized_prompt = tokenizer(
        [sample['prompt_text']],
        return_tensors='pt',
        truncation=True,
        max_length=max_seq_length,
    ).to(device)

    outputs = llm(
        input_ids=tokenized_prompt['input_ids'],
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
    explanation_loss_weight: float = 0.5,
    step_class_weights: Optional[torch.Tensor] = None,
    mcp_pos_weights: Optional[torch.Tensor] = None,
):
    step_logits, mcp_logits = classify_sample(
        policy, llm, tokenizer, text_model, sample, device
    )
    step_target = torch.tensor([sample['step_label']], dtype=torch.long, device=device)
    mcp_target = torch.tensor(sample['mcp_multihot'], dtype=torch.float32, device=device).unsqueeze(0)
    step_loss = F.cross_entropy(step_logits, step_target, weight=step_class_weights)
    mcp_loss = F.binary_cross_entropy_with_logits(mcp_logits, mcp_target, pos_weight=mcp_pos_weights)
    
    # Explanation generation loss (for training only, not evaluated)
    explanation_loss = torch.tensor(0.0, device=device)
    if explanation_loss_weight > 0.0 and sample.get('step_explanation'):
        explanation_text = sample['step_explanation']
        if explanation_text.strip():
            explanation_tokens = tokenizer(
                explanation_text,
                return_tensors='pt',
                truncation=True,
                max_length=256,
            ).to(device)
            explanation_input_ids = explanation_tokens['input_ids']
            explanation_attention_mask = explanation_tokens['attention_mask']
            
            # Create a simple prompt for explanation generation
            explanation_prompt = f"Step: {step_id_to_label(sample['step_label'])}\nExplanation:"
            prompt_tokens = tokenizer(
                explanation_prompt,
                return_tensors='pt',
                truncation=True,
                max_length=128,
            ).to(device)
            
            # Concatenate prompt with target explanation for language modeling
            combined_input_ids = torch.cat([prompt_tokens['input_ids'], explanation_input_ids], dim=1)
            combined_attention_mask = torch.cat([prompt_tokens['attention_mask'], explanation_attention_mask], dim=1)
            
            # Shift for causal language modeling
            labels = combined_input_ids.clone()
            labels[:, :prompt_tokens['input_ids'].size(1)] = -100  # Ignore prompt tokens in loss
            
            outputs = llm(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                labels=labels,
            )
            explanation_loss = outputs.loss
    
    total_loss = step_loss_weight * step_loss + mcp_loss_weight * mcp_loss + explanation_loss_weight * explanation_loss
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
                "train_llm_rl.py:classification-rollout",
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
    was_training = policy.training
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
    policy.train(was_training)
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
    was_training = policy.training
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

    policy.train(was_training)
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
    was_training = policy.training
    policy.eval()
    llm.eval()
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
    label_probs = []
    label_targets = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            _, mcp_logits = classify_sample(policy, llm, tokenizer, text_model, sample, device)
            label_probs.append(torch.sigmoid(mcp_logits.squeeze(0)).detach().cpu().numpy())
            label_targets.append(np.asarray(sample['mcp_multihot'], dtype=np.float32))
    if label_probs:
        prob_matrix = np.stack(label_probs, axis=0)
        target_matrix = np.stack(label_targets, axis=0)
        per_label_thresholds = []
        for label_idx in range(len(MCP_LABELS)):
            best_label_threshold = 0.5
            best_label_f1 = -1.0
            for threshold in candidate_thresholds:
                pred = (prob_matrix[:, label_idx] >= threshold).astype(np.float32)
                true = target_matrix[:, label_idx]
                tp = float((pred * true).sum())
                fp = float((pred * (1.0 - true)).sum())
                fn = float(((1.0 - pred) * true).sum())
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0)
                denom = precision + recall
                label_f1 = 0.0 if denom == 0.0 else 2.0 * precision * recall / denom
                if label_f1 > best_label_f1:
                    best_label_f1 = label_f1
                    best_label_threshold = threshold
            per_label_thresholds.append(best_label_threshold)

        per_label_metrics = evaluate_metrics_on_dataset(
            dataset,
            policy,
            llm,
            tokenizer,
            text_model,
            device,
            threshold=per_label_thresholds,
            selection_step_weight=selection_step_weight,
            selection_mcp_weight=selection_mcp_weight,
            selection_both_exact_weight=selection_both_exact_weight,
        )
        per_label_metrics["threshold"] = per_label_thresholds
        if (
            best_metrics is None
            or per_label_metrics["selection_score"] > best_metrics["selection_score"]
            or (
                per_label_metrics["selection_score"] == best_metrics["selection_score"]
                and per_label_metrics["mcp_micro_f1"] > best_metrics["mcp_micro_f1"]
            )
        ):
            best_metrics = per_label_metrics
    policy.train(was_training)
    llm.eval()
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

    # Text model not needed for text-only approach
    text_model = None

    # Load data
    data_dir = os.path.join(base_dir, config['paths']['data_dir'])
    full_train_data = load_processed_data(os.path.join(data_dir, "training_data.csv"))
    test_data = load_processed_data(os.path.join(data_dir, "test_data.csv"))
    train_data, val_data = split_train_val(full_train_data, val_split=config['training']['validation_split'])

    # Load tokenizer and LLM
    llm_name = config['model']['llm_name']
    trust_remote_code = bool(config.get('model', {}).get('trust_remote_code', False))

    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded")

    torch_dtype_name = str(config.get('model', {}).get('torch_dtype', 'float16')).lower()
    if torch_dtype_name in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif torch_dtype_name in {"fp16", "float16"}:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    load_in_4bit = bool(config.get('model', {}).get('load_in_4bit', False))
    use_lora = bool(config.get('model', {}).get('use_lora', False))

    if use_lora and not PEFT_AVAILABLE:
        raise ImportError(
            "LoRA is enabled in config.json, but the `peft` package is not installed. "
            "Install it with `pip install peft` or set `model.use_lora` to false."
        )

    quant_config = None
    device_map = None
    if load_in_4bit:
        bnb_ok, bnb_msg = get_bitsandbytes_4bit_status()
        if not bnb_ok:
            print(f"Warning: {bnb_msg}")
            load_in_4bit = False
        else:
            print(bnb_msg)
    if load_in_4bit:
        compute_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        device_map = "auto"

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quant_config,
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.gradient_checkpointing_enable()
    if device_map is None:
        llm.to(device)

    if use_lora:
        if load_in_4bit:
            llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
        lora_target_modules = list(config.get('model', {}).get('lora_target_modules', []))
        lora_config = LoraConfig(
            r=int(config.get('model', {}).get('lora_r', 16)),
            lora_alpha=int(config.get('model', {}).get('lora_alpha', 32)),
            lora_dropout=float(config.get('model', {}).get('lora_dropout', 0.05)),
            target_modules=lora_target_modules if lora_target_modules else None,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)

    llm_hidden_size = llm.config.hidden_size
    pooling_strategy = str(config.get('model', {}).get('pooling_strategy', 'hybrid')).lower()
    prompt_style = str(config.get('training', {}).get('prompt_style', 'compact')).lower()

    # Initialize policy
    policy = LLMPolicy(
        llm_hidden_size=llm_hidden_size,
        pooling_strategy=pooling_strategy,
    ).to(device)

    # Training setup first (define batch_size before creating loader)
    num_supervised_epochs = config['training']['num_supervised_epochs']
    num_grpo_epochs = config['training']['num_grpo_epochs']
    batch_size = config['training']['batch_size']
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 4)
    
    # Create datasets and dataloaders
    max_seq_length = config.get('training', {}).get('max_seq_length', 1024)
    train_dataset = PenTestDataset(
        train_data,
        text_model,
        max_seq_length=max_seq_length,
        prompt_style=prompt_style,
    )
    val_dataset = PenTestDataset(
        val_data,
        text_model,
        max_seq_length=max_seq_length,
        prompt_style=prompt_style,
    )
    test_dataset = PenTestDataset(
        test_data,
        text_model,
        max_seq_length=max_seq_length,
        prompt_style=prompt_style,
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
    generate_top_p = config['training']['generate_top_p']
    patience = config['training']['patience']
    step_loss_weight = config['training'].get('step_loss_weight', 1.5)
    mcp_loss_weight = config['training'].get('mcp_loss_weight', 1.5)
    explanation_loss_weight = config['training'].get('explanation_loss_weight', 0.5)
    rl_aux_supervised_weight = float(config['training'].get('rl_aux_supervised_weight', 0.1))
    step_class_weighting = bool(config['training'].get('step_class_weighting', True))
    step_class_weight_power = float(config['training'].get('step_class_weight_power', 0.5))
    max_step_class_weight = float(config['training'].get('max_step_class_weight', 4.0))
    mcp_class_weighting = bool(config['training'].get('mcp_class_weighting', True))
    mcp_class_weight_power = float(config['training'].get('mcp_class_weight_power', 0.5))
    max_mcp_class_weight = float(config['training'].get('max_mcp_class_weight', 8.0))
    use_weighted_sampler = bool(config['training'].get('use_weighted_sampler', True))
    sampler_power = float(config['training'].get('sampler_power', 1.0))
    selection_step_weight = float(config['training'].get('selection_step_weight', 0.75))
    selection_mcp_weight = float(config['training'].get('selection_mcp_weight', 0.25))
    selection_both_exact_weight = float(config['training'].get('selection_both_exact_weight', 0.0))
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
    mcp_pos_weights = None
    if mcp_class_weighting:
        mcp_pos_weights = compute_mcp_pos_weights(
            train_dataset,
            power=mcp_class_weight_power,
            max_weight=max_mcp_class_weight,
        ).to(device)
        print(f"Using MCP positive weights: {mcp_pos_weights.detach().cpu().tolist()}")
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
    
    # Calculate total steps and amp_enabled BEFORE printing configuration
    supervised_updates_per_epoch = len(train_loader)
    total_supervised_steps = num_supervised_epochs * supervised_updates_per_epoch
    total_grpo_steps = num_grpo_epochs * len(train_loader)
    total_steps = total_supervised_steps + total_grpo_steps
    
    # Determine if AMP is enabled (needed for config display)
    amp_enabled = device.type == "cuda" and not load_in_4bit and torch_dtype != torch.float16
    
    # Print training configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {llm_name}")
    print(f"Device: {device}")
    print(f"Pooling Strategy: {pooling_strategy}")
    print(f"Prompt Style: {prompt_style}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"\nSupervised Training:")
    print(f"  Epochs: {num_supervised_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Step Loss Weight: {step_loss_weight}")
    print(f"  MCP Loss Weight: {mcp_loss_weight}")
    print(f"  Explanation Loss Weight: {explanation_loss_weight}")
    print(f"\nGRPO Training:")
    print(f"  Epochs: {num_grpo_epochs}")
    print(f"  Generations per Sample: {num_generations_per_sample}")
    print(f"  Temperature: {generate_temperature}")
    print(f"  Clip Epsilon: {clip_eps}")
    print(f"  RL Aux Supervised Weight: {rl_aux_supervised_weight}")
    print(f"\nOther:")
    print(f"  Total Steps: {total_steps}")
    print(f"  Warmup Steps: {num_warmup_steps}")
    print(f"  Max Grad Norm: {max_grad_norm}")
    print(f"  Patience: {patience}")
    print(f"  AMP Enabled: {amp_enabled}")
    print(f"  4-bit Quantization: {load_in_4bit}")
    print(f"  LoRA: {use_lora}")
    print("="*60 + "\n")

    # Optimizer and scheduler
    llm_trainable_params = [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + llm_trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )
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
    print(f"Starting training with {len(train_dataset)} samples...")
    print(f"Progress will be shown every 10 samples (initially) and every 50 samples thereafter.\n")
    policy.train()
    llm.train()

    for epoch in range(num_supervised_epochs):
        print(f"Starting Supervised Epoch {epoch+1}/{num_supervised_epochs}...")
        epoch_start_time = time.time()
        total_loss = 0.0
        total_step_loss = 0.0
        total_mcp_loss = 0.0
        num_samples = 0

        for batch_idx, batch_samples in enumerate(train_loader):
            for sample in batch_samples:
                # Show progress for first few samples and periodically
                if num_samples < 10 or num_samples % 10 == 0:
                    print(f"  Processing sample {num_samples + 1}/{len(train_dataset)}...", end='\r', flush=True)
                
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    loss, step_loss, mcp_loss = compute_supervised_loss_for_sample(
                        policy, llm, tokenizer, text_model, sample, device,
                        step_loss_weight=step_loss_weight,
                        mcp_loss_weight=mcp_loss_weight,
                        explanation_loss_weight=explanation_loss_weight,
                        step_class_weights=step_class_weights,
                        mcp_pos_weights=mcp_pos_weights,
                    )
                if not torch.isfinite(loss):
                    print(f"\n  WARNING: Non-finite loss at sample {num_samples + 1}, skipping...")
                    _debug_report(
                        "F",
                        "train_llm_rl.py:supervised-loss",
                        "[DEBUG] Skipping non-finite supervised loss",
                        {"epoch": epoch + 1, "sample_index": num_samples + 1},
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

                total_loss += loss.item() * gradient_accumulation_steps
                total_step_loss += step_loss.item()
                total_mcp_loss += mcp_loss.item()
                num_samples += 1

                # Step optimizer after accumulating gradients
                if (num_samples % gradient_accumulation_steps == 0):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(policy.parameters()) + llm_trainable_params, max_norm=max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    
                    # Clear GPU cache
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                if writer is not None and num_samples % gradient_accumulation_steps == 0:
                    writer.add_scalar("Supervised/loss", loss.item() * gradient_accumulation_steps, global_step)
                    writer.add_scalar("Supervised/step_loss", step_loss.item(), global_step)
                    writer.add_scalar("Supervised/mcp_loss", mcp_loss.item(), global_step)
                if num_samples > 0 and num_samples % 50 == 0:
                    elapsed = time.time() - epoch_start_time
                    samples_per_sec = num_samples / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(train_dataset) - num_samples) / samples_per_sec if samples_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60
                    avg_loss = total_loss / num_samples
                    print(f"\n[Supervised] Epoch {epoch+1}/{num_supervised_epochs}, "
                        f"Sample {num_samples}/{len(train_dataset)}, "
                        f"Avg Loss: {avg_loss:.4f}, "
                        f"Step CE: {total_step_loss / num_samples:.4f}, "
                        f"MCP BCE: {total_mcp_loss / num_samples:.4f}, "
                        f"Speed: {samples_per_sec:.2f} samples/sec, "
                        f"ETA: {eta_minutes:.1f}m"
                    )

        print(f"\n  Epoch {epoch+1} training completed in {(time.time() - epoch_start_time) / 60:.1f} minutes")
        avg_epoch_loss = total_loss / max(num_samples, 1)
        avg_epoch_step_loss = total_step_loss / max(num_samples, 1)
        avg_epoch_mcp_loss = total_mcp_loss / max(num_samples, 1)
        
        if num_samples == 0:
            print(f"WARNING: All samples in supervised epoch {epoch+1} were skipped due to non-finite losses!")
        
        print(f"  Evaluating on validation set...")
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
            f"\n{'='*60}\n"
            f"Supervised Epoch {epoch+1}/{num_supervised_epochs} Complete!\n"
            f"{'='*60}\n"
            f"  Training Metrics:\n"
            f"    Avg Loss: {avg_epoch_loss:.4f}\n"
            f"    Step CE Loss: {avg_epoch_step_loss:.4f}\n"
            f"    MCP BCE Loss: {avg_epoch_mcp_loss:.4f}\n"
            f"    Loss Weights: step={step_loss_weight:.2f}, mcp={mcp_loss_weight:.2f}\n"
            f"  Validation Metrics:\n"
            f"    Val Reward: {val_reward:.4f}\n"
            f"    Val Step Accuracy: {val_metrics['step_acc']:.4f}\n"
            f"    Val MCP F1: {val_metrics['mcp_f1']:.4f}\n"
            f"    Val MCP Exact: {val_metrics['mcp_exact']:.4f}\n"
            f"    Selection Score: {val_metrics['selection_score']:.4f}\n"
            f"    Best Threshold: {format_threshold(val_metrics['threshold'])}\n"
            f"{'='*60}\n"
        )

        # Early stopping check
        if val_metrics["selection_score"] > best_val_combined:
            best_val_combined = val_metrics["selection_score"]
            best_val_reward = val_reward
            best_mcp_threshold = val_metrics["threshold"]
            patience_counter = 0
            print(f"✓ New best model! Saving checkpoint (Selection Score: {val_metrics['selection_score']:.4f})")
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
                        "pooling_strategy": pooling_strategy,
                        "prompt_style": prompt_style,
                    },
                ),
                os.path.join(output_dir, "best_supervised_checkpoint.pt"),
            )
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{patience})")

        atomic_torch_save(
            build_checkpoint_payload(
                policy,
                llm,
                llm_name,
                llm_hidden_size,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={
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
    print(f"Starting GRPO training with {len(train_loader)} batches per epoch...\n")
    policy.train()
    llm.train()
    patience_counter = 0  # Reset patience

    for epoch in range(num_grpo_epochs):
        print(f"Starting GRPO Epoch {epoch+1}/{num_grpo_epochs}...")
        epoch_start_time = time.time()
        total_loss = 0.0
        total_grpo_loss = 0.0
        total_aux_sup_loss = 0.0
        total_reward = 0.0
        num_updates = 0

        for batch_idx, batch_samples in enumerate(train_loader):
            print(f"  Processing batch {batch_idx + 1}/{len(train_loader)}, generating rollouts...", end='\r', flush=True)

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
                "train_llm_rl.py:grpo-reward",
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

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                grpo_loss = compute_grpo_loss(
                    policy, llm, tokenizer, text_model, all_rollouts, device, clip_eps=clip_eps
                )
                aux_sup_loss = torch.zeros((), device=device, dtype=grpo_loss.dtype)
                if rl_aux_supervised_weight > 0.0:
                    aux_loss_terms = []
                    for sample in batch_samples:
                        sup_loss, _, _ = compute_supervised_loss_for_sample(
                            policy,
                            llm,
                            tokenizer,
                            text_model,
                            sample,
                            device,
                            step_loss_weight=step_loss_weight,
                            mcp_loss_weight=mcp_loss_weight,
                            explanation_loss_weight=explanation_loss_weight,
                            step_class_weights=step_class_weights,
                            mcp_pos_weights=mcp_pos_weights,
                        )
                        aux_loss_terms.append(sup_loss)
                    if aux_loss_terms:
                        aux_sup_loss = torch.stack(aux_loss_terms).mean()
                loss = grpo_loss + rl_aux_supervised_weight * aux_sup_loss
            if not torch.isfinite(loss):
                print(f"\n  WARNING: Non-finite GRPO loss at update {num_updates + 1}, skipping...")
                _debug_report(
                    "F",
                    "train_llm_rl.py:grpo-loss",
                    "[DEBUG] Skipping non-finite GRPO loss",
                    {"epoch": epoch + 1, "update": num_updates + 1},
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + llm_trainable_params, max_norm=max_grad_norm
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
            
            # Print progress AFTER update is complete
            elapsed = time.time() - epoch_start_time
            updates_per_sec = num_updates / elapsed if elapsed > 0 else 0
            eta_seconds = (len(train_loader) - num_updates) / updates_per_sec if updates_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            print(f"\n[GRPO] Epoch {epoch+1}/{num_grpo_epochs}, "
                  f"Update {num_updates}/{len(train_loader)}, "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Speed: {updates_per_sec:.2f} updates/sec, "
                  f"ETA: {eta_minutes:.1f}m")

        print(f"\n  GRPO Epoch {epoch+1} training completed in {(time.time() - epoch_start_time) / 60:.1f} minutes")
        avg_epoch_loss = total_loss / max(num_updates, 1)
        avg_epoch_grpo_loss = total_grpo_loss / max(num_updates, 1)
        avg_epoch_aux_sup_loss = total_aux_sup_loss / max(num_updates, 1)
        avg_epoch_reward = total_reward / max(num_updates, 1)
        
        print(f"  Evaluating on validation set...")
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

        print(
            f"\n{'='*60}\n"
            f"GRPO Epoch {epoch+1}/{num_grpo_epochs} Complete!\n"
            f"{'='*60}\n"
            f"  Training Metrics:\n"
            f"    Avg Loss: {avg_epoch_loss:.4f}\n"
            f"    Avg GRPO Loss: {avg_epoch_grpo_loss:.4f}\n"
            f"    Avg Aux Sup Loss: {avg_epoch_aux_sup_loss:.4f}\n"
            f"    Avg Train Reward: {avg_epoch_reward:.4f}\n"
            f"  Validation Metrics:\n"
            f"    Val Reward: {val_reward:.4f}\n"
            f"    Val Step Accuracy: {val_metrics['step_acc']:.4f}\n"
            f"    Val MCP F1: {val_metrics['mcp_f1']:.4f}\n"
            f"    Val MCP Exact: {val_metrics['mcp_exact']:.4f}\n"
            f"    Selection Score: {val_metrics['selection_score']:.4f}\n"
            f"    Best Threshold: {format_threshold(val_metrics['threshold'])}\n"
            f"{'='*60}\n"
        )

        if val_metrics["selection_score"] > best_val_combined:
            best_val_combined = val_metrics["selection_score"]
            best_val_reward = val_reward
            best_mcp_threshold = val_metrics["threshold"]
            patience_counter = 0
            print(f"✓ New best model! Saving checkpoint (Selection Score: {val_metrics['selection_score']:.4f})")
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
                        "pooling_strategy": pooling_strategy,
                        "prompt_style": prompt_style,
                    },
                ),
                os.path.join(output_dir, "best_checkpoint.pt"),
            )
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} GRPO epochs (no improvement for {patience} epochs)")
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

    final_eval_checkpoint = os.path.join(output_dir, "best_checkpoint.pt")
    if not os.path.exists(final_eval_checkpoint):
        final_eval_checkpoint = os.path.join(output_dir, "best_supervised_checkpoint.pt")
    checkpoint = torch.load(final_eval_checkpoint, map_location=device, weights_only=False)
    missing, unexpected = policy.load_state_dict(checkpoint["policy"], strict=False)
    if missing:
        print(f"Policy checkpoint missing {len(missing)} newly initialized keys.")
    if unexpected:
        print(f"Policy checkpoint had {len(unexpected)} unexpected keys.")
    llm_checkpoint_mode = checkpoint.get("llm_checkpoint_mode", "full")
    if llm_checkpoint_mode == "full":
        llm.load_state_dict(checkpoint["llm"])
    else:
        llm.load_state_dict(checkpoint["llm"], strict=False)
    best_mcp_threshold = checkpoint.get("mcp_threshold", best_mcp_threshold)

    test_metrics = evaluate_metrics_on_dataset(
        test_dataset, policy, llm, tokenizer, text_model, device, threshold=best_mcp_threshold
    )
    print(f"{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Step Metrics:")
    print(f"    Accuracy: {test_metrics['step_acc']:.4f}")
    print(f"    Micro F1: {test_metrics['step_micro_f1']:.4f}")
    print(f"\n  MCP Metrics:")
    print(f"    F1 Score: {test_metrics['mcp_f1']:.4f}")
    print(f"    Micro F1: {test_metrics['mcp_micro_f1']:.4f}")
    print(f"    Accuracy: {test_metrics['mcp_acc']:.4f}")
    print(f"    Exact Match: {test_metrics['mcp_exact']:.4f}")
    print(f"\n  Combined:")
    print(f"    Average Reward: {test_metrics['avg_reward']:.4f}")
    print(f"    Both Exact: {test_metrics['both_exact']:.4f}")
    print(f"    MCP Threshold: {format_threshold(best_mcp_threshold)}")
    print(f"{'='*60}\n")

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
                "test_mcp_f1": test_metrics["mcp_f1"],
                "test_mcp_micro_f1": test_metrics["mcp_micro_f1"],
                "test_mcp_exact": test_metrics["mcp_exact"],
                "mcp_threshold": best_mcp_threshold,
                "pooling_strategy": pooling_strategy,
                "prompt_style": prompt_style,
            },
        ),
        os.path.join(output_dir, "final_checkpoint.pt"),
    )
    llm.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if writer is not None:
        writer.close()
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best checkpoint saved to: {output_dir}/best_checkpoint.pt")
    print(f"Final checkpoint saved to: {output_dir}/final_checkpoint.pt")
    print(f"Model saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
