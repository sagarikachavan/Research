#!/usr/bin/env python3
"""
Evaluation script for the label-only Step/MCP version of stepmodel.
"""

import os
import json
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tokenizers.models import Model as TokenizersModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer

from label_space import (
    STEP_LABELS,
    MCP_LABELS,
    multihot_to_mcp_tools,
    set_f1,
)
from train_gnn_rl import (
    GNNLLMPolicy,
    PenTestDataset,
    classify_sample,
    compute_reward,
    predict_mcp_multihot,
)


def infer_llm_name_from_state_dict(llm_state_dict):
    if not llm_state_dict:
        return None

    layer_ids = set()
    pattern = re.compile(r"transformer\.h\.(\d+)\.")
    for key in llm_state_dict.keys():
        match = pattern.match(key)
        if match:
            layer_ids.add(int(match.group(1)))

    num_layers = (max(layer_ids) + 1) if layer_ids else None
    hidden_size = None
    if "transformer.wte.weight" in llm_state_dict:
        hidden_size = llm_state_dict["transformer.wte.weight"].shape[1]

    if num_layers == 6 and hidden_size == 768:
        return "distilgpt2"
    if num_layers == 12 and hidden_size == 768:
        return "gpt2"
    if num_layers == 24 and hidden_size == 1024:
        return "gpt2-medium"
    if num_layers == 36 and hidden_size == 1280:
        return "gpt2-large"
    if num_layers == 48 and hidden_size == 1600:
        return "gpt2-xl"
    return None


def infer_policy_hidden_size(policy_state_dict):
    weight = policy_state_dict.get("combine.0.weight")
    if weight is None:
        return None
    return weight.shape[0]


def checkpoint_has_label_heads(policy_state_dict):
    required = {
        "step_head.weight",
        "step_head.bias",
        "mcp_head.weight",
        "mcp_head.bias",
    }
    return required.issubset(set(policy_state_dict.keys()))


def find_compatible_checkpoint(checkpoints_dir: str, device):
    candidate_names = [
        "best_checkpoint.pt",
        "best_supervised_checkpoint.pt",
    ]
    candidate_paths = [Path(checkpoints_dir) / name for name in candidate_names]
    candidate_paths.extend(sorted(Path(checkpoints_dir).glob("grpo_checkpoint_epoch_*.pt"), reverse=True))
    candidate_paths.extend(sorted(Path(checkpoints_dir).glob("supervised_checkpoint_epoch_*.pt"), reverse=True))

    checked = []
    for path in candidate_paths:
        if not path.exists():
            continue
        checked.append(str(path))
        with torch.serialization.safe_globals([GPT2Tokenizer, Tokenizer, TokenizersModel]):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        policy_state = checkpoint.get("policy", {})
        if checkpoint_has_label_heads(policy_state):
            return str(path), checkpoint, checked
    return None, None, checked


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "config.json"), "r") as f:
        config = json.load(f)

    embeddings_dir = os.path.join(base_dir, config['paths']['embeddings_dir'])
    checkpoints_dir = os.path.join(base_dir, config['paths']['output_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_model = SentenceTransformer(config['model']['text_embedding_model'])
    text_emb_dim = text_model.get_embedding_dimension()

    with open(os.path.join(embeddings_dir, "test", "all_processed.json"), "r") as f:
        test_data = json.load(f)

    checkpoint_path, checkpoint, checked_paths = find_compatible_checkpoint(checkpoints_dir, device)
    if checkpoint_path is None:
        if checked_paths:
            print("No compatible label-only checkpoint found.")
            print("Checked:")
            for path in checked_paths:
                print(f"  - {path}")
            print("Run `python3 train_gnn_rl.py` to create a new compatible checkpoint, then rerun evaluation.")
            return
        print("No checkpoint found. Run `python3 train_gnn_rl.py` first, then rerun evaluation.")
        return

    checkpoint_llm_name = config['model']['llm_name']
    if checkpoint is not None:
        checkpoint_llm_name = (
            checkpoint.get("llm_name")
            or infer_llm_name_from_state_dict(checkpoint.get("llm", {}))
            or checkpoint_llm_name
        )

    checkpoint_tokenizer = checkpoint.get("tokenizer") if checkpoint is not None else None
    if checkpoint_tokenizer is not None:
        tokenizer = checkpoint_tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_llm_name)
        if '[GRAPH]' not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': ['[GRAPH]']})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(checkpoint_llm_name)
    llm.resize_token_embeddings(len(tokenizer))
    llm.to(device)
    llm.eval()

    policy_hidden_size = llm.config.hidden_size
    if checkpoint is not None:
        inferred_policy_hidden_size = infer_policy_hidden_size(checkpoint.get("policy", {}))
        if inferred_policy_hidden_size is not None:
            policy_hidden_size = inferred_policy_hidden_size

    policy = GNNLLMPolicy(
        gnn_out_dim=config['model']['gnn_out_dim'],
        text_emb_dim=text_emb_dim,
        llm_hidden_size=policy_hidden_size,
        use_gat=config['model']['use_gat']
    ).to(device)

    if checkpoint is not None:
        policy_state = checkpoint["policy"]
        policy.load_state_dict(policy_state)
        llm.load_state_dict(checkpoint["llm"])
        print(f"Loaded compatible checkpoint: {checkpoint_path}")

    policy.eval()
    test_dataset = PenTestDataset(
        test_data,
        text_model,
        max_seq_length=config.get('training', {}).get('max_seq_length', 1024),
    )

    total_reward = 0.0
    total_step_correct = 0
    total_mcp_correct = 0
    total_both_correct = 0
    total_mcp_exact = 0
    total_both_exact = 0
    total_mcp_f1 = 0.0
    mcp_tp = 0
    mcp_fp = 0
    mcp_fn = 0

    print("Evaluating on full test dataset...")

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            step_logits, mcp_logits = classify_sample(
                policy, llm, tokenizer, text_model, sample, device
            )
            pred_step_id = int(step_logits.argmax(dim=-1).item())
            pred_mcp_multihot = predict_mcp_multihot(mcp_logits.squeeze(0))
            true_mcp_multihot = sample['mcp_multihot']

            reward = compute_reward(
                pred_step_id,
                int(sample['step_label']),
                pred_mcp_multihot,
                true_mcp_multihot,
            )

            pred_tools = multihot_to_mcp_tools(pred_mcp_multihot)
            true_tools = multihot_to_mcp_tools(true_mcp_multihot)
            mcp_f1 = set_f1(pred_tools, true_tools)
            step_correct = int(pred_step_id == int(sample['step_label']))
            mcp_correct = int(mcp_f1 >= 0.5)
            mcp_exact = int(pred_tools == true_tools)

            total_reward += reward
            total_step_correct += step_correct
            total_mcp_correct += mcp_correct
            total_both_correct += int(step_correct and mcp_correct)
            total_mcp_exact += mcp_exact
            total_both_exact += int(step_correct and mcp_exact)
            total_mcp_f1 += mcp_f1

            pred_arr = pred_mcp_multihot.astype(int)
            true_arr = true_mcp_multihot.astype(int)
            mcp_tp += int(((pred_arr == 1) & (true_arr == 1)).sum())
            mcp_fp += int(((pred_arr == 1) & (true_arr == 0)).sum())
            mcp_fn += int(((pred_arr == 0) & (true_arr == 1)).sum())

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1} samples, current average: {total_reward / (idx + 1):.4f}")

    num_samples = len(test_dataset)
    avg_reward = total_reward / max(num_samples, 1)
    step_acc = total_step_correct / max(num_samples, 1)
    mcp_acc = total_mcp_correct / max(num_samples, 1)
    both_acc = total_both_correct / max(num_samples, 1)
    mcp_exact_acc = total_mcp_exact / max(num_samples, 1)
    both_exact_acc = total_both_exact / max(num_samples, 1)
    avg_mcp_f1 = total_mcp_f1 / max(num_samples, 1)
    micro_precision = mcp_tp / max(mcp_tp + mcp_fp, 1)
    micro_recall = mcp_tp / max(mcp_tp + mcp_fn, 1)
    micro_denom = micro_precision + micro_recall
    mcp_micro_f1 = 0.0 if micro_denom == 0 else 2 * micro_precision * micro_recall / micro_denom

    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION ON FULL DATASET ({num_samples} samples):")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Step Accuracy: {step_acc:.4f}")
    print(f"MCP F1 (set): {avg_mcp_f1:.4f}")
    print(f"MCP Micro F1 (global): {mcp_micro_f1:.4f}")
    print(f"MCP Accuracy (F1>=0.5): {mcp_acc:.4f}")
    print(f"Both Step+MCP Accuracy: {both_acc:.4f}")
    print(f"MCP Exact Set Match: {mcp_exact_acc:.4f}")
    print(f"Both Exact Match: {both_exact_acc:.4f}")
    print(f"Fixed Step Labels: {len(STEP_LABELS)}")
    print(f"Fixed MCP Labels: {len(MCP_LABELS)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
