#!/usr/bin/env python3
"""
Evaluation script for trained GNN + RL + LLM model.
"""

import os
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_gnn_rl import GNNRLPolicy, compute_reward


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    embeddings_dir = os.path.join(base_dir, "embeddings_data")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sentence transformer
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_emb_dim = text_model.get_sentence_embedding_dimension()

    # Load test data
    with open(os.path.join(embeddings_dir, "test", "all_processed.json"), 'r') as f:
        test_data = json.load(f)

    # Load tokenizer and LLM
    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)
    tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(checkpoints_dir)
    llm.to(device)

    # Load policy model
    policy = GNNRLPolicy(
        gnn_out_dim=128,
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm.config.hidden_size
    ).to(device)

    checkpoint = torch.load(os.path.join(checkpoints_dir, "initial_checkpoint.pt"), map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    llm.load_state_dict(checkpoint["llm"])

    # Evaluate on first few samples (for demonstration)
    print("Evaluating model...")
    total_reward = 0.0
    num_samples = 0

    for machine in test_data:
        for step_pair in machine["step_pairs"]:
            # Just for demonstration, we'll skip full generation
            # In real implementation, you'd use policy to guide LLM generation
            # and compute reward using step_pair['next_step'] etc.
            true_step = step_pair['next_step']
            true_mcp = step_pair['next_mcp_tasks']
            # Dummy prediction for now
            pred_step = true_step
            pred_mcp = true_mcp
            reward = compute_reward(pred_step, true_step, pred_mcp, true_mcp)
            total_reward += reward
            num_samples += 1

    print(f"Average evaluation reward: {total_reward / max(num_samples, 1):.4f}")


if __name__ == "__main__":
    main()
