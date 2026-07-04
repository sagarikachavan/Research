#!/usr/bin/env python3
"""
Evaluation script for trained GNN + RL + LLM model.
"""

import os
import json
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_gnn_rl import GNNRLPolicy, compute_reward


def parse_prediction(pred_text):
    strategy = ""
    step = ""
    mcp_tasks = ""

    strategy_match = re.search(r"Strategy:(.*?)(?=Strategy Explanation:|Step:|Step Explanation:|MCP Tasks:|$)", pred_text, re.DOTALL)
    if strategy_match:
        strategy = strategy_match.group(1).strip()

    step_match = re.search(r"Step:(.*?)(?=Step Explanation:|MCP Tasks:|$)", pred_text, re.DOTALL)
    if step_match:
        step = step_match.group(1).strip()

    mcp_match = re.search(r"MCP Tasks:(.*?)$", pred_text, re.DOTALL)
    if mcp_match:
        mcp_tasks = mcp_match.group(1).strip()

    return strategy, step, mcp_tasks


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    embeddings_dir = os.path.join(base_dir, "embeddings_data")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sentence transformer
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_emb_dim = text_model.get_embedding_dimension()

    # Load test data
    with open(os.path.join(embeddings_dir, "test", "all_processed.json"), 'r') as f:
        test_data = json.load(f)

    # Load tokenizer and LLM
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)
    tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(checkpoints_dir)
    llm.to(device)
    llm.eval()

    # Load policy model
    policy = GNNRLPolicy(
        gnn_out_dim=128,
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm.config.hidden_size
    ).to(device)

    checkpoint_path = os.path.join(checkpoints_dir, "final_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        llm.load_state_dict(checkpoint["llm"])
        print("Loaded checkpoint successfully!")
    else:
        print("Warning: No checkpoint found, using initial weights.")

    policy.eval()

    # Evaluate on test samples
    print("Evaluating model...")
    total_reward = 0.0
    num_samples = 0
    sample_predictions = []

    # Prepare test samples
    test_samples = []
    for machine in test_data:
        nodes = machine['nodes']
        edges = machine['edges']
        for step_pair in machine['step_pairs']:
            test_samples.append({
                'nodes': nodes,
                'edges': edges,
                'step_pair': step_pair
            })

    # Evaluate first 10 samples for demonstration (or all if there are fewer)
    num_eval_samples = min(10, len(test_samples))
    for sample in test_samples[:num_eval_samples]:
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

        with torch.no_grad():
            # Get text embeddings for previous step
            previous_emb = torch.tensor(text_model.encode([previous_text], convert_to_numpy=True), dtype=torch.float32).to(device)

            # Forward pass through policy
            policy_out = policy(sample['nodes'], sample['edges'], previous_emb, device)

            # Generate prediction using the same prompt as training
            prompt = (
                f"Previous penetration testing context:\n"
                f"Strategy: {step_pair['previous_strategy']}\n"
                f"Step: {step_pair['previous_step']}\n"
                f"Result: {step_pair['previous_step_result']}\n\n"
                f"Next:\n"
            )
            
            tokenized_prompt = tokenizer([prompt], return_tensors='pt').to(device)
            inputs_embeds = llm.get_input_embeddings()(tokenized_prompt['input_ids'])
            
            # Add policy output as an additional embedding at the beginning (to condition the LLM)
            policy_emb_expanded = policy_out.unsqueeze(1)
            inputs_embeds = torch.cat([policy_emb_expanded, inputs_embeds], dim=1)
            
            # Create attention mask (all ones for all tokens, including the policy embedding)
            attention_mask = torch.ones(1, inputs_embeds.size(1), dtype=torch.long).to(device)

            # Generate
            output_ids = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Parse prediction
        _, pred_step, pred_mcp = parse_prediction(pred_text)
        if not pred_step:
            pred_step = pred_text
        if not pred_mcp:
            pred_mcp = pred_text

        # Compute reward
        reward = compute_reward(pred_step, true_step, pred_mcp, true_mcp)
        total_reward += reward
        num_samples += 1

        sample_predictions.append({
            'previous': previous_text[:100] + "...",
            'true_step': true_step,
            'pred_step': pred_step,
            'true_mcp': true_mcp,
            'pred_mcp': pred_mcp,
            'reward': reward
        })

        print(f"Sample {num_samples}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  True Step: {true_step[:80]}...")
        print(f"  Pred Step: {pred_step[:80]}...")

    avg_reward = total_reward / max(num_samples, 1)
    print(f"\nAverage Evaluation Reward over {num_samples} samples: {avg_reward:.4f}")


if __name__ == "__main__":
    main()
