#!/usr/bin/env python3
"""
Evaluation script for trained GNN + LLM model.
Uses [GRAPH] token approach to condition LLM on graph info.
"""

import os
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_gnn_rl import GNNLLMPolicy, compute_reward, parse_prediction


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Load config first to get paths
    with open(os.path.join(base_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    # Use config paths instead of hardcoding
    embeddings_dir = os.path.join(base_dir, config['paths']['embeddings_dir'])
    checkpoints_dir = os.path.join(base_dir, config['paths']['output_dir'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sentence transformer
    text_model = SentenceTransformer(config['model']['text_embedding_model'])
    text_emb_dim = text_model.get_embedding_dimension()

    # Load test data
    with open(os.path.join(embeddings_dir, "test", "all_processed.json"), "r") as f:
        test_data = json.load(f)

    # Load tokenizer and LLM
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir)
    tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(checkpoints_dir)
    llm.to(device)
    llm.eval()

    # Initialize policy
    policy = GNNLLMPolicy(
        gnn_out_dim=config['model']['gnn_out_dim'],
        text_emb_dim=text_emb_dim,
        llm_hidden_size=llm.config.hidden_size,
        use_gat=config['model']['use_gat']
    ).to(device)

    checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        llm.load_state_dict(checkpoint["llm"])
        print("Loaded best checkpoint successfully!")
    else:
        print("Warning: No best checkpoint found, using initial weights.")

    policy.eval()

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

    total_reward = 0.0
    num_samples = 0

    print("Evaluating on full test dataset...")

    with torch.no_grad():
        for sample in test_samples:
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

            # Compute policy output
            previous_emb = torch.tensor(text_model.encode([previous_text], convert_to_numpy=True), dtype=torch.float32).to(device)
            policy_out = policy(sample['nodes'], sample['edges'], previous_emb, device)

            # Prepare prompt and generate
            prompt_text = (
                "[GRAPH] "
                f"Previous penetration testing context:\n"
                f"Strategy: {step_pair['previous_strategy']}\n"
                f"Step: {step_pair['previous_step']}\n"
                f"Result: {step_pair['previous_step_result']}\n\n"
                f"Next:\n"
            )
            tokenized_prompt = tokenizer([prompt_text], return_tensors='pt').to(device)
            inputs_embeds = llm.get_input_embeddings()(tokenized_prompt['input_ids'])
            inputs_embeds[:, 0, :] = policy_out

            output_ids = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=tokenized_prompt['attention_mask'],
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            reward = compute_reward(pred_text, true_step, true_mcp, text_model, step_pair)
            total_reward += reward
            num_samples += 1

            if num_samples % 50 == 0:
                print(f"Processed {num_samples} samples, current average: {total_reward/num_samples:.4f}")

    avg_reward = total_reward / num_samples
    print("\n" + "="*60)
    print(f"FINAL TEST EVALUATION ON FULL DATASET ({num_samples} samples):")
    print(f"Average Reward: {avg_reward:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
