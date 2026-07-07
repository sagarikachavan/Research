#!/usr/bin/env python3
import json
from transformers import AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    with open("embeddings_data/train/all_processed.json", "r") as f:
        data = json.load(f)
    
    total_samples = 0
    max_target_tokens = 0
    total_target_tokens = 0
    max_prompt_tokens = 0
    
    for machine in data:
        for step_pair in machine['step_pairs']:
            prompt_text = (
                "[GRAPH] "
                f"Previous penetration testing context:\n"
                f"Strategy: {step_pair['previous_strategy']}\n"
                f"Step: {step_pair['previous_step']}\n"
                f"Result: {step_pair['previous_step_result']}\n\n"
                f"Next:\n"
            )
            target_text = (
                "Strategy: " + step_pair['next_strategy'] + "\n"
                "Strategy Explanation: " + step_pair['next_strategy_explanation'] + "\n"
                "Step: " + step_pair['next_step'] + "\n"
                "Step Explanation: " + step_pair['next_step_explanation'] + "\n"
                "MCP Tasks: " + step_pair['next_mcp_tasks']
            )
            
            target_len = len(tokenizer.encode(target_text))
            prompt_len = len(tokenizer.encode(prompt_text))
            
            total_target_tokens += target_len
            if target_len > max_target_tokens:
                max_target_tokens = target_len
            if prompt_len > max_prompt_tokens:
                max_prompt_tokens = prompt_len
            
            total_samples += 1
    
    print(f"Total samples: {total_samples}")
    print(f"Max prompt tokens: {max_prompt_tokens}")
    print(f"Max target tokens: {max_target_tokens}")
    print(f"Average target tokens: {total_target_tokens / total_samples:.2f}")
    print(f"Max full sequence (prompt + target) tokens: {max_prompt_tokens + max_target_tokens}")

if __name__ == "__main__":
    main()
