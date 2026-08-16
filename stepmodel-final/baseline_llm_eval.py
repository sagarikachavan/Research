"""
Baseline LLM Evaluation: Zero-shot and Few-shot comparison

This script evaluates LLM performance on the penetration testing task without
any graph conditioning or fine-tuning. It provides baseline comparisons for:
- Zero-shot: Direct LLM query with no examples
- Few-shot: LLM query with N examples from training data

Usage:
    python baseline_llm_eval.py --num_shots 0    # Zero-shot
    python baseline_llm_eval.py --num_shots 3    # 3-shot
    python baseline_llm_eval.py --num_shots 5    # 5-shot

Output:
    output/baseline_zeroshot.csv or output/baseline_3shot.csv
"""
import argparse
import json
import csv
import os
import random
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, QWEN_MODEL_NAME,
    STEP_LABELS, MCP_LABELS, ROOT, RANDOM_SEED,
)
from data_utils import load_from_input_json, StepLabelNormalizer, extract_mcp_labels

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

SYSTEM_PROMPT = (
    "You are an autonomous penetration-testing planning assistant operating "
    "strictly within an authorized lab environment. Given the current "
    "reconnaissance graph state and the previous/new strategy context, "
    "choose exactly one next-step type from the fixed taxonomy, exactly one "
    "or more tool(s) from the fixed MCP taxonomy, and explain your reasoning. "
    "IMPORTANT: Your step explanation MUST explicitly mention the chosen step "
    "type by name to justify why that specific step is appropriate."
)

STEP_TAXONOMY = ", ".join(STEP_LABELS)
MCP_TAXONOMY = ", ".join(MCP_LABELS)


def build_prompt(ex: dict) -> str:
    """Build the prompt for a single example."""
    ctx = ex["context"]
    lines = [
        f"Machine: {ex['machine']}",
        f"Previous strategy: {ctx['Previous strategy']}",
        f"Previous step: {ctx['Previous step']}",
        f"Previous step result: {ctx['Previous step result']}",
        f"New strategy: {ctx['New strategy']}",
        f"Strategy explanation: {ctx['Strategy explanation']}",
    ]
    return "\n".join(lines)


def build_target(ex: dict) -> str:
    """Build the target response for an example."""
    mcp_dict = {
        label: f"Use {label} as part of: {ex['step_label']}"
        for label in ex["mcp_labels"]
    }
    return json.dumps(
        {
            "New step": ex["step_label"],
            "Step explanation": ex["gold_step_explanation"],
            "MCP_tasks": mcp_dict,
        },
        ensure_ascii=False,
    )


def build_few_shot_prompt(examples: List[dict], test_ex: dict, num_shots: int) -> str:
    """Build a few-shot prompt with examples."""
    # System prompt with taxonomy
    prompt = f"{SYSTEM_PROMPT}\n\n"
    prompt += f"Available step types: {STEP_TAXONOMY}\n"
    prompt += f"Available MCP tools: {MCP_TAXONOMY}\n\n"
    
    # Add few-shot examples
    if num_shots > 0:
        prompt += "Here are some examples:\n\n"
        for i, ex in enumerate(examples[:num_shots]):
            prompt += f"Example {i+1}:\n"
            prompt += f"{build_prompt(ex)}\n"
            prompt += f"Response: {build_target(ex)}\n\n"
    
    # Add test example
    prompt += "Now, your task:\n"
    prompt += f"{build_prompt(test_ex)}\n"
    prompt += "Response:"
    
    return prompt


def parse_response(response_text: str, normalizer: StepLabelNormalizer) -> Dict:
    """Parse the LLM response to extract predictions."""
    obj = {}
    try:
        # Try to parse JSON
        if "{" in response_text and "}" in response_text:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            obj = json.loads(response_text[start:end])
    except:
        pass
    
    return obj


def evaluate_baseline(
    num_shots: int,
    model_name: str = QWEN_MODEL_NAME,
    max_new_tokens: int = 500,
    device: str = "cuda",
):
    """Run baseline evaluation with specified number of shots."""
    
    # Load data
    print(f"[Baseline] Loading data...")
    train_examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
    test_examples = load_from_input_json(INPUT_TEST_JSON, "test")
    
    # Sample few-shot examples from training data
    if num_shots > 0:
        few_shot_examples = random.sample(train_examples, min(num_shots, len(train_examples)))
        print(f"[Baseline] Using {len(few_shot_examples)} few-shot examples")
    else:
        few_shot_examples = []
        print(f"[Baseline] Zero-shot evaluation (no examples)")
    
    # Load model and tokenizer
    print(f"[Baseline] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    
    model.eval()
    normalizer = StepLabelNormalizer()
    csv_rows = []
    
    print(f"[Baseline] Evaluating {len(test_examples)} test examples...")
    
    with torch.no_grad():
        for idx, test_ex in enumerate(test_examples):
            if idx % 10 == 0:
                print(f"[Baseline] Progress: {idx}/{len(test_examples)}")
            
            # Build prompt
            prompt_text = build_few_shot_prompt(few_shot_examples, test_ex, num_shots)
            
            # Tokenize
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            ).to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                temperature=1.0,
            )
            
            # Decode response
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            if prompt_text in response_text:
                response_text = response_text.replace(prompt_text, "").strip()
            
            # Parse response
            obj = parse_response(response_text, normalizer)
            
            # Extract step prediction
            pred_step_raw = obj.get("New step", "")
            pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
            
            # Try multiple fallback strategies for step prediction
            pred_step_label = "UNPARSEABLE"
            if pred_step_norm and pred_step_norm in STEP_LABELS:
                pred_step_label = pred_step_norm
            elif pred_step_raw:
                # Try direct match
                if pred_step_raw in STEP_LABELS:
                    pred_step_label = pred_step_raw
                else:
                    # Try fuzzy match - find closest label
                    import difflib
                    closest_match = difflib.get_close_matches(pred_step_raw, STEP_LABELS, n=1, cutoff=0.6)
                    if closest_match:
                        pred_step_label = closest_match[0]
            
            gold_step_label = STEP_LABELS[test_ex["step_idx"]]
            
            # Extract MCP predictions
            pred_mcp_keys = list(obj.get("MCP_tasks", {}).keys()) if isinstance(obj.get("MCP_tasks"), dict) else []
            pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
            pred_mcp_tools = "|".join(pred_mcp_labels)
            gold_mcp_tools = "|".join(test_ex["mcp_labels"])
            
            # Extract explanations
            pred_expl = str(obj.get("Step explanation", "")).strip()
            gold_expl = test_ex.get("gold_step_explanation", "")
            
            csv_rows.append({
                "machine": test_ex.get("machine", ""),
                "new_strategy": test_ex["context"].get("New strategy", ""),
                "strategy_explanation": test_ex["context"].get("Strategy explanation", ""),
                "step_prediction": pred_step_label,
                "gold_new_step": gold_step_label,
                "mcp_tool_prediction": pred_mcp_tools,
                "mcp_tool_gold": gold_mcp_tools,
                "step_explanation_predicted": pred_expl,
                "step_explanation_gold": gold_expl,
                "raw_response": response_text[:500],  # Truncate for CSV
            })
    
    # Save CSV
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    if num_shots == 0:
        csv_filename = "baseline_zeroshot.csv"
    else:
        csv_filename = f"baseline_{num_shots}shot.csv"
    
    csv_path = os.path.join(output_dir, csv_filename)
    
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Baseline] Evaluation CSV saved to: {csv_path}")
        print(f"[Baseline] Total test samples evaluated: {len(csv_rows)}")
    else:
        print("[Baseline] Warning: No CSV rows generated")


def main():
    parser = argparse.ArgumentParser(description="Baseline LLM Evaluation")
    parser.add_argument(
        "--num_shots",
        type=int,
        default=0,
        help="Number of few-shot examples (0 for zero-shot, default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=QWEN_MODEL_NAME,
        help=f"Model name or path (default: {QWEN_MODEL_NAME})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    print(f"[Baseline] Configuration:")
    print(f"  Number of shots: {args.num_shots}")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    
    evaluate_baseline(
        num_shots=args.num_shots,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
