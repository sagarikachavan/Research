"""
Comprehensive evaluation script for the multi-stage training pipeline.

This script evaluates the final model (Stage 3 checkpoint) on:
- Step prediction: accuracy, micro-F1
- MCP tool prediction: accuracy, F1, micro-F1, missing tools, extra tools, exact match
- Explanation quality: multi-dimensional LLM judge evaluation

Usage:
    python eval/comprehensive_evaluation.py --checkpoint <stage3_checkpoint_path>
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm

# Add parent directories to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    INPUT_TEST_JSON, STAGE3_ADAPTER_DIR, STAGE2_ADAPTER_DIR,
    STEP_LABELS, MCP_LABELS, STEP2IDX, MCP2IDX,
    QWEN_MODEL_NAME, LLM_JUDGE_MODEL_NAME,
)
from comprehensive_evaluator import ComprehensiveEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


SYSTEM_PROMPT = """You are an expert penetration testing assistant. Given a strategy and explanation, determine the next step, the tools needed, and explain your reasoning.

Respond in JSON format with the following structure:
{
    "New step": "<one of the 10 step labels>",
    "MCP_tasks": {
        "<tool_name>": "<short action description>",
        ...
    },
    "Step explanation": "<detailed explanation of why this step is appropriate>"
}

Available step labels:
- Do a google search for more information
- Enumerate further on the X service to find software versions, hidden directories and file.
- Explore the suspicious files, commands and create a summary of the findings.
- Further Enumerate the website. - hidden directories, links and software
- Enumerate the domain
- Exploit the selected exploitations
- Analyze the outcomes of the previous step and find an attack path
- Ask for human assistant
- Explore the source code for vulnerabilities.
- End task and ask permission to generate the report

Available MCP tools: Nmap, Metasploit, Netcat, Dirbuster, SQLmap, Smb client, hydra, John-the-ripper, Google search, Interactive CLI, Web page interaction
"""


def build_prompt(ex: dict) -> str:
    """Build prompt from example."""
    ctx = f"Strategy: {ex.get('new_strategy', '')}\nExplanation: {ex.get('strategy_explanation', '')}"
    lines = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>",
        f"<|im_start|>user\n{ctx}<|im_end|>",
        f"<|im_start|>assistant\n",
    ]
    return "\n".join(lines)


def parse_response(response_text: str):
    """Parse model response into step, MCP, and explanation."""
    try:
        obj = json.loads(response_text)
        step = obj.get("New step", "")
        mcp_tasks = obj.get("MCP_tasks", {})
        explanation = obj.get("Step explanation", "")
        return step, mcp_tasks, explanation
    except Exception:
        return "", {}, ""


def load_test_data(path: str) -> List[Dict]:
    """Load test data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} test examples from {path}")
    return examples


def load_model_and_tokenizer(checkpoint_path: str, device: str):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model, checkpoint_path, is_trainable=False
    )
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_llm_judge(device: str):
    """Load the LLM judge model for explanation evaluation."""
    try:
        from transformers import AutoModel
        from llm_judge import set_llm_judge_model
        
        print(f"Loading LLM judge model: {LLM_JUDGE_MODEL_NAME}")
        judge_model = AutoModelForCausalLM.from_pretrained(
            LLM_JUDGE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(LLM_JUDGE_MODEL_NAME, trust_remote_code=True)
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
        
        set_llm_judge_model(judge_model, judge_tokenizer, device, model_id=LLM_JUDGE_MODEL_NAME)
        print("LLM judge loaded successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not load LLM judge: {e}")
        print("Explanation evaluation will use structural metrics only")
        return False


def generate_predictions(model, tokenizer, examples: List[Dict], 
                        device: str, max_samples: int = None) -> List[Dict]:
    """Generate predictions for all test examples."""
    if max_samples:
        examples = examples[:max_samples]
    
    predictions = []
    
    for ex in tqdm(examples, desc="Generating predictions"):
        prompt = build_prompt(ex)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,  # Use greedy decoding for evaluation
                pad_token_id=tokenizer.pad_token_id,
            )
        
        prompt_len = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        pred_step, pred_mcp, pred_expl = parse_response(response)
        
        predictions.append({
            "step": pred_step,
            "mcp": pred_mcp,
            "explanation": pred_expl,
            "raw_response": response,
        })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of multi-stage model")
    parser.add_argument("--checkpoint", type=str, default=STAGE3_ADAPTER_DIR,
                        help="Path to Stage 3 checkpoint")
    parser.add_argument("--test-data", type=str, default=INPUT_TEST_JSON,
                        help="Path to test data JSON")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for debugging)")
    parser.add_argument("--use-llm-judge", action="store_true", default=True,
                        help="Use LLM judge for explanation evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    test_examples = load_test_data(args.test_data)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    
    # Load LLM judge if requested
    if args.use_llm_judge:
        load_llm_judge(device)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, tokenizer, test_examples, device, args.max_samples)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Prepare data for evaluation
    pred_steps = [p["step"] for p in predictions]
    pred_mcps = [p["mcp"] for p in predictions]
    pred_expls = [p["explanation"] for p in predictions]
    
    gold_steps = [ex.get('gold_new_step', '') for ex in test_examples[:len(predictions)]]
    gold_mcp_texts = [ex.get('gold_mcp_tasks', '') for ex in test_examples[:len(predictions)]]
    gold_expls = [ex.get('gold_step_explanation', '') for ex in test_examples[:len(predictions)]]
    
    # Evaluate step predictions
    print("\nEvaluating step predictions...")
    step_metrics = evaluator.evaluate_step(pred_steps, gold_steps)
    
    # Evaluate MCP predictions
    print("Evaluating MCP predictions...")
    mcp_metrics = evaluator.evaluate_mcp(pred_mcps, gold_mcp_texts)
    
    # Evaluate explanation quality
    print("Evaluating explanation quality...")
    contexts = test_examples[:len(predictions)]
    expl_metrics = evaluator.evaluate_explanation_quality(
        pred_expls, gold_expls, pred_steps, contexts, use_llm_judge=args.use_llm_judge
    )
    
    # Compile comprehensive results
    results = {
        "checkpoint_path": args.checkpoint,
        "test_data_path": args.test_data,
        "n_samples": len(predictions),
        "step_metrics": step_metrics,
        "mcp_metrics": mcp_metrics,
        "explanation_metrics": expl_metrics,
        "overall_score": (
            step_metrics["accuracy"] * 0.4 +
            mcp_metrics["micro_f1"] * 0.4 +
            expl_metrics.get("overall_quality", expl_metrics["length_similarity"]) * 0.2
        )
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    
    print("\nStep Prediction Metrics:")
    print(f"  Accuracy: {step_metrics['accuracy']:.4f}")
    print(f"  Micro-F1: {step_metrics['micro_f1']:.4f}")
    print(f"  Valid samples: {step_metrics['n_valid']}")
    
    print("\nMCP Tool Prediction Metrics:")
    print(f"  Accuracy: {mcp_metrics['accuracy']:.4f}")
    print(f"  Micro-F1: {mcp_metrics['micro_f1']:.4f}")
    print(f"  Macro-F1: {mcp_metrics['macro_f1']:.4f}")
    print(f"  Avg Missing Tools: {mcp_metrics['avg_missing_tools']:.4f}")
    print(f"  Avg Extra Tools: {mcp_metrics['avg_extra_tools']:.4f}")
    print(f"  Exact Match Rate: {mcp_metrics['exact_match_rate']:.4f}")
    
    print("\nPer-Tool MCP Metrics:")
    for tool, metrics in mcp_metrics['per_tool_metrics'].items():
        print(f"  {tool:<20}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
    
    print("\nExplanation Quality Metrics:")
    print(f"  Length Similarity: {expl_metrics['length_similarity']:.4f}")
    print(f"  Structural Completeness: {expl_metrics['structural_completeness']:.4f}")
    print(f"  Avg Predicted Length: {expl_metrics['avg_pred_length']:.2f}")
    print(f"  Avg Gold Length: {expl_metrics['avg_gold_length']:.2f}")
    
    if "relevance" in expl_metrics:
        print(f"  Relevance: {expl_metrics['relevance']:.4f}")
        print(f"  Technical Accuracy: {expl_metrics['technical_accuracy']:.4f}")
        print(f"  Completeness: {expl_metrics['completeness']:.4f}")
        print(f"  Clarity: {expl_metrics['clarity']:.4f}")
        print(f"  Faithfulness: {expl_metrics['faithfulness']:.4f}")
        print(f"  Plausibility: {expl_metrics['plausibility']:.4f}")
        print(f"  Correctness: {expl_metrics['correctness']:.4f}")
        print(f"  Overall Quality: {expl_metrics['overall_quality']:.4f}")
        print(f"  Judge Error Count: {expl_metrics['judge_error_count']}")
    
    print(f"\nOverall Score: {results['overall_score']:.4f}")
    print("=" * 80)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
