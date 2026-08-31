"""
evaluate.py
===========
Evaluation script for text-only experiment (no graph).
Evaluates all three stages on test set and outputs metrics.

Usage:
    python evaluate.py --stage 1  # Evaluate Stage 1
    python evaluate.py --stage 2  # Evaluate Stage 2
    python evaluate.py --stage 3  # Evaluate Stage 3
    python evaluate.py --stage all  # Evaluate all stages

--------------------------------------------------------------------------
FIXES:
1. Added an existence check for the adapter directory before calling
   PeftModel.from_pretrained(). If a local path doesn't exist, PEFT
   silently treats it as a Hugging Face Hub repo ID and tries to resolve
   it over the network -- which can hang for a long time (or fail with a
   confusing error) instead of clearly telling you the checkpoint isn't
   there yet.
2. `--stage all` loads a full 14B model for Stage 2, then loads ANOTHER
   full 14B model for Stage 3 without freeing the first one. Added
   explicit cleanup (del + gc.collect + torch.cuda.empty_cache) between
   stages so the second load doesn't OOM.
--------------------------------------------------------------------------
"""

import os
import sys
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Add parent directories to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    ROOT, INPUT_TEST_JSON, STAGE2_ADAPTER_DIR, STAGE3_ADAPTER_DIR, OUTPUT_DIR,
    STEP_LABELS, MCP_LABELS, STEP2IDX, MCP2IDX, IDX2STEP, IDX2MCP,
    QWEN_MODEL_NAME,
)


def load_from_input_json(path, split):
    """Load examples from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"[{split}] Loaded {len(examples)} examples from {path}")
    return examples


def evaluate_stage2(model, tokenizer, examples, device):
    """Evaluate Stage 2 LLM SFT."""
    model.eval()
    results = []
    
    prompt_template = """<|im_start|>system
You are an expert penetration testing assistant. Given a strategy and explanation, determine the next step, the tools needed, and explain your reasoning.

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
<|im_end|>
<|im_start|>user
Strategy: {strategy}
Explanation: {explanation}
<|im_end|>
<|im_start|>assistant
"""
    
    for ex in tqdm(examples, desc="Evaluating Stage 2"):
        prompt = prompt_template.format(
            strategy=ex.get('new_strategy', ''),
            explanation=ex.get('strategy_explanation', '')
        )
        
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        try:
            obj = json.loads(response)
            pred_step = obj.get("New step", "")
            pred_mcp = obj.get("MCP_tasks", {})
            pred_expl = obj.get("Step explanation", "")
        except:
            pred_step = ""
            pred_mcp = {}
            pred_expl = ""
        
        results.append({
            'machine': ex.get('machine', ''),
            'pred_step': pred_step,
            'gold_step': ex.get('gold_new_step', ''),
            'pred_mcp': list(pred_mcp.keys()),
            'gold_mcp': ex.get('gold_mcp_tasks', ''),
            'pred_expl': pred_expl,
            'gold_expl': ex.get('gold_step_explanation', ''),
        })
    
    # Compute metrics
    step_correct = sum(1 for r in results if r['pred_step'] == r['gold_step'])
    step_acc = step_correct / len(results) if results else 0
    
    all_step_preds = [STEP2IDX.get(r['pred_step'], -1) for r in results if r['pred_step'] in STEP2IDX]
    all_step_labels = [STEP2IDX.get(r['gold_step'], -1) for r in results if r['gold_step'] in STEP2IDX]
    step_macro_f1 = f1_score(all_step_labels, all_step_preds, average='macro', zero_division=0) if all_step_labels else 0
    
    metrics = {
        'step_accuracy': step_acc,
        'step_macro_f1': step_macro_f1,
    }
    
    return metrics, results


def evaluate_stage3(model, tokenizer, examples, device):
    """Evaluate Stage 3 GRPO RL (same as Stage 2)."""
    return evaluate_stage2(model, tokenizer, examples, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["2", "3", "all"], default="all",
                         help="Which stage to evaluate (Stage 1 removed - not needed for text-only baseline)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_examples = load_from_input_json(INPUT_TEST_JSON, "test")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    stages_to_eval = ["2", "3"] if args.stage == "all" else [args.stage]
    
    for stage in stages_to_eval:
        print(f"\n{'='*50}")
        print(f"Evaluating Stage {stage}")
        print(f"{'='*50}")

        adapter_dir = STAGE2_ADAPTER_DIR if stage == "2" else STAGE3_ADAPTER_DIR
        if not os.path.isdir(adapter_dir) or not os.listdir(adapter_dir):
            print(f"⚠️  Skipping Stage {stage}: no checkpoint found at {adapter_dir}. "
                  f"(Run the Stage {stage} training script first.)")
            continue

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model = model.merge_and_unload()
        
        # Evaluate
        metrics, results = evaluate_stage2(model, tokenizer, test_examples, device)
        
        # Print metrics
        print(f"\nStage {stage} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save results
        output_path = os.path.join(OUTPUT_DIR, f"stage{stage}_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'results': results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_path}")

        # Free the full 14B model before the next stage's model is loaded --
        # without this, evaluating "all" loads a second full copy on top of
        # the first and risks an OOM right at the last step of the pipeline.
        del model, base_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
