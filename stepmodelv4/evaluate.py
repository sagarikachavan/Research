"""
evaluate.py — Evaluation for stepmodelv4 with graph embeddings
Evaluates both supervised and GRPO models
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from data_utils import (
    load_json_data, parse_completion, classify_step, mcp_labels_from_dict,
    mcp_multihot, STEP_LABELS, MCP_LABELS, get_sentence_encoder,
)
from prompts import build_chat_prompt
from config import (
    INPUT_TEST_JSON, QWEN_MODEL_NAME, SUPERVISED_ADAPTER_DIR, GRPO_ADAPTER_DIR,
    MAX_PROMPT_TOKENS, MAX_NEW_TOKENS, GNN_CKPT, ADAPTER_CKPT,
)
from graph_prefix_adapter import load_graph_encoder_and_adapter

INPUT_TEST_JSON = "input/test.json"


def eval_model(adapter_dir, max_new_tokens=350, save_explanations=None):
    """Evaluate a model with graph embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[eval] Test input : {INPUT_TEST_JSON}")
    print(f"[eval] Device     : {device}")
    print(f"[eval] Adapter    : {adapter_dir}")

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir).eval()

    # Load graph encoder and adapter
    graph_encoder, graph_adapter = load_graph_encoder_and_adapter(GNN_CKPT, ADAPTER_CKPT, device)
    print("[eval] Graph encoder and adapter loaded")

    # Load test data
    examples = load_json_data(INPUT_TEST_JSON)
    print(f"[eval] Loaded {len(examples)} test examples")

    # Storage for predictions
    step_preds, step_gold = [], []
    mcp_preds, mcp_gold = [], []
    expl_preds, expl_gold = [], []
    parse_failures = 0

    for ex in examples:
        # Build prompt
        prompt_text = build_chat_prompt(ex)
        
        # Get graph embedding
        with torch.no_grad():
            graph = ex["graph"].to(device)
            field_embs = torch.tensor(ex["field_embs"], dtype=torch.float32).unsqueeze(0).to(device)
            _, _, graph_emb = graph_encoder(
                graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long, device=device),
                field_embs
            )
            soft_prompt = graph_adapter(graph_emb)

        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        ).input_ids.to(device)
        L_prompt = prompt_ids.shape[1]
        attn_prompt = torch.ones(1, L_prompt, dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=prompt_ids,
                attention_mask=attn_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_only = gen_out[:, L_prompt:]
        completion_text = tokenizer.decode(completion_only[0], skip_special_tokens=True)

        # Parse completion
        parsed = parse_completion(completion_text)
        if not parsed:
            parse_failures += 1
            continue

        # Step classification
        pred_step = classify_step(parsed.get("New step", ""))
        gold_step = ex["gold_step_category"]
        step_preds.append(pred_step)
        step_gold.append(gold_step)

        # MCP classification
        pred_mcp = mcp_labels_from_dict(parsed.get("MCP_tasks", {}))
        gold_mcp = ex["gold_mcp_labels"]
        mcp_preds.append(pred_mcp)
        mcp_gold.append(gold_mcp)

        # Explanation
        pred_expl = parsed.get("Step explanation", "")
        gold_expl = ex["gold_step_explanation"]
        expl_preds.append(pred_expl)
        expl_gold.append(gold_expl)

    print(f"[eval] Parse failures: {parse_failures}/{len(examples)}")

    # Compute metrics
    metrics = compute_metrics(step_preds, step_gold, mcp_preds, mcp_gold, expl_preds, expl_gold)
    print_metrics(metrics)

    if save_explanations:
        save_predictions(examples, expl_preds, save_explanations)


def compute_metrics(step_preds, step_gold, mcp_preds, mcp_gold, expl_preds, expl_gold):
    """Compute evaluation metrics."""
    metrics = {}

    # Step classification
    metrics["step_accuracy"] = accuracy_score(step_gold, step_preds)
    metrics["step_macro_f1"] = f1_score(step_gold, step_preds, average="macro", zero_division=0)
    metrics["step_weighted_f1"] = f1_score(step_gold, step_preds, average="weighted", zero_division=0)

    # MCP classification
    mcp_pred_vecs = [mcp_multihot(labels) for labels in mcp_preds]
    mcp_gold_vecs = [mcp_multihot(labels) for labels in mcp_gold]
    mcp_pred_vecs = np.array(mcp_pred_vecs)
    mcp_gold_vecs = np.array(mcp_gold_vecs)

    metrics["mcp_subset_accuracy"] = accuracy_score(mcp_gold_vecs, mcp_pred_vecs)
    metrics["mcp_micro_f1"] = f1_score(mcp_gold_vecs, mcp_pred_vecs, average="micro", zero_division=0)
    metrics["mcp_macro_f1"] = f1_score(mcp_gold_vecs, mcp_pred_vecs, average="macro", zero_division=0)
    metrics["mcp_samples_f1"] = f1_score(mcp_gold_vecs, mcp_pred_vecs, average="samples", zero_division=0)

    # Explanation quality
    encoder = get_sentence_encoder()
    expl_emb_pred = encoder.encode(expl_preds, convert_to_numpy=True)
    expl_emb_gold = encoder.encode(expl_gold, convert_to_numpy=True)
    from sklearn.metrics.pairwise import cosine_similarity
    bertscore = np.mean([cosine_similarity([p], [g])[0][0] for p, g in zip(expl_emb_pred, expl_emb_gold)])
    metrics["bertscore_f1"] = bertscore

    # Length stats
    expl_lengths = [len(e.split()) for e in expl_preds]
    metrics["avg_length"] = np.mean(expl_lengths)
    metrics["empty_rate"] = sum(1 for e in expl_preds if not e.strip()) / len(expl_preds)

    return metrics


def print_metrics(metrics):
    """Print evaluation metrics."""
    print("\n" + "=" * 60)
    print("STEP CLASSIFICATION")
    print("=" * 60)
    print(f"  Accuracy      : {metrics['step_accuracy']:.4f}")
    print(f"  Macro F1      : {metrics['step_macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['step_weighted_f1']:.4f}")

    print("\n" + "=" * 60)
    print("MCP TOOL CLASSIFICATION")
    print("=" * 60)
    print(f"  Subset accuracy : {metrics['mcp_subset_accuracy']:.4f}")
    print(f"  Micro F1       : {metrics['mcp_micro_f1']:.4f}")
    print(f"  Macro F1       : {metrics['mcp_macro_f1']:.4f}")
    print(f"  Samples F1     : {metrics['mcp_samples_f1']:.4f}")

    print("\n" + "=" * 60)
    print("EXPLANATION QUALITY")
    print("=" * 60)
    print(f"  BERTScore F1   : {metrics['bertscore_f1']:.4f}")
    print(f"  Avg length     : {metrics['avg_length']:.1f}")
    print(f"  Empty rate     : {metrics['empty_rate']:.1%}")


def save_predictions(examples, predictions, output_path):
    """Save predictions to CSV."""
    import csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["gold_step", "gold_explanation", "pred_explanation"])
        for ex, pred in zip(examples, predictions):
            writer.writerow([ex["gold_step_text"], ex["gold_step_explanation"], pred])
    print(f"[eval] Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stepmodelv4 models")
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="grpo", choices=["supervised", "grpo"])
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--save-explanations", type=str, default=None)
    args = parser.parse_args()

    if args.adapter_dir:
        adapter_dir = args.adapter_dir
    elif args.model_type == "supervised":
        adapter_dir = SUPERVISED_ADAPTER_DIR
    else:
        adapter_dir = GRPO_ADAPTER_DIR

    if not os.path.exists(adapter_dir):
        print(f"[eval] ERROR: Adapter directory not found: {adapter_dir}")
        exit(1)

    eval_model(adapter_dir, args.max_new_tokens, args.save_explanations)
