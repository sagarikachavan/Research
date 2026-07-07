#!/usr/bin/env python3
"""
Evaluation script for trained GNN + LLM model.
Uses [GRAPH] token approach to condition LLM on graph info.
"""

import os
import json
import re
import ast
import difflib
import string
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tokenizers.models import Model as TokenizersModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from train_gnn_rl import GNNLLMPolicy, compute_reward, parse_prediction


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


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    text = " ".join(text.split())
    return text


def _token_set(text: str):
    norm = _normalize_text(text)
    return set(_WORD_RE.findall(norm))


def _f1_from_sets(pred_set, true_set) -> float:
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    inter = len(pred_set & true_set)
    precision = inter / max(len(pred_set), 1)
    recall = inter / max(len(true_set), 1)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def _split_mcp_candidates(text: str):
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    parts = re.split(r"[\n,;|]+", text)
    out = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if s.startswith("-"):
            s = s[1:].strip()
        if ":" in s:
            s = s.split(":", 1)[0].strip()
        s = re.sub(r"^\d+\)?\.?\s*", "", s).strip()
        if s:
            out.append(s)
    return out


def _extract_mcp_tools(raw) -> set:
    if raw is None:
        return set()
    value = raw
    if isinstance(raw, str):
        s = raw.strip()
        try:
            value = ast.literal_eval(s)
        except Exception:
            value = raw

    tools = set()
    if isinstance(value, dict):
        for k in value.keys():
            nk = _normalize_text(k)
            if nk:
                tools.add(nk)
        return tools

    if isinstance(value, (list, tuple, set)):
        for it in value:
            nk = _normalize_text(it)
            if nk:
                tools.add(nk)
        return tools

    for cand in _split_mcp_candidates(str(value)):
        nk = _normalize_text(cand)
        if nk:
            tools.add(nk)
    return tools


def _normalize_to_canonical(tool: str, canonical_tools: set) -> str:
    t = _normalize_text(tool)
    if not t:
        return ""
    if t in canonical_tools:
        return t
    matches = difflib.get_close_matches(t, list(canonical_tools), n=1, cutoff=0.84)
    return matches[0] if matches else t


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

    checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pt")
    checkpoint = None
    if os.path.exists(checkpoint_path):
        with torch.serialization.safe_globals([GPT2Tokenizer, Tokenizer, TokenizersModel]):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        print("Warning: No best checkpoint found, using initial weights.")

    checkpoint_llm_name = config['model']['llm_name']
    if checkpoint is not None:
        checkpoint_llm_name = (
            checkpoint.get("llm_name")
            or infer_llm_name_from_state_dict(checkpoint.get("llm", {}))
            or checkpoint_llm_name
        )
        if checkpoint_llm_name != config['model']['llm_name']:
            print(
                f"Checkpoint was trained with '{checkpoint_llm_name}', "
                f"overriding config model '{config['model']['llm_name']}' for evaluation."
            )

    checkpoint_tokenizer = checkpoint.get("tokenizer") if checkpoint is not None else None
    if checkpoint_tokenizer is not None:
        tokenizer = checkpoint_tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_llm_name)
        if '[GRAPH]' not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': ['[GRAPH]']})
    tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(checkpoint_llm_name)
    llm.resize_token_embeddings(len(tokenizer))
    llm.to(device)
    llm.eval()

    policy_hidden_size = llm.config.hidden_size
    if checkpoint is not None:
        inferred_policy_hidden_size = infer_policy_hidden_size(checkpoint.get("policy", {}))
        if inferred_policy_hidden_size is not None:
            policy_hidden_size = inferred_policy_hidden_size

    # Initialize policy
    policy = GNNLLMPolicy(
        gnn_out_dim=config['model']['gnn_out_dim'],
        text_emb_dim=text_emb_dim,
        llm_hidden_size=policy_hidden_size,
        use_gat=config['model']['use_gat']
    ).to(device)

    if checkpoint is not None:
        if llm.config.hidden_size != policy_hidden_size:
            raise RuntimeError(
                f"Checkpoint mismatch: policy expects hidden size {policy_hidden_size}, "
                f"but loaded LLM '{checkpoint_llm_name}' has hidden size {llm.config.hidden_size}."
            )
        policy.load_state_dict(checkpoint["policy"])
        llm.load_state_dict(checkpoint["llm"])
        print("Loaded best checkpoint successfully!")

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
    total_step_correct = 0
    total_mcp_correct = 0
    total_both_correct = 0
    total_step_exact = 0
    total_mcp_exact = 0
    total_both_exact = 0
    total_step_f1 = 0.0
    total_mcp_f1 = 0.0
    total_parse_ok = 0

    canonical_tools = set()
    for s in test_samples:
        sp = s["step_pair"]
        canonical_tools |= _extract_mcp_tools(sp.get("next_mcp_tasks", ""))

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
                "[GRAPH]\n"
                "### Previous Penetration Testing Context ###\n"
                f"Strategy: {step_pair['previous_strategy']}\n"
                f"Step: {step_pair['previous_step']}\n"
                f"Result: {step_pair['previous_step_result']}\n\n"
                "### Generate Next Step ###\n"
                "Respond with exactly these five labeled lines and do not repeat the prompt:\n"
                "Strategy:\n"
                "Strategy Explanation:\n"
                "Step:\n"
                "Step Explanation:\n"
                "MCP Tasks:"
            )
            max_new_tokens = config['training'].get('generate_max_new_tokens', 256)
            max_prompt_tokens = llm.config.n_positions - max_new_tokens
            if max_prompt_tokens <= 0:
                max_prompt_tokens = llm.config.n_positions - 1

            tokenized_prompt = tokenizer(
                [prompt_text],
                return_tensors='pt',
                truncation=True,
                max_length=max_prompt_tokens
            ).to(device)
            inputs_embeds = llm.get_input_embeddings()(tokenized_prompt['input_ids'])
            inputs_embeds[:, 0, :] = policy_out

            output_ids = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=tokenized_prompt['attention_mask'],
                max_new_tokens=max(1, min(max_new_tokens, 256)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            reward = compute_reward(pred_text, true_step, true_mcp, text_model, step_pair)
            # Parse predicted structured fields for accuracy metrics
            _, pred_step, pred_mcp = parse_prediction(pred_text)

            parse_ok = bool(pred_step) and bool(pred_mcp)
            total_parse_ok += int(parse_ok)

            pred_step_norm = _normalize_text(pred_step)
            true_step_norm = _normalize_text(true_step)
            step_exact = int(bool(pred_step_norm) and pred_step_norm == true_step_norm)
            pred_step_tokens = _token_set(pred_step)
            true_step_tokens = _token_set(true_step)
            step_f1 = _f1_from_sets(pred_step_tokens, true_step_tokens)
            step_correct = step_f1 >= 0.5

            true_tools = _extract_mcp_tools(true_mcp)
            pred_tools_raw = _extract_mcp_tools(pred_mcp)
            pred_tools = {_normalize_to_canonical(t, canonical_tools) for t in pred_tools_raw}
            pred_tools = {t for t in pred_tools if t}
            mcp_exact = int(bool(pred_tools) and pred_tools == true_tools)
            mcp_f1 = _f1_from_sets(pred_tools, true_tools)
            mcp_correct = mcp_f1 >= 0.5

            total_step_f1 += step_f1
            total_mcp_f1 += mcp_f1
            total_step_correct += int(step_correct)
            total_mcp_correct += int(mcp_correct)
            total_both_correct += int(step_correct and mcp_correct)
            total_step_exact += step_exact
            total_mcp_exact += mcp_exact
            total_both_exact += int(step_exact and mcp_exact)
            total_reward += reward
            num_samples += 1

            if num_samples % 50 == 0:
                print(f"Processed {num_samples} samples, current average: {total_reward/num_samples:.4f}")

    avg_reward = total_reward / num_samples
    print("\n" + "="*60)
    print(f"FINAL TEST EVALUATION ON FULL DATASET ({num_samples} samples):")
    print(f"Average Reward: {avg_reward:.4f}")
    step_acc = total_step_correct / num_samples if num_samples > 0 else 0.0
    mcp_acc = total_mcp_correct / num_samples if num_samples > 0 else 0.0
    both_acc = total_both_correct / num_samples if num_samples > 0 else 0.0
    step_exact_acc = total_step_exact / num_samples if num_samples > 0 else 0.0
    mcp_exact_acc = total_mcp_exact / num_samples if num_samples > 0 else 0.0
    both_exact_acc = total_both_exact / num_samples if num_samples > 0 else 0.0
    avg_step_f1 = total_step_f1 / num_samples if num_samples > 0 else 0.0
    avg_mcp_f1 = total_mcp_f1 / num_samples if num_samples > 0 else 0.0
    parse_rate = total_parse_ok / num_samples if num_samples > 0 else 0.0
    print(f"Step F1 (token): {avg_step_f1:.4f}")
    print(f"MCP F1 (set): {avg_mcp_f1:.4f}")
    print(f"Step Accuracy (F1>=0.5): {step_acc:.4f}")
    print(f"MCP Accuracy (F1>=0.5): {mcp_acc:.4f}")
    print(f"Both Step+MCP Accuracy (F1>=0.5): {both_acc:.4f}")
    print(f"Step Exact Match: {step_exact_acc:.4f}")
    print(f"MCP Exact Set Match: {mcp_exact_acc:.4f}")
    print(f"Both Exact Match: {both_exact_acc:.4f}")
    print(f"Parse Success Rate: {parse_rate:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
