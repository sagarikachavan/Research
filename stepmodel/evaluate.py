#!/usr/bin/env python3
"""
Evaluation script for the label-only Step/MCP version of stepmodel.
"""

import os
import json
import re
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tokenizers.models import Model as TokenizersModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, BitsAndBytesConfig

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PEFT_AVAILABLE = False

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
    evaluate_metrics_on_dataset,
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
    step_head_weight = policy_state_dict.get("step_head.weight")
    if step_head_weight is not None:
        return step_head_weight.shape[1]

    mcp_head_weight = policy_state_dict.get("mcp_head.weight")
    if mcp_head_weight is not None:
        return mcp_head_weight.shape[1]

    readout_weight = policy_state_dict.get("readout_projection.1.weight")
    if readout_weight is not None:
        return readout_weight.shape[0]

    graph_token_projector_weight = policy_state_dict.get("graph_token_projector.weight")
    if graph_token_projector_weight is not None:
        graph_token_count = infer_graph_token_count(policy_state_dict)
        if graph_token_count:
            return graph_token_projector_weight.shape[0] // graph_token_count

    return None


def infer_graph_token_count(policy_state_dict):
    graph_token_projector_weight = policy_state_dict.get("graph_token_projector.weight")
    hidden_size = infer_policy_hidden_size_from_heads(policy_state_dict)
    if graph_token_projector_weight is None or hidden_size in {None, 0}:
        return None
    if graph_token_projector_weight.shape[0] % hidden_size != 0:
        return None
    return graph_token_projector_weight.shape[0] // hidden_size


def infer_policy_hidden_size_from_heads(policy_state_dict):
    step_head_weight = policy_state_dict.get("step_head.weight")
    if step_head_weight is not None:
        return step_head_weight.shape[1]

    mcp_head_weight = policy_state_dict.get("mcp_head.weight")
    if mcp_head_weight is not None:
        return mcp_head_weight.shape[1]

    return None


def infer_pooling_strategy(policy_state_dict):
    if "readout_projection.1.weight" in policy_state_dict:
        return "hybrid"
    return None


def checkpoint_has_label_heads(policy_state_dict):
    required = {
        "step_head.weight",
        "step_head.bias",
        "mcp_head.weight",
        "mcp_head.bias",
    }
    return required.issubset(set(policy_state_dict.keys()))


def checkpoint_uses_lora(llm_state_dict):
    return any("lora_" in key for key in llm_state_dict.keys())


def llm_name_supports_qwen_kbit(llm_name: str) -> bool:
    return "qwen" in str(llm_name or "").lower()


def _parse_version_tuple(version_str: str):
    parts = []
    for token in str(version_str).replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def get_bitsandbytes_4bit_status():
    try:
        version = importlib_metadata.version("bitsandbytes")
    except importlib_metadata.PackageNotFoundError:
        return False, (
            "4-bit quantization is enabled in config.json, but `bitsandbytes` is not installed. "
            "Falling back to non-4bit loading. Install it with `pip install -U bitsandbytes>=0.46.1` "
            "to re-enable 4-bit quantization."
        )

    if _parse_version_tuple(version) < (0, 46, 1):
        return False, (
            f"4-bit quantization requires `bitsandbytes>=0.46.1`, but found {version}. "
            "Falling back to non-4bit loading. Upgrade it with `pip install -U bitsandbytes>=0.46.1` "
            "to re-enable 4-bit quantization."
        )
    return True, f"bitsandbytes {version} detected; using 4-bit quantization."


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
    policy_state = checkpoint.get("policy", {}) if checkpoint is not None else {}
    inferred_pooling_strategy = infer_pooling_strategy(policy_state)
    checkpoint_pooling_strategy = (
        checkpoint.get("pooling_strategy")
        if checkpoint is not None and checkpoint.get("pooling_strategy") is not None
        else inferred_pooling_strategy
        or str(config.get('model', {}).get('pooling_strategy', 'mean')).lower()
    )
    checkpoint_gnn_type = (
        checkpoint.get("gnn_type")
        if checkpoint is not None and checkpoint.get("gnn_type") is not None
        else str(config.get('model', {}).get('gnn_type', 'gcn')).lower()
    )
    inferred_graph_token_count = infer_graph_token_count(policy_state)
    checkpoint_graph_token_count = (
        int(checkpoint.get("graph_token_count"))
        if checkpoint is not None and checkpoint.get("graph_token_count") is not None
        else inferred_graph_token_count
        or int(config.get('model', {}).get('graph_token_count', 4))
    )
    checkpoint_prompt_style = (
        checkpoint.get("prompt_style")
        if checkpoint is not None and checkpoint.get("prompt_style") is not None
        else "full"
    )

    trust_remote_code = bool(config.get('model', {}).get('trust_remote_code', False))

    checkpoint_tokenizer = checkpoint.get("tokenizer") if checkpoint is not None else None
    if checkpoint_tokenizer is not None:
        tokenizer = checkpoint_tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_llm_name, trust_remote_code=trust_remote_code)
        if '[GRAPH]' not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': ['[GRAPH]']})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype_name = str(config.get('model', {}).get('torch_dtype', 'float16')).lower()
    if torch_dtype_name in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif torch_dtype_name in {"fp16", "float16"}:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    llm_state_dict = checkpoint.get("llm", {}) if checkpoint is not None else {}
    llm_checkpoint_mode = checkpoint.get("llm_checkpoint_mode", "full") if checkpoint is not None else "full"
    checkpoint_has_lora = checkpoint_uses_lora(llm_state_dict)
    load_in_4bit = (
        bool(config.get('model', {}).get('load_in_4bit', False))
        and llm_name_supports_qwen_kbit(checkpoint_llm_name)
    )
    use_lora = (
        bool(config.get('model', {}).get('use_lora', False))
        and checkpoint_has_lora
    )

    if bool(config.get('model', {}).get('load_in_4bit', False)) and not load_in_4bit:
        print(f"Disabling 4-bit loading for checkpoint backbone `{checkpoint_llm_name}`.")
    if bool(config.get('model', {}).get('use_lora', False)) and not use_lora:
        print("Disabling LoRA adapter injection because the checkpoint does not contain LoRA weights.")
    if checkpoint_has_lora and not PEFT_AVAILABLE:
        raise ImportError(
            "This checkpoint contains LoRA weights, but the `peft` package is not installed. "
            "Install it with `pip install peft` before running evaluation."
        )

    quant_config = None
    device_map = None
    if load_in_4bit:
        bnb_ok, bnb_msg = get_bitsandbytes_4bit_status()
        if not bnb_ok:
            print(f"Warning: {bnb_msg}")
            load_in_4bit = False
        else:
            print(bnb_msg)
    if load_in_4bit:
        compute_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        device_map = "auto"

    llm = AutoModelForCausalLM.from_pretrained(
        checkpoint_llm_name,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quant_config,
    )
    llm.resize_token_embeddings(len(tokenizer))
    if device_map is None:
        llm.to(device)
    llm.eval()

    if use_lora:
        if load_in_4bit:
            llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
        lora_target_modules = list(config.get('model', {}).get('lora_target_modules', []))
        lora_config = LoraConfig(
            r=int(config.get('model', {}).get('lora_r', 16)),
            lora_alpha=int(config.get('model', {}).get('lora_alpha', 32)),
            lora_dropout=float(config.get('model', {}).get('lora_dropout', 0.05)),
            target_modules=lora_target_modules if lora_target_modules else None,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)

    policy_hidden_size = llm.config.hidden_size
    if checkpoint is not None:
        inferred_policy_hidden_size = infer_policy_hidden_size(policy_state)
        if inferred_policy_hidden_size is not None:
            policy_hidden_size = inferred_policy_hidden_size

    policy = GNNLLMPolicy(
        gnn_out_dim=config['model']['gnn_out_dim'],
        text_emb_dim=text_emb_dim,
        llm_hidden_size=policy_hidden_size,
        gnn_type=checkpoint_gnn_type,
        use_gat=config['model']['use_gat'],
        pooling_strategy=checkpoint_pooling_strategy,
        graph_token_count=checkpoint_graph_token_count,
    ).to(device)

    if checkpoint is not None:
        policy.load_state_dict(policy_state)
        if llm_checkpoint_mode == "full":
            llm.load_state_dict(checkpoint["llm"])
        else:
            llm.load_state_dict(checkpoint["llm"], strict=False)
        print(f"Loaded compatible checkpoint: {checkpoint_path}")

    policy.eval()
    test_dataset = PenTestDataset(
        test_data,
        text_model,
        max_seq_length=config.get('training', {}).get('max_seq_length', 1024),
        prompt_style=checkpoint_prompt_style,
        graph_token_count=checkpoint_graph_token_count,
    )

    mcp_threshold = float(checkpoint.get("mcp_threshold", 0.5)) if checkpoint is not None else 0.5

    print("Evaluating on full test dataset...")
    metrics = evaluate_metrics_on_dataset(
        test_dataset, policy, llm, tokenizer, text_model, device, threshold=mcp_threshold
    )
    num_samples = len(test_dataset)

    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION ON FULL DATASET ({num_samples} samples):")
    print(f"Average Reward: {metrics['avg_reward']:.4f}")
    print(f"Step Accuracy: {metrics['step_acc']:.4f}")
    print(f"Step Micro F1: {metrics['step_micro_f1']:.4f}")
    print(f"MCP Accuracy: {metrics['mcp_acc']:.4f}")
    print(f"MCP Micro F1 (global): {metrics['mcp_micro_f1']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
