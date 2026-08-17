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
import re
import difflib
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

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


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


def build_user_content(ex: dict, include_taxonomy: bool = True) -> str:
    """Build the user-visible content for a query, optionally with taxonomy."""
    content = ""
    if include_taxonomy:
        content += f"Available step types: {STEP_TAXONOMY}\n"
        content += f"Available MCP tools: {MCP_TAXONOMY}\n\n"
    content += f"{build_prompt(ex)}\n"
    content += "IMPORTANT: Respond with ONLY a single JSON object. No other text.\n"
    content += "Format: {\"New step\": \"...\", \"Step explanation\": \"...\", \"MCP_tasks\": {...}}"
    return content


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
    """Build a few-shot prompt using the correct Qwen2.5 ChatML format.

    Qwen2.5-Instruct uses ChatML tokens: <|im_start|>role and <|im_end|>.
    The eos_token is '<|im_end|>'. Few-shot examples MUST be formatted as
    separate user → assistant dialogue turns so the chat-tuned model
    recognises them as demonstration pairs rather than raw text to continue.

    Format (ChatML):
      <|im_start|>system
      {system}<|im_end|>
      <|im_start|>user
      {example_1_query}<|im_end|>
      <|im_start|>assistant
      {example_1_response}<|im_end|>
      ... (repeated for each few-shot example, taxonomy only on first user msg)
      <|im_start|>user
      {test_query}<|im_end|>
      <|im_start|>assistant
      (model generates here — no trailing <|im_end|>)
    """
    segments = []

    segments.append(f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n")

    shots = examples[:num_shots]

    for i, ex in enumerate(shots):
        include_taxonomy = (i == 0)
        user_msg = build_user_content(ex, include_taxonomy=include_taxonomy)
        assistant_msg = build_target(ex)
        segments.append(f"{IM_START}user\n{user_msg}{IM_END}\n")
        segments.append(f"{IM_START}assistant\n{assistant_msg}{IM_END}\n")

    include_taxonomy = (num_shots == 0)
    test_user_msg = build_user_content(test_ex, include_taxonomy=include_taxonomy)
    segments.append(f"{IM_START}user\n{test_user_msg}{IM_END}\n")
    segments.append(f"{IM_START}assistant\n")

    return "".join(segments)


def parse_response(response_text: str, normalizer: StepLabelNormalizer) -> Dict:
    """Robust response parser — matches evaluate.py style.

    Tries (in order):
      1. Regex-based nested JSON extraction (handles JSON with inner objects)
      2. Simple brace-based JSON extraction
      3. Regex fallbacks for individual fields ("New step", "Step explanation", etc.)
    """
    obj = {}
    try:
        json_candidates = []
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL):
            json_candidates.append(match.group())

        for candidate in json_candidates:
            try:
                obj = json.loads(candidate)
                break
            except:
                continue

        if not obj:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    obj = json.loads(response_text[start:end])
                except:
                    pass

    except Exception:
        pass

    if not obj or "New step" not in obj:
        m = re.search(r'"?New step"?\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
        if m:
            obj["New step"] = m.group(1)
        m = re.search(r'"?new_step"?\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
        if m and "New step" not in obj:
            obj["New step"] = m.group(1)
        m = re.search(r'"?step"?\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
        if m and "New step" not in obj:
            obj["New step"] = m.group(1)
        m = re.search(r'(?:step|next step|action)(?:\s*:| is)\s*["\']?([^"\':.]+)["\']?', response_text, re.IGNORECASE)
        if m and "New step" not in obj:
            obj["New step"] = m.group(1).strip()

    if "Step explanation" not in obj:
        m2 = re.search(r'"?Step explanation"?\s*:\s*"([^"]*)"', response_text, re.DOTALL | re.IGNORECASE)
        if m2:
            obj["Step explanation"] = m2.group(1)
        m2 = re.search(r'"?step_explanation"?\s*:\s*"([^"]*)"', response_text, re.DOTALL | re.IGNORECASE)
        if m2 and "Step explanation" not in obj:
            obj["Step explanation"] = m2.group(1)

    if not obj or "MCP_tasks" not in obj or not isinstance(obj.get("MCP_tasks"), dict):
        m3 = re.search(r'"?MCP_tasks"?\s*:\s*(\{[^}]*(?:\{[^}]*\}[^}]*)*\})', response_text, re.DOTALL | re.IGNORECASE)
        if m3:
            try:
                obj["MCP_tasks"] = json.loads(m3.group(1))
            except:
                pass
        if not isinstance(obj.get("MCP_tasks"), dict):
            mcp_keys = re.findall(r'"([A-Za-z][A-Za-z\s\-]{2,})"\s*:', response_text)
            if mcp_keys:
                filtered = [k for k in mcp_keys if k in MCP_LABELS]
                if filtered:
                    obj["MCP_tasks"] = {k: "" for k in filtered}

    if not obj or "New step" not in obj:
        m = re.search(
            r'(?:next[_\s-]?step(?:\s*type)?|step\s*(?:type|choice)?|action)\s*[:\-–]\s*["\']?\s*([^"\':;.\n][^\n:;]{3,150}?)\s*(?:\.|,|\n|"|Tools|Tool|Reasoning|Explanation|$)',
            response_text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().strip('"').strip("'").rstrip(".")
            if candidate and len(candidate) > 3:
                obj["New step"] = candidate

    if "Step explanation" not in obj:
        m2 = re.search(
            r'(?:reasoning|explanation|justification|why|step\s*explanation)\s*[:\-–]\s*["\']?\s*(.{5,400}?)\s*(?:\n\n|\Z|"Tools|Tool:|Step)',
            response_text, re.DOTALL | re.IGNORECASE)
        if m2:
            obj["Step explanation"] = m2.group(1).strip().strip('"').strip("'")

    if not isinstance(obj.get("MCP_tasks"), dict) or not obj.get("MCP_tasks"):
        m4 = re.search(
            r'(?:tools?|mcp(?:[_\s-]?tasks)?)\s*[:\-–]\s*(.{3,200}?)\s*(?:\n\n|\Z|Reasoning|Explanation|Step explanation|Next)',
            response_text, re.DOTALL | re.IGNORECASE)
        if m4:
            segment = m4.group(1)
            found = _extract_mcp_labels_from_text(segment)
            if found:
                obj["MCP_tasks"] = {k: "" for k in found}

    if not isinstance(obj.get("MCP_tasks"), dict) or not obj.get("MCP_tasks"):
        full_text_tools = _extract_mcp_labels_from_text(response_text)
        if full_text_tools:
            obj["MCP_tasks"] = {k: "" for k in full_text_tools}

    return obj


_MCP_PATTERNS = {
    "Nmap": re.compile(r"\bnmap\b", re.I),
    "Metasploit": re.compile(r"\bmetasploit|msfconsole|msfvenom\b", re.I),
    "Netcat": re.compile(r"\bnetcat|\bnc\b", re.I),
    "Dirbuster": re.compile(r"\bdirbuster|gobuster|dirb|netexec\b", re.I),
    "SQLmap": re.compile(r"\bsqlmap\b", re.I),
    "Smb client": re.compile(r"smb\s*client|smbclient|\bsmb\b", re.I),
    "hydra": re.compile(r"\bhydra\b", re.I),
    "John-the-ripper": re.compile(r"john[\s\-]?the[\s\-]?ripper|\bjohn\b", re.I),
    "Google search": re.compile(r"google\s*search|\bgoogle\b", re.I),
    "Interactive CLI": re.compile(r"interactive\s*cli|\bssh\b|\bbash\b|\bshell\b", re.I),
    "Web page interaction": re.compile(r"web\s*page\s*interaction|\bbrowser\b|\bcurl\b", re.I),
}


def _extract_mcp_labels_from_text(text: str) -> list:
    """Return canonical MCP_LABELS found by regex scanning free-form text."""
    if not text:
        return []
    return [label for label, pat in _MCP_PATTERNS.items() if pat.search(text)]


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
    # Use diverse examples with REASONABLE-length explanations to avoid
    # prompt truncation. (Previously we picked LONGEST explanations, which
    # caused massive truncation of the actual test query.)
    if num_shots > 0:
        # Cap individual example size so the combined prompt fits comfortably.
        MAX_EXPL_CHARS = 600
        MAX_PROMPT_CHARS = 1500

        scored_examples = []
        for ex in train_examples:
            expl = ex.get("gold_step_explanation", "")
            prompt_text = build_prompt(ex)
            target_text = build_target(ex)
            total_chars = len(prompt_text) + len(target_text)
            # Prefer medium-length examples: informative but not huge
            if len(expl) <= MAX_EXPL_CHARS and total_chars <= MAX_PROMPT_CHARS and len(expl) > 80:
                score = len(expl)
                scored_examples.append((score, ex))

        # If not enough medium examples, fall back to any examples (still capped)
        if len(scored_examples) < num_shots * 10:
            scored_examples = []
            for ex in train_examples:
                expl = ex.get("gold_step_explanation", "")
                prompt_text = build_prompt(ex)
                target_text = build_target(ex)
                total_chars = len(prompt_text) + len(target_text)
                if total_chars <= MAX_PROMPT_CHARS * 1.5 and len(expl) > 40:
                    score = len(expl)
                    scored_examples.append((score, ex))

        # Sort by explanation length (descending) within size cap for quality
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        top_pool = [ex for _, ex in scored_examples[:max(200, num_shots * 20)]]

        # Select diverse examples from different step groups
        step_groups = {}
        for ex in top_pool:
            step = ex["step_label"]
            if step not in step_groups:
                step_groups[step] = []
            step_groups[step].append(ex)

        available_steps = list(step_groups.keys())
        random.shuffle(available_steps)
        few_shot_examples = []
        for i in range(min(num_shots, len(top_pool))):
            step = available_steps[i % len(available_steps)]
            group = step_groups[step]
            ex = random.choice(group)
            few_shot_examples.append(ex)

        print(f"[Baseline] Using {len(few_shot_examples)} few-shot examples (diverse, size-capped)")
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
        dtype=torch.bfloat16,
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
            
            prompt_text = build_few_shot_prompt(few_shot_examples, test_ex, num_shots)
            
            # Qwen2.5-Instruct has 131072-token context window. Use a generous
            # limit so the prompt + few-shot examples are never truncated.
            # (Old limit of 2048 silently removed the actual test query and
            # the <|im_start|>assistant generation marker from the prompt.)
            PROMPT_MAX_LEN = 16384
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=PROMPT_MAX_LEN,
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            
            input_len = inputs.input_ids.shape[1]
            # Detect and warn if the prompt was actually truncated — this would
            # silently invalidate the generation result.
            full_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            if full_len > PROMPT_MAX_LEN:
                print(f"[WARN] Sample {idx}: prompt truncated ({full_len} > {PROMPT_MAX_LEN} tokens) — generation will be wrong")
            if idx < 3:
                print(f"[Debug] Sample {idx} - prompt tokens: {input_len} / {PROMPT_MAX_LEN} (full untruncated would be {full_len})")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
            )
            
            generated_ids = outputs[:, input_len:]
            gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            
            obj = parse_response(gen_text, normalizer)
            
            pred_step_raw = obj.get("New step", "")
            pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
            
            if idx < 3:
                print(f"[Debug] Sample {idx} - raw_gen_text[:300]: '{gen_text[:300]}'")
                print(f"[Debug] Sample {idx} - pred_step_raw: '{pred_step_raw}'")
                print(f"[Debug] Sample {idx} - pred_step_norm: '{pred_step_norm}'")
                print(f"[Debug] Sample {idx} - pred_step_norm in STEP_LABELS: {pred_step_norm in STEP_LABELS if pred_step_norm else False}")
            
            pred_step_label = "UNPARSEABLE"
            if pred_step_norm and pred_step_norm in STEP_LABELS:
                pred_step_label = pred_step_norm
            elif pred_step_raw:
                if pred_step_raw in STEP_LABELS:
                    pred_step_label = pred_step_raw
                else:
                    closest_match = difflib.get_close_matches(pred_step_raw, STEP_LABELS, n=1, cutoff=0.6)
                    if closest_match:
                        pred_step_label = closest_match[0]
                        if idx < 3:
                            print(f"[Debug] Sample {idx} - Fuzzy matched: '{pred_step_raw}' -> '{pred_step_label}'")
            
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
                "raw_response": gen_text[:500],  # Truncate for CSV
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
