"""
evaluate.py — Comprehensive evaluation for all three training stages.

Metrics reported for every model:

  STEP CLASSIFICATION
    Accuracy, Macro-F1, Weighted-F1, per-class precision/recall/F1,
    confusion matrix

  MCP TOOL CLASSIFICATION  (multi-label)
    Subset (exact-match) accuracy, Micro-F1, Macro-F1, Samples-F1,
    per-label precision/recall/F1

  STEP EXPLANATION QUALITY  (LLM stages only — GNN doesn't generate text)
    LLM Judge Evaluation  — teacher-style evaluation comparing predicted vs gold explanation
                            Returns correctness score (0.0-1.0) and binary correctness (>= 0.6)
    Overall Accuracy       — percentage of explanations deemed correct by LLM judge

Usage:
    python evaluate.py                         # evaluate all available models
    python evaluate.py --model gnn             # GNN only
    python evaluate.py --model llm             # best available LLM adapter
    python evaluate.py --model llm \\
        --adapter-dir checkpoints/stage3_qwen_grpo
    python evaluate.py --threshold 0.5         # override MCP threshold
    python evaluate.py --save-explanations out.csv   # dump predictions to CSV
    python evaluate.py --llm-judge-model gpt-4o  # specify LLM judge model (default: gpt-4o)
    python evaluate.py --llm-judge-samples 100   # limit number of samples for LLM judge evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

# ── Path bootstrap (folder was restructured into core/ data_prep/ training/ eval/) ──
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "core"), _os.path.join(_ROOT, "data_prep"), _os.path.join(_ROOT, "training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from config import (
    INPUT_TEST_JSON, STAGE1_CKPT, STEP_LABELS, MCP_LABELS, MCP_DECISION_THRESHOLD,
    QWEN_MODEL_NAME, ROOT, LLM_JUDGE_MODEL_NAME,
)
from data_utils import (
    load_from_input_json, CONTEXT_COLUMNS, _embed_texts,
    mcp_multihot, StepLabelNormalizer, extract_mcp_labels,
)
from graph_encoder import Stage1Classifier
from mcp_threshold_search import predict_with_per_class_thresholds
from llm_judge import batch_evaluate_explanations, print_llm_judge_results


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_stage1_checkpoint(ckpt_path: str, device: str):
    """
    Handles both checkpoint formats:
      - New (Improvement 2): dict with 'model_state_dict' + 'mcp_thresholds'
      - Legacy: plain state dict
    Returns (model, mcp_thresholds).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        mcp_thresholds = ckpt.get(
            "mcp_thresholds", [MCP_DECISION_THRESHOLD] * len(MCP_LABELS)
        )
        print(
            f"[eval] Loaded checkpoint "
            f"(epoch={ckpt.get('best_epoch','?')}, "
            f"score={ckpt.get('best_score','?'):.4f})"
            if isinstance(ckpt.get("best_score"), float)
            else f"[eval] Loaded checkpoint (epoch={ckpt.get('best_epoch','?')})"
        )
        print(
            f"[eval] Per-class MCP thresholds: "
            f"{[round(t, 2) for t in mcp_thresholds]}"
        )
    else:
        state_dict = ckpt
        mcp_thresholds = [MCP_DECISION_THRESHOLD] * len(MCP_LABELS)
        print("[eval] Legacy checkpoint — using uniform threshold=0.5 for all MCP labels.")

    model = Stage1Classifier().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, mcp_thresholds


# ---------------------------------------------------------------------------
# Explanation quality metrics - LLM Judge only
# ---------------------------------------------------------------------------

def compute_explanation_metrics_with_llm_judge(
    pred_explanations: list[str],
    gold_explanations: list[str],
    step_preds: list[int] | None = None,
    examples: list | None = None,
    model: str = "gpt-4o",
    max_samples: int = None,
) -> dict:
    """
    Compute explanation quality metrics using LLM judge only.
    
    Returns:
        Dictionary with LLM judge results including correctness scores and accuracy
    """
    if examples is None:
        raise ValueError("examples list must be provided for LLM judge evaluation")
    
    # Prepare examples for LLM judge
    llm_examples = []
    for i, ex in enumerate(examples):
        if i < len(pred_explanations):
            pred_step_label = "UNPARSEABLE"
            if len(step_preds) > i and step_preds[i] >= 0:
                pred_step_label = STEP_LABELS[step_preds[i]]
            
            llm_examples.append({
                "pred_explanation": pred_explanations[i],
                "gold_explanation": gold_explanations[i],
                "pred_step": pred_step_label,
                "gold_step": STEP_LABELS[ex["step_idx"]],
                "context": ex["context"],
                "machine": ex["machine"],
            })
    
    # Run LLM judge evaluation
    llm_results = batch_evaluate_explanations(
        examples=llm_examples,
        model=model,
        max_samples=max_samples,
        verbose=True,
    )
    
    return llm_results


# ---------------------------------------------------------------------------
# GNN evaluation  (classification only — no text generation)
# ---------------------------------------------------------------------------

def eval_gnn(threshold_override=None, auto_save_csv=False) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Test input: {INPUT_TEST_JSON}")
    examples = load_from_input_json(INPUT_TEST_JSON, "test")

    if not os.path.exists(STAGE1_CKPT):
        print(f"[eval] Checkpoint not found at {STAGE1_CKPT}. Run stage1_gnn_train.py first.")
        return

    model, ckpt_thresholds = load_stage1_checkpoint(STAGE1_CKPT, device)
    use_thresholds = (
        [float(threshold_override)] * len(MCP_LABELS)
        if threshold_override is not None
        else ckpt_thresholds
    )

    graphs, field_embs_list, step_gold, mcp_gold = [], [], [], []
    for ex in examples:
        # Graph is already a torch_geometric Data object from load_from_input_json
        graphs.append(ex["graph"])
        field_embs_list.append(
            _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS])
        )
        step_gold.append(ex["step_idx"])
        mcp_gold.append(ex["mcp_vec"])

    step_preds, mcp_preds = [], []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            from torch_geometric.data import Batch as PyGBatch
            batch_graphs = PyGBatch.from_data_list(graphs[i : i + bs]).to(device)
            batch_fe = torch.tensor(
                np.stack(field_embs_list[i : i + bs]), dtype=torch.float32
            ).to(device)
            edge_attr = getattr(batch_graphs, 'edge_attr', None)
            step_logits, mcp_logits, _ = model(
                batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch, batch_fe,
                edge_attr=edge_attr,
            )
            step_preds.append(step_logits.argmax(-1).cpu().numpy())
            probs = torch.sigmoid(mcp_logits).cpu().numpy()
            mcp_preds.append(predict_with_per_class_thresholds(probs, use_thresholds))

    step_preds = np.concatenate(step_preds)
    mcp_preds  = np.concatenate(mcp_preds)
    step_gold  = np.array(step_gold)
    mcp_gold   = np.stack(mcp_gold)

    report_classification(step_preds, step_gold, mcp_preds, mcp_gold)

    # GNN has no text generation — note this explicitly
    print("\n  [Explanation quality: N/A — GNN is a classifier, not a text generator]")

    # Print thresholds used
    print("\n[eval] MCP thresholds used:")
    for label, thr in zip(MCP_LABELS, use_thresholds):
        marker = "  <-- non-default" if abs(thr - 0.5) > 0.05 else ""
        print(f"  {label:<22}  {thr:.2f}{marker}")

    # ── CSV dump for GNN predictions ────────────────────────────────────────
    if auto_save_csv:
        import csv
        
        # Create output directory
        output_dir = os.path.join(ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "stage1_gnn_predictions.csv")
        
        # Build CSV rows (without explanation fields)
        csv_rows = []
        for i, ex in enumerate(examples):
            pred_step_label = STEP_LABELS[step_preds[i]] if step_preds[i] >= 0 else "UNPARSEABLE"
            gold_step_label = STEP_LABELS[ex["step_idx"]]
            
            # Convert MCP vectors to label lists
            pred_mcp_labels = [MCP_LABELS[j] for j, val in enumerate(mcp_preds[i]) if val == 1]
            gold_mcp_labels = [MCP_LABELS[j] for j, val in enumerate(mcp_gold[i]) if val == 1]
            
            csv_row = {
                "machine": ex["machine"],
                "new_strategy": ex["context"].get("New strategy", ""),
                "strategy_explanation": ex["context"].get("Strategy explanation", ""),
                "gold_new_step": gold_step_label,
                "predicted_new_step": pred_step_label,
                "gold_mcp_tasks": "|".join(gold_mcp_labels),
                "predicted_mcp_tasks": "|".join(pred_mcp_labels),
                "step_correct": int(step_preds[i] == ex["step_idx"]),
            }
            csv_rows.append(csv_row)
        
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[eval] GNN prediction CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# LLM evaluation  (classification + explanation quality)
# ---------------------------------------------------------------------------

def eval_llm(adapter_dir: str, threshold_override=None,
             max_new_tokens: int = 200,
             save_explanations: str | None = None,
             auto_save_csv: bool = False,
             llm_judge_model_name: str | None = None,
             llm_judge_samples: int | None = None,
             use_llm_judge: bool = True) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from torch_geometric.data import Batch as PyGBatch
    from stage2_sft_qwen import (
        build_prompt, SYSTEM_PROMPT, GraphPrefixAdapter,
        GRAPH_PREFIX_SRC_DIM, precompute_stage1_hints,
    )
    from graph_encoder import Stage1Classifier
    from data_utils import _embed_texts, CONTEXT_COLUMNS
    from llm_judge import set_llm_judge_model

    # Resolve LLM judge model name
    if llm_judge_model_name is None:
        llm_judge_model_name = LLM_JUDGE_MODEL_NAME

    # MCP thresholds — not used for LLM (tools come from parsed JSON text),
    # but loaded for reporting consistency.
    if threshold_override is not None:
        use_thresholds = [float(threshold_override)] * len(MCP_LABELS)
    elif os.path.exists(STAGE1_CKPT):
        _, use_thresholds = load_stage1_checkpoint(STAGE1_CKPT, "cpu")
        print("[eval] MCP thresholds loaded from Stage-1 checkpoint (for reference).")
    else:
        use_thresholds = [MCP_DECISION_THRESHOLD] * len(MCP_LABELS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16
    print(f"[eval] Test input: {INPUT_TEST_JSON}")
    print(f"[eval] Device    : {device}")

    # ── ⚠ CRITICAL: Load SEPARATE LLM judge model BEFORE evaluation ────
    # Must use a DIFFERENT model from training base (QWEN_MODEL_NAME)
    # to prevent self-deception / reward hacking.
    if use_llm_judge:
        print(f"\n[eval] ⚠ Loading SEPARATE LLM judge: {llm_judge_model_name}")
        print(f"[eval]   (training base = {QWEN_MODEL_NAME} — must be different)")
        try:
            j_tok = AutoTokenizer.from_pretrained(llm_judge_model_name)
            if j_tok.pad_token is None:
                j_tok.pad_token = j_tok.eos_token
            j_model = AutoModelForCausalLM.from_pretrained(
                llm_judge_model_name, torch_dtype=dtype, device_map=None
            ).to(device)
            j_model.eval()
            for p in j_model.parameters():
                p.requires_grad_(False)
            set_llm_judge_model(j_model, j_tok, device)
            print(f"[eval] ✓ Separate LLM judge loaded ({llm_judge_model_name})")
        except Exception as e:
            print(f"[eval] ⚠ Failed to load LLM judge model: {e}")
            print("[eval]   Continuing without LLM judge — explanation quality will use heuristics.")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype
    ).to(device)
    llm_model = PeftModel.from_pretrained(base, adapter_dir).eval()

    # ── Graph prefix adapter ──────────────────────────────────────────────
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
    else:
        stage1.load_state_dict(ckpt)
    # FIX: keep the whole frozen Stage-1 classifier (not just graph_encoder)
    # so generation-time inference matches what Stage 2/3 training actually
    # conditioned on -- see graph_encoder.Stage1Classifier.encode_and_predict.
    # Previously this used graph_encoder ALONE (no context/strategy fusion
    # at all), which was out-of-distribution relative to both Stage 2's
    # training-time fusion and this fix's own training-time fusion, and is
    # the most likely single cause of Stage 2/3's generation-time accuracy
    # being far below their own training-time (teacher-forced) metrics.
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    from config import GRAPH_PREFIX_TOKENS
    llm_hidden = llm_model.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).to(dtype)
    adapter_ckpt = os.path.join(adapter_dir, "graph_adapter.pt")
    if os.path.exists(adapter_ckpt):
        adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
        print(f"[eval] Loaded GraphPrefixAdapter from {adapter_ckpt}")
    else:
        print(f"[eval] WARNING: graph_adapter.pt not found in {adapter_dir} — "
              f"using randomly initialised adapter (results will be worse)")
    adapter.eval()

    embed_layer = llm_model.get_input_embeddings()

    examples = load_from_input_json(INPUT_TEST_JSON, "test")
    # REMOVED: precompute_stage1_hints to force model to decode graph prefix tokens
    # instead of copying Stage 1 predictions. This is critical for Stage 2/3 to
    # actually improve over Stage 1.
    normalizer = StepLabelNormalizer()

    step_preds, mcp_preds, step_gold, mcp_gold       = [], [], [], []
    pred_explanations, gold_explanations               = [], []
    parse_failures                                     = 0
    # rows saved to CSV if --save-explanations is set
    csv_rows: list[dict]                               = []

    for ex in tqdm(examples, desc="Generating", unit="sample"):
        prompt = build_prompt(ex, mask_hint=True)  # Force model to decode graph tokens
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )

        with torch.no_grad():
            pyg_batch = PyGBatch.from_data_list([ex["graph"]]).to(device)
            edge_attr = getattr(pyg_batch, 'edge_attr', None)
            field_embs = torch.tensor(
                _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
                dtype=torch.float32,
            ).unsqueeze(0).to(device)
            combined_emb, _, _ = stage1.encode_and_predict(
                pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch, field_embs, edge_attr=edge_attr
            )
            prefix_embeds = adapter(combined_emb.to(dtype))
            ids = tokenizer(
                full_prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=900,
            ).input_ids.to(device)
            token_embeds  = embed_layer(ids).to(dtype)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            out = llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # ── Parse JSON ────────────────────────────────────────────────────
        obj = {}
        try:
            # Try to find and parse JSON object - more robust extraction
            json_candidates = []
            # Find all potential JSON objects
            for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', gen_text, re.DOTALL):
                json_candidates.append(match.group())
            
            # Try each candidate
            for candidate in json_candidates:
                try:
                    obj = json.loads(candidate)
                    break  # Successfully parsed
                except:
                    continue
            
            # If JSON parsing failed, try the original method as fallback
            if not obj:
                start = gen_text.find("{")
                end = gen_text.rfind("}") + 1
                if start != -1 and end > start:
                    try:
                        obj = json.loads(gen_text[start:end])
                    except:
                        pass
                        
        except Exception:
            parse_failures += 1
            
        # Enhanced regex fallback with multiple patterns
        if not obj:
            # Pattern 1: "New step": "value"
            m = re.search(r'"?New step"?\s*:\s*"([^"]+)"', gen_text, re.IGNORECASE)
            if m:
                obj["New step"] = m.group(1)
            # Pattern 2: "new_step": "value" (underscore variant)
            m = re.search(r'"?new_step"?\s*:\s*"([^"]+)"', gen_text, re.IGNORECASE)
            if m and "New step" not in obj:
                obj["New step"] = m.group(1)
            # Pattern 3: "step": "value"
            m = re.search(r'"?step"?\s*:\s*"([^"]+)"', gen_text, re.IGNORECASE)
            if m and "New step" not in obj:
                obj["New step"] = m.group(1)
            # Pattern 4: Look for step-like patterns in text
            m = re.search(r'(?:step|next step|action)(?:\s*:| is)\s*["\']?([^"\':.]+)["\']?', gen_text, re.IGNORECASE)
            if m and "New step" not in obj:
                obj["New step"] = m.group(1).strip()
            
            # Step explanation patterns - more comprehensive
            m2 = re.search(r'"?Step explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL | re.IGNORECASE)
            if m2:
                obj["Step explanation"] = m2.group(1)
            # Pattern 2: "step_explanation": "value"
            m2 = re.search(r'"?step_explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL | re.IGNORECASE)
            if m2 and "Step explanation" not in obj:
                obj["Step explanation"] = m2.group(1)
            # Pattern 3: "explanation": "value"
            m2 = re.search(r'"?explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL | re.IGNORECASE)
            if m2 and "Step explanation" not in obj:
                obj["Step explanation"] = m2.group(1)
            # Pattern 4: Look for explanation-like text
            m2 = re.search(r'(?:explanation|reasoning|rationale)(?:\s*:| is)\s*["\']?([^"\':.]+(?:\s+[^"\':.]+)*)["\']?', gen_text, re.IGNORECASE)
            if m2 and "Step explanation" not in obj:
                obj["Step explanation"] = m2.group(1).strip()
            
            # MCP tasks patterns - try to extract tool names
            mcp_pattern = re.search(r'"?MCP[_ ]tasks"?\s*:\s*\[([^\]]*)\]', gen_text, re.DOTALL | re.IGNORECASE)
            if mcp_pattern:
                # Extract quoted tool names from the array
                tools = re.findall(r'"([^"]+)"', mcp_pattern.group(1))
                if tools:
                    obj["MCP_tasks"] = {tool: True for tool in tools}
            # Alternative MCP pattern: object format
            mcp_obj_pattern = re.search(r'"?MCP[_ ]tasks"?\s*:\s*\{([^}]+)\}', gen_text, re.DOTALL | re.IGNORECASE)
            if mcp_obj_pattern and "MCP_tasks" not in obj:
                # Extract keys from object format
                tools = re.findall(r'"([^"]+)"\s*:', mcp_obj_pattern.group(1))
                if tools:
                    obj["MCP_tasks"] = {tool: True for tool in tools}
            # Pattern 3: Look for tool names directly in text
            if "MCP_tasks" not in obj:
                # Look for known tool names
                known_tools = ["Nmap", "Metasploit", "Netcat", "Dirbuster", "SQLmap", "Smb client", "hydra", "John-the-ripper", "Google search", "Interactive CLI", "Web page interaction"]
                found_tools = []
                for tool in known_tools:
                    if tool.lower() in gen_text.lower():
                        found_tools.append(tool)
                if found_tools:
                    obj["MCP_tasks"] = {tool: True for tool in found_tools}
        
        # Final fallback: if still no explanation, use the generated text as explanation
        if "Step explanation" not in obj or not obj["Step explanation"]:
            # Extract any meaningful text after the prompt
            if "Response:" in gen_text:
                fallback_text = gen_text.split("Response:")[-1].strip()
            else:
                fallback_text = gen_text.strip()
            # Clean up the fallback text
            fallback_text = re.sub(r'[{}\[\]"\'`]', '', fallback_text)
            fallback_text = fallback_text[:500]  # Limit length
            if fallback_text:
                obj["Step explanation"] = fallback_text

        # ── Step classification ───────────────────────────────────────────
        pred_step_raw  = obj.get("New step", "")
        pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
        s_idx = (
            STEP_LABELS.index(pred_step_norm)
            if pred_step_norm in STEP_LABELS
            else -1
        )
        step_preds.append(s_idx)
        step_gold.append(ex["step_idx"])

        # ── MCP classification ────────────────────────────────────────────
        pred_mcp_keys   = (
            list(obj.get("MCP_tasks", {}).keys())
            if isinstance(obj.get("MCP_tasks"), dict)
            else []
        )
        pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
        mcp_preds.append(mcp_multihot(pred_mcp_labels))
        mcp_gold.append(ex["mcp_vec"])

        # ── Explanation ───────────────────────────────────────────────────
        pred_expl = str(obj.get("Step explanation", "")).strip()
        gold_expl = ex.get("gold_step_explanation", "")
        if not isinstance(gold_expl, str):
            gold_expl = ""
        pred_explanations.append(pred_expl)
        gold_explanations.append(gold_expl)

        # Accumulate CSV row
        if save_explanations or auto_save_csv:
            csv_rows.append({
                "machine":          ex["machine"],
                "gold_step":        STEP_LABELS[ex["step_idx"]],
                "pred_step":        pred_step_norm or "UNPARSEABLE",
                "step_correct":     int(s_idx == ex["step_idx"]),
                "gold_mcp":         "|".join(ex["mcp_labels"]),
                "pred_mcp":         "|".join(pred_mcp_labels),
                "gold_explanation": gold_expl,
                "pred_explanation": pred_expl,
            })

    if parse_failures:
        print(
            f"\n[eval] Note: {parse_failures}/{len(examples)} responses had no "
            f"parseable JSON — regex fallback applied where possible."
        )

    step_preds_arr = np.array(step_preds)
    step_gold_arr  = np.array(step_gold)
    mcp_preds_arr  = np.stack(mcp_preds)
    mcp_gold_arr   = np.stack(mcp_gold)

    # ── Classification report ─────────────────────────────────────────────
    report_classification(step_preds_arr, step_gold_arr, mcp_preds_arr, mcp_gold_arr)

    # ── Explanation quality report (LLM Judge) ────────────────────────────────
    print("\n\n" + "=" * 60)
    print("STEP EXPLANATION QUALITY - LLM JUDGE")
    print("=" * 60)
    if use_llm_judge:
        print("Using LLM to evaluate explanation quality...")
        # Run LLM judge evaluation
        llm_results = compute_explanation_metrics_with_llm_judge(
            pred_explanations=pred_explanations,
            gold_explanations=gold_explanations,
            step_preds=step_preds_arr,
            examples=examples,
            model=llm_judge_model_name,
            max_samples=llm_judge_samples,
        )
        # Print LLM judge results
        print_llm_judge_results(llm_results)
    else:
        print("LLM judge evaluation disabled (--no-llm-judge).")
        # Compute heuristic explanation quality as fallback
        expl_lens = [len(p) for p in pred_explanations]
        if expl_lens:
            avg_len = float(np.mean(expl_lens))
            print(f"  Avg prediction length: {avg_len:.0f} chars")
            print("  (LLM judge disabled — use --use-llm-judge for semantic evaluation)")

    # ── Optional CSV dump ─────────────────────────────────────────────────
    if (save_explanations or auto_save_csv) and csv_rows:
        import csv
        
        # Create output directory
        output_dir = os.path.join(ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        if save_explanations:
            csv_path = save_explanations
        else:
            # Auto-generate path based on adapter directory name
            if adapter_dir:
                stage_name = os.path.basename(adapter_dir)
                # Map adapter directory names to expected CSV filenames
                if stage_name == "stage2_qwen_lora":
                    csv_filename = "stage2.csv"
                elif stage_name == "stage3_qwen_grpo":
                    csv_filename = "stage3.csv"
                else:
                    csv_filename = f"{stage_name}_predictions.csv"
                csv_path = os.path.join(output_dir, csv_filename)
            else:
                csv_path = os.path.join(output_dir, "llm_predictions.csv")
        
        # Enhance CSV rows with all requested fields
        enhanced_rows = []
        for i, row in enumerate(csv_rows):
            enhanced_row = {
                "machine": row["machine"],
                "new_strategy": examples[i]["context"].get("New strategy", ""),
                "strategy_explanation": examples[i]["context"].get("Strategy explanation", ""),
                "gold_new_step": row["gold_step"],
                "predicted_new_step": row["pred_step"],
                "gold_step_explanation": row["gold_explanation"],
                "predicted_step_explanation": row["pred_explanation"],
                "gold_mcp_tasks": row["gold_mcp"],
                "predicted_mcp_tasks": row["pred_mcp"],
                "step_correct": row["step_correct"],
            }
            enhanced_rows.append(enhanced_row)
        
        fieldnames = list(enhanced_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enhanced_rows)
        print(f"\n[eval] Prediction CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _compute_jaccard(pred_set: set, gold_set: set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    union = pred_set | gold_set
    return len(pred_set & gold_set) / len(union) if union else 0.0


def report_classification(
    step_preds: np.ndarray,
    step_gold: np.ndarray,
    mcp_preds: np.ndarray,
    mcp_gold: np.ndarray,
) -> None:
    # ── Jaccard metrics (consistent with Stage 2/3 evaluation) ──
    step_jaccards = []
    mcp_jaccards = []
    for i in range(len(step_gold)):
        step_j = 1.0 if step_preds[i] == step_gold[i] else 0.0
        step_jaccards.append(step_j)
        pred_mcp_set = set(MCP_LABELS[j] for j, v in enumerate(mcp_preds[i]) if v == 1)
        gold_mcp_set = set(MCP_LABELS[j] for j, v in enumerate(mcp_gold[i]) if v == 1)
        mcp_jaccards.append(_compute_jaccard(pred_mcp_set, gold_mcp_set))
    mean_step_jac = float(np.mean(step_jaccards))
    mean_mcp_jac = float(np.mean(mcp_jaccards))
    mcp_jac_pass = sum(1 for j in mcp_jaccards if j >= 0.5)
    combined_jac = (mean_step_jac + mean_mcp_jac) / 2.0

    print("\n" + "=" * 60)
    print("STEP CLASSIFICATION")
    print("=" * 60)
    print(f"  Accuracy      : {accuracy_score(step_gold, step_preds):.4f}")
    print(f"  Macro F1      : {f1_score(step_gold, step_preds, average='macro',    zero_division=0):.4f}")
    print(f"  Weighted F1   : {f1_score(step_gold, step_preds, average='weighted', zero_division=0):.4f}")
    print(f"  [Jaccard] Step: {mean_step_jac:.4f}  (exact match ratio)")

    labels_present = sorted(
        set(step_gold.tolist()) | set(int(p) for p in step_preds if p >= 0)
    )
    print("\n  Per-class report:")
    print(
        classification_report(
            step_gold, step_preds,
            labels=labels_present,
            target_names=[
                STEP_LABELS[i] if 0 <= i < len(STEP_LABELS) else "UNPARSEABLE"
                for i in labels_present
            ],
            zero_division=0,
        )
    )
    print("  Confusion matrix (rows=gold, cols=pred):")
    cm = confusion_matrix(step_gold, step_preds, labels=list(range(len(STEP_LABELS))))
    print(cm)

    print("\n" + "=" * 60)
    print("MCP TOOL CLASSIFICATION  (multi-label)")
    print("=" * 60)
    print(f"  Subset (exact-match) accuracy : {accuracy_score(mcp_gold, mcp_preds):.4f}")
    print(f"  Micro F1                      : {f1_score(mcp_gold, mcp_preds, average='micro',   zero_division=0):.4f}")
    print(f"  Macro F1                      : {f1_score(mcp_gold, mcp_preds, average='macro',   zero_division=0):.4f}")
    print(f"  Samples F1                    : {f1_score(mcp_gold, mcp_preds, average='samples', zero_division=0):.4f}")
    print(f"  [Jaccard] MCP mean: {mean_mcp_jac:.4f}  "
          f"(≥0.5 pass: {mcp_jac_pass}/{len(mcp_jaccards)} = {mcp_jac_pass/len(mcp_jaccards)*100:.2f}%)")
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  [Jaccard Combined (Step+MCP)/2]: {combined_jac:.4f}")
    print(f"  ═══════════════════════════════════════════════")

    prec, rec, f1, support = precision_recall_fscore_support(
        mcp_gold, mcp_preds, average=None, zero_division=0
    )
    print("\n  Per-label metrics:")
    print(f"  {'Label':<22}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Sup':>5}")
    print("  " + "-" * 52)
    for i, label in enumerate(MCP_LABELS):
        flag = "  ← low recall" if rec[i] < 0.3 and support[i] > 0 else ""
        print(
            f"  {label:<22}  {prec[i]:>6.3f}  {rec[i]:>6.3f}  "
            f"{f1[i]:>6.3f}  {int(support[i]):>5}{flag}"
        )


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def check_model_availability() -> list[tuple[str, str | None]]:
    available = []
    if os.path.exists(STAGE1_CKPT):
        available.append(("gnn", None))
    ckpt_dir = os.path.dirname(STAGE1_CKPT)
    for subdir, label in [
        ("stage2_qwen_lora", "Stage 2 SFT"),
        ("stage3_qwen_grpo", "Stage 3 GRPO"),
    ]:
        d = os.path.join(ckpt_dir, subdir)
        if os.path.exists(d):
            available.append(("llm", d))
    return available


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate stepmodelv2 — step classification, MCP tools, explanation quality"
    )
    parser.add_argument(
        "--model", choices=["gnn", "llm", "all"], default="all",
        help="Which model(s) to evaluate (default: all available)",
    )
    parser.add_argument(
        "--adapter-dir", default=None,
        help="Specific LLM adapter directory. Ignored when --model=all.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override MCP threshold (single float). "
             "Omit to use per-class thresholds from Stage-1 checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=500,
        help="Max tokens to generate per sample in LLM mode (default: 500).",
    )
    parser.add_argument(
        "--save-explanations", default=None, metavar="PATH",
        help="Save per-sample explanation predictions to a CSV file at PATH.",
    )
    parser.add_argument(
        "--auto-save-csv", action="store_true", default=None,
        help="Automatically save prediction CSVs for LLM models (stage 2/3). "
             "Automatically enabled for LLM models. Use --no-auto-save-csv to disable.",
    )
    parser.add_argument(
        "--no-auto-save-csv", dest="auto_save_csv", action="store_false",
        help="Disable automatic CSV saving for LLM models.",
    )
    parser.add_argument(
        "--use-llm-judge", action="store_true", default=None,
        help="Use LLM judge to evaluate explanation quality (requires OPENAI_API_KEY). "
             "Automatically enabled for LLM models (stage 2/3). Use --no-llm-judge to disable.",
    )
    parser.add_argument(
        "--no-llm-judge", dest="use_llm_judge", action="store_false",
        help="Disable automatic LLM judge evaluation for LLM models.",
    )
    parser.add_argument(
        "--llm-judge-model", default=LLM_JUDGE_MODEL_NAME,
        help=f"Model to use for LLM judge evaluation (default: {LLM_JUDGE_MODEL_NAME}). NOTE: Uses SEPARATE model from training base {QWEN_MODEL_NAME} to prevent self-deception reward hacking.",
    )
    parser.add_argument(
        "--llm-judge-samples", type=int, default=None,
        help="Maximum number of samples to evaluate with LLM judge (for testing).",
    )
    args = parser.parse_args()

    # Auto-enable LLM judge and CSV saving for LLM models unless explicitly disabled
    if args.use_llm_judge is None and args.model in ["llm", "all"]:
        args.use_llm_judge = True
    if args.auto_save_csv is None and args.model in ["llm", "all"]:
        args.auto_save_csv = True

    if args.model == "all":
        available = check_model_availability()
        if not available:
            print("[eval] No trained models found. Run the pipeline first.")
            sys.exit(1)

        print(f"[eval] Found {len(available)} model(s) to evaluate:")
        for mtype, adir in available:
            label = "Stage 1 GNN" if mtype == "gnn" else adir
            print(f"  • {label}")
        print()

        for mtype, adir in available:
            header = "Stage 1 GNN" if mtype == "gnn" else adir
            print(f"\n{'═' * 60}")
            print(f"  MODEL: {header}")
            print(f"{'═' * 60}")
            if mtype == "gnn":
                eval_gnn(threshold_override=args.threshold, auto_save_csv=args.auto_save_csv)
            else:
                eval_llm(
                    adir,
                    threshold_override=args.threshold,
                    max_new_tokens=args.max_new_tokens,
                    save_explanations=args.save_explanations,
                    auto_save_csv=args.auto_save_csv,
                    llm_judge_model_name=args.llm_judge_model,
                    llm_judge_samples=args.llm_judge_samples,
                    use_llm_judge=args.use_llm_judge if args.use_llm_judge is not None else True,
                )

    elif args.model == "gnn":
        if not os.path.exists(STAGE1_CKPT):
            print(f"[eval] Stage-1 checkpoint not found: {STAGE1_CKPT}")
            sys.exit(1)
        print(f"\n{'═' * 60}\n  MODEL: Stage 1 GNN\n{'═' * 60}")
        eval_gnn(threshold_override=args.threshold, auto_save_csv=args.auto_save_csv)

    else:  # llm
        adapter = args.adapter_dir
        if adapter is None:
            ckpt_dir = os.path.dirname(STAGE1_CKPT)
            for subdir in ["stage3_qwen_grpo", "stage2_qwen_lora"]:
                d = os.path.join(ckpt_dir, subdir)
                if os.path.exists(d):
                    adapter = d
                    break
        if adapter is None or not os.path.exists(adapter):
            print("[eval] No LLM adapter found. Train Stage 2/3 first, or pass --adapter-dir.")
            sys.exit(1)
        print(f"\n{'═' * 60}\n  MODEL: {adapter}\n{'═' * 60}")
        eval_llm(
            adapter,
            threshold_override=args.threshold,
            max_new_tokens=args.max_new_tokens,
            save_explanations=args.save_explanations,
            auto_save_csv=args.auto_save_csv,
            llm_judge_model_name=args.llm_judge_model,
            llm_judge_samples=args.llm_judge_samples,
            use_llm_judge=args.use_llm_judge if args.use_llm_judge is not None else True,
        )