"""
evaluate.py — Comprehensive evaluation for all three training stages.

Metrics reported for every model:

  STEP CLASSIFICATION
    Accuracy, Macro-F1, Weighted-F1, per-class precision/recall/F1,
    confusion matrix

  MCP TOOL CLASSIFICATION  (multi-label)
    Subset (exact-match) accuracy, Micro-F1 (sklearn, pooled over the whole
    matrix), Macro-F1 (per-label average), Samples-F1/-precision/-recall
    (per-row average — the paper-comparable "Micro F1" definition used by
    train_step_CNN.py / test_step_CNN.py and arXiv:2605.04499 Table 3 — NOT
    the same number as sklearn micro-F1 above), Hamming loss, Jaccard mean,
    micro-precision vs micro-recall (tells you whether errors skew toward
    extra predicted tools or missing ones), per-label precision/recall/F1.
    A machine-readable summary of every metric above is also written to
    `output/eval_metrics_<model_tag>.json` on each run.

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
    precision_recall_fscore_support, hamming_loss, precision_score, recall_score,
)
from datetime import datetime, timezone

# ── Path bootstrap (folder was restructured into core/ data_prep/ training/ eval/) ──
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "core"), _os.path.join(_ROOT, "data_prep"), _os.path.join(_ROOT, "training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from config import (
    INPUT_TEST_JSON, STAGE1_CKPT, STEP_LABELS, MCP_LABELS, MCP_DECISION_THRESHOLD,
    QWEN_MODEL_NAME, ROOT, LLM_JUDGE_MODEL_NAME,
    SEMANTIC_LM_NAME, SEMANTIC_MAX_TOKENS,
)
from data_utils import (
    load_from_input_json, mcp_multihot, StepLabelNormalizer, extract_mcp_labels,
    precompute_semantic_tokens,
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




def compute_reference_explanation_metrics(pred_explanations, gold_explanations):
    """Reference-based explanation metrics.

    BERTScore and BLEURT are preferred automatic metrics for semantic
    explanation similarity; both are optional dependencies so evaluation
    remains runnable in minimal environments. Scores are reported separately
    from the LLM judge because embedding metrics can reward lexical/semantic
    similarity without proving factual correctness.
    """
    result = {"bertscore_f1": None, "bleurt": None}
    try:
        from bert_score import score as bert_score
        P, R, F = bert_score(pred_explanations, gold_explanations,
                              lang="en", rescale_with_baseline=True,
                              verbose=False)
        result["bertscore_f1"] = float(F.mean().item())
    except Exception as e:
        result["bertscore_error"] = str(e)
    try:
        from bleurt import score as bleurt_score
        checkpoint = os.environ.get("BLEURT_CHECKPOINT", "BLEURT-20")
        scorer = bleurt_score.BleurtScorer(checkpoint)
        vals = scorer.score(references=gold_explanations,
                            candidates=pred_explanations)
        result["bleurt"] = float(np.mean(vals))
    except Exception as e:
        result["bleurt_error"] = str(e)
    return result


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

    # BUG FIX: this used to build a `field_embs` tensor (a BGE embedding of
    # CONTEXT_COLUMNS) and pass it as the model's 4th positional arg. That
    # parameter is accepted by Stage1Classifier.forward() for backward
    # compatibility but never actually used inside encode_and_predict() --
    # what the model actually requires (and raises ValueError without) is
    # `semantic_tokens`/`semantic_mask`: frozen-GPT-2 token embeddings of
    # "New strategy" + "Strategy explanation", built by
    # precompute_semantic_tokens() exactly the way
    # training/stage1_gnn_train.py's Stage1Dataset builds them. This whole
    # code path had apparently never been run end-to-end before -- the bug
    # surfaced as an immediate crash, not a silent wrong answer.
    for ex in examples:
        texts = [ex["context"].get("New strategy", "") or "empty",
                 ex["context"].get("Strategy explanation", "") or "empty"]
        ex["semantic_text"] = f"{texts[0]} {texts[1]}"
    precompute_semantic_tokens(examples, model_name=SEMANTIC_LM_NAME,
                                max_tokens=SEMANTIC_MAX_TOKENS, device=device)

    graphs, step_gold, mcp_gold = [], [], []
    for ex in examples:
        # Graph is already a torch_geometric Data object from load_from_input_json
        graphs.append(ex["graph"])
        step_gold.append(ex["step_idx"])
        mcp_gold.append(ex["mcp_vec"])

    step_preds, mcp_preds = [], []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            from torch_geometric.data import Batch as PyGBatch
            batch_examples = examples[i : i + bs]
            batch_graphs = PyGBatch.from_data_list(graphs[i : i + bs]).to(device)

            # Pad this batch's variable-length semantic token sequences into
            # one (B, max_len, D) tensor + a validity mask -- same padding
            # logic as training/stage1_gnn_train.py's collate().
            tokens = [ex["semantic_tokens"] for ex in batch_examples]
            max_len = max(t.shape[0] for t in tokens)
            d = tokens[0].shape[1]
            sem = torch.zeros(len(tokens), max_len, d, dtype=torch.float32)
            mask = torch.zeros(len(tokens), max_len, dtype=torch.bool)
            for j, t in enumerate(tokens):
                L = t.shape[0]
                sem[j, :L] = t
                mask[j, :L] = True
            sem = sem.to(device)
            mask = mask.to(device)

            edge_attr = getattr(batch_graphs, 'edge_attr', None)
            step_logits, mcp_logits, _ = model(
                batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch,
                semantic_tokens=sem, semantic_mask=mask, edge_attr=edge_attr,
            )
            step_preds.append(step_logits.argmax(-1).cpu().numpy())
            probs = torch.sigmoid(mcp_logits).cpu().numpy()
            mcp_preds.append(predict_with_per_class_thresholds(probs, use_thresholds))

    step_preds = np.concatenate(step_preds)
    mcp_preds  = np.concatenate(mcp_preds)
    step_gold  = np.array(step_gold)
    mcp_gold   = np.stack(mcp_gold)

    report_classification(step_preds, step_gold, mcp_preds, mcp_gold,
                           model_tag="stage1_gnn", mcp_thresholds=use_thresholds)

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
        GRAPH_PREFIX_SRC_DIM,
    )
    from graph_encoder import Stage1Classifier
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

    # Load tokenizer from adapter directory
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype
    ).to(device)
    
    # Load adapter using load_adapter method to bypass HuggingFace hub validation
    base.load_adapter(adapter_dir)
    llm_model = base
    llm_model.eval()

    # ── Graph prefix adapter ──────────────────────────────────────────────
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        stage1.load_state_dict(ckpt["model_state_dict"])
    else:
        stage1.load_state_dict(ckpt)
    # Stage 2/3 graph conditioning uses ONLY the raw 512-d GINE representation.
    # Stage-1 fusion/classification outputs are not part of the LLM interface.
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    from config import GRAPH_PREFIX_TOKENS
    llm_hidden = llm_model.config.hidden_size
    # FIX (architecture re-audit): this used to build the adapter directly in
    # bf16 (`.to(dtype)`) before loading its checkpoint. Training deliberately
    # keeps the adapter itself in fp32 throughout and only casts its OUTPUT to
    # bf16 right before concatenation (see stage2_sft_qwen.py's forward_batch
    # -- an explicit comment there documents this as the fix for
    # "intermittent NaNs" seen when the adapter was trained directly in
    # bf16). Loading fp32-trained weights into a module whose parameters are
    # already bf16 silently downcasts them in place, so evaluation was
    # running the adapter's LayerNorms/GELUs/matmuls in a precision it was
    # deliberately trained to avoid. Keep the adapter in fp32 and only cast
    # its output below, exactly matching training's forward_batch.
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device)
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
    # NOTE: precompute_stage1_hints has been removed. Per the architecture
    # contract, the graph prefix tokens are the ONLY graph-derived signal;
    # no classifier predictions are leaked as text into the prompt. The LLM
    # must decode graph structure from the soft-prompt tokens and combine it
    # with the strategy text itself, rather than copying a provided hint.
    normalizer = StepLabelNormalizer()

    step_preds, mcp_preds, step_gold, mcp_gold       = [], [], [], []
    pred_explanations, gold_explanations               = [], []
    parse_failures                                     = 0
    csv_rows: list[dict]                               = []

    for ex in tqdm(examples, desc="Generating", unit="sample"):
        prompt = build_prompt(ex)
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{prompt}\n"
            f"<|assistant|>\n"
        )

        with torch.no_grad():
            pyg_batch = PyGBatch.from_data_list([ex["graph"]]).to(device)
            edge_attr = getattr(pyg_batch, 'edge_attr', None)
            # Stage 2/3 graph conditioning uses ONLY the raw 512-d GINE
            # representation (see stage2_sft_qwen.py's forward_batch comment
            # and module docstring) -- the private Stage-1 fusion/classifier
            # vector is never exposed to the LLM. Reproduce that exact
            # interface at evaluation time.
            #
            # CLEANUP (architecture re-audit): removed a `context_texts`
            # variable and a comment claiming Stage 2/3 were trained from a
            # "classification-calibrated fused Stage-1 representation" --
            # both were dead/stale. context_texts was computed and never
            # used, and the code beneath it has always called
            # stage1.graph_encoder(...) (the raw GINE output), matching
            # training exactly; the comment described a different, earlier
            # design that isn't what this code (or training) actually does.
            graph_h = stage1.graph_encoder(
                pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch,
                edge_attr=edge_attr
            )
            expected_dim = adapter.proj[0].in_features
            if graph_h.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"Evaluation graph-prefix dimension mismatch: Stage-1 GINE produced {graph_h.shape[-1]} dims, "
                    f"but the adapter expects {expected_dim}."
                )
            # fp32 forward through the adapter (matching training's
            # forward_batch), cast only the output to the model's dtype.
            prefix_embeds = adapter(graph_h.float()).to(dtype)
            # BUG FIX (train/eval mismatch): this used
            # `truncation=True, max_length=900`, which keeps the FIRST 900
            # tokens and discards the tail -- but the tail is where the
            # actual "# Strategy" text and "# Task" instruction live
            # (build_prompt puts Machine first, Strategy/Task last). Stage 2
            # TRAINING truncates the other way (`prompt_ids[-max_prompt_len:]`
            # in SFTDataset, keeping the end), so any prompt over 900 tokens
            # was evaluated with its most important content deleted -- a
            # deletion that never happens during training. Now matches
            # training: keep the END, and use the same 1536 budget
            # SFTDataset uses.
            ids = tokenizer(
                full_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            if ids.shape[1] > 1536:
                ids = ids[:, -1536:]
            ids = ids.to(device)
            token_embeds  = embed_layer(ids).to(dtype)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            # BUG FIX (train/eval mismatch): `repetition_penalty=1.1` was
            # applied here but NOWHERE in training or in stage2_sft_qwen.py's
            # own test loop. It is actively harmful for this task: the model
            # must reproduce a long canonical STEP_LABELS string verbatim,
            # and those labels repeat vocabulary that already appears in the
            # prompt/taxonomy ("Enumerate", "the", "further", ...), so a
            # repetition penalty systematically pushes the decoder AWAY from
            # the exact label text that exact-match scoring requires. Dropped
            # (along with `temperature=1.0`, which is meaningless and emits a
            # warning under `do_sample=False`) so evaluation decodes exactly
            # the way training/Stage-2's own evaluation does.
            out = llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
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
    llm_model_tag = os.path.basename(os.path.normpath(adapter_dir)) or "llm"
    metrics_summary = report_classification(
        step_preds_arr, step_gold_arr, mcp_preds_arr, mcp_gold_arr,
        model_tag=llm_model_tag, mcp_thresholds=use_thresholds,
    )

    # ── Explanation quality report (LLM Judge) ────────────────────────────────
    print("\n\n" + "=" * 60)
    print("STEP EXPLANATION QUALITY - LLM JUDGE")
    print("=" * 60)
    if use_llm_judge:
        print("Computing reference-based explanation metrics (BERTScore/BLEURT when installed)...")
        ref_metrics = compute_reference_explanation_metrics(pred_explanations, gold_explanations)
        print(f"  BERTScore F1 : {ref_metrics.get('bertscore_f1')}  "
              f"(secondary signal -- semantic similarity only, not factual "
              f"correctness; see the LLM judge below for that)")
        print(f"  BLEURT       : {ref_metrics.get('bleurt')}")
        # Persist alongside the step/MCP summary so BERTScore/BLEURT are
        # comparable across runs without re-parsing console output, same as
        # every other metric in this file.
        metrics_summary["explanation_reference_metrics"] = ref_metrics
        out_path = os.path.join(ROOT, "output", f"eval_metrics_{llm_model_tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
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
        # Print LLM judge results, plus save a qualitative markdown report
        # (full predicted vs. gold explanation text for a stratified sample)
        judge_report_path = os.path.join(ROOT, "output", f"llm_judge_examples_{llm_model_tag}.md")
        os.makedirs(os.path.dirname(judge_report_path), exist_ok=True)
        print_llm_judge_results(llm_results, n_examples=8, save_path=judge_report_path)
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
    model_tag: str = "model",
    mcp_thresholds: list | None = None,
) -> dict:
    """
    Prints the full metric set for both heads and also returns (and saves to
    `output/eval_metrics_<model_tag>.json`) a plain-dict summary, so results
    are comparable across runs/models without re-parsing console output.

    STEP (single-label): accuracy, macro-F1, weighted-F1.
    MCP (multi-label, "which tools" set prediction): subset (exact-match)
    accuracy, micro-F1 (sklearn's pooled-over-the-whole-matrix definition),
    samples-F1 (per-row F1 averaged across rows — this is the number the
    Pen-Strategist paper / train_step_CNN.py / test_step_CNN.py report as
    "Micro F1"; it is NOT the same number as sklearn micro-F1 above, and the
    two can diverge under label imbalance — compare against the paper using
    samples-F1, not micro-F1), macro-F1 (per-label average, catches silent
    failure on rare tools like hydra/John-the-ripper), Hamming loss (fraction
    of individual tool-slots wrong), Jaccard mean (intersection/union per
    row, a stricter middle ground between exact-match and F1), and
    micro-precision/micro-recall reported separately so you can tell whether
    errors skew toward extra predicted tools (low precision) or missing ones
    (low recall) -- F1 alone hides that direction.
    """
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

    # Tool-set error analysis requested for the final evaluation: how many
    # gold tools were omitted and how many non-gold tools were added.
    missing_counts = []
    extra_counts = []
    missing_rates = []  # per-row: |missing| / |gold| -- "what fraction of the
                         # tools we needed did we forget", only defined for
                         # rows with a non-empty gold set.
    extra_rates = []    # per-row: |extra| / |predicted| -- "what fraction of
                         # what we predicted was wrong", only defined for
                         # rows where we predicted at least one tool.
    missing_total = 0
    extra_total = 0
    exact_match_count = 0
    for pred_row, gold_row in zip(mcp_preds, mcp_gold):
        pred_set = {j for j, v in enumerate(pred_row) if v == 1}
        gold_set = {j for j, v in enumerate(gold_row) if v == 1}
        missing = gold_set - pred_set
        extra = pred_set - gold_set
        missing_counts.append(len(missing))
        extra_counts.append(len(extra))
        if gold_set:
            missing_rates.append(len(missing) / len(gold_set))
        if pred_set:
            extra_rates.append(len(extra) / len(pred_set))
        missing_total += len(missing)
        extra_total += len(extra)
        exact_match_count += int(pred_set == gold_set)
    avg_missing_tools = float(np.mean(missing_counts)) if missing_counts else 0.0
    avg_extra_tools = float(np.mean(extra_counts)) if extra_counts else 0.0
    missing_tool_rate = float(np.mean(missing_rates)) if missing_rates else 0.0
    extra_tool_rate = float(np.mean(extra_rates)) if extra_rates else 0.0
    exact_match_rate = float(exact_match_count / len(mcp_preds)) if len(mcp_preds) else 0.0

    # ── STEP metrics ──
    step_acc = float(accuracy_score(step_gold, step_preds))
    step_macro_f1 = float(f1_score(step_gold, step_preds, average='macro', zero_division=0))
    step_weighted_f1 = float(f1_score(step_gold, step_preds, average='weighted', zero_division=0))

    print("\n" + "=" * 60)
    print("STEP CLASSIFICATION")
    print("=" * 60)
    print(f"  Accuracy      : {step_acc:.4f}")
    print(f"  Macro F1      : {step_macro_f1:.4f}")
    print(f"  Weighted F1   : {step_weighted_f1:.4f}")
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

    # ── MCP metrics ──
    subset_acc = float(accuracy_score(mcp_gold, mcp_preds))
    micro_f1 = float(f1_score(mcp_gold, mcp_preds, average='micro', zero_division=0))
    macro_f1 = float(f1_score(mcp_gold, mcp_preds, average='macro', zero_division=0))
    samples_f1 = float(f1_score(mcp_gold, mcp_preds, average='samples', zero_division=0))
    samples_precision = float(precision_score(mcp_gold, mcp_preds, average='samples', zero_division=0))
    samples_recall = float(recall_score(mcp_gold, mcp_preds, average='samples', zero_division=0))
    micro_precision = float(precision_score(mcp_gold, mcp_preds, average='micro', zero_division=0))
    micro_recall = float(recall_score(mcp_gold, mcp_preds, average='micro', zero_division=0))
    hamming = float(hamming_loss(mcp_gold, mcp_preds))

    print("\n" + "=" * 60)
    print("MCP TOOL CLASSIFICATION  (multi-label)")
    print("=" * 60)
    print(f"  Subset (exact-match) accuracy : {subset_acc:.4f}")
    print(f"  Exact MCP set match rate      : {exact_match_rate:.4f}")
    print(f"  Avg missing gold tools / row  : {avg_missing_tools:.3f}  (total={missing_total})")
    print(f"  Missing-tool rate             : {missing_tool_rate:.4f}  "
          f"(mean of |missing|/|gold| per row)")
    print(f"  Avg extra predicted tools/row : {avg_extra_tools:.3f}  (total={extra_total})")
    print(f"  Extra-tool rate               : {extra_tool_rate:.4f}  "
          f"(mean of |extra|/|predicted| per row)")
    print(f"  Micro F1  (pooled over matrix): {micro_f1:.4f}")
    print(f"  Macro F1  (per-label avg)     : {macro_f1:.4f}")
    print(f"  Samples F1 (per-row avg, ***paper-comparable Micro F1***): {samples_f1:.4f}")
    print(f"    Samples precision           : {samples_precision:.4f}")
    print(f"    Samples recall              : {samples_recall:.4f}")
    print(f"  Micro precision                : {micro_precision:.4f}  "
          f"(low -> over-predicting / extra tools)")
    print(f"  Micro recall                   : {micro_recall:.4f}  "
          f"(low -> under-predicting / missing tools)")
    print(f"  Hamming loss                   : {hamming:.4f}  "
          f"(fraction of tool-slots wrong, lower is better)")
    print(f"  [Jaccard] MCP mean: {mean_mcp_jac:.4f}  "
          f"(≥0.5 pass: {mcp_jac_pass}/{len(mcp_jaccards)} = {mcp_jac_pass/len(mcp_jaccards)*100:.2f}%)")
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  [Jaccard Combined (Step+MCP)/2]: {combined_jac:.4f}")
    print(f"  ═══════════════════════════════════════════════")

    prec, rec, f1, support = precision_recall_fscore_support(
        mcp_gold, mcp_preds, average=None, zero_division=0
    )
    per_label = {}
    print("\n  Per-label metrics:")
    print(f"  {'Label':<22}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Sup':>5}")
    print("  " + "-" * 52)
    for i, label in enumerate(MCP_LABELS):
        flag = "  ← low recall" if rec[i] < 0.3 and support[i] > 0 else ""
        print(
            f"  {label:<22}  {prec[i]:>6.3f}  {rec[i]:>6.3f}  "
            f"{f1[i]:>6.3f}  {int(support[i]):>5}{flag}"
        )
        per_label[label] = {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }

    # ── save a single machine-readable summary alongside the console report ──
    summary = {
        "model_tag": model_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(step_gold)),
        "mcp_thresholds": [float(t) for t in mcp_thresholds] if mcp_thresholds else None,
        "step": {
            "accuracy": step_acc,
            "macro_f1": step_macro_f1,
            "weighted_f1": step_weighted_f1,
            "jaccard_exact_match": mean_step_jac,
        },
        "mcp": {
            "subset_accuracy": subset_acc,
            "exact_match_rate": exact_match_rate,
            "avg_missing_tools": avg_missing_tools,
            "avg_extra_tools": avg_extra_tools,
            "missing_tool_rate": missing_tool_rate,
            "extra_tool_rate": extra_tool_rate,
            "total_missing_tools": int(missing_total),
            "total_extra_tools": int(extra_total),
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "samples_f1": samples_f1,
            "samples_precision": samples_precision,
            "samples_recall": samples_recall,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "hamming_loss": hamming,
            "jaccard_mean": mean_mcp_jac,
            "jaccard_pass_rate_at_0.5": mcp_jac_pass / len(mcp_jaccards) if mcp_jaccards else 0.0,
            "per_label": per_label,
        },
        "combined_jaccard_step_mcp": combined_jac,
    }
    out_dir = os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_metrics_{model_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [eval] Metrics summary saved to: {out_path}")

    return summary


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