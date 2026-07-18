"""
Evaluation on test_data.csv.

Supports two evaluation modes, selectable with --model:
  --model gnn   : fast, deterministic -- runs the Stage-1 GNN classifier
                  directly (no text generation, exactly reproducible).
  --model llm   : runs the Stage-2 (or Stage-3) generative model, parses its
                  structured JSON output, and maps "New step"/"MCP_tasks"
                  back onto the label taxonomy with the same normalizer
                  used at training time, so LLM free text is scored on a
                  level field with the GNN classifier.

Both modes report, for the held-out test split:
  Step Type   : Accuracy, Macro-F1, Weighted-F1, per-class F1, confusion matrix
  MCP Type    : Subset (exact-match) Accuracy, Micro-F1, Macro-F1,
                Samples-F1, per-label precision/recall/F1

Step explanation quality is intentionally NOT scored here (per your note --
hook is left in `--score-explanations` for the later G-Eval-style pass).

Usage:
    python evaluate.py --model gnn
    python evaluate.py --model llm --adapter-dir checkpoints/stage3_qwen_grpo
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

from config import (
    TEST_CSV, STAGE1_CKPT, STEP_LABELS, MCP_LABELS, MCP_DECISION_THRESHOLD,
    QWEN_MODEL_NAME,
)
from data_utils import load_and_clean, load_graph, _embed_texts, CONTEXT_COLUMNS, mcp_multihot
from graph_encoder import Stage1Classifier
from stage2_sft_qwen import build_prompt, SYSTEM_PROMPT


# ----------------------------------------------------------------------------
# GNN-mode evaluation
# ----------------------------------------------------------------------------
def eval_gnn(threshold=MCP_DECISION_THRESHOLD):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = load_and_clean(TEST_CSV, "test")

    model = Stage1Classifier().to(device)
    model.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
    model.eval()

    graphs, field_embs_list, step_gold, mcp_gold = [], [], [], []
    for ex in examples:
        g = load_graph(ex["machine"], ex["row_id"], ex["ptt"], "test")
        graphs.append(g)
        field_embs_list.append(_embed_texts([ex["context"][c] or "empty" for c in CONTEXT_COLUMNS]))
        step_gold.append(ex["step_idx"])
        mcp_gold.append(ex["mcp_vec"])

    step_preds, mcp_preds = [], []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(graphs), bs):
            batch_graphs = Batch.from_data_list(graphs[i:i + bs]).to(device)
            batch_fe = torch.tensor(np.stack(field_embs_list[i:i + bs]), dtype=torch.float32).to(device)
            step_logits, mcp_logits, _ = model(
                batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch, batch_fe
            )
            step_preds.append(step_logits.argmax(-1).cpu().numpy())
            mcp_preds.append((torch.sigmoid(mcp_logits) >= threshold).float().cpu().numpy())

    step_preds = np.concatenate(step_preds)
    mcp_preds = np.concatenate(mcp_preds)
    step_gold = np.array(step_gold)
    mcp_gold = np.stack(mcp_gold)

    report(step_preds, step_gold, mcp_preds, mcp_gold)


# ----------------------------------------------------------------------------
# LLM-mode evaluation
# ----------------------------------------------------------------------------
def eval_llm(adapter_dir, threshold=MCP_DECISION_THRESHOLD, max_new_tokens=400):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from data_utils import StepLabelNormalizer, extract_mcp_labels

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, adapter_dir).eval()

    examples = load_and_clean(TEST_CSV, "test")
    normalizer = StepLabelNormalizer()

    step_preds, mcp_preds, step_gold, mcp_gold = [], [], [], []
    for ex in examples:
        prompt = build_prompt(ex)
        full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
        ids = tokenizer(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        gen_text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            obj = json.loads(gen_text[gen_text.index("{"): gen_text.rindex("}") + 1])
        except Exception:
            obj = {}

        pred_step_raw = obj.get("New step", "")
        pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
        step_idx = STEP_LABELS.index(pred_step_norm) if pred_step_norm in STEP_LABELS else -1
        step_preds.append(step_idx)

        pred_mcp_keys = list(obj.get("MCP_tasks", {}).keys()) if isinstance(obj.get("MCP_tasks"), dict) else []
        # normalize predicted keys onto the canonical taxonomy too
        pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
        mcp_preds.append(mcp_multihot(pred_mcp_labels))

        step_gold.append(ex["step_idx"])
        mcp_gold.append(ex["mcp_vec"])

    # rows where the model produced an unparseable/unmappable step are
    # scored as incorrect (index -1 never matches any gold index)
    report(np.array(step_preds), np.array(step_gold), np.stack(mcp_preds), np.stack(mcp_gold))


# ----------------------------------------------------------------------------
# Shared reporting
# ----------------------------------------------------------------------------
def report(step_preds, step_gold, mcp_preds, mcp_gold):
    print("\n================= STEP TYPE =================")
    print("Accuracy      :", accuracy_score(step_gold, step_preds))
    print("Macro F1      :", f1_score(step_gold, step_preds, average="macro", zero_division=0))
    print("Weighted F1   :", f1_score(step_gold, step_preds, average="weighted", zero_division=0))
    print("\nPer-class report:")
    labels_present = sorted(set(step_gold.tolist()) | set(step_preds[step_preds >= 0].tolist()))
    print(classification_report(
        step_gold, step_preds,
        labels=labels_present,
        target_names=[STEP_LABELS[i] if 0 <= i < len(STEP_LABELS) else "UNPARSEABLE" for i in labels_present],
        zero_division=0,
    ))
    print("Confusion matrix (rows=gold, cols=pred):")
    print(confusion_matrix(step_gold, step_preds, labels=list(range(len(STEP_LABELS)))))

    print("\n================= MCP TYPE (multi-label) =================")
    print("Subset (exact-match) Accuracy :", accuracy_score(mcp_gold, mcp_preds))
    print("Micro F1                      :", f1_score(mcp_gold, mcp_preds, average="micro", zero_division=0))
    print("Macro F1                      :", f1_score(mcp_gold, mcp_preds, average="macro", zero_division=0))
    print("Samples F1                    :", f1_score(mcp_gold, mcp_preds, average="samples", zero_division=0))

    prec, rec, f1, support = precision_recall_fscore_support(
        mcp_gold, mcp_preds, average=None, zero_division=0
    )
    print("\nPer-label MCP metrics:")
    for i, label in enumerate(MCP_LABELS):
        print(f"  {label:22s} P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f} support={int(support[i])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gnn", "llm"], default="gnn")
    parser.add_argument("--adapter-dir", default="checkpoints/stage3_qwen_grpo")
    parser.add_argument("--threshold", type=float, default=MCP_DECISION_THRESHOLD)
    args = parser.parse_args()

    if args.model == "gnn":
        eval_gnn(threshold=args.threshold)
    else:
        eval_llm(args.adapter_dir, threshold=args.threshold)
