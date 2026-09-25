"""
Generate step/explanation/MCP predictions on the held-out test set using
commercial LLMs (GPT-5, Claude, Gemini, ...), as extra baseline rows
alongside baseline_zeroshot/3shot/5shot and stage1/2/3 -- following the
Pen-Strategist paper's Table 2 methodology (arxiv.org/pdf/2605.04499):
swap the model at inference time, same test set, same task.

Reuses baseline_llm_eval.py's prompt (explicit taxonomy + JSON-format
instructions in-context) rather than stage2_sft_qwen.py's bare prompt,
because stage2/3's prompt relies on the graph reaching the model as
soft-prompt embeddings (GraphPrefixAdapter) -- commercial APIs have no
such channel, so they need the taxonomy spelled out in text, same as the
project's own zero/few-shot baselines already do.

Usage:
    python commercial_explanation_baselines.py --model gpt-5
    python commercial_explanation_baselines.py --model claude-sonnet --max-samples 20   # smoke test

Output:
    output/commercial_<model>.csv -- same schema as output/stage2.csv, so it
    plugs directly into comparison_report.py / multi_judge_explanation_eval.py.
"""
import argparse
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "data_prep"), os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import INPUT_TEST_JSON, ROOT, STEP_LABELS, MCP_LABELS
from data_utils import load_from_input_json, StepLabelNormalizer, extract_mcp_labels
from baseline_llm_eval import SYSTEM_PROMPT, build_user_content
from stage2_sft_qwen import build_obj_parser
from commercial_llm import generate_text, MODEL_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    ap.add_argument("--max-samples", type=int, default=None, help="Smoke-test on a subset first")
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    examples = load_from_input_json(INPUT_TEST_JSON, "test")
    if args.max_samples:
        examples = examples[: args.max_samples]

    normalizer = StepLabelNormalizer()
    parse_obj = build_obj_parser()

    rows = []
    n_unparseable = 0
    for i, ex in enumerate(examples):
        user_content = build_user_content(ex, include_taxonomy=True)
        try:
            raw = generate_text(SYSTEM_PROMPT, user_content, args.model, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"[{i+1}/{len(examples)}] generation failed: {e}")
            raw = ""

        obj = parse_obj(raw, normalizer) or {}
        pred_step = str(obj.get("New step", "")).strip() or "UNPARSEABLE"
        pred_expl = str(obj.get("Step explanation", "")).strip()
        mcp_val = obj.get("MCP_tasks", {})
        pred_mcp = extract_mcp_labels(str(mcp_val)) if isinstance(mcp_val, dict) else []
        if pred_step == "UNPARSEABLE":
            n_unparseable += 1

        gold_step = ex["step_label"]
        gold_mcp = ex["mcp_labels"]
        rows.append({
            "machine": ex["machine"],
            "new_strategy": ex["context"]["New strategy"],
            "strategy_explanation": ex["context"]["Strategy explanation"],
            "gold_new_step": gold_step,
            "predicted_new_step": pred_step,
            "gold_step_explanation": ex["gold_step_explanation"],
            "predicted_step_explanation": pred_expl,
            "gold_mcp_tasks": "|".join(gold_mcp),
            "predicted_mcp_tasks": "|".join(pred_mcp),
            "step_correct": int(normalizer.normalize(pred_step) == gold_step),
        })
        print(f"[{i+1}/{len(examples)}] step_correct={rows[-1]['step_correct']}")

    out_path = os.path.join(ROOT, "output", f"commercial_{args.model}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[commercial_baseline] {args.model}: {len(rows)} rows, "
          f"{n_unparseable} unparseable ({100*n_unparseable/len(rows):.1f}%)")
    print(f"[commercial_baseline] Saved to {out_path}")


if __name__ == "__main__":
    main()
