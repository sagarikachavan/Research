"""
Real graph-conditioning ablation on the actual production checkpoint.

Question: does the trained Stage 1 -> Stage 2/3 pipeline actually use the
PTT graph, or could the same predictions be produced from the strategy text
alone? `graph_adapter_experiments/eval_graph_ablation.py` and
`eval_right_vs_wrong_graph.py` do NOT answer this -- they are a separate,
smaller standalone proof-of-concept (their own tiny GINE, their own
Qwen2.5-1.5B, and they explicitly never load STAGE1_CKPT / STAGE2_ADAPTER_DIR
/ STAGE3_ADAPTER_DIR). This script runs the ACTUAL trained pipeline: for each
test example, generate a prediction twice -- once with its own correct
graph, once with a different (different-machine) example's graph substituted
in -- while holding the strategy/explanation text exactly fixed. If the model
is genuinely graph-grounded, predictions should differ meaningfully between
the two conditions, and the correct-graph condition should score closer to
the gold answer than the wrong-graph condition. A model that ignores the
graph would show near-zero difference between the two conditions.

Usage:
    python eval/graph_conditioning_ablation.py [--adapter-dir DIR] [--n 60] [--seed 42]

--adapter-dir defaults to STAGE3_ADAPTER_DIR if it holds a real adapter
checkpoint, else falls back to STAGE2_ADAPTER_DIR.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "data_prep"), os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    INPUT_TEST_JSON, QWEN_MODEL_NAME, STAGE1_CKPT, STAGE2_ADAPTER_DIR,
    STAGE3_ADAPTER_DIR, ROOT, RANDOM_SEED,
)
from data_utils import load_from_input_json, StepLabelNormalizer, extract_mcp_labels, _embed_texts
from graph_encoder import Stage1Classifier
from stage2_sft_qwen import GraphPrefixAdapter, build_prompt, SYSTEM_PROMPT, GRAPH_PREFIX_SRC_DIM
from stage3_grpo_rl import (
    build_prefix_embeds, build_prompt_embeds, trim_generated_row, _parse_completion,
)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _resolve_adapter_dir(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.isfile(os.path.join(STAGE3_ADAPTER_DIR, "adapter_config.json")):
        return STAGE3_ADAPTER_DIR
    return STAGE2_ADAPTER_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", default=None,
                     help="Policy adapter directory. Defaults to STAGE3_ADAPTER_DIR if it exists, else STAGE2_ADAPTER_DIR.")
    ap.add_argument("--n", type=int, default=60, help="Number of test examples to ablate.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    args = ap.parse_args()

    adapter_dir = _resolve_adapter_dir(args.adapter_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    rng = random.Random(args.seed)

    print(f"[ablation] Policy adapter dir : {adapter_dir}")
    print(f"[ablation] Device             : {device}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map=None
    ).to(device)
    base.config.use_cache = False
    policy = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    policy.eval()
    embed_layer = policy.get_input_embeddings()

    print(f"[ablation] Loading Stage-1 GNN checkpoint: {STAGE1_CKPT}")
    stage1 = Stage1Classifier()
    ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    stage1.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    stage1 = stage1.to(device).eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    # fp32 adapter, output-only cast -- matches training's forward_batch and
    # the fixed eval/evaluate.py eval_llm() dtype path (see
    # STAGE2_STAGE3_IMPROVEMENTS.md for why this matters).
    llm_hidden = policy.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).float()
    adapter_ckpt = os.path.join(adapter_dir, "graph_adapter.pt")
    if not os.path.isfile(adapter_ckpt):
        raise FileNotFoundError(f"graph_adapter.pt not found in {adapter_dir}")
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device, weights_only=False))
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    test_examples = load_from_input_json(INPUT_TEST_JSON, "test")
    if len(test_examples) < 2:
        raise RuntimeError("Need at least 2 test examples to substitute a different graph.")
    n = min(args.n, len(test_examples))
    sample = rng.sample(test_examples, n)
    normalizer = StepLabelNormalizer()

    def generate(ex_for_graph: dict, ex_for_text: dict):
        prefix = build_prefix_embeds(ex_for_graph, stage1, adapter, device, dtype)
        prompt = build_prompt(ex_for_text)
        full_prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
        p_emb, p_len = build_prompt_embeds(full_prompt, tokenizer, embed_layer, prefix, device, dtype)
        attn = torch.ones(1, p_len, dtype=torch.long, device=device)
        with torch.no_grad():
            out = policy.generate(
                inputs_embeds=p_emb, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        ids = trim_generated_row(out[0], tokenizer.eos_token_id, tokenizer.pad_token_id)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        obj = _parse_completion(text) or {}
        pred_step = normalizer.normalize(str(obj.get("New step", "")))
        mcp_val = obj.get("MCP_tasks", {})
        pred_mcp = set(extract_mcp_labels(str(mcp_val))) if isinstance(mcp_val, dict) else set()
        pred_expl = str(obj.get("Step explanation", ""))
        return pred_step, pred_mcp, pred_expl

    other_pool = list(test_examples)
    rows = []
    step_changed = 0
    mcp_jaccard_between = []
    correct_step_hits = 0
    wrong_step_hits = 0
    correct_mcp_jac = []
    wrong_mcp_jac = []
    expl_cosine = []

    for i, ex in enumerate(sample):
        others = [o for o in other_pool if o.get("machine") != ex.get("machine")]
        if not others:
            others = [o for o in other_pool if o is not ex]
        wrong_graph_ex = rng.choice(others)

        gold_step = normalizer.normalize(ex["step_label"])
        gold_mcp = set(ex["mcp_labels"])

        correct_step, correct_mcp, correct_expl = generate(ex, ex)
        wrong_ex_for_graph = dict(ex)
        wrong_ex_for_graph["graph"] = wrong_graph_ex["graph"]
        wrong_step, wrong_mcp, wrong_expl = generate(wrong_ex_for_graph, ex)

        step_changed += int(correct_step != wrong_step)
        j_between = _jaccard(correct_mcp, wrong_mcp)
        mcp_jaccard_between.append(j_between)
        correct_step_hits += int(correct_step == gold_step)
        wrong_step_hits += int(wrong_step == gold_step)
        correct_mcp_jac.append(_jaccard(correct_mcp, gold_mcp))
        wrong_mcp_jac.append(_jaccard(wrong_mcp, gold_mcp))

        cos = None
        try:
            emb = _embed_texts([correct_expl or "empty", wrong_expl or "empty"])
            a, b = emb[0], emb[1]
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 0:
                cos = float(np.dot(a, b) / denom)
        except Exception:
            pass
        if cos is not None:
            expl_cosine.append(cos)

        rows.append({
            "machine": ex.get("machine", ""),
            "wrong_graph_source_machine": wrong_graph_ex.get("machine", ""),
            "gold_step": gold_step,
            "correct_graph_step": correct_step,
            "wrong_graph_step": wrong_step,
            "step_changed": correct_step != wrong_step,
            "gold_mcp": "|".join(sorted(gold_mcp)),
            "correct_graph_mcp": "|".join(sorted(correct_mcp)),
            "wrong_graph_mcp": "|".join(sorted(wrong_mcp)),
            "mcp_jaccard_correct_vs_wrong": round(j_between, 3),
            "mcp_jaccard_correct_vs_gold": round(_jaccard(correct_mcp, gold_mcp), 3),
            "mcp_jaccard_wrong_vs_gold": round(_jaccard(wrong_mcp, gold_mcp), 3),
            "explanation_cosine_correct_vs_wrong": round(cos, 3) if cos is not None else "",
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"[ablation] {i + 1}/{n} examples done")

    print("\n" + "=" * 70)
    print("GRAPH-CONDITIONING ABLATION  (real Stage 1 -> 2/3 production checkpoint)")
    print("=" * 70)
    print(f"  Examples evaluated                         : {n}")
    print(f"  Step prediction CHANGED when graph swapped : {step_changed}/{n} "
          f"({100 * step_changed / n:.1f}%)")
    print(f"  Mean MCP Jaccard (correct vs wrong graph)  : {np.mean(mcp_jaccard_between):.4f}  "
          f"(1.0 = identical regardless of graph -> would suggest the model ignores the graph)")
    print()
    print(f"  Step accuracy  with CORRECT graph : {correct_step_hits / n:.4f}")
    print(f"  Step accuracy  with WRONG graph   : {wrong_step_hits / n:.4f}")
    print(f"  MCP Jaccard    with CORRECT graph : {float(np.mean(correct_mcp_jac)):.4f}")
    print(f"  MCP Jaccard    with WRONG graph   : {float(np.mean(wrong_mcp_jac)):.4f}")
    if expl_cosine:
        print(f"  Mean explanation cosine similarity (correct vs wrong graph): "
              f"{float(np.mean(expl_cosine)):.4f}  (lower = explanation text "
              f"changes more when the graph changes)")
    gap_step = correct_step_hits / n - wrong_step_hits / n
    gap_mcp = float(np.mean(correct_mcp_jac)) - float(np.mean(wrong_mcp_jac))
    print()
    print(f"  Graph-grounding gap: step {gap_step:+.4f}, MCP Jaccard {gap_mcp:+.4f}")
    print("  A meaningfully positive gap is evidence the model uses the graph;")
    print("  a gap near zero suggests it may be ignoring graph conditioning.")

    out_dir = os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "graph_conditioning_ablation.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[ablation] Per-example detail saved to: {out_path}")


if __name__ == "__main__":
    main()
