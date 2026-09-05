"""
train_multitask_adapter.py
============================

Answers "Next step #3" from the original report: trains the GraphPrefixAdapter
+ LoRA on step-prediction (the real task) and graph-structure probes
(adjacency/node_type/edge_type/two_hop/graph_aggregate) TOGETHER, in one
LoRA fine-tune, rather than as two separate models.

Each optimizer step randomly draws either a step-prediction training example
(same format Stage 2 uses -- reuses stage2_sft_qwen.forward_batch verbatim,
not a re-implementation) or a structure-task item (reuses
train_structure_adapter.forward_item), mixed at `--structure_frac`
(default 0.3: mostly step-prediction, with structure tasks as an auxiliary
signal -- this is a knob you should sweep, see RESEARCH_PLAN.md).

Checkpoint selection tracks BOTH held-out step-prediction accuracy (the
"New step" field, matching Stage 2's own metric) and held-out structure
score, and only promotes a checkpoint that doesn't regress step-prediction
accuracy below the Stage-2 baseline -- so this script can tell you "yes,
multi-task training helps structure understanding without hurting the real
task" or "no, it doesn't" with a number either way, rather than just one
combined number that could hide a regression on the task you actually care
about.

Usage:
    python graph_adapter_experiments/train_multitask_adapter.py \
        --init_from checkpoints/stage2_qwen_lora \
        --out_dir   checkpoints/multitask \
        --steps 2000 --structure_frac 0.3
"""
import os
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    CKPT_DIR, TASKS_DIR, RANDOM_SEED, GRAPH_PREFIX_SRC_DIM,
    load_all_examples, index_examples_by_key, load_task_items, machine_level_split,
)
from train_structure_adapter import forward_item as structure_forward_item, evaluate_on_items

import torch
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default=None, help="default: STAGE2_ADAPTER_DIR")
    ap.add_argument("--out_dir", default=os.path.join(CKPT_DIR, "multitask"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--structure_frac", type=float, default=0.3,
                     help="fraction of steps drawn from structure tasks vs step-prediction")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import config
    from stage2_sft_qwen import (
        GraphPrefixAdapter, build_prompt, build_target, SYSTEM_PROMPT, forward_batch,
    )
    from data_utils import load_from_input_json, _embed_texts, CONTEXT_COLUMNS
    import common as C

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    init_from = args.init_from or config.STAGE2_ADAPTER_DIR
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ── Step-prediction data (same source as Stage 2/3) ─────────────────────
    step_examples = load_from_input_json(config.INPUT_TRAIN_JSON, "train")
    step_train, step_val = machine_level_split(step_examples, val_frac=config.STAGE2_VAL_SPLIT, seed=args.seed)
    print(f"[train_multitask] Step-prediction: {len(step_train)} train / {len(step_val)} held-out")

    # ── Structure task data (from build_structure_tasks.py) ─────────────────
    struct_train = load_task_items(os.path.join(TASKS_DIR, "train.jsonl"))
    struct_held_out = load_task_items(os.path.join(TASKS_DIR, "held_out.jsonl"))
    ex_index = index_examples_by_key(load_all_examples())
    print(f"[train_multitask] Structure tasks: {len(struct_train)} train / {len(struct_held_out)} held-out")

    tokenizer = AutoTokenizer.from_pretrained(init_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config.QWEN_MODEL_NAME, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, init_from, is_trainable=True, local_files_only=True)
    model.gradient_checkpointing_enable()
    model.train()

    stage1 = C.load_stage1(device)
    embed_layer = model.get_input_embeddings()

    llm_hidden = model.config.hidden_size
    adapter = GraphPrefixAdapter(GRAPH_PREFIX_SRC_DIM, llm_hidden).to(device).to(dtype)
    init_adapter_ckpt = os.path.join(init_from, "graph_adapter.pt")
    if os.path.exists(init_adapter_ckpt):
        adapter.load_state_dict(torch.load(init_adapter_ckpt, map_location=device))
    adapter.train()

    params = [p for p in model.parameters() if p.requires_grad] + list(adapter.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    def step_prediction_loss(ex):
        prompt_text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{build_prompt(ex, mask_hint=True)}\n<|assistant|>\n"
        target_text = build_target(ex)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = torch.tensor([(prompt_ids + target_ids)[:1400]])
        labels = torch.tensor([([-100] * len(prompt_ids) + target_ids)[:1400]])
        attn = torch.ones_like(input_ids)
        from torch_geometric.data import Batch as PyGBatch
        graphs = PyGBatch.from_data_list([ex["graph"]])
        field_embs = torch.tensor(
            _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
            dtype=torch.float32,
        ).unsqueeze(0)
        return forward_batch(input_ids, attn, labels, graphs, field_embs,
                              model, stage1, adapter, embed_layer, device, dtype)

    def held_out_step_accuracy(max_items=40):
        """Exact-match accuracy on the 'New step' field, greedy decode --
        the same metric Stage 2 itself uses for checkpoint selection, so
        this is directly comparable to the Stage-2-only baseline."""
        import re, json as _json
        from graph_encoder import Stage1Classifier  # noqa
        from torch_geometric.data import Batch as PyGBatch
        rng = random.Random(args.seed)
        sample = step_val if len(step_val) <= max_items else rng.sample(step_val, max_items)
        was_training = model.training
        model.eval()
        correct = 0
        with torch.no_grad():
            for ex in sample:
                prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{build_prompt(ex, mask_hint=True)}\n<|assistant|>\n"
                ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                graphs = PyGBatch.from_data_list([ex["graph"]]).to(device)
                field_embs = torch.tensor(
                    _embed_texts([ex["context"].get(c, "") or "empty" for c in CONTEXT_COLUMNS]),
                    dtype=torch.float32,
                ).unsqueeze(0).to(device)
                edge_attr = getattr(graphs, "edge_attr", None)
                combined_emb, _, _ = stage1.encode_and_predict(
                    graphs.x, graphs.edge_index, graphs.batch, field_embs, edge_attr=edge_attr
                )
                prefix = adapter(combined_emb.to(dtype))
                tok_emb = embed_layer(ids).to(dtype)
                inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
                attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
                out = model.generate(inputs_embeds=inputs_embeds, attention_mask=attn,
                                      max_new_tokens=200, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                m = re.search(r'\{.*\}', text, re.DOTALL)
                pred_step = None
                if m:
                    try:
                        pred_step = _json.loads(m.group()).get("New step")
                    except Exception:
                        pass
                if pred_step == ex["step_label"]:
                    correct += 1
        if was_training:
            model.train()
        return correct / max(1, len(sample))

    os.makedirs(args.out_dir, exist_ok=True)
    best_dir = os.path.join(args.out_dir, "best")

    print("[train_multitask] Scoring Stage-2 baseline on both metrics before training...")
    baseline_step_acc = held_out_step_accuracy()
    baseline_struct_score, _ = evaluate_on_items(
        struct_held_out, ex_index, stage1, adapter, model, tokenizer, embed_layer, device, dtype
    )
    print(f"[train_multitask] Baseline: step_acc={baseline_step_acc:.4f}  "
          f"structure_score={baseline_struct_score:.4f}")
    best_combo = baseline_step_acc + baseline_struct_score
    best_step_acc, best_struct_score, best_step = baseline_step_acc, baseline_struct_score, 0

    rng = random.Random(args.seed)
    step, accum_loss = 0, 0.0
    optim.zero_grad()
    while step < args.steps:
        if rng.random() < args.structure_frac:
            item = rng.choice(struct_train)
            loss = structure_forward_item(item, ex_index, stage1, adapter, model, tokenizer,
                                           embed_layer, device, dtype)
        else:
            ex = rng.choice(step_train)
            loss = step_prediction_loss(ex)
        if loss is None:
            continue
        (loss / args.grad_accum).backward()
        accum_loss += loss.item()
        step += 1

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            optim.zero_grad()

        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps} | loss (last 50) ≈ {accum_loss / 50:.4f}")
            accum_loss = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            step_acc = held_out_step_accuracy()
            struct_score, _ = evaluate_on_items(
                struct_held_out, ex_index, stage1, adapter, model, tokenizer, embed_layer, device, dtype
            )
            combo = step_acc + struct_score
            flag = ""
            # Only promote if BOTH: combined score improves AND step-prediction
            # accuracy doesn't fall below the Stage-2 baseline -- prevents
            # "multitask" from quietly meaning "structure got better because
            # we let the real task get worse".
            if combo > best_combo and step_acc >= baseline_step_acc - 0.02:
                best_combo = combo
                best_step_acc, best_struct_score, best_step = step_acc, struct_score, step
                model.save_pretrained(best_dir)
                torch.save(adapter.state_dict(), os.path.join(best_dir, "graph_adapter.pt"))
                tokenizer.save_pretrained(best_dir)
                flag = "  <-- new best, saved"
            print(f"[train_multitask] step {step:5d} | step_acc={step_acc:.4f} "
                  f"(baseline {baseline_step_acc:.4f}) | structure={struct_score:.4f} "
                  f"(baseline {baseline_struct_score:.4f}) | best combo @ step {best_step}{flag}")

    print("\n" + "=" * 70)
    print(f"[train_multitask] DONE.")
    print(f"  Stage-2 baseline : step_acc={baseline_step_acc:.4f}  structure={baseline_struct_score:.4f}")
    print(f"  Best multitask   : step_acc={best_step_acc:.4f}  structure={best_struct_score:.4f}  "
          f"(step {best_step}) -> {best_dir}")
    if best_step == 0:
        print("[train_multitask] ⚠ Never found a checkpoint that both improved the combined "
              "score AND kept step-prediction accuracy within 2pp of baseline. Try a lower "
              "--structure_frac, more --steps, or check per-task structure scores in "
              "analyze_results.py to see which task is dragging the average down.")
    print("=" * 70)


if __name__ == "__main__":
    main()
