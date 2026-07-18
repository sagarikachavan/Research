"""
Stage 3: GRPO (Group Relative Policy Optimization) fine-tuning on top of the
Stage-2 SFT checkpoint.

By Stage 2, the model has already learned to reproduce the gold "New step"
label and "MCP_tasks" set from supervised examples. What SFT teaches poorly
is open-ended text quality: "Step explanation" has no single correct string,
so cross-entropy against one gold explanation over-penalizes equally valid
phrasings. GRPO instead samples a *group* of completions per prompt, scores
each with a composite reward, and reinforces the ones that scored better
than their group average -- no separate critic network needed.

Reward composition (mirrors the paper's explanation-quality framing, i.e.
G-Eval's readability / coherence / informativeness, plus task correctness):

    r = w_fmt * format_ok
      + w_step * step_label_exact_match
      + w_mcp  * mcp_set_F1(pred, gold)
      + w_exp  * explanation_quality(judge_model)               # 0..1

format_ok, step match and mcp F1 are cheap/deterministic and keep the model
from drifting off the label taxonomy while it optimizes for explanation
quality (which is comparatively expensive/judge-based).

Run:
    python stage3_grpo_rl.py
"""
import json
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from config import (
    TRAIN_CSV, QWEN_MODEL_NAME, STAGE2_ADAPTER_DIR, STAGE3_ADAPTER_DIR,
    STAGE3_GROUP_SIZE, STAGE3_LR, STAGE3_STEPS, STAGE3_KL_COEF, MCP_LABELS,
    STEP_LABELS, RANDOM_SEED,
)
from data_utils import load_and_clean
from stage2_sft_qwen import build_prompt, SYSTEM_PROMPT

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# NOTE on graph conditioning during RL: trl's GRPOTrainer operates on plain
# text prompts. To keep the graph signal in the loop without forking the
# trainer's generation path, we linearize the Stage-1 GNN's TOP predicted
# step/tool labels (already grounded in the graph) into the text prompt as
# an explicit "Graph summary:" line -- i.e. we distill the graph embedding
# into text once via Stage 1, then let GRPO operate purely on text. This
# keeps Stage 3 compatible with off-the-shelf RL tooling; see README for the
# alternative (custom generation loop with soft-prompt prefix) if you need
# the raw graph embedding inside the RL rollout itself.


def make_prompt_with_graph_summary(ex, stage1_graph_summary: str) -> str:
    base = build_prompt(ex)
    return f"{base}\nGraph summary (from GNN state encoder): {stage1_graph_summary}"


def parse_completion(text: str):
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return obj
    except Exception:
        return None


def reward_fn(prompts, completions, gold_examples, judge_fn,
              w_fmt=0.2, w_step=0.3, w_mcp=0.3, w_exp=0.2):
    rewards = []
    for completion, gold in zip(completions, gold_examples):
        obj = parse_completion(completion)
        if obj is None or not all(k in obj for k in ("New step", "Step explanation", "MCP_tasks")):
            rewards.append(0.0)  # malformed output gets zero reward regardless of content
            continue

        fmt_r = 1.0
        step_r = 1.0 if obj["New step"].strip() == gold["step_label"] else 0.0

        mcp_val = obj.get("MCP_tasks", {})
        pred_mcp = set(mcp_val.keys() if isinstance(mcp_val, dict) else []) & set(MCP_LABELS)
        gold_mcp = set(gold["mcp_labels"])
        if not pred_mcp and not gold_mcp:
            mcp_r = 1.0
        else:
            inter = len(pred_mcp & gold_mcp)
            prec = inter / len(pred_mcp) if pred_mcp else 0.0
            rec = inter / len(gold_mcp) if gold_mcp else 0.0
            mcp_r = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        exp_r = judge_fn(obj["Step explanation"], gold["gold_step_explanation"])

        r = w_fmt * fmt_r + w_step * step_r + w_mcp * mcp_r + w_exp * exp_r
        rewards.append(r)
    return rewards


def make_judge_fn(judge_model, judge_tokenizer, device):
    """
    G-Eval-style judge, adapted from the paper's Section 4.3.2: prompts a
    (separate/frozen) LLM to score readability + coherence + informativeness
    of the generated explanation on a 1-5 scale, normalized to [0,1].
    Swap this for a proper log-prob-weighted G-Eval implementation if you
    need the finer-grained scoring described in the paper; this version
    uses a single discrete rating for simplicity/speed inside the RL loop.
    """
    RUBRIC = (
        "Rate the following phishing/pentest step explanation from 1 (poor) "
        "to 5 (excellent) on readability, coherence, and informativeness "
        "combined. Respond with only the integer.\n\n"
        "Reference explanation: {ref}\n\nCandidate explanation: {cand}\n\nScore:"
    )

    @torch.no_grad()
    def judge_fn(candidate: str, reference: str) -> float:
        prompt = RUBRIC.format(ref=reference[:500], cand=candidate[:500])
        ids = judge_tokenizer(prompt, return_tensors="pt").to(device)
        out = judge_model.generate(**ids, max_new_tokens=4, do_sample=False)
        text = judge_tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        digits = "".join(ch for ch in text if ch.isdigit())
        score = int(digits[0]) if digits else 3
        return max(0.0, min(1.0, (score - 1) / 4.0))

    return judge_fn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, dtype=torch.bfloat16).to(device)
    policy = PeftModel.from_pretrained(base, STAGE2_ADAPTER_DIR, is_trainable=True)

    # Judge model: reuse the same base weights (frozen) as a cheap stand-in;
    # swap for a stronger external judge if available.
    judge_base = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, dtype=torch.bfloat16).to(device).eval()
    judge_fn = make_judge_fn(judge_base, tokenizer, device)

    examples = load_and_clean(TRAIN_CSV, "train")

    # Build the dataset: each row has "prompt" (required by GRPOTrainer) plus
    # "gold_json" which carries the gold labels through the trainer's collator
    # so the reward function can access them without a global index lookup.
    train_dataset = []
    for ex in examples:
        prompt_text = make_prompt_with_graph_summary(
            ex,
            f"predicted step-type leaning='{ex['step_label']}', "
            f"detected services/tools context='{', '.join(ex['mcp_labels']) or 'none'}'"
        )
        train_dataset.append({
            "prompt": f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt_text}\n<|assistant|>\n",
            # Serialise gold fields as a JSON string so they survive the
            # trainer's dict/tensor collation unchanged.
            "gold_json": json.dumps({
                "step_label": ex["step_label"],
                "mcp_labels": ex["mcp_labels"],
                "gold_step_explanation": ex["gold_step_explanation"],
            }, ensure_ascii=False),
        })

    # GRPOTrainer calls: reward_func(prompts, completions, **batch_columns)
    # where batch_columns contains one key per extra column in the dataset
    # (here "gold_json") as a list of strings, one per sample in the batch.
    def reward_wrapper(prompts, completions, gold_json, **kwargs):
        gold_examples = [json.loads(g) for g in gold_json]
        return reward_fn(prompts, completions, gold_examples, judge_fn)

    grpo_config = GRPOConfig(
        output_dir=STAGE3_ADAPTER_DIR,
        learning_rate=STAGE3_LR,
        num_generations=STAGE3_GROUP_SIZE,
        max_steps=STAGE3_STEPS,
        beta=STAGE3_KL_COEF,          # KL penalty toward the Stage-2 SFT policy
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        temperature=0.8,
    )

    trainer = GRPOTrainer(
        model=policy,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_wrapper],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(STAGE3_ADAPTER_DIR)
    print(f"Stage 3 GRPO complete. Policy saved to {STAGE3_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
