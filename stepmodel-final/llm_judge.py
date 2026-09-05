"""
llm_judge.py — LLM-based evaluation of step explanations.

Uses a QWEN model to evaluate the quality of predicted step explanations by
comparing them to ground truth explanations.

CHANGES vs the original version (see rationale in DOCUMENTATION / chat):

1. RUBRIC INSTEAD OF ONE FREE FLOAT.
   The judge no longer picks a single continuous "correctness_score" out of
   thin air. It scores 4 independent sub-dimensions on a small integer
   Likert scale (0-3): relevance, technical_accuracy, completeness, clarity.
   Small discrete choices are far more reproducible for an LLM than an
   unconstrained float — there are only 4^4 = 256 possible outputs instead
   of a continuum, which collapses most of the run-to-run wobble.

2. is_correct IS COMPUTED IN CODE, NOT BY THE MODEL.
   Previously the prompt asked the model to independently invent both a
   score AND a boolean "is_correct" flag, with no guarantee the two agreed
   (e.g. score=0.55 but is_correct=true). Now correctness is always
   `final_score >= CORRECTNESS_THRESHOLD`, computed deterministically from
   the rubric sub-scores after generation. One source of truth.

3. GREEDY DECODING, EXPLICITLY PINNED.
   do_sample=False (already true before) + num_beams=1 + model forced into
   eval() with no_grad, so there is no sampling randomness in generation at
   all — the only remaining variance is hardware/kernel non-determinism,
   which is negligible for short generations.

4. PERSISTENT DISK CACHE, KEYED ON EXACT INPUTS.
   Every judge call is cached to .llm_judge_cache/<hash>.json (same pattern
   already used by llm_ptt_parser.py's .llm_cache/). Re-running evaluate.py
   on the same predictions now returns bit-identical numbers instead of a
   fresh (and possibly slightly different) LLM call every time, and it's
   also much cheaper to re-run.

5. NO SILENT 0.5 ON PARSE FAILURE.
   If the model's output can't be parsed as the rubric JSON, the code now
   retries once with a stricter "JSON ONLY" instruction, and if that still
   fails the sample is marked as a judge_error (excluded from the accuracy
   denominator by default, reported separately) instead of being silently
   scored 0.5 — a hidden constant that used to quietly bias the aggregate
   percentage.

Usage:
    from llm_judge import evaluate_explanation_with_llm
    scores, raw = evaluate_explanation_with_llm(
        predicted_explanation="...",
        ground_truth_explanation="...",
        predicted_step="...",
        context={...}
    )
"""
import hashlib
import json
import os
import pathlib
import re
import torch
from typing import Dict, Any, Tuple, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Single source of truth for the pass/fail cut — used everywhere instead of
# letting the LLM decide its own threshold.
CORRECTNESS_THRESHOLD = 0.6

CACHE_DIR = pathlib.Path(__file__).parent / ".llm_judge_cache"

# Rubric sub-dimensions, each scored 0-3 by the judge, then averaged and
# normalized to 0-1. Weight them evenly by default; adjust WEIGHTS if you
# want e.g. technical_accuracy to matter more than clarity.
RUBRIC_DIMS = ["relevance", "technical_accuracy", "completeness", "clarity"]
RUBRIC_WEIGHTS = {"relevance": 1.0, "technical_accuracy": 1.5, "completeness": 1.0, "clarity": 0.5}
_WEIGHT_SUM = sum(RUBRIC_WEIGHTS[d] for d in RUBRIC_DIMS)

JUDGE_SYSTEM_PROMPT = """You are an expert penetration-testing instructor grading a student's step explanation against a reference answer, like grading short-answer exam responses.

You will be given:
1. A predicted step explanation (the student's answer)
2. A ground truth step explanation (the reference answer)
3. The predicted step (the action being explained)
4. Context from the previous step and strategy

Score the predicted explanation on FOUR separate rubric dimensions, each as an INTEGER from 0 to 3. Do not average them yourself — just give the four integers, the code will combine them.

RELEVANCE (0-3): Does the explanation justify the SAME predicted step / action as the reference?
  0 = talks about a different step or is off-topic
  1 = loosely related
  2 = clearly about the right step, minor drift
  3 = squarely justifies the same step as the reference

TECHNICAL_ACCURACY (0-3): Are the technical claims (services, vulnerabilities, tools, reasoning) correct and consistent with the reference?
  0 = technically wrong or contradicts the reference
  1 = several inaccuracies or unsupported claims
  2 = mostly correct, one minor inaccuracy
  3 = technically sound, consistent with the reference

COMPLETENESS (0-3): Does it cover the key justification points the reference makes (why this step, given what was found)?
  0 = missing the core reasoning entirely
  1 = captures a small part of the reasoning
  2 = captures most of the key points
  3 = captures all key reasoning points the reference makes

CLARITY (0-3): Is it well-structured and unambiguous?
  0 = incoherent or self-contradictory
  1 = hard to follow
  2 = mostly clear
  3 = clear and well-structured

Respond with ONLY this JSON object and nothing else — no markdown fences, no commentary before or after:
{
    "relevance": <int 0-3>,
    "technical_accuracy": <int 0-3>,
    "completeness": <int 0-3>,
    "clarity": <int 0-3>,
    "justification": "<one sentence>"
}"""

_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Respond with ONLY the JSON object, no other text, no markdown fences."
)


# Global LLM judge model reference (separate from training model)
_llm_judge_model = None
_llm_judge_tokenizer = None
_llm_judge_device = None
_llm_judge_model_id = "unset"  # part of the cache key, so switching judge models invalidates the cache


def set_llm_judge_model(model, tokenizer, device, model_id: Optional[str] = None):
    """Set the LLM judge model reference (separate from training model)."""
    global _llm_judge_model, _llm_judge_tokenizer, _llm_judge_device, _llm_judge_model_id
    _llm_judge_model = model
    _llm_judge_tokenizer = tokenizer
    _llm_judge_device = device
    _llm_judge_model_id = model_id or getattr(getattr(model, "config", None), "_name_or_path", "unset")
    model.eval()


# ---------------------------------------------------------------------------
# Disk cache — same convention as llm_ptt_parser.py's .llm_cache/
# ---------------------------------------------------------------------------

def _cache_key(pred_expl: str, gold_expl: str, pred_step: str, context: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(_llm_judge_model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update((pred_expl or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((gold_expl or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((pred_step or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(json.dumps(context or {}, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _load_cache(key: str) -> Optional[dict]:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(key: str, payload: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass  # cache is a best-effort speed/reproducibility optimization, never fatal


# ---------------------------------------------------------------------------
# Rubric parsing + deterministic scoring
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[str]:
    # Strip any thinking block first — even with enable_thinking=False some
    # tokenizer/model combos still emit a stray <think>...</think> (e.g. if
    # it was left open across the max_new_tokens budget on an earlier bug).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)  # unterminated leading think block

    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    # last resort: grab the LAST {...} block (prefer the final answer over
    # any JSON-shaped example the model may have echoed while reasoning)
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    if matches:
        return matches[-1].group()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _parse_rubric(raw_response: str) -> Optional[Dict[str, int]]:
    json_str = _extract_json_block(raw_response)
    if json_str is None:
        return None
    try:
        parsed = json.loads(json_str)
    except Exception:
        return None
    out = {}
    for dim in RUBRIC_DIMS:
        v = parsed.get(dim)
        if v is None:
            return None
        try:
            v = int(round(float(v)))
        except Exception:
            return None
        out[dim] = max(0, min(3, v))  # clamp defensively — model should stay in-range anyway
    out["justification"] = str(parsed.get("justification", ""))[:500]
    return out


def _score_from_rubric(rubric: Dict[str, int]) -> float:
    """Deterministic weighted average of the 0-3 sub-scores, normalized to 0-1."""
    weighted = sum(rubric[d] * RUBRIC_WEIGHTS[d] for d in RUBRIC_DIMS)
    return round(weighted / (_WEIGHT_SUM * 3.0), 4)


def _run_judge_generation(user_prompt: str) -> str:
    """
    BUG FIX: this used to tokenize `user_prompt` as raw text and hand it
    straight to `.generate()` — i.e. plain next-token completion on an
    instruction-tuned/chat model, with no chat template at all. The judge
    model here (Qwen3) is a *thinking* model: without the chat template
    telling it that thinking is off, or without the special tokens that
    mark the assistant turn, it doesn't reliably follow "respond with ONLY
    this JSON" — it either free-associates past the "ground truth"/"context"
    text in the raw prompt, or (worse) emits a `<think>...</think>` block
    that alone can eat the whole 200-token budget, leaving zero JSON tokens
    generated. That is the direct cause of the near-100% judge parse-failure
    rate seen in eval logs (264/268 and 262/268 unparseable).
    Fix: build a proper chat-formatted prompt via apply_chat_template, with
    thinking explicitly disabled (enable_thinking=False) so the model goes
    straight to the JSON answer, and give it a larger token budget as a
    safety margin in case a particular tokenizer/model combo still emits a
    short thinking preamble despite the flag.
    """
    messages = [
        {"role": "system", "content": "Respond with ONLY the requested JSON object. Do not think out loud, do not use <think> tags, do not add commentary."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        chat_text = _llm_judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        # tokenizer's template doesn't accept enable_thinking (non-Qwen3 model)
        chat_text = _llm_judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = _llm_judge_tokenizer(
        chat_text, return_tensors="pt", truncation=True, max_length=1536, add_special_tokens=False,
    ).to(_llm_judge_device)
    with torch.no_grad():
        outputs = _llm_judge_model.generate(
            **inputs,
            max_new_tokens=500,  # headroom for an occasional thinking preamble + the JSON itself
            do_sample=False,   # greedy — no sampling randomness
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=_llm_judge_tokenizer.pad_token_id or _llm_judge_tokenizer.eos_token_id,
        )
    return _llm_judge_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def _build_user_prompt(predicted_explanation, ground_truth_explanation, predicted_step, context, retry=False):
    prompt = f"""{JUDGE_SYSTEM_PROMPT}

PREDICTED STEP: {predicted_step}

PREDICTED EXPLANATION: {predicted_explanation}

GROUND TRUTH EXPLANATION: {ground_truth_explanation}

CONTEXT:
- New strategy: {context.get('New strategy', 'N/A')}
- Strategy explanation: {context.get('Strategy explanation', 'N/A')}
"""
    if retry:
        prompt += _RETRY_SUFFIX
    return prompt


def evaluate_explanation_with_llm(
    predicted_explanation: str,
    ground_truth_explanation: str,
    predicted_step: str,
    context: Dict[str, Any],
    model: str = "qwen",
    api_key: str = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """
    Evaluate a predicted explanation using the QWEN LLM judge on a fixed,
    4-dimension integer rubric. Deterministic: same inputs -> same cached
    output; is_correct is always computed in code from the final score.

    Returns:
        Tuple of (scores_dict, raw_response) where scores_dict has:
          correctness_score (float, 0-1), is_correct (bool),
          rubric (dict of the 4 sub-scores), justification (str),
          judge_error (bool, True only if parsing failed twice)
    """
    if _llm_judge_model is None or _llm_judge_tokenizer is None:
        return _heuristic_fallback(predicted_explanation, reason="no judge model loaded")

    cache_key = _cache_key(predicted_explanation, ground_truth_explanation, predicted_step, context)
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached["scores"], cached["raw_response"]

    try:
        prompt = _build_user_prompt(predicted_explanation, ground_truth_explanation, predicted_step, context)
        raw_response = _run_judge_generation(prompt)
        rubric = _parse_rubric(raw_response)

        if rubric is None:
            # one deterministic retry with a stricter instruction, not a silent default
            retry_prompt = _build_user_prompt(
                predicted_explanation, ground_truth_explanation, predicted_step, context, retry=True
            )
            raw_response_retry = _run_judge_generation(retry_prompt)
            rubric = _parse_rubric(raw_response_retry)
            raw_response = raw_response + "\n---RETRY---\n" + raw_response_retry

        if rubric is None:
            scores, raw = _judge_error(raw_response)
        else:
            final_score = _score_from_rubric(rubric)
            scores = {
                "correctness_score": final_score,
                "is_correct": final_score >= CORRECTNESS_THRESHOLD,  # computed in code, not by the model
                "rubric": {d: rubric[d] for d in RUBRIC_DIMS},
                "justification": rubric["justification"],
                "judge_error": False,
            }
            raw = raw_response

        if use_cache:
            _save_cache(cache_key, {"scores": scores, "raw_response": raw})
        return scores, raw

    except Exception as e:
        print(f"[LLM Judge] QWEN evaluation failed: {e}")
        return _judge_error(str(e))


def _judge_error(raw_response: str) -> Tuple[Dict[str, Any], str]:
    """
    Explicit failure marker — NOT scored as 0.5. Kept out of the accuracy
    denominator by batch_evaluate_explanations unless include_errors=True,
    so a parsing failure can never silently nudge the reported percentage.
    """
    return {
        "correctness_score": None,
        "is_correct": None,
        "rubric": None,
        "justification": "Judge output could not be parsed as valid rubric JSON after retry.",
        "judge_error": True,
    }, raw_response


def _heuristic_fallback(predicted_explanation: str, reason: str) -> Tuple[Dict[str, Any], str]:
    """
    Only used when no judge model is loaded at all. Clearly labeled as
    non-authoritative so it's never confused with a real judge score.
    """
    print(f"[LLM Judge] Falling back to heuristic scoring ({reason}) — NOT a real quality signal.")
    expl_len = len(predicted_explanation or "")
    if expl_len < 20:
        score = 0.3
    elif expl_len < 50:
        score = 0.5
    elif expl_len < 100:
        score = 0.7
    else:
        score = 0.8
    return {
        "correctness_score": score,
        "is_correct": score >= CORRECTNESS_THRESHOLD,
        "rubric": None,
        "justification": f"Heuristic fallback ({reason}) — length-based, not a real judgment.",
        "judge_error": False,
        "heuristic": True,
    }, "heuristic fallback, no judge model loaded"


def batch_evaluate_explanations(
    examples: list,
    model: str = "qwen",
    api_key: str = None,
    max_samples: int = None,
    verbose: bool = True,
    include_errors_in_accuracy: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate multiple explanations using the QWEN LLM judge.

    include_errors_in_accuracy: if False (default), samples where the judge
    output couldn't be parsed (judge_error=True) are excluded from the
    correctness/accuracy aggregates and reported separately as
    `judge_error_count`, instead of being silently folded in as 0.5/incorrect.
    """
    if max_samples:
        examples = examples[:max_samples]

    results = []
    all_scores = {"correctness": [], "is_correct": []}
    judge_error_count = 0

    total = len(examples)
    for i, ex in enumerate(examples, 1):
        if verbose:
            print(f"[LLM Judge] Evaluating {i}/{total}...")

        try:
            scores, raw = evaluate_explanation_with_llm(
                predicted_explanation=ex.get("pred_explanation", ""),
                ground_truth_explanation=ex.get("gold_explanation", ""),
                predicted_step=ex.get("pred_step", ex.get("gold_step", "")),
                context=ex.get("context", {}),
                model=model,
                api_key=api_key,
            )

            results.append({
                "index": i,
                "machine": ex.get("machine", ""),
                "scores": scores,
                "raw_response": raw,
            })

            if scores.get("judge_error"):
                judge_error_count += 1
                if include_errors_in_accuracy:
                    all_scores["correctness"].append(0.0)
                    all_scores["is_correct"].append(False)
                continue

            all_scores["correctness"].append(scores["correctness_score"])
            all_scores["is_correct"].append(scores["is_correct"])

        except Exception as e:
            print(f"[LLM Judge] Error evaluating sample {i}: {e}")
            judge_error_count += 1
            results.append({"index": i, "machine": ex.get("machine", ""), "error": str(e)})

    import numpy as np

    aggregates = {}
    for key, values in all_scores.items():
        if values:
            if key == "is_correct":
                accuracy = sum(values) / len(values) * 100
                aggregates[key] = {
                    "accuracy_percent": float(accuracy),
                    "correct_count": int(sum(values)),
                    "total_count": len(values),
                }
            else:
                aggregates[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                }

    return {
        "aggregates": aggregates,
        "individual_results": results,
        "total_evaluated": len(results),
        "total_errors": sum(1 for r in results if "error" in r),
        "judge_error_count": judge_error_count,
        "correctness_threshold": CORRECTNESS_THRESHOLD,
    }


def print_llm_judge_results(results: Dict[str, Any]):
    """Print formatted LLM judge results."""
    print("\n" + "=" * 80)
    print("LLM JUDGE EVALUATION RESULTS")
    print("=" * 80)

    aggregates = results["aggregates"]

    print(f"\nCorrectness threshold: score >= {results.get('correctness_threshold', CORRECTNESS_THRESHOLD)} "
          f"(computed in code from the 4-dim rubric, not self-reported by the judge)")

    print("\nAggregate Scores:")
    print("-" * 80)

    if "correctness" in aggregates:
        stats = aggregates["correctness"]
        print(f"Correctness Score | Mean: {stats['mean']:.3f} ± {stats['std']:.3f} | "
              f"Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Median: {stats['median']:.3f}")

    if "is_correct" in aggregates:
        stats = aggregates["is_correct"]
        print(f"\nAccuracy: {stats['accuracy_percent']:.2f}% ({stats['correct_count']}/{stats['total_count']} correct)")

    judge_errors = results.get("judge_error_count", 0)
    if judge_errors:
        print(f"\n⚠ {judge_errors} sample(s) excluded from accuracy — judge output could not be parsed "
              f"as valid rubric JSON even after retry. These are NOT counted as 0.5 or incorrect; "
              f"fix the judge model/prompt if this count is large.")

    print(f"\nTotal evaluated: {results['total_evaluated']}")
    print(f"Total errors: {results['total_errors']}")

    print("\nSample Feedback:")
    print("-" * 80)
    for i, result in enumerate(results["individual_results"][:3]):
        if "scores" in result and not result["scores"].get("judge_error"):
            s = result["scores"]
            print(f"\nSample {i+1} (Machine: {result['machine']}):")
            print(f"  Correctness: {s['correctness_score']:.3f}  (rubric: {s.get('rubric')})")
            print(f"  Is Correct: {s['is_correct']}")
            print(f"  Justification: {s['justification']}")

    print("=" * 80 + "\n")