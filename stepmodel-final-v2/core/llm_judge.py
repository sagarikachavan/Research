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

2. is_correct IS A GATE ON THE RAW RUBRIC INTEGERS, NOT A THRESHOLD ON A
   NORMALIZED SCORE.
   Previously the prompt asked the model to independently invent both a
   score AND a boolean "is_correct" flag, with no guarantee the two agreed
   (e.g. score=0.55 but is_correct=true). Then this was fixed to derive
   is_correct from `final_score >= 0.6`, where final_score was a weighted
   average of the four 0-3 rubric dimensions normalized to 0-1 — better, but
   still let dimensions compensate for each other (e.g. perfect relevance/
   completeness/clarity could outweigh technical_accuracy=0 and still clear
   0.6, marking a technically WRONG explanation "correct"). Now is_correct
   requires relevance>=2 AND technical_accuracy>=2 AND completeness>=1,
   evaluated directly on the raw integers the judge output — no averaging,
   no normalization, no compensating dimension. `correctness_score` (the
   old weighted-average float) is still computed and reported, but purely
   as a diagnostic trend indicator, not as what decides correctness.

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

# Only used by the heuristic (no-judge-model-loaded) fallback path below,
# where there's no rubric to gate on. The real judge path never uses this —
# see CRITICAL_DIMS / _is_correct_from_rubric instead.
CORRECTNESS_THRESHOLD = 0.6

CACHE_DIR = pathlib.Path(__file__).parent / ".llm_judge_cache"

# Rubric sub-dimensions, each scored 0-3 by the judge. `correctness_score`
# below is now a DIAGNOSTIC/trend number only (still a 0-1 weighted average,
# still useful for "is quality drifting up or down across runs") — it no
# longer decides is_correct. See _is_correct_from_rubric().
RUBRIC_DIMS = ["relevance", "technical_accuracy", "completeness", "clarity"]
RUBRIC_WEIGHTS = {"relevance": 1.0, "technical_accuracy": 1.5, "completeness": 1.0, "clarity": 0.5}
_WEIGHT_SUM = sum(RUBRIC_WEIGHTS[d] for d in RUBRIC_DIMS)

# ---------------------------------------------------------------------------
# is_correct is a GATE on the raw 0-3 integers, not a threshold on a
# normalized/weighted float.
#
# Why: averaging four 0-3 scores into one 0-1 number and then cutting at 0.6
# lets dimensions compensate for each other — e.g. relevance=3,
# technical_accuracy=0, completeness=3, clarity=3 weighted-averages to
# ~0.69, clearing the old 0.6 bar, even though the explanation is
# technically WRONG. A composite score can't tell you that; a gate can.
# "Accuracy" should mean "the explanation is substantively right", and
# substantively right specifically requires: same step (relevance) AND
# technically correct claims (technical_accuracy) AND at least some of the
# actual reasoning present (completeness). Clarity is prose quality, not
# correctness — a correct-but-clunky explanation still gets marked correct.
#
# Each cutoff is an integer already on the rubric's own 0-3 scale (2 = the
# rubric's own "mostly/clearly right" band), so there's no new arbitrary
# constant introduced, no normalization step, and the number is directly
# legible from the rubric prompt itself.
# ---------------------------------------------------------------------------
CRITICAL_DIMS = ["relevance", "technical_accuracy"]
CRITICAL_MIN = 2       # both must be at least "mostly/clearly right" (rubric's own 2)
COMPLETENESS_MIN = 1   # must capture at least SOME of the actual reasoning

# Per-dimension "pass" bar used only for the diagnostic per-dimension
# accuracy breakdown (batch_evaluate_explanations aggregates["dimension_pass_rates"]).
DIMENSION_PASS_MIN = 2

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
    """Deterministic weighted average of the 0-3 sub-scores, normalized to
    0-1. DIAGNOSTIC ONLY — a continuous trend indicator, not what decides
    is_correct (see _is_correct_from_rubric)."""
    weighted = sum(rubric[d] * RUBRIC_WEIGHTS[d] for d in RUBRIC_DIMS)
    return round(weighted / (_WEIGHT_SUM * 3.0), 4)


def _is_correct_from_rubric(rubric: Dict[str, int]) -> bool:
    """
    The actual accuracy decision. A gate on the raw integers, not a
    threshold on a normalized composite — see the module-level comment by
    CRITICAL_DIMS for why. "Correct" = same step (relevance) AND technically
    sound (technical_accuracy) AND covers at least some real reasoning
    (completeness). Clarity never blocks correctness; it's prose quality.
    """
    return (
        all(rubric[d] >= CRITICAL_MIN for d in CRITICAL_DIMS)
        and rubric["completeness"] >= COMPLETENESS_MIN
    )


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
    output; is_correct is always computed in code, directly from the raw
    rubric integers (a pass/fail gate), never from a normalized/thresholded
    composite score.

    Returns:
        Tuple of (scores_dict, raw_response) where scores_dict has:
          correctness_score (float, 0-1, DIAGNOSTIC trend metric only),
          is_correct (bool, the actual accuracy decision — see
          _is_correct_from_rubric),
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
                "correctness_score": final_score,  # diagnostic trend metric only
                "is_correct": _is_correct_from_rubric(rubric),  # the actual accuracy decision — a gate on the raw ints
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
    dim_passes: Dict[str, list] = {}
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
                "pred_step": ex.get("pred_step", ""),
                "gold_step": ex.get("gold_step", ""),
                "pred_explanation": ex.get("pred_explanation", ""),
                "gold_explanation": ex.get("gold_explanation", ""),
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
            if scores.get("rubric"):
                for dim in RUBRIC_DIMS:
                    dim_passes.setdefault(dim, []).append(
                        1 if scores["rubric"][dim] >= DIMENSION_PASS_MIN else 0
                    )

        except Exception as e:
            print(f"[LLM Judge] Error evaluating sample {i}: {e}")
            judge_error_count += 1
            results.append({"index": i, "machine": ex.get("machine", ""), "error": str(e)})

    import numpy as np

    # Per-dimension accuracy: the % of samples that were "mostly/clearly
    # right" (rubric >= 2) on EACH dimension separately, instead of folding
    # all four into one number. This is usually more actionable than a
    # single blended accuracy — e.g. relevance_pass_rate=95% but
    # technical_accuracy_pass_rate=60% tells you the model is confidently
    # explaining the WRONG technical reasoning for the right step, which a
    # single composite accuracy would hide.
    dimension_pass_rates = {
        dim: {
            "pass_rate_percent": float(sum(vals) / len(vals) * 100) if vals else 0.0,
            "n": len(vals),
        }
        for dim, vals in dim_passes.items()
    }

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
        "dimension_pass_rates": dimension_pass_rates,
        "individual_results": results,
        "total_evaluated": len(results),
        "total_errors": sum(1 for r in results if "error" in r),
        "judge_error_count": judge_error_count,
        "correctness_threshold": CORRECTNESS_THRESHOLD,
    }


def print_llm_judge_results(results: Dict[str, Any], n_examples: int = 6,
                             save_path: Optional[str] = None):
    """
    Print formatted LLM judge results, including a stratified sample of
    full ground-truth-vs-predicted explanation pairs (not just the score).

    n_examples: how many qualitative examples to show/save, split roughly
        half correct / half incorrect (whatever mix is available) so you
        see both what the judge accepts and what it rejects, not just
        whichever 3 happened to come first in the list.
    save_path: if given, also write the qualitative examples (plus the
        aggregate numbers) to this path as Markdown, so they can be
        reviewed outside the console / attached to a PR or report.
    """
    print("\n" + "=" * 80)
    print("LLM JUDGE EVALUATION RESULTS")
    print("=" * 80)

    aggregates = results["aggregates"]

    print(f"\nis_correct definition: relevance>=2 AND technical_accuracy>=2 AND "
          f"completeness>=1 (gate on the raw 0-3 rubric integers — see "
          f"CRITICAL_DIMS in llm_judge.py, not a threshold on a normalized score)")

    print("\nAggregate Scores:")
    print("-" * 80)

    if "correctness" in aggregates:
        stats = aggregates["correctness"]
        print(f"Correctness score (diagnostic, 0-1 weighted avg — NOT what decides "
              f"is_correct) | Mean: {stats['mean']:.3f} ± {stats['std']:.3f} | "
              f"Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Median: {stats['median']:.3f}")

    if "is_correct" in aggregates:
        stats = aggregates["is_correct"]
        print(f"\nAccuracy (gate-based): {stats['accuracy_percent']:.2f}% "
              f"({stats['correct_count']}/{stats['total_count']} correct)")

    dim_rates = results.get("dimension_pass_rates", {})
    if dim_rates:
        print("\nPer-dimension pass rate (rubric >= 2 on that dimension alone):")
        for dim in RUBRIC_DIMS:
            if dim in dim_rates:
                d = dim_rates[dim]
                print(f"  {dim:<20}: {d['pass_rate_percent']:6.2f}%  (n={d['n']})")

    judge_errors = results.get("judge_error_count", 0)
    if judge_errors:
        print(f"\n⚠ {judge_errors} sample(s) excluded from accuracy — judge output could not be parsed "
              f"as valid rubric JSON even after retry. These are NOT counted as 0.5 or incorrect; "
              f"fix the judge model/prompt if this count is large.")

    print(f"\nTotal evaluated: {results['total_evaluated']}")
    print(f"Total errors: {results['total_errors']}")

    examples_md = _format_qualitative_examples(results, n_examples)
    print(examples_md)

    print("=" * 80 + "\n")

    if save_path:
        try:
            header = (
                f"# LLM Judge Report\n\n"
                f"Accuracy (gate-based): "
                f"{aggregates.get('is_correct', {}).get('accuracy_percent', float('nan')):.2f}%  "
                f"({aggregates.get('is_correct', {}).get('correct_count', '?')}/"
                f"{aggregates.get('is_correct', {}).get('total_count', '?')})\n\n"
                f"Per-dimension pass rate:\n\n"
                + "\n".join(
                    f"- {dim}: {dim_rates[dim]['pass_rate_percent']:.2f}% (n={dim_rates[dim]['n']})"
                    for dim in RUBRIC_DIMS if dim in dim_rates
                )
                + "\n\n"
            )
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(examples_md.replace("\n\nSample Feedback", "\n\n## Sample Feedback"))
            print(f"[LLM Judge] Qualitative report saved to: {save_path}")
        except Exception as e:
            print(f"[LLM Judge] Could not save report to {save_path}: {e}")


def _format_qualitative_examples(results: Dict[str, Any], n_examples: int) -> str:
    """
    Build the human-readable "here's what the judge actually saw" section:
    full predicted vs. ground-truth explanation text side by side, the
    predicted vs. gold step, the rubric breakdown, and the verdict —
    stratified so you see both accepted and rejected examples, not just
    whichever came first.
    """
    scored = [
        r for r in results["individual_results"]
        if "scores" in r and not r["scores"].get("judge_error")
    ]
    correct = [r for r in scored if r["scores"]["is_correct"]]
    incorrect = [r for r in scored if not r["scores"]["is_correct"]]

    n_incorrect = min(len(incorrect), max(1, n_examples // 2)) if incorrect else 0
    n_correct = min(len(correct), n_examples - n_incorrect)
    # backfill from whichever bucket has more, so we still hit n_examples
    # even if one bucket is small (e.g. very few incorrect samples)
    remaining = n_examples - n_correct - n_incorrect
    if remaining > 0 and len(incorrect) > n_incorrect:
        extra = min(remaining, len(incorrect) - n_incorrect)
        n_incorrect += extra
    remaining = n_examples - n_correct - n_incorrect
    if remaining > 0 and len(correct) > n_correct:
        n_correct += min(remaining, len(correct) - n_correct)

    picked = correct[:n_correct] + incorrect[:n_incorrect]

    lines = ["\nSample Feedback  (full explanation text, stratified correct/incorrect):", "-" * 80]
    if not picked:
        lines.append("(no non-error samples to show)")
        return "\n".join(lines)

    for i, result in enumerate(picked, 1):
        s = result["scores"]
        verdict = "✓ CORRECT" if s["is_correct"] else "✗ INCORRECT"
        step_match = "" if result.get("pred_step") == result.get("gold_step") else "  ⚠ STEP MISMATCH"
        lines.append(f"\n[{i}] {verdict}   (Machine: {result.get('machine', '?')}){step_match}")
        lines.append(f"    Predicted step : {result.get('pred_step', '?')}")
        if step_match:
            lines.append(f"    Gold step      : {result.get('gold_step', '?')}")
        lines.append(f"    Rubric         : {s.get('rubric')}")
        lines.append(f"    Justification  : {s['justification']}")
        lines.append(f"    ── Predicted explanation ──")
        lines.append(f"    {result.get('pred_explanation', '') or '(empty)'}")
        lines.append(f"    ── Ground truth explanation ──")
        lines.append(f"    {result.get('gold_explanation', '') or '(empty)'}")

    return "\n".join(lines)