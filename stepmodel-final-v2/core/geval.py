"""
Shared G-Eval-STYLE rubric for explanation scoring.

CHANGED (2026-09-30, user-directed): now uses the SAME 4 criteria as
core/llm_judge.py's rubric (relevance/technical_accuracy/completeness/
clarity, 0-3 scale, same wording) instead of the Pen-Strategist paper's own
G-Eval criteria (logical_alignment/evidence_reference/decision_consistency/
tool_technique_use, 1-5). This means this module NO LONGER replicates the
paper's R_s (arxiv.org/pdf/2605.04499, Section 4.2.1) -- it's now this
project's own rubric, applied via the G-Eval-style "have an LLM score it"
mechanism. Whatever uses this module (stage3's training reward when
STAGE3_EXPLANATION_REWARD_MODE=geval_gpt4o, and --rubric geval in
eval/multi_judge_explanation_eval.py) now optimizes/reports against THIS
rubric, not the paper's.

NOTE: is_correct here is still a simple `score >= IS_CORRECT_THRESHOLD` on
the AVERAGED score -- NOT core/llm_judge.py's actual gate logic (relevance>=2
AND technical_accuracy>=2 AND completeness>=1 on the raw ints, clarity never
gates). Same 4 criteria/definitions/scale as llm_judge.py now, but not an
identical decision rule -- an average-based pass can differ from the AND-gate
in edge cases (e.g. one very high clarity score compensating for a low
completeness score here, which the AND-gate would never allow).

Single source of truth so the stage3 GRPO reward (training/stage3_grpo_rl.py)
and the commercial-LLM test-time comparison (eval/multi_judge_explanation_eval.py)
score explanations with the IDENTICAL rubric.

Deliberately generation-backend-agnostic (takes a `generate_fn` callback)
so callers can use either a local HF model (core/llm_judge.py's loaded judge)
or a commercial API (core/commercial_llm.py) with the same prompt/parsing.
"""
import hashlib
import json
import pathlib
import re

CACHE_DIR = pathlib.Path(__file__).parent / ".geval_cache"

CRITERIA = ["relevance", "technical_accuracy", "completeness", "clarity"]

PROMPT_TEMPLATE = """You are an expert penetration-testing instructor grading a student's step explanation against a reference answer, like grading short-answer exam responses.

You will be given:
1. A predicted step explanation (the student's answer)
2. A ground truth step explanation (the reference answer)
3. The predicted step (the action being explained)
4. Context from the previous step and strategy

Score the predicted explanation on FOUR separate rubric dimensions, each as an INTEGER from 0 to 3.

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

CONTEXT:
- New strategy: {new_strategy}
- Strategy explanation: {strategy_explanation}
- Predicted step: {pred_step}

PREDICTED EXPLANATION: {pred_expl}

REFERENCE EXPLANATION: {gold_expl}

Respond with ONLY this JSON object, no other text:
{{"relevance": <0-3>, "technical_accuracy": <0-3>, "completeness": <0-3>, "clarity": <0-3>}}"""

SYSTEM_PROMPT = "Respond with ONLY the requested JSON object."

# OUR threshold for producing an accuracy-style number alongside the
# continuous score -- an average-based pass/fail, not llm_judge.py's actual
# AND-gate (see module docstring). Matches core/llm_judge.py's own fallback
# threshold constant so it's at least consistent with that convention.
IS_CORRECT_THRESHOLD = 0.6


def build_prompt(pred_expl: str, gold_expl: str, pred_step: str, context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        new_strategy=context.get("New strategy", ""),
        strategy_explanation=context.get("Strategy explanation", ""),
        pred_step=pred_step, pred_expl=pred_expl, gold_expl=gold_expl,
    )


def parse_raw(raw_text: str) -> dict | None:
    """Extract the 4 raw 0-3 criteria ints, unaveraged. None on parse failure."""
    try:
        match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        return {c: max(0, min(3, int(round(float(parsed[c]))))) for c in CRITERIA}
    except Exception:
        return None


def parse_score(raw_text: str) -> float | None:
    """Extract the 4 criteria and return the averaged, normalized-to-[0,1]
    score. Returns None (not 0.0) on parse failure so callers can tell
    'judge said it was bad' apart from 'we couldn't read the judge'."""
    raw = parse_raw(raw_text)
    if raw is None:
        return None
    vals = [raw[c] for c in CRITERIA]
    result = (sum(vals) / len(vals)) / 3.0  # 0-3 -> 0-1
    return max(0.0, min(1.0, result))


def _cache_key(judge_id: str, pred_expl: str, gold_expl: str, pred_step: str, context: dict) -> str:
    h = hashlib.sha256()
    for part in (judge_id, pred_expl, gold_expl, pred_step,
                 context.get("New strategy", ""), context.get("Strategy explanation", "")):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def score(pred_expl: str, gold_expl: str, pred_step: str, context: dict,
          generate_fn, judge_id: str, use_cache: bool = True,
          on_error=None, return_raw: bool = False):
    """generate_fn(system_prompt, user_prompt) -> raw text. judge_id
    identifies the model/backend for cache-key purposes (switching judges
    must invalidate the cache, same reasoning as core/llm_judge.py). Returns
    None on failure (see parse_score); on_error(exception) is called if the
    generate_fn call itself raises, so callers can log/handle it.
    return_raw=True returns (score, raw_criteria_dict) instead of just score
    -- both are None together on failure."""
    cache_key = _cache_key(judge_id, pred_expl, gold_expl, pred_step, context)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            return (cached["score"], cached.get("raw")) if return_raw else cached["score"]
        except Exception:
            pass

    prompt = build_prompt(pred_expl, gold_expl, pred_step, context)
    try:
        raw_text = generate_fn(SYSTEM_PROMPT, prompt)
    except Exception as e:
        if on_error:
            on_error(e)
        return (None, None) if return_raw else None

    raw_dict = parse_raw(raw_text)
    result = parse_score(raw_text)
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"score": result, "raw": raw_dict}))
        except Exception:
            pass
    return (result, raw_dict) if return_raw else result
