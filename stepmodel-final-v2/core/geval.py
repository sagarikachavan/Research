"""
Shared G-Eval rubric, matching the Pen-Strategist paper's R_s exactly
(arxiv.org/pdf/2605.04499, Section 4.2.1): 4 fixed criteria, each scored
1-5 by an LLM, averaged and normalized to [0,1].

Single source of truth so the stage3 GRPO reward (training/stage3_grpo_rl.py)
and the commercial-LLM test-time comparison (eval/multi_judge_explanation_eval.py)
score explanations with the IDENTICAL rubric -- previously these had drifted:
the reward used this G-Eval rubric while the test-time comparison used
core/llm_judge.py's older project-specific rubric (relevance/technical_accuracy/
completeness/clarity, 0-3 gate), so a stage3 checkpoint's training reward and
its held-out test score were measuring different things.

Deliberately generation-backend-agnostic (takes a `generate_fn` callback)
so callers can use either a local HF model (core/llm_judge.py's loaded judge)
or a commercial API (core/commercial_llm.py) with the same prompt/parsing.
"""
import hashlib
import json
import pathlib
import re

CACHE_DIR = pathlib.Path(__file__).parent / ".geval_cache"

CRITERIA = [
    "logical_alignment",    # does the reasoning logically follow from the same rationale as the reference?
    "evidence_reference",   # does it reference similar evidence / the same primary task as the reference?
    "decision_consistency", # is the final step decision consistent with the reference, given the context?
    "tool_technique_use",   # does it invoke similar tools/techniques as the reference?
]

PROMPT_TEMPLATE = """You are scoring a penetration-testing step explanation against a reference (ground truth) explanation, using the G-Eval methodology.

Score the PREDICTED explanation on FOUR criteria, each as an INTEGER from 1 (poor) to 5 (excellent), by comparing it to the REFERENCE explanation:
1. logical_alignment: Does the predicted explanation's reasoning logically align with the reference's rationale?
2. evidence_reference: Does it reference similar evidence and the same primary task as the reference?
3. decision_consistency: Is the final step decision consistent with the reference's decision, given the context?
4. tool_technique_use: Does it invoke similar tools/techniques as the reference?

CONTEXT:
- New strategy: {new_strategy}
- Strategy explanation: {strategy_explanation}
- Predicted step: {pred_step}

PREDICTED EXPLANATION: {pred_expl}

REFERENCE EXPLANATION: {gold_expl}

Respond with ONLY this JSON object, no other text:
{{"logical_alignment": <1-5>, "evidence_reference": <1-5>, "decision_consistency": <1-5>, "tool_technique_use": <1-5>}}"""

SYSTEM_PROMPT = "Respond with ONLY the requested JSON object."

# Not part of the paper (Table 2 reports the continuous score only, never a
# binarized rate) -- this is OUR threshold for producing an accuracy-style
# number alongside the continuous score, since the rest of the project's
# reporting expects both. Matches core/llm_judge.py's own fallback threshold
# so it's at least consistent with the rest of the codebase's convention.
IS_CORRECT_THRESHOLD = 0.6


def build_prompt(pred_expl: str, gold_expl: str, pred_step: str, context: dict) -> str:
    return PROMPT_TEMPLATE.format(
        new_strategy=context.get("New strategy", ""),
        strategy_explanation=context.get("Strategy explanation", ""),
        pred_step=pred_step, pred_expl=pred_expl, gold_expl=gold_expl,
    )


def parse_raw(raw_text: str) -> dict | None:
    """Extract the 4 raw 1-5 criteria ints, unaveraged. None on parse failure."""
    try:
        match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {}
        return {c: int(round(float(parsed[c]))) for c in CRITERIA}
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
    result = (sum(vals) / len(vals) - 1.0) / 4.0  # 1-5 -> 0-1
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
