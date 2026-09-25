"""
Unified client for commercial LLM APIs (OpenAI, Anthropic, Google).

Used in two places, matching the methodology in the Pen-Strategist paper
(arxiv.org/pdf/2605.04499, Table 2 / Section 5.5):
  1. eval/commercial_explanation_baselines.py -- generate step/explanation/
     MCP predictions on the test set with a commercial model, as an extra
     baseline row alongside baseline_zeroshot/3shot/5shot.
  2. eval/multi_judge_explanation_eval.py -- use a commercial model as an
     LLM judge, cross-checked against the project's own local Qwen2.5-7B
     judge (core/llm_judge.py) and against manual human scoring.

Deliberately NOT a training-time dependency -- these are test-set-only,
API-key-gated calls, isolated in this one module so nothing else in the
pipeline needs network access.
"""
import os
import time

# Friendly name -> (provider, actual API model id). Add/remove entries here
# rather than scattering model ids through the eval scripts.
MODEL_REGISTRY = {
    "gpt-5":        ("openai",    "gpt-5"),
    "gpt-5-mini":   ("openai",    "gpt-5-mini"),
    "gpt-4o":       ("openai",    "gpt-4o"),
    "claude-sonnet":("anthropic", "claude-sonnet-4-5"),
    "gemini-flash": ("google",    "gemini-2.5-flash"),
}

_openai_client = None
_anthropic_client = None
_google_configured = False


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_google_client():
    global _google_configured
    import google.generativeai as genai
    if not _google_configured:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        _google_configured = True
    return genai


def generate_text(system_prompt: str, user_content: str, model_key: str,
                   max_tokens: int = 800, retries: int = 3) -> str:
    """Call `model_key` (a MODEL_REGISTRY key) and return its raw text
    response. Retries on transient API errors with linear backoff; raises
    on the final failure so callers can log/skip that row explicitly rather
    than silently treating a network blip as a bad generation."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key {model_key!r}; add it to MODEL_REGISTRY")
    provider, api_model = MODEL_REGISTRY[model_key]

    last_err = None
    for attempt in range(retries):
        try:
            if provider == "openai":
                client = _get_openai_client()
                resp = client.chat.completions.create(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_completion_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""

            if provider == "anthropic":
                client = _get_anthropic_client()
                resp = client.messages.create(
                    model=api_model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=max_tokens,
                )
                return "".join(b.text for b in resp.content if hasattr(b, "text"))

            if provider == "google":
                genai = _get_google_client()
                model = genai.GenerativeModel(api_model, system_instruction=system_prompt)
                resp = model.generate_content(
                    user_content,
                    generation_config={"max_output_tokens": max_tokens},
                )
                return resp.text or ""

            raise ValueError(f"Unhandled provider {provider!r}")

        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"generate_text failed for {model_key} after {retries} attempts: {last_err}")
