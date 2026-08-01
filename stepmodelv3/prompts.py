"""
prompts.py — shared prompt construction for stepmodelv3 (raw-JSON input,
no graph embedding — the full graph dict is serialized to text and fed to
Qwen3-14B directly, per the task requirement).
"""
import json

SYSTEM_PROMPT = (
    "You are an expert penetration testing AI assistant. Given the current "
    "state of a penetration test represented as a graph structure (a "
    "Penetration Testing Tree of states, actions and findings), predict the "
    "single next step to take, explain the reasoning behind it, and specify "
    "which tools (MCP tasks) should be used to carry it out."
)

_FEWSHOT = """Example 1:
{
  "New step": "Enumerate further on the HTTP service to find software versions, hidden directories and file",
  "Step explanation": "The previous steps identified an open port (80) running HTTP. It is logical to enumerate this service further to uncover hidden directories, files, and version information that could reveal exploitable weaknesses.",
  "MCP_tasks": {
    "Dirbuster": "Enumerate hidden directories and files on the HTTP service.",
    "Google search": "Look for known vulnerabilities related to the identified service version."
  }
}

Example 2:
{
  "New step": "Exploit the selected exploitation",
  "Step explanation": "A vulnerable service and matching public exploit have been identified. We can now attempt to exploit it to gain a foothold on the target.",
  "MCP_tasks": {
    "Metasploit": "Use Metasploit to exploit the identified vulnerability for a reverse shell.",
    "Netcat": "Set up a listener with Netcat to receive the reverse shell connection."
  }
}"""


def build_prompt(example: dict) -> str:
    """
    Build the user-turn prompt. The full raw graph JSON is embedded as text
    (NOT vector-embedded) — this is passed directly to the LLM as its input,
    exactly as required.
    """
    graph_json_str = json.dumps(example["graph_json"], indent=2)
    return f"""Current penetration test state for machine: {example['machine']}

Graph data (JSON format, raw — states/actions/findings of the Penetration Testing Tree):
{graph_json_str}

Candidate next step drafted by a heuristic (for reference only, may be wrong or incomplete):
"{example.get('candidate_step', '')}"
Candidate reasoning: "{example.get('candidate_step_explanation', '')}"

Based on the current graph state, predict the correct next step. You must
respond ONLY with a valid JSON object containing exactly these three keys:
- "New step": the name of the next step to take
- "Step explanation": a brief explanation of why this step is appropriate
- "MCP_tasks": a JSON object with tool names as keys and their parameters/instructions as values

{_FEWSHOT}

Your response (JSON only):"""


def build_chat_prompt(example: dict) -> str:
    """Full Qwen chat-formatted prompt string."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{build_prompt(example)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
