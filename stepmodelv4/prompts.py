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
    "which tools (MCP tasks) should be used to carry it out. "
    "Your response must be valid JSON only, with no additional text or explanation outside the JSON."
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
}

Example 3:
{
  "New step": "Enumerate the domain",
  "Step explanation": "We have identified the target domain. The next logical step is to enumerate DNS records and subdomains to discover additional attack surface.",
  "MCP_tasks": {
    "Google search": "Search for subdomain enumeration techniques and known subdomains.",
    "Interactive CLI": "Use tools like dig or nslookup to query DNS records."
  }
}

Example 4:
{
  "New step": "Analyze the outcomes of the previous step and find an attack path",
  "Step explanation": "After gathering information about open ports, services, and potential vulnerabilities, we need to analyze these findings to identify the most promising attack path forward.",
  "MCP_tasks": {
    "Google search": "Research potential attack chains based on discovered services and versions.",
    "Interactive CLI": "Correlate findings from different reconnaissance steps."
  }
}"""


def build_prompt(example: dict) -> str:
    """
    Build the user-turn prompt. The full raw graph JSON is embedded as text
    (NOT vector-embedded) — this is passed directly to the LLM as its input,
    exactly as required.
    """
    # Use plain text summary instead of JSON to avoid continuation
    graph = example["graph_json"]
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    
    recent_states_text = ""
    if nodes:
        recent = nodes[-3:]
        recent_states_text = "\n".join([
            f"- {n.get('label', 'Unknown')} (status: {n.get('status', 'unknown')})"
            for n in recent
        ])
    
    graph_summary = f"""Total nodes: {len(nodes)}
Total edges: {len(edges)}
Recent states:
{recent_states_text}"""

    # Access context fields from the nested context dict
    context = example.get("context", {})
    machine = context.get("machine", "Unknown")
    candidate_step = context.get("candidate_step", "N/A")
    candidate_step_explanation = context.get("candidate_step_explanation", "N/A")

    return f"""Machine: {machine}

Current penetration test state:
{graph_summary}

Heuristic suggestion: {candidate_step}
Heuristic reasoning: {candidate_step_explanation}

TASK: Predict the next penetration testing step.

IMPORTANT: Your response must be valid JSON only. Do not include any text before or after the JSON.

Output format:
{{
  "New step": "exact step name from the canonical categories",
  "Step explanation": "detailed reasoning explaining why this step is appropriate given the current state",
  "MCP_tasks": {{"tool_name": "description of how to use this tool"}}
}}

Examples:
{_FEWSHOT}

Your response (JSON only):"""


def build_chat_prompt(example: dict) -> str:
    """Full Qwen chat-formatted prompt string."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{build_prompt(example)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
