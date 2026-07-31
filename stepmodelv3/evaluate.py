"""
evaluate.py — Evaluation for stepmodelv3 GRPO RL stage.

Metrics reported:
  STEP CLASSIFICATION
    Accuracy, Macro-F1, Weighted-F1, per-class precision/recall/F1,
    confusion matrix

  MCP TOOL CLASSIFICATION  (multi-label)
    Subset (exact-match) accuracy, Micro-F1, Macro-F1, Samples-F1,
    per-label precision/recall/F1

  STEP EXPLANATION QUALITY
    BLEU-1/2/4          — n-gram overlap with gold explanation
    ROUGE-L             — longest common subsequence recall
    BERTScore F1        — semantic similarity via sentence transformer
    Step Alignment      — do explanation keywords match the predicted step?
    Context Grounding   — does the explanation reference observed findings?
    Reasoning Density   — causal/justification language fraction
    Avg / P25 / P75 token length — completeness sanity checks
    Empty rate          — fraction of blank/missing explanations

Usage:
    python evaluate.py                         # evaluate trained GRPO model
    python evaluate.py --adapter-dir checkpoints/stage1_grpo_rl
    python evaluate.py --save-explanations out.csv   # dump predictions to CSV
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
INPUT_TEST_JSON = "input/test.json"
QWEN_MODEL_NAME = "Qwen/Qwen3-14B-Instruct"
STAGE1_ADAPTER_DIR = "checkpoints/stage1_grpo_rl"
MCP_LABELS = [
    "nmap", "ssh", "ftp", "smbclient", "hydra", "john", "hashcat", 
    "sqlmap", "metasploit", "netcat", "burpsuite", "gobuster", "nikto",
    "responder", "autopsy", "git-dumper", "smtp-user-enum"
]

# Step labels (same as stepmodelv2)
STEP_LABELS = [
    "google search",
    "enumerate further",
    "explore files",
    "enumerate website",
    "enumerate domain",
    "exploit",
    "analyze outcomes",
    "ask human",
    "source code",
    "end task",
]

# ---------------------------------------------------------------------------
# Graph to text conversion (same as training script)
# ---------------------------------------------------------------------------

def graph_to_text(graph_data: dict) -> str:
    """Convert graph JSON structure to readable text format."""
    if not graph_data or "nodes" not in graph_data:
        return "No graph data available"
    
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    
    # Group nodes by type
    agent_nodes = [n for n in nodes if n["type"] == "Agent"]
    search_nodes = [n for n in nodes if n["type"] == "Search"]
    track_nodes = [n for n in nodes if n["type"] == "Track"]
    
    text_parts = []
    text_parts.append(f"Graph for machine: {graph_data.get('machine', 'unknown')}")
    text_parts.append(f"Total nodes: {len(nodes)}, Total edges: {len(edges)}")
    text_parts.append("\n=== AGENT NODES (States) ===")
    for node in agent_nodes[:10]:  # Limit to first 10
        text_parts.append(f"- {node['label']}: {node.get('title', '')[:100]}")
    
    text_parts.append("\n=== SEARCH NODES (Actions) ===")
    for node in search_nodes[:10]:
        text_parts.append(f"- {node['label']}: {node.get('title', '')[:100]}")
    
    text_parts.append("\n=== TRACK NODES (Findings) ===")
    for node in track_nodes[:10]:
        text_parts.append(f"- {node['label']}: {node.get('title', '')[:100]}")
    
    text_parts.append("\n=== RECENT EDGES ===")
    for edge in edges[-5:]:  # Last 5 edges
        text_parts.append(f"- {edge['from']} -> {edge['to']} ({edge['type']})")
    
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Load data from JSON
# ---------------------------------------------------------------------------

def load_json_data(json_path: str):
    """Load test data from JSON file - pass raw JSON directly."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # Pass raw graph JSON directly without conversion
        examples.append({
            "machine": item.get("Machine", ""),
            "graph_json": item.get("Graph", {}),
            "step_label": item.get("step_label", ""),
            "mcp_labels": item.get("mcp_labels", []),
            "gold_step_explanation": item.get("gold_step_explanation", ""),
        })
    
    print(f"[eval] Loaded {len(examples)} examples from {json_path}")
    return examples


# ---------------------------------------------------------------------------
# Step label normalization
# ---------------------------------------------------------------------------

class StepLabelNormalizer:
    """Normalize step labels to match STEP_LABELS."""
    def normalize(self, label: str) -> str:
        if not label:
            return None
        label_lower = label.lower().strip()
        for canonical in STEP_LABELS:
            if canonical in label_lower or label_lower in canonical:
                return canonical
        return label_lower  # return original if no match


# ---------------------------------------------------------------------------
# MCP utilities
# ---------------------------------------------------------------------------

def mcp_multihot(labels: list) -> np.ndarray:
    """Convert list of MCP labels to multi-hot vector."""
    vec = np.zeros(len(MCP_LABELS), dtype=int)
    for label in labels:
        if label in MCP_LABELS:
            vec[MCP_LABELS.index(label)] = 1
    return vec


def extract_mcp_labels(text: str) -> list:
    """Extract MCP tool labels from text."""
    found = []
    text_lower = text.lower()
    for label in MCP_LABELS:
        if label in text_lower:
            found.append(label)
    return found


# ---------------------------------------------------------------------------
# Explanation quality metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def bleu_n(hypothesis: list[str], reference: list[str], n: int) -> float:
    """Sentence-level BLEU-n (no brevity penalty, just n-gram precision)."""
    if len(hypothesis) < n or len(reference) < n:
        return 0.0
    ref_ngrams: dict[tuple, int] = defaultdict(int)
    for i in range(len(reference) - n + 1):
        ref_ngrams[tuple(reference[i : i + n])] += 1
    clipped = 0
    for i in range(len(hypothesis) - n + 1):
        ng = tuple(hypothesis[i : i + n])
        if ref_ngrams.get(ng, 0) > 0:
            clipped += 1
            ref_ngrams[ng] -= 1
    return clipped / max(len(hypothesis) - n + 1, 1)


def rouge_l(hypothesis: list[str], reference: list[str]) -> float:
    """ROUGE-L F1 via LCS dynamic programming."""
    m, n = len(reference), len(hypothesis)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    recall    = lcs / m
    precision = lcs / max(n, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bertscore_f1_batch(
    hypotheses: list[str],
    references: list[str],
    encoder: SentenceTransformer,
    batch_size: int = 64,
) -> list[float]:
    """Sentence-level BERTScore using sentence transformer."""
    scores = []
    for i in range(0, len(hypotheses), batch_size):
        hyp_batch = [h if h.strip() else "empty" for h in hypotheses[i : i + batch_size]]
        ref_batch = [r if r.strip() else "empty" for r in references[i : i + batch_size]]
        hyp_emb = encoder.encode(hyp_batch)
        ref_emb = encoder.encode(ref_batch)
        scores.extend((hyp_emb * ref_emb).sum(axis=1).tolist())
    return scores


# Step keywords for alignment
_STEP_KEYWORDS: list[list[str]] = [
    ["google", "search", "googled"],
    ["enumerate", "enumerat", "version", "hidden", "directory", "service"],
    ["explore", "suspicious", "file", "command", "summary", "finding"],
    ["website", "web", "dirb", "gobuster", "directory", "link"],
    ["domain", "dns", "subdomain", "ldap"],
    ["exploit", "exploitation", "payload", "metasploit", "shell"],
    ["analyz", "outcome", "attack", "path", "vector", "assess"],
    ["human", "assist", "help", "stuck", "unclear"],
    ["source", "code", "review", "vuln", "injection", "xss", "sqli"],
    ["report", "end", "task", "complet", "finish"],
]

_REASONING_TOKENS: set[str] = {
    "because", "since", "therefore", "thus", "hence", "as", "so",
    "indicates", "suggests", "shows", "reveals", "found", "identified",
    "discovered", "confirm", "allows", "enables", "in order", "to", "which",
    "given", "based", "due", "result", "led", "lead", "indicate",
}


def step_alignment_score(explanation: str, step_idx: int) -> float:
    """Fraction of step-specific keywords present in the explanation."""
    if step_idx < 0 or step_idx >= len(_STEP_KEYWORDS):
        return 0.0
    tokens = set(_tokenize(explanation))
    keywords = _STEP_KEYWORDS[step_idx]
    if not keywords:
        return 0.0
    hits = sum(
        1 for kw in keywords
        if any(tok.startswith(kw) for tok in tokens)
    )
    return hits / len(keywords)


def context_grounding_score(explanation: str, prev_step_result: str) -> float:
    """Measures how much explanation references previous step result."""
    _STOP = {
        "that", "this", "with", "from", "have", "been", "were", "they",
        "their", "will", "would", "could", "should", "about", "into",
        "which", "when", "also", "more", "some", "such", "than", "then",
        "there", "these", "those", "your", "what", "where", "here",
    }
    if not prev_step_result or not prev_step_result.strip():
        return 0.0
    result_toks  = {t for t in _tokenize(prev_step_result) if len(t) >= 4 and t not in _STOP}
    expl_toks    = set(_tokenize(explanation))
    if not result_toks:
        return 0.0
    overlap = result_toks & expl_toks
    return len(overlap) / len(result_toks)


def reasoning_density_score(explanation: str) -> float:
    """Fraction of tokens that are reasoning/causal connectors."""
    tokens = _tokenize(explanation)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in _REASONING_TOKENS)
    return hits / len(tokens)


def compute_explanation_metrics(
    pred_explanations: list[str],
    gold_explanations: list[str],
    step_preds: list[int] | None = None,
    prev_step_results: list[str] | None = None,
    encoder: SentenceTransformer = None,
) -> dict:
    """Compute all explanation quality metrics."""
    bleu1, bleu2, bleu4, rougeL = [], [], [], []
    valid_preds, valid_refs = [], []
    lengths = []
    alignment, grounding, reasoning = [], [], []
    empty_count = 0

    for idx, (pred, gold) in enumerate(zip(pred_explanations, gold_explanations)):
        pred_str = pred.strip()

        # Length & empty rate
        tok_len = len(pred_str.split())
        lengths.append(tok_len)
        if tok_len == 0:
            empty_count += 1

        # Intrinsic metrics
        s_idx = step_preds[idx] if step_preds is not None else -1
        alignment.append(step_alignment_score(pred_str, s_idx))
        prev_res = prev_step_results[idx] if prev_step_results is not None else ""
        grounding.append(context_grounding_score(pred_str, prev_res))
        reasoning.append(reasoning_density_score(pred_str))

        # Reference-based metrics
        if not gold.strip():
            continue
        hyp_tok = _tokenize(pred_str)
        ref_tok = _tokenize(gold)
        bleu1.append(bleu_n(hyp_tok, ref_tok, 1))
        bleu2.append(bleu_n(hyp_tok, ref_tok, 2))
        bleu4.append(bleu_n(hyp_tok, ref_tok, 4))
        rougeL.append(rouge_l(hyp_tok, ref_tok))
        valid_preds.append(pred_str)
        valid_refs.append(gold)

    bert_scores = bertscore_f1_batch(valid_preds, valid_refs, encoder) if (valid_preds and encoder) else []

    def avg(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    lengths_arr = np.array(lengths) if lengths else np.array([0])

    return {
        "bleu1":            avg(bleu1),
        "bleu2":            avg(bleu2),
        "bleu4":            avg(bleu4),
        "rouge_l":          avg(rougeL),
        "bertscore_f1":     avg(bert_scores),
        "step_alignment":   avg(alignment),
        "ctx_grounding":    avg(grounding),
        "reasoning_density": avg(reasoning),
        "avg_length":       avg(lengths),
        "p25_length":       float(np.percentile(lengths_arr, 25)),
        "p75_length":       float(np.percentile(lengths_arr, 75)),
        "empty_rate":       empty_count / max(len(pred_explanations), 1),
        "n_scored":         len(valid_preds),
        "n_total":          len(pred_explanations),
    }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def eval_grpo(adapter_dir: str, max_new_tokens: int = 200,
              save_explanations: str | None = None) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[eval] Test input: {INPUT_TEST_JSON}")
    print(f"[eval] Device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=dtype, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir).eval()

    # Load sentence transformer for BERTScore
    print("[eval] Loading sentence transformer for BERTScore...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load test data
    examples = load_json_data(INPUT_TEST_JSON)
    normalizer = StepLabelNormalizer()

    # System prompt (same as training)
    SYSTEM_PROMPT = """You are an expert penetration testing AI assistant. Given the current state of a penetration test represented as a graph structure, predict the next step to take, explain why, and specify which tools to use."""

    step_preds, mcp_preds, step_gold, mcp_gold = [], [], [], []
    pred_explanations, gold_explanations = [], []
    prev_step_results = []
    parse_failures = 0
    csv_rows = []

    for ex in tqdm(examples, desc="Evaluating", unit="sample"):
        # Build prompt with raw JSON
        graph_json_str = json.dumps(ex['graph_json'], indent=2)
        prompt_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nCurrent penetration test state for machine: {ex['machine']}\n\nGraph data (JSON format):\n{graph_json_str}\n\nBased on the current graph state, predict the next step. You must respond ONLY with a valid JSON object containing exactly these three keys:\n- \"New step\": the name of the next step to take\n- \"Step explanation\": a brief explanation of why this step is appropriate\n- \"MCP_tasks\": a JSON object with tool names as keys and their parameters as values\n\nExample 1:\n{{\n  \"New step\": \"enumerate further\",\n  \"Step explanation\": \"We need to enumerate further services on the target machine to identify potential vulnerabilities.\",\n  \"MCP_tasks\": {{\n    \"nmap\": \"Execute port scan with service version detection\",\n    \"nikto\": \"Scan for web vulnerabilities\"\n  }}\n}}\n\nYour response (JSON only):<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        with torch.no_grad():
            ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=900,
            ).input_ids.to(device)

            embed_layer = model.get_input_embeddings()
            token_embeds = embed_layer(ids).to(dtype)
            attn = torch.ones(token_embeds.shape[:2], dtype=torch.long, device=device)

            out = model.generate(
                inputs_embeds=token_embeds,
                attention_mask=attn,
                max_new_tokens=350,  # Increased for complete JSON
                do_sample=True,  # Use sampling like training
                temperature=0.3,  # Low temperature for more deterministic output
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Parse JSON response
        obj = {}
        try:
            start = gen_text.index("{")
            end = gen_text.rindex("}") + 1
            obj = json.loads(gen_text[start:end])
        except Exception:
            parse_failures += 1
            m = re.search(r'"?New step"?\s*:\s*"([^"]+)"', gen_text)
            if m:
                obj["New step"] = m.group(1)
            m2 = re.search(r'"?Step explanation"?\s*:\s*"([^"]*)"', gen_text, re.DOTALL)
            if m2:
                obj["Step explanation"] = m2.group(1)

        # Step classification
        pred_step_raw = obj.get("New step", "")
        pred_step_norm = normalizer.normalize(pred_step_raw) if pred_step_raw else None
        s_idx = (
            STEP_LABELS.index(pred_step_norm)
            if pred_step_norm in STEP_LABELS
            else -1
        )
        step_preds.append(s_idx)
        step_gold.append(STEP_LABELS.index(ex["step_label"]) if ex["step_label"] in STEP_LABELS else -1)

        # MCP classification
        pred_mcp_keys = (
            list(obj.get("MCP_tasks", {}).keys())
            if isinstance(obj.get("MCP_tasks"), dict)
            else []
        )
        pred_mcp_labels = extract_mcp_labels(str(pred_mcp_keys))
        mcp_preds.append(mcp_multihot(pred_mcp_labels))
        mcp_gold.append(mcp_multihot(ex["mcp_labels"]))

        # Explanation
        pred_expl = str(obj.get("Step explanation", "")).strip()
        gold_expl = ex.get("gold_step_explanation", "")
        pred_explanations.append(pred_expl)
        gold_explanations.append(gold_expl)
        prev_step_results.append("")  # No previous step result in stepmodelv3 format

        # CSV row
        if save_explanations:
            csv_rows.append({
                "machine": ex["machine"],
                "gold_step": ex["step_label"],
                "pred_step": pred_step_norm or "UNPARSEABLE",
                "step_correct": int(s_idx == step_gold[-1]),
                "gold_mcp": "|".join(ex["mcp_labels"]),
                "pred_mcp": "|".join(pred_mcp_labels),
                "gold_explanation": gold_expl,
                "pred_explanation": pred_expl,
            })

    if parse_failures:
        print(f"\n[eval] Note: {parse_failures}/{len(examples)} responses had no parseable JSON")

    step_preds_arr = np.array(step_preds)
    step_gold_arr = np.array(step_gold)
    mcp_preds_arr = np.stack(mcp_preds)
    mcp_gold_arr = np.stack(mcp_gold)

    # Classification report
    report_classification(step_preds_arr, step_gold_arr, mcp_preds_arr, mcp_gold_arr)

    # Explanation quality report
    print("\n\n" + "=" * 60)
    print("STEP EXPLANATION QUALITY")
    print("=" * 60)
    expl_metrics = compute_explanation_metrics(
        pred_explanations,
        gold_explanations,
        step_preds=step_preds,
        prev_step_results=prev_step_results,
        encoder=encoder,
    )
    report_explanation(expl_metrics)

    # CSV dump
    if save_explanations and csv_rows:
        import csv
        fieldnames = list(csv_rows[0].keys())
        with open(save_explanations, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[eval] Explanation predictions saved to: {save_explanations}")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def report_classification(
    step_preds: np.ndarray,
    step_gold: np.ndarray,
    mcp_preds: np.ndarray,
    mcp_gold: np.ndarray,
) -> None:
    print("\n" + "=" * 60)
    print("STEP CLASSIFICATION")
    print("=" * 60)
    print(f"  Accuracy      : {accuracy_score(step_gold, step_preds):.4f}")
    print(f"  Macro F1      : {f1_score(step_gold, step_preds, average='macro',    zero_division=0):.4f}")
    print(f"  Weighted F1   : {f1_score(step_gold, step_preds, average='weighted', zero_division=0):.4f}")

    labels_present = sorted(
        set(step_gold.tolist()) | set(int(p) for p in step_preds if p >= 0)
    )
    print("\n  Per-class report:")
    print(
        classification_report(
            step_gold, step_preds,
            labels=labels_present,
            target_names=[
                STEP_LABELS[i] if 0 <= i < len(STEP_LABELS) else "UNPARSEABLE"
                for i in labels_present
            ],
            zero_division=0,
        )
    )
    print("  Confusion matrix (rows=gold, cols=pred):")
    # Only compute confusion matrix if we have valid labels
    valid_labels = [i for i in step_gold if i >= 0] + [i for i in step_preds if i >= 0]
    if valid_labels:
        cm = confusion_matrix(step_gold, step_preds, labels=list(range(len(STEP_LABELS))))
        print(cm)
    else:
        print("  Skipped: No valid predictions (all unparseable)")

    print("\n" + "=" * 60)
    print("MCP TOOL CLASSIFICATION  (multi-label)")
    print("=" * 60)
    print(f"  Subset (exact-match) accuracy : {accuracy_score(mcp_gold, mcp_preds):.4f}")
    print(f"  Micro F1                      : {f1_score(mcp_gold, mcp_preds, average='micro',   zero_division=0):.4f}")
    print(f"  Macro F1                      : {f1_score(mcp_gold, mcp_preds, average='macro',   zero_division=0):.4f}")
    print(f"  Samples F1                    : {f1_score(mcp_gold, mcp_preds, average='samples', zero_division=0):.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(
        mcp_gold, mcp_preds, average=None, zero_division=0
    )
    print("\n  Per-label metrics:")
    print(f"  {'Label':<22}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Sup':>5}")
    print("  " + "-" * 52)
    for i, label in enumerate(MCP_LABELS):
        flag = "  ← low recall" if rec[i] < 0.3 and support[i] > 0 else ""
        print(
            f"  {label:<22}  {prec[i]:>6.3f}  {rec[i]:>6.3f}  "
            f"{f1[i]:>6.3f}  {int(support[i]):>5}{flag}"
        )


def report_explanation(metrics: dict) -> None:
    n_scored = metrics["n_scored"]
    n_total = metrics["n_total"]

    if n_scored < n_total:
        print(
            f"  (Reference-based metrics scored on {n_scored}/{n_total} pairs — "
            f"{n_total - n_scored} skipped because gold explanation was empty)"
        )

    print(f"\n  {'Metric':<26}  {'Score':>8}  Interpretation")
    print("  " + "-" * 62)

    def _bar(score: float, thresholds: tuple, icons: tuple = ("✗", "~", "✓")) -> str:
        if score >= thresholds[1]:
            return icons[2]
        if score >= thresholds[0]:
            return icons[1]
        return icons[0]

    b1 = metrics["bleu1"]
    b2 = metrics["bleu2"]
    b4 = metrics["bleu4"]
    rl = metrics["rouge_l"]
    bs = metrics["bertscore_f1"]
    sa = metrics["step_alignment"]
    cg = metrics["ctx_grounding"]
    rd = metrics["reasoning_density"]

    print(f"  {'BLEU-1':<26}  {b1:>8.4f}  {_bar(b1,(0.20,0.35))}  n-gram overlap with gold")
    print(f"  {'BLEU-2':<26}  {b2:>8.4f}  {_bar(b2,(0.10,0.20))}")
    print(f"  {'BLEU-4':<26}  {b4:>8.4f}  {_bar(b4,(0.03,0.10))}")
    print(f"  {'ROUGE-L':<26}  {rl:>8.4f}  {_bar(rl,(0.20,0.35))}  longest common subsequence")
    print(f"  {'BERTScore-F1':<26}  {bs:>8.4f}  {_bar(bs,(0.60,0.75))}  semantic similarity to gold")

    print()
    print(f"  {'Step Alignment':<26}  {sa:>8.4f}  {_bar(sa,(0.20,0.35))}  "
          f"explanation mentions step keywords")
    print(f"  {'Context Grounding':<26}  {cg:>8.4f}  {_bar(cg,(0.05,0.15))}  "
          f"references previous step findings")
    print(f"  {'Reasoning Density':<26}  {rd:>8.4f}  {_bar(rd,(0.03,0.07))}  "
          f"causal/justification language")

    avg_l = metrics["avg_length"]
    p25_l = metrics["p25_length"]
    p75_l = metrics["p75_length"]
    empty = metrics["empty_rate"]
    print()
    print(f"  {'Avg length (tokens)':<26}  {avg_l:>8.1f}")
    print(f"  {'P25 length':<26}  {p25_l:>8.1f}")
    print(f"  {'P75 length':<26}  {p75_l:>8.1f}")
    print(f"  {'Empty rate':<26}  {empty:>8.1%}  "
          f"{'⚠ high — JSON parsing likely failing' if empty > 0.1 else 'OK'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate stepmodelv3 GRPO RL model"
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=STAGE1_ADAPTER_DIR,
        help="Path to trained GRPO model checkpoint directory"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate for explanations"
    )
    parser.add_argument(
        "--save-explanations",
        type=str,
        default=None,
        help="Path to save prediction CSV with explanations"
    )

    args = parser.parse_args()

    if not os.path.exists(args.adapter_dir):
        print(f"[eval] ERROR: Adapter directory not found: {args.adapter_dir}")
        print("[eval] Please train the model first by running: python stage1_grpo_rl.py")
        exit(1)

    eval_grpo(args.adapter_dir, args.max_new_tokens, args.save_explanations)
