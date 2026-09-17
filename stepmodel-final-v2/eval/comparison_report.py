"""
Comprehensive Comparison Report Generator

This script compares baseline (zero-shot, few-shot) results against your model
evaluation results (stage1, stage2, stage3) and generates:
1. Consolidated metrics comparison table
2. Visual graphs showing performance differences
3. Summary report with key improvements
4. Combined CSV with all predictions for detailed analysis

Usage:
    python comparison_report.py
"""
import os
import csv
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, hamming_loss,
)
from pathlib import Path

# ── Path bootstrap (folder was restructured into core/ data_prep/ training/ eval/) ──
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "core"), _os.path.join(_ROOT, "data_prep"), _os.path.join(_ROOT, "training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from config import ROOT, STEP_LABELS, MCP_LABELS


# ---------------------------------------------------------------------------
# Pen-Strategist (arXiv 2605.04499) Step Model -- the paper's OWN reported
# numbers. Used verbatim as the baseline row; we do not re-train a replica.
# NaN for metrics the paper does not report, so the comparison table never
# fabricates a figure on their behalf.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# THE REPORTED METRIC SET. Everything else evaluate_model computes is kept as
# a diagnostic but is NOT put in the comparison table or the charts, so the
# headline comparison stays readable and matches what the write-up claims.
#   step: accuracy (headline, vs the paper's 82.87%) + macro-F1 (imbalance)
#   mcp : samples-F1 (headline, vs the paper's 0.64) + macro-F1 per tool,
#         plus the two error-direction rates (missing vs extra)
#   expl: LLM-judge gate accuracy
# ---------------------------------------------------------------------------
REPORTED_METRICS = [
    "step_accuracy",
    "step_macro_f1",
    "mcp_samples_f1",
    "mcp_macro_f1",
    "mcp_missing_tool_rate",
    "mcp_extra_tool_rate",
    "explanation_judge_accuracy",
]

# Lower is better for these two -- the charts label them so nobody reads a
# tall bar as good.
LOWER_IS_BETTER = {"mcp_missing_tool_rate", "mcp_extra_tool_rate"}


def load_judge_accuracy(output_dir, model_name):
    """Read 'Accuracy (gate-based): NN.NN%' from a model's LLM-judge report.

    Returns None when the model has no judge report (Stage 1 is a classifier
    and generates no explanation, and the baselines may not have been judged).
    None keeps the cell empty rather than implying a measured zero.
    """
    import glob as _glob
    tag = {"stage2": "stage2_qwen_lora", "stage3": "stage3_qwen_grpo"}.get(model_name, model_name)
    hits = _glob.glob(os.path.join(output_dir, f"llm_judge_examples_{tag}*.md"))
    if not hits:
        return None
    hits.sort(key=os.path.getmtime)
    try:
        txt = open(hits[-1], encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"Accuracy \(gate-based\):\s*([0-9.]+)%", txt)
    return float(m.group(1)) / 100.0 if m else None


PAPER_ROW_NAME = "pen_strategist_paper_reported"
PAPER_REPORTED = {
    "step_accuracy":      0.8287,   # 82.87%
    "step_micro_f1":      0.80,
    "mcp_subset_accuracy": 0.4888,  # 48.88% (their 'MCP accuracy')
    "mcp_micro_f1":       0.64,
}


def load_csv_data(csv_path):
    """Load CSV data and return as list of dictionaries."""
    if not os.path.exists(csv_path):
        print(f"[Warning] CSV file not found: {csv_path}")
        return None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def calculate_metrics(predictions, gold_labels, task_type='step'):
    """Calculate metrics for predictions vs gold labels."""
    if task_type == 'step':
        # Single-label classification
        return {
            'accuracy': accuracy_score(gold_labels, predictions),
            'macro_f1': f1_score(gold_labels, predictions, average='macro', zero_division=0),
            'weighted_f1': f1_score(gold_labels, predictions, average='weighted', zero_division=0),
        }
    else:
        # Multi-label classification (MCP)
        return {
            'micro_f1': f1_score(gold_labels, predictions, average='micro', zero_division=0),
            'macro_f1': f1_score(gold_labels, predictions, average='macro', zero_division=0),
            'subset_accuracy': accuracy_score(gold_labels, predictions),
        }


def _jaccard(pred_set, gold_set) -> float:
    """1.0 if both empty, else |intersection| / |union|. Symmetric, punishes
    both extra (FP) and missing (FN) tools without the all-or-nothing
    harshness of exact-match / subset accuracy."""
    if not pred_set and not gold_set:
        return 1.0
    union = pred_set | gold_set
    return len(pred_set & gold_set) / len(union) if union else 0.0


def parse_mcp_tools(mcp_string):
    """Parse MCP tool string into set of tools."""
    if not mcp_string or mcp_string == '':
        return set()
    return set(tool.strip() for tool in mcp_string.split('|') if tool.strip())


def extract_mcp_from_text(text: str):
    """Fallback: extract MCP tool names from free-form text using regex patterns.

    Uses the same pattern dictionary as data_utils.extract_mcp_labels so tool
    names match MCP_LABELS exactly.
    """
    if not text:
        return set()
    import re as _re
    patterns = {
        "Nmap": _re.compile(r"\bnmap\b", _re.I),
        "Metasploit": _re.compile(r"\bmetasploit|msfconsole|msfvenom\b", _re.I),
        "Netcat": _re.compile(r"\bnetcat|\bnc\b", _re.I),
        "Dirbuster": _re.compile(r"\bdirbuster|gobuster|dirb|netexec\b", _re.I),
        "SQLmap": _re.compile(r"\bsqlmap\b", _re.I),
        "Smb client": _re.compile(r"smb\s*client|smbclient|\bsmb\b", _re.I),
        "hydra": _re.compile(r"\bhydra\b", _re.I),
        "John-the-ripper": _re.compile(r"john[\s\-]?the[\s\-]?ripper|\bjohn\b", _re.I),
        "Google search": _re.compile(r"google\s*search|\bgoogle\b", _re.I),
        "Interactive CLI": _re.compile(r"interactive\s*cli|\bssh\b|\bbash\b|\bshell\b", _re.I),
        "Web page interaction": _re.compile(r"web\s*page\s*interaction|\bbrowser\b|\bcurl\b", _re.I),
    }
    return {label for label, pat in patterns.items() if pat.search(text)}


def evaluate_model(csv_data, model_name):
    """Evaluate a model's predictions from CSV data."""
    if csv_data is None:
        return None

    n_labels = len(STEP_LABELS)
    UNKNOWN_LABEL = n_labels  # used for UNPARSEABLE / wrong-format preds

    # Step classification metrics
    step_preds = []
    step_gold = []

    # MCP classification metrics
    mcp_preds = []
    mcp_gold = []

    print(f"[Debug] {model_name} CSV fields: {list(csv_data[0].keys()) if csv_data else 'No data'}")

    unparseable_count = 0
    total_count = 0

    for row in csv_data:
        pred_step = row.get('step_prediction', row.get('predicted_new_step', row.get('pred_step', ''))) or ''
        gold_step = row.get('gold_new_step', row.get('gold_step', '')) or ''

        total_count += 1

        # -------- step label mapping --------
        if pred_step == "UNPARSEABLE" or pred_step not in STEP_LABELS:
            if pred_step == "UNPARSEABLE":
                unparseable_count += 1
            # Map anything that is not a valid STEP_LABEL to the sentinel class.
            # sklearn's accuracy + f1 can handle this consistently without crash.
            p_idx = UNKNOWN_LABEL
        else:
            p_idx = STEP_LABELS.index(pred_step)

        if gold_step in STEP_LABELS:
            g_idx = STEP_LABELS.index(gold_step)
        else:
            # If gold format is wrong, skip to avoid corrupting metrics
            continue

        step_preds.append(p_idx)
        step_gold.append(g_idx)

        # -------- MCP --------
        pred_mcp_str = row.get('mcp_tool_prediction', row.get('predicted_mcp_tasks', row.get('pred_mcp_tasks', ''))) or ''
        gold_mcp_str = row.get('mcp_tool_gold', row.get('gold_mcp_tasks', row.get('gold_mcp', ''))) or ''

        pred_mcp = parse_mcp_tools(pred_mcp_str)
        gold_mcp = parse_mcp_tools(gold_mcp_str)

        # Secondary pass: if MCP was empty but raw_response / other fields mention tools, try to catch them.
        if not pred_mcp:
            rr = (row.get('raw_response') or row.get('step_explanation_predicted') or '')
            if rr:
                rr_set = extract_mcp_from_text(rr)
                if rr_set:
                    pred_mcp = rr_set

        pred_vec = [1 if tool in pred_mcp else 0 for tool in MCP_LABELS]
        gold_vec = [1 if tool in gold_mcp else 0 for tool in MCP_LABELS]

        mcp_preds.append(pred_vec)
        mcp_gold.append(gold_vec)

    print(f"[Debug] {model_name} - Unparseable predictions: {unparseable_count}/{total_count} "
          f"({100 * unparseable_count / total_count if total_count > 0 else 0:.1f}%)")

    labels_all = list(range(n_labels + 1))  # include UNKNOWN sentinel
    step_metrics = {}
    if step_preds and step_gold:
        step_metrics['accuracy'] = accuracy_score(step_gold, step_preds)
        step_metrics['macro_f1'] = f1_score(step_gold, step_preds, average='macro',
                                            labels=labels_all, zero_division=0)
        step_metrics['weighted_f1'] = f1_score(step_gold, step_preds, average='weighted',
                                               labels=labels_all, zero_division=0)

    mcp_metrics = {}
    if mcp_preds and mcp_gold:
        p = np.array(mcp_preds)
        g = np.array(mcp_gold)
        mcp_metrics['micro_f1'] = f1_score(g, p, average='micro', zero_division=0)
        mcp_metrics['macro_f1'] = f1_score(g, p, average='macro', zero_division=0)
        mcp_metrics['subset_accuracy'] = accuracy_score(g, p)

        # --- additional metrics for evaluating a "which tools" multi-label task ---
        # samples_f1: precision/recall/F1 computed PER ROW then averaged across rows.
        mcp_metrics['samples_f1'] = f1_score(g, p, average='samples', zero_division=0)
        mcp_metrics['samples_precision'] = precision_score(g, p, average='samples', zero_division=0)
        mcp_metrics['samples_recall'] = recall_score(g, p, average='samples', zero_division=0)

        # hamming_loss: fraction of individual tool slots (present/absent)
        # that are wrong, across all rows and all 11 tools. Lower is better.
        # Good "how far off, on average" summary that neither exact-match nor
        # F1 gives you directly.
        mcp_metrics['hamming_loss'] = hamming_loss(g, p)

        # jaccard_mean: |predicted ∩ gold| / |predicted ∪ gold| per row, averaged.
        # A stricter middle ground between subset_accuracy (all-or-nothing) and
        # samples_f1 (rewards partial overlap fairly generously). Extra tools
        # in the prediction and missing tools both shrink the union, so both
        # error types are penalized symmetrically.
        jaccards = []
        for pi, gi in zip(mcp_preds, mcp_gold):
            pred_set = {MCP_LABELS[j] for j, v in enumerate(pi) if v == 1}
            gold_set = {MCP_LABELS[j] for j, v in enumerate(gi) if v == 1}
            jaccards.append(_jaccard(pred_set, gold_set))
        mcp_metrics['jaccard_mean'] = float(np.mean(jaccards)) if jaccards else 0.0

        # Global micro precision/recall make the extra-vs-missing tradeoff
        # explicit: low precision means the model over-predicts tools (lots of
        # FPs / extra tools not in gold), low recall means it under-predicts
        # (lots of FNs / missing tools).
        mcp_metrics['micro_precision'] = precision_score(g, p, average='micro', zero_division=0)
        mcp_metrics['micro_recall'] = recall_score(g, p, average='micro', zero_division=0)

        # Missing- and extra-tool rates, per the reported metric set.
        # missing = of the tools that SHOULD have been named, what fraction
        #           were forgotten:  mean(|gold - pred| / |gold|)
        # extra   = of the tools that WERE named, what fraction should not
        #           have been there: mean(|pred - gold| / |pred|)
        # Rows with an empty denominator are skipped, not counted as 0.
        miss_r, extra_r = [], []
        for gi, pi in zip(g, p):
            gold_set = {j for j, v in enumerate(gi) if v}
            pred_set = {j for j, v in enumerate(pi) if v}
            if gold_set:
                miss_r.append(len(gold_set - pred_set) / len(gold_set))
            if pred_set:
                extra_r.append(len(pred_set - gold_set) / len(pred_set))
        mcp_metrics['missing_tool_rate'] = float(np.mean(miss_r)) if miss_r else 0.0
        mcp_metrics['extra_tool_rate'] = float(np.mean(extra_r)) if extra_r else 0.0

    metrics = {}
    metrics.update({f'step_{k}': v for k, v in step_metrics.items()})
    metrics.update({f'mcp_{k}': v for k, v in mcp_metrics.items()})
    return metrics


def generate_comparison_table(model_metrics):
    """Comparison table over REPORTED_METRICS only, in declared order.

    Everything else evaluate_model produces stays available as a
    diagnostic but is deliberately kept out of the headline table.
    """
    models = list(model_metrics.keys())
    metrics_names = list(REPORTED_METRICS)
    
    # Create comparison DataFrame
    comparison_data = []
    for model in models:
        row = {'Model': model}
        if model_metrics[model]:
            for metric in metrics_names:
                # float('nan'), not 0.0: Stage 1 generates no explanation
                # and the paper does not report every metric. A 0.0 would
                # read as 'measured and terrible' instead of 'not measured'.
                row[metric] = model_metrics[model].get(metric, float('nan'))
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def create_visualizations(comparison_df, output_dir):
    """Charts for the REPORTED metric set only: step, MCP, explanation.

    One figure per group so each stays readable, and the two error-rate bars
    are labelled lower-is-better so a tall bar is never misread as good.
    """
    plt.style.use('seaborn-v0_8-darkgrid')

    groups = [
        ('step_comparison.png', 'Step Classification',
         [m for m in ('step_accuracy', 'step_macro_f1') if m in comparison_df.columns]),
        ('mcp_comparison.png', 'MCP Tool Classification',
         [m for m in ('mcp_samples_f1', 'mcp_macro_f1',
                      'mcp_missing_tool_rate', 'mcp_extra_tool_rate')
          if m in comparison_df.columns]),
        ('explanation_comparison.png', 'Step Explanation (LLM judge)',
         [m for m in ('explanation_judge_accuracy',) if m in comparison_df.columns]),
    ]

    for fname, title, metrics in groups:
        cols = [m for m in metrics if comparison_df[m].notna().any()]
        if not cols:
            continue
        fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
        if len(cols) == 1:
            axes = [axes]
        for ax, metric in zip(axes, cols):
            sub = comparison_df[['Model', metric]].dropna(subset=[metric])
            colour = 'salmon' if metric in LOWER_IS_BETTER else 'skyblue'
            ax.bar(sub['Model'], sub[metric], color=colour)
            label = metric.replace('step_', '').replace('mcp_', '').replace('_', ' ')
            suffix = '  (lower is better)' if metric in LOWER_IS_BETTER else ''
            ax.set_title(f'{label}{suffix}', fontsize=13, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')
        fig.suptitle(title, fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

def generate_consolidated_csv(model_data, output_dir):
    """Generate a consolidated CSV with all model predictions."""
    # Find the model with the most complete data as reference
    reference_model = None
    max_length = 0
    
    for model_name, data in model_data.items():
        if data and len(data) > max_length:
            max_length = len(data)
            reference_model = model_name
    
    if not reference_model:
        print("[Warning] No valid model data found for consolidation")
        return
    
    reference_data = model_data[reference_model]
    
    # Build consolidated rows
    consolidated = []
    for i in range(len(reference_data)):
        row = {
            'index': i,
            'machine': reference_data[i].get('machine', ''),
            'new_strategy': reference_data[i].get('new_strategy', ''),
            'strategy_explanation': reference_data[i].get('strategy_explanation', ''),
        }
        
        # Add gold labels
        row['gold_step'] = reference_data[i].get('gold_new_step', '')
        row['gold_mcp'] = reference_data[i].get('mcp_tool_gold', '')
        
        # Add predictions from each model
        for model_name, data in model_data.items():
            if data and i < len(data):
                pred_step = data[i].get('step_prediction', data[i].get('predicted_new_step', data[i].get('pred_step', '')))
                pred_mcp = data[i].get('mcp_tool_prediction', data[i].get('predicted_mcp_tasks', data[i].get('pred_mcp_tasks', '')))
                
                row[f'{model_name}_step'] = pred_step
                row[f'{model_name}_mcp'] = pred_mcp
                
                # Add explanation if available
                if 'step_explanation_predicted' in data[i]:
                    row[f'{model_name}_explanation'] = data[i]['step_explanation_predicted']
                if 'step_explanation_gold' in data[i]:
                    row['gold_explanation'] = data[i]['step_explanation_gold']
        
        consolidated.append(row)
    
    # Save consolidated CSV
    output_path = os.path.join(output_dir, 'consolidated_predictions.csv')
    fieldnames = list(consolidated[0].keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(consolidated)
    
    print(f"[Report] Consolidated predictions saved to: {output_path}")


def generate_summary_report(comparison_df, output_dir):
    """Generate a summary report with key metrics and improvements."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE MODEL COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall comparison table
    report_lines.append("OVERALL PERFORMANCE METRICS")
    report_lines.append("-" * 80)
    report_lines.append(comparison_df.to_string(index=False))
    report_lines.append("")
    
    # Key improvements
    report_lines.append("KEY IMPROVEMENTS (Stage 2 vs Zero-shot Baseline)")
    report_lines.append("-" * 80)
    
    if 'stage2' in comparison_df['Model'].values and 'baseline_zeroshot' in comparison_df['Model'].values:
        stage2_row = comparison_df[comparison_df['Model'] == 'stage2'].iloc[0]
        baseline_row = comparison_df[comparison_df['Model'] == 'baseline_zeroshot'].iloc[0]
        
        for metric in comparison_df.columns:
            if metric != 'Model' and pd.notna(stage2_row[metric]) and pd.notna(baseline_row[metric]):
                delta = stage2_row[metric] - baseline_row[metric]
                # For error rates a NEGATIVE delta is the improvement, so flip
                # the sign before calling it one -- otherwise a rising
                # missing-tool rate prints as '+27.0%' improvement.
                gain = -delta if metric in LOWER_IS_BETTER else delta
                denom = abs(baseline_row[metric])
                pct = (gain / denom * 100) if denom > 0 else 0.0
                verdict = 'better' if gain > 0 else ('worse' if gain < 0 else 'same')
                report_lines.append(
                    f"{metric:25s}: {stage2_row[metric]:.4f} vs {baseline_row[metric]:.4f} "
                    f"({delta:+.4f}, {pct:+.1f}% {verdict})")
    
    report_lines.append("")
    
    # Best performing model per metric
    report_lines.append("BEST MODEL PER METRIC")
    report_lines.append("-" * 80)
    
    for metric in comparison_df.columns:
        if metric == 'Model':
            continue
        col = comparison_df[metric]
        if not col.notna().any():
            continue
        # DIRECTION MATTERS. missing_tool_rate / extra_tool_rate are ERROR
        # rates -- lower is better. Using idxmax() on them reported the
        # WORST model as the best (e.g. extra_tool_rate 0.5457 for the
        # zero-shot baseline was printed as the winner).
        lower_better = metric in LOWER_IS_BETTER
        idx = col.idxmin() if lower_better else col.idxmax()
        best_model = comparison_df.loc[idx, 'Model']
        best_score = col.min() if lower_better else col.max()
        arrow = ' (lower is better)' if lower_better else ''
        report_lines.append(f"{metric:25s}: {best_model} ({best_score:.4f}){arrow}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    output_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[Report] Summary report saved to: {output_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


def main():
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("[Report] Generating comprehensive comparison report...")
    
    # Define models to compare
    models = {
        'baseline_zeroshot': 'baseline_zeroshot.csv',
        'baseline_3shot': 'baseline_3shot.csv', 
        'baseline_5shot': 'baseline_5shot.csv',
        'stage1': 'stage1.csv',
        'stage2': 'stage2.csv',
        'stage3': 'stage3.csv',
    }
    
    # Load all model data
    model_data = {}
    model_metrics = {}
    
    for model_name, csv_filename in models.items():
        csv_path = os.path.join(output_dir, csv_filename)
        data = load_csv_data(csv_path)
        model_data[model_name] = data
        
        if data:
            metrics = evaluate_model(data, model_name)
            model_metrics[model_name] = metrics
            print(f"[Report] Loaded {model_name}: {len(data)} samples")
        else:
            print(f"[Warning] Could not load {model_name}")
    
    # Attach LLM-judge explanation accuracy (None where a model produces no
    # explanation, e.g. the Stage-1 classifier).
    for _m in list(model_metrics):
        _acc = load_judge_accuracy(output_dir, _m)
        if _acc is not None and model_metrics[_m]:
            model_metrics[_m]['explanation_judge_accuracy'] = _acc

    # ---- Pen-Strategist paper reference (reported, NOT re-trained) ----
    # We no longer train a GPT-2 TextCNN replica: reproducing someone else's
    # model introduces its own confounds (our split, our label normalisation,
    # our tokenizer) and the replica's numbers were not comparable to the
    # published ones anyway. The paper's OWN reported figures are used
    # directly, which is the honest comparison. Source: arXiv 2605.04499.
    # Only the four metrics the paper reports are filled in; everything else
    # is left as NaN so the table never implies we measured something they
    # did not publish.
    model_metrics[PAPER_ROW_NAME] = dict(PAPER_REPORTED)
    print(f"[Report] Added paper reference row '{PAPER_ROW_NAME}' "
          f"(reported values from arXiv 2605.04499, not re-trained)")

    # Generate comparison table
    comparison_df = generate_comparison_table(model_metrics)
    
    # Save comparison table as CSV
    comparison_path = os.path.join(output_dir, 'metrics_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[Report] Metrics comparison saved to: {comparison_path}")
    
    # Generate visualizations
    create_visualizations(comparison_df, output_dir)
    print(f"[Report] Visualizations saved to: {output_dir}")
    
    # Generate consolidated CSV
    generate_consolidated_csv(model_data, output_dir)
    
    # Generate summary report
    generate_summary_report(comparison_df, output_dir)
    
    print(f"[Report] Comprehensive comparison report complete!")


if __name__ == "__main__":
    main()