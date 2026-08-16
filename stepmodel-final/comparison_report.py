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
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path

from config import ROOT, STEP_LABELS, MCP_LABELS


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

    metrics = {}
    metrics.update({f'step_{k}': v for k, v in step_metrics.items()})
    metrics.update({f'mcp_{k}': v for k, v in mcp_metrics.items()})
    return metrics


def generate_comparison_table(model_metrics):
    """Generate a comparison table of all models."""
    models = list(model_metrics.keys())
    metrics_names = set()
    for metrics in model_metrics.values():
        if metrics:
            metrics_names.update(metrics.keys())
    
    metrics_names = sorted(metrics_names)
    
    # Create comparison DataFrame
    comparison_data = []
    for model in models:
        row = {'Model': model}
        if model_metrics[model]:
            for metric in metrics_names:
                row[metric] = model_metrics[model].get(metric, 0.0)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def create_visualizations(comparison_df, output_dir):
    """Create visualization graphs for performance comparison."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Step classification comparison
    step_metrics = [col for col in comparison_df.columns if 'step_' in col]
    if step_metrics:
        fig, axes = plt.subplots(1, len(step_metrics), figsize=(6*len(step_metrics), 5))
        if len(step_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(step_metrics):
            ax = axes[i]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'Step Classification: {metric.replace("step_", "").upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.legend().remove()
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # MCP classification comparison
    mcp_metrics = [col for col in comparison_df.columns if 'mcp_' in col]
    if mcp_metrics:
        fig, axes = plt.subplots(1, len(mcp_metrics), figsize=(6*len(mcp_metrics), 5))
        if len(mcp_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(mcp_metrics):
            ax = axes[i]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, color='lightcoral')
            ax.set_title(f'MCP Classification: {metric.replace("mcp_", "").upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.legend().remove()
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mcp_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Combined radar chart
    all_metrics = step_metrics + mcp_metrics
    if all_metrics:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Normalize metric names for display
        display_names = [m.replace('step_', '').replace('mcp_', '').upper() for m in all_metrics]
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(comparison_df)))
        for idx, (_, row) in enumerate(comparison_df.iterrows()):
            values = [row[m] for m in all_metrics]
            values += values[:1]  # Close the radar
            angles = np.linspace(0, 2*np.pi, len(all_metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
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
                improvement = stage2_row[metric] - baseline_row[metric]
                pct_improvement = (improvement / baseline_row[metric] * 100) if baseline_row[metric] > 0 else 0
                report_lines.append(f"{metric:25s}: {stage2_row[metric]:.4f} vs {baseline_row[metric]:.4f} ({improvement:+.4f}, {pct_improvement:+.1f}%)")
    
    report_lines.append("")
    
    # Best performing model per metric
    report_lines.append("BEST MODEL PER METRIC")
    report_lines.append("-" * 80)
    
    for metric in comparison_df.columns:
        if metric != 'Model':
            best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
            best_score = comparison_df[metric].max()
            report_lines.append(f"{metric:25s}: {best_model} ({best_score:.4f})")
    
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
