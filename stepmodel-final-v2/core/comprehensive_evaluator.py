"""
Comprehensive evaluation metrics for multi-stage training pipeline.

Evaluates:
- Step prediction: accuracy, micro-F1
- MCP tool prediction: accuracy, F1, micro-F1, missing tools, extra tools, exact match
- Explanation quality: multi-dimensional LLM judge evaluation
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import json

from config import STEP_LABELS, MCP_LABELS, STEP2IDX, MCP2IDX


class ComprehensiveEvaluator:
    """Comprehensive evaluation for step, MCP, and explanation predictions."""
    
    def __init__(self):
        self.step_labels = STEP_LABELS
        self.mcp_labels = MCP_LABELS
        self.step2idx = STEP2IDX
        self.mcp2idx = MCP2IDX
    
    def evaluate_step(self, pred_steps: List[str], gold_steps: List[str]) -> Dict[str, float]:
        """
        Evaluate step predictions.
        
        Args:
            pred_steps: List of predicted step labels
            gold_steps: List of ground truth step labels
            
        Returns:
            Dictionary with accuracy and micro-F1 scores
        """
        # Convert to indices
        pred_indices = [self.step2idx.get(s, -1) for s in pred_steps]
        gold_indices = [self.step2idx.get(s, -1) for s in gold_steps]
        
        # Filter out invalid indices
        valid_mask = [p != -1 and g != -1 for p, g in zip(pred_indices, gold_indices)]
        pred_indices = [p for p, m in zip(pred_indices, valid_mask) if m]
        gold_indices = [g for g, m in zip(gold_indices, valid_mask) if m]
        
        if len(pred_indices) == 0:
            return {"accuracy": 0.0, "micro_f1": 0.0, "n_valid": 0}
        
        accuracy = accuracy_score(gold_indices, pred_indices)
        micro_f1 = f1_score(gold_indices, pred_indices, average='micro')
        
        return {
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "n_valid": len(pred_indices)
        }
    
    def evaluate_mcp(self, pred_mcps: List[Dict[str, str]], 
                     gold_mcp_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate MCP tool predictions with comprehensive metrics.
        
        Args:
            pred_mcps: List of predicted MCP tool dictionaries
            gold_mcp_texts: List of ground truth MCP tool text descriptions
            
        Returns:
            Dictionary with accuracy, F1, micro-F1, missing tools, extra tools, exact match
        """
        # Parse ground truth
        gold_mcp_labels = []
        for text in gold_mcp_texts:
            labels = [0] * len(self.mcp_labels)
            if text:
                for i, tool in enumerate(self.mcp_labels):
                    if tool.lower() in text.lower():
                        labels[i] = 1
            gold_mcp_labels.append(labels)
        
        # Parse predictions
        pred_mcp_labels = []
        for pred in pred_mcps:
            labels = [0] * len(self.mcp_labels)
            if pred:
                for tool in pred.keys():
                    if tool in self.mcp2idx:
                        labels[self.mcp2idx[tool]] = 1
            pred_mcp_labels.append(labels)
        
        gold_mcp_labels = np.array(gold_mcp_labels)
        pred_mcp_labels = np.array(pred_mcp_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(gold_mcp_labels.flatten(), pred_mcp_labels.flatten())
        micro_f1 = f1_score(gold_mcp_labels, pred_mcp_labels, average='micro')
        
        # Per-class F1 (macro)
        macro_f1 = f1_score(gold_mcp_labels, pred_mcp_labels, average='macro')
        
        # Calculate missing and extra tools
        missing_tools = []
        extra_tools = []
        exact_matches = []
        
        for gold, pred in zip(gold_mcp_labels, pred_mcp_labels):
            gold_set = set(i for i, v in enumerate(gold) if v == 1)
            pred_set = set(i for i, v in enumerate(pred) if v == 1)
            
            missing = gold_set - pred_set
            extra = pred_set - gold_set
            
            missing_tools.append(len(missing))
            extra_tools.append(len(extra))
            exact_matches.append(1 if gold_set == pred_set else 0)
        
        avg_missing = np.mean(missing_tools)
        avg_extra = np.mean(extra_tools)
        exact_match_rate = np.mean(exact_matches)
        
        # Per-tool breakdown
        precision, recall, f1_per_tool, _ = precision_recall_fscore_support(
            gold_mcp_labels, pred_mcp_labels, average=None, zero_division=0
        )
        
        per_tool_metrics = {
            tool: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1_per_tool[i])
            }
            for i, tool in enumerate(self.mcp_labels)
        }
        
        return {
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "avg_missing_tools": avg_missing,
            "avg_extra_tools": avg_extra,
            "exact_match_rate": exact_match_rate,
            "per_tool_metrics": per_tool_metrics,
            "n_samples": len(gold_mcp_labels)
        }
    
    def evaluate_explanation_quality(self, pred_explanations: List[str],
                                     gold_explanations: List[str],
                                     pred_steps: Optional[List[str]] = None,
                                     contexts: Optional[List[Dict]] = None,
                                     use_llm_judge: bool = True) -> Dict[str, float]:
        """
        Evaluate explanation quality using multiple dimensions.
        
        Integrates the existing LLM judge (llm_judge.py) which uses a rubric-based
        evaluation with 4 dimensions: relevance, technical_accuracy, completeness, clarity.
        Based on research findings, this multi-dimensional approach is superior to
        single-score metrics like BLEU/F1.
        
        Research-based dimensions from llm_judge.py:
        - Relevance: Does explanation justify the predicted step?
        - Technical Accuracy: Are technical claims correct?
        - Completeness: Does it cover key justification points?
        - Clarity: Is it well-structured and unambiguous?
        
        Args:
            pred_explanations: List of predicted explanations
            gold_explanations: List of ground truth explanations
            pred_steps: Optional list of predicted step labels
            contexts: Optional list of context dictionaries for LLM judge
            use_llm_judge: Whether to use LLM judge (requires model loaded)
            
        Returns:
            Dictionary with multi-dimensional scores
        """
        # Structural metrics (can be computed without LLM judge)
        pred_lengths = [len(e.split()) for e in pred_explanations]
        gold_lengths = [len(e.split()) for e in gold_explanations]
        
        length_similarity = [
            1 - abs(p - g) / max(g, 1) for p, g in zip(pred_lengths, gold_lengths)
        ]
        avg_length_similarity = np.mean(length_similarity)
        
        # Completeness (non-empty explanations)
        completeness = [1 if len(e.strip()) > 0 else 0 for e in pred_explanations]
        avg_completeness = np.mean(completeness)
        
        result = {
            "length_similarity": avg_length_similarity,
            "structural_completeness": avg_completeness,
            "avg_pred_length": np.mean(pred_lengths),
            "avg_gold_length": np.mean(gold_lengths),
        }
        
        # Use LLM judge if available and requested
        if use_llm_judge and contexts is not None and pred_steps is not None:
            try:
                from llm_judge import batch_evaluate_explanations
                
                # Prepare examples for LLM judge
                examples = []
                for i, (pred_expl, gold_expl, pred_step, ctx) in enumerate(
                    zip(pred_explanations, gold_explanations, pred_steps, contexts)
                ):
                    examples.append({
                        "pred_explanation": pred_expl,
                        "gold_explanation": gold_expl,
                        "pred_step": pred_step,
                        "gold_step": ctx.get('gold_new_step', ''),
                        "context": ctx,
                        "machine": ctx.get('machine', f'sample_{i}')
                    })
                
                # Run LLM judge evaluation
                judge_results = batch_evaluate_explanations(
                    examples, 
                    verbose=False,
                    include_errors_in_accuracy=False
                )
                
                # Extract aggregate metrics
                aggregates = judge_results.get("aggregates", {})
                dim_rates = judge_results.get("dimension_pass_rates", {})
                
                # Map LLM judge dimensions to research-based terminology
                result.update({
                    # LLM judge rubric dimensions (0-3 scale, normalized to 0-1)
                    "relevance": dim_rates.get("relevance", {}).get("pass_rate_percent", 0.0) / 100.0,
                    "technical_accuracy": dim_rates.get("technical_accuracy", {}).get("pass_rate_percent", 0.0) / 100.0,
                    "completeness": dim_rates.get("completeness", {}).get("pass_rate_percent", 0.0) / 100.0,
                    "clarity": dim_rates.get("clarity", {}).get("pass_rate_percent", 0.0) / 100.0,
                    
                    # Research-based terminology mapping
                    "faithfulness": dim_rates.get("technical_accuracy", {}).get("pass_rate_percent", 0.0) / 100.0,
                    "plausibility": dim_rates.get("relevance", {}).get("pass_rate_percent", 0.0) / 100.0,
                    "correctness": aggregates.get("is_correct", {}).get("accuracy_percent", 0.0) / 100.0,
                    
                    # Overall quality (weighted average of dimensions)
                    "overall_quality": aggregates.get("correctness", {}).get("mean", 0.0),
                    
                    # Judge metadata
                    "judge_error_count": judge_results.get("judge_error_count", 0),
                    "n_evaluated": judge_results.get("total_evaluated", 0),
                })
                
            except ImportError:
                print("[Warning] llm_judge not available, using structural metrics only")
            except Exception as e:
                print(f"[Warning] LLM judge evaluation failed: {e}, using structural metrics only")
        
        return result
    
    def compute_comprehensive_reward(self, pred_step: str, pred_mcp: Dict[str, str],
                                     pred_expl: str, gold_step: str,
                                     gold_mcp_text: str, gold_expl: str,
                                     step_weight: float = 1.0,
                                     mcp_weight: float = 1.0,
                                     expl_weight: float = 1.0) -> Dict[str, float]:
        """
        Compute comprehensive reward for GRPO training.
        
        Args:
            pred_step: Predicted step label
            pred_mcp: Predicted MCP tools dictionary
            pred_expl: Predicted explanation
            gold_step: Ground truth step label
            gold_mcp_text: Ground truth MCP tools text
            gold_expl: Ground truth explanation
            step_weight: Weight for step component
            mcp_weight: Weight for MCP component
            expl_weight: Weight for explanation component
            
        Returns:
            Dictionary with individual component scores and total reward
        """
        # Step reward (exact match)
        step_reward = 1.0 if pred_step == gold_step else 0.0
        
        # MCP reward (F1 score)
        gold_mcp_labels = [0] * len(self.mcp_labels)
        if gold_mcp_text:
            for i, tool in enumerate(self.mcp_labels):
                if tool.lower() in gold_mcp_text.lower():
                    gold_mcp_labels[i] = 1
        
        pred_mcp_labels = [0] * len(self.mcp_labels)
        if pred_mcp:
            for tool in pred_mcp.keys():
                if tool in self.mcp2idx:
                    pred_mcp_labels[self.mcp2idx[tool]] = 1
        
        if sum(gold_mcp_labels) > 0:
            tp = sum(1 for g, p in zip(gold_mcp_labels, pred_mcp_labels) if g == 1 and p == 1)
            fp = sum(1 for g, p in zip(gold_mcp_labels, pred_mcp_labels) if g == 0 and p == 1)
            fn = sum(1 for g, p in zip(gold_mcp_labels, pred_mcp_labels) if g == 1 and p == 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            mcp_reward = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            mcp_reward = 1.0 if sum(pred_mcp_labels) == 0 else 0.0
        
        # Explanation reward (multi-dimensional)
        # For now, use length and completeness as proxy
        # This should be enhanced with LLM judge integration
        expl_length_score = min(1.0, len(pred_expl) / 50.0)  # Normalize to ~50 chars
        expl_completeness = 1.0 if len(pred_expl.strip()) > 10 else 0.0
        expl_reward = 0.5 * expl_length_score + 0.5 * expl_completeness
        
        # Weighted total reward
        total_weight = step_weight + mcp_weight + expl_weight
        total_reward = (
            step_weight * step_reward +
            mcp_weight * mcp_reward +
            expl_weight * expl_reward
        ) / total_weight
        
        return {
            "step_reward": step_reward,
            "mcp_reward": mcp_reward,
            "explanation_reward": expl_reward,
            "total_reward": total_reward
        }
    
    def evaluate_batch(self, predictions: List[Dict], gold_data: List[Dict]) -> Dict[str, any]:
        """
        Evaluate a batch of predictions comprehensively.
        
        Args:
            predictions: List of prediction dictionaries with keys:
                - step: predicted step label
                - mcp: predicted MCP tools dict
                - explanation: predicted explanation
            gold_data: List of ground truth dictionaries with keys:
                - gold_new_step: ground truth step
                - gold_mcp_tasks: ground truth MCP text
                - gold_step_explanation: ground truth explanation
                
        Returns:
            Comprehensive evaluation results
        """
        pred_steps = [p.get("step", "") for p in predictions]
        gold_steps = [g.get("gold_new_step", "") for g in gold_data]
        
        pred_mcps = [p.get("mcp", {}) for p in predictions]
        gold_mcp_texts = [g.get("gold_mcp_tasks", "") for g in gold_data]
        
        pred_expls = [p.get("explanation", "") for p in predictions]
        gold_expls = [g.get("gold_step_explanation", "") for g in gold_data]
        
        step_metrics = self.evaluate_step(pred_steps, gold_steps)
        mcp_metrics = self.evaluate_mcp(pred_mcps, gold_mcp_texts)
        expl_metrics = self.evaluate_explanation_quality(pred_expls, gold_expls)
        
        return {
            "step_metrics": step_metrics,
            "mcp_metrics": mcp_metrics,
            "explanation_metrics": expl_metrics,
            "overall_score": (
                step_metrics["accuracy"] * 0.4 +
                mcp_metrics["micro_f1"] * 0.4 +
                expl_metrics["length_similarity"] * 0.2
            )
        }
