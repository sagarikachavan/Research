#!/usr/bin/env python3
"""
Performance Investigation Framework

This script systematically investigates whether graph structure training improves
overall model performance by comparing different training approaches:

1. Baseline: Stage 2 trained directly on step prediction (no graph structure pre-training)
2. Graph Structure Pre-trained: Stage 2 trained after graph structure pre-training
3. Multi-task: Simultaneous training on step prediction + graph structure tasks

The framework evaluates:
- Step prediction accuracy
- MCP tool classification performance  
- Graph structure understanding (adjacency, node type, edge type, path prediction)
- Overall model performance across all metrics

Usage:
    python investigate_performance.py --baseline_checkpoint /path/to/stage2
                                       --graph_structure_checkpoint /path/to/graph_structure
                                       --multi_task_checkpoint /path/to/multi_task
                                       --output_dir /path/to/results
"""
import json
import os
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

from config import (
    QWEN_MODEL_NAME, GRAPH_PREFIX_TOKENS, GNN_OUT_DIM,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, STAGE1_CKPT,
    ROOT, INPUT_TRAIN_JSON, INPUT_TEST_JSON,
)
from graph_encoder import Stage1Classifier, GraphEncoder
from data_utils import build_graph_from_input_json_graph, load_from_input_json
from prompts import build_prompt
from test_graph_prefix_adapter import (
    GraphPrefixAdapter,
    run_adjacency_test,
    run_node_type_test,
    run_edge_type_test,
    run_path_test,
    run_step_prediction_test,
    load_graph_from_json,
    build_adjacency_dict,
    build_node_content_dict,
)


class PerformanceInvestigator:
    """Systematically investigate performance across different training approaches."""
    
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.tokenizer = None
        self.graph_encoder = None
        self.graph_dict = None
        self.graph_data = None
        self.adjacency = None
        self.node_id_to_label = None
        self.node_id_to_content = None
        
    def setup(self):
        """Setup common components."""
        print("Setting up performance investigation framework...")
        
        # Load tokenizer
        print(f"Loading tokenizer: {QWEN_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Stage-1 graph encoder
        print(f"Loading Stage-1 graph encoder...")
        if os.path.exists(STAGE1_CKPT):
            stage1 = Stage1Classifier()
            stage1.load_state_dict(torch.load(STAGE1_CKPT, map_location=self.device))
            self.graph_encoder = stage1.graph_encoder.to(self.device).eval()
            for p in self.graph_encoder.parameters():
                p.requires_grad_(False)
        else:
            print(f"Warning: Stage-1 checkpoint not found at {STAGE1_CKPT}")
            self.graph_encoder = GraphEncoder().to(self.device).eval()
        
        # Load test graph from training data
        print(f"Loading test graph from training data...")
        examples = load_from_input_json(INPUT_TRAIN_JSON, "train")
        if examples and len(examples) > 0:
            # Use first example's graph as test graph
            graph_data = examples[0].get("graph", {})
            if graph_data and hasattr(graph_data, 'x'):
                self.graph_data = graph_data.to(self.device)
                # Convert torch_geometric Data to dict format for testing functions
                self.graph_dict = self._convert_torch_geo_to_dict(graph_data)
            else:
                print("Warning: No valid graph found in training data")
                self.graph_data = None
                self.graph_dict = {}
        else:
            print("Warning: No training data found")
            self.graph_data = None
            self.graph_dict = {}
        
        if self.graph_dict:
            # Build graph structures
            self.adjacency, self.node_id_to_label = build_adjacency_dict(self.graph_dict)
            self.node_id_to_content = build_node_content_dict(self.graph_dict)
            print(f"Graph statistics: {len(self.node_id_to_label)} nodes, {len(self.graph_dict.get('edges', []))} edges")
        else:
            self.adjacency = {}
            self.node_id_to_label = {}
            self.node_id_to_content = {}
    
    def _convert_torch_geo_to_dict(self, graph_data):
        """Convert torch_geometric Data to dict format."""
        x = graph_data.x.cpu().numpy() if hasattr(graph_data.x, 'cpu') else graph_data.x
        edge_index = graph_data.edge_index.cpu().numpy() if hasattr(graph_data.edge_index, 'cpu') else graph_data.edge_index
        
        nodes = [{"id": str(i), "label": f"node_{i}"} for i in range(x.shape[0])]
        edges = []
        for i in range(edge_index.shape[1]):
            src = str(edge_index[0, i])
            tgt = str(edge_index[1, i])
            edges.append({"from": src, "to": tgt, "type": "StateTransition"})
        
        return {"nodes": nodes, "edges": edges}
    
    def load_model(self, checkpoint_dir: str, model_name: str):
        """Load a trained model from checkpoint directory."""
        print(f"\nLoading model: {model_name}")
        print(f"Checkpoint: {checkpoint_dir}")
        
        if not os.path.exists(checkpoint_dir):
            print(f"Warning: Checkpoint not found at {checkpoint_dir}")
            return None, None
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME, torch_dtype=self.dtype, device_map=None
        ).to(self.device)
        
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.eval()
        
        # Load Graph Prefix Adapter
        llm_hidden = model.config.hidden_size
        adapter = GraphPrefixAdapter(GNN_OUT_DIM, llm_hidden).to(self.device).to(self.dtype)
        
        adapter_path = os.path.join(checkpoint_dir, "graph_adapter.pt")
        if os.path.exists(adapter_path):
            adapter.load_state_dict(torch.load(adapter_path, map_location=self.device))
            print(f"Loaded adapter weights from: {adapter_path}")
        else:
            print(f"Warning: Adapter weights not found at {adapter_path}")
        
        adapter.eval()
        
        return model, adapter
    
    def evaluate_model(self, model, adapter, model_name: str) -> Dict[str, float]:
        """Evaluate a model on all metrics."""
        print(f"\n{'=' * 60}")
        print(f"Evaluating Model: {model_name}")
        print(f"{'=' * 60}")
        
        results = {"model_name": model_name}
        
        if model is None or adapter is None:
            print("Model or adapter not available, skipping evaluation")
            return results
        
        # Graph structure tests
        print("\n--- Graph Structure Understanding ---")
        
        try:
            adjacency_recall = run_adjacency_test(
                self.device, self.dtype, self.graph_dict, self.graph_data, 
                self.adjacency, self.node_id_to_label,
                self.graph_encoder, model, adapter, self.tokenizer, model_name
            )
            results["adjacency_recall"] = adjacency_recall
        except Exception as e:
            print(f"Error in adjacency test: {e}")
            results["adjacency_recall"] = None
        
        try:
            node_type_accuracy = run_node_type_test(
                self.device, self.dtype, self.graph_dict, self.graph_data,
                self.node_id_to_label,
                self.graph_encoder, model, adapter, self.tokenizer, model_name
            )
            results["node_type_accuracy"] = node_type_accuracy
        except Exception as e:
            print(f"Error in node type test: {e}")
            results["node_type_accuracy"] = None
        
        try:
            edge_type_accuracy = run_edge_type_test(
                self.device, self.dtype, self.graph_dict, self.graph_data,
                self.node_id_to_label,
                self.graph_encoder, model, adapter, self.tokenizer, model_name
            )
            results["edge_type_accuracy"] = edge_type_accuracy
        except Exception as e:
            print(f"Error in edge type test: {e}")
            results["edge_type_accuracy"] = None
        
        try:
            path_accuracy = run_path_test(
                self.device, self.dtype, self.graph_dict, self.graph_data,
                self.adjacency, self.node_id_to_label,
                self.graph_encoder, model, adapter, self.tokenizer, model_name
            )
            results["path_accuracy"] = path_accuracy
        except Exception as e:
            print(f"Error in path test: {e}")
            results["path_accuracy"] = None
        
        # Step prediction test
        print("\n--- Step Prediction Performance ---")
        try:
            step_accuracy = run_step_prediction_test(
                self.device, self.dtype, self.graph_encoder, model, adapter, self.tokenizer, model_name
            )
            results["step_prediction_accuracy"] = step_accuracy
        except Exception as e:
            print(f"Error in step prediction test: {e}")
            results["step_prediction_accuracy"] = None
        
        return results
    
    def compare_models(self, results_list: List[Dict]) -> pd.DataFrame:
        """Compare results across models."""
        df = pd.DataFrame(results_list)
        
        # Calculate improvement over baseline
        if len(results_list) >= 2:
            baseline = results_list[0]
            for i, result in enumerate(results_list[1:], 1):
                for key in ["adjacency_recall", "node_type_accuracy", "edge_type_accuracy", 
                           "path_accuracy", "step_prediction_accuracy"]:
                    if baseline.get(key) is not None and result.get(key) is not None:
                        improvement_key = f"{key}_improvement"
                        result[improvement_key] = result[key] - baseline[key]
        
        return df
    
    def generate_report(self, df: pd.DataFrame, output_dir: str):
        """Generate comprehensive performance report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results CSV
        csv_path = os.path.join(output_dir, "performance_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Generate text report
        report_path = os.path.join(output_dir, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PERFORMANCE INVESTIGATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Models Evaluated:\n")
            for i, row in df.iterrows():
                f.write(f"  {i+1}. {row['model_name']}\n")
            f.write("\n")
            
            f.write("Performance Metrics:\n")
            f.write("-" * 60 + "\n")
            
            for metric in ["adjacency_recall", "node_type_accuracy", "edge_type_accuracy", 
                          "path_accuracy", "step_prediction_accuracy"]:
                f.write(f"\n{metric}:\n")
                for i, row in df.iterrows():
                    value = row.get(metric)
                    if value is not None:
                        f.write(f"  {row['model_name']}: {value:.2%}\n")
                        
                        # Show improvement if available
                        improvement_key = f"{metric}_improvement"
                        if improvement_key in row and row[improvement_key] != 0:
                            f.write(f"    (Improvement: {row[improvement_key]:+.2%})\n")
                    else:
                        f.write(f"  {row['model_name']}: N/A\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("CONCLUSIONS\n")
            f.write("=" * 60 + "\n")
            
            # Analyze which model performs best overall
            best_model = None
            best_score = -1
            
            for i, row in df.iterrows():
                # Calculate average score across available metrics
                scores = [row.get(m) for m in ["adjacency_recall", "node_type_accuracy", 
                                              "edge_type_accuracy", "path_accuracy", 
                                              "step_prediction_accuracy"] if row.get(m) is not None]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = row['model_name']
            
            if best_model:
                f.write(f"\nBest overall performer: {best_model} (avg score: {best_score:.2%})\n")
            
            # Check if graph structure pre-training helps
            if len(df) >= 2:
                baseline_step = df.iloc[0].get("step_prediction_accuracy")
                if baseline_step is not None:
                    for i in range(1, len(df)):
                        model_step = df.iloc[i].get("step_prediction_accuracy")
                        if model_step is not None:
                            improvement = model_step - baseline_step
                            if improvement > 0:
                                f.write(f"\n{df.iloc[i]['model_name']} improves step prediction by {improvement:.2%}\n")
                            else:
                                f.write(f"\n{df.iloc[i]['model_name']} does not improve step prediction ({improvement:.2%})\n")
        
        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Investigate performance across training approaches")
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                       help="Path to baseline Stage 2 checkpoint (no graph structure pre-training)")
    parser.add_argument("--graph_structure_checkpoint", type=str, default=None,
                       help="Path to graph structure pre-trained checkpoint")
    parser.add_argument("--multi_task_checkpoint", type=str, default=None,
                       help="Path to multi-task trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="/tmp/performance_investigation",
                       help="Output directory for results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("PERFORMANCE INVESTIGATION FRAMEWORK")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Initialize investigator
    investigator = PerformanceInvestigator(device, dtype)
    investigator.setup()
    
    # Define models to evaluate
    models_to_evaluate = []
    
    if args.baseline_checkpoint:
        models_to_evaluate.append(("Baseline (Stage 2)", args.baseline_checkpoint))
    else:
        # Try to find default Stage 2 checkpoint
        default_stage2 = os.path.join(ROOT, "checkpoints", "stage2_qwen_lora")
        if os.path.exists(default_stage2):
            models_to_evaluate.append(("Baseline (Stage 2)", default_stage2))
            print(f"Using default Stage 2 checkpoint: {default_stage2}")
    
    if args.graph_structure_checkpoint:
        models_to_evaluate.append(("Graph Structure Pre-trained", args.graph_structure_checkpoint))
    
    if args.multi_task_checkpoint:
        models_to_evaluate.append(("Multi-task Trained", args.multi_task_checkpoint))
    
    if not models_to_evaluate:
        print("No checkpoints specified. Please provide at least one checkpoint.")
        print("Usage: python investigate_performance.py --baseline_checkpoint /path/to/checkpoint")
        return
    
    # Evaluate all models
    results = []
    for model_name, checkpoint_dir in models_to_evaluate:
        model, adapter = investigator.load_model(checkpoint_dir, model_name)
        result = investigator.evaluate_model(model, adapter, model_name)
        results.append(result)
        
        # Clean up
        if model is not None:
            del model
        if adapter is not None:
            del adapter
        torch.cuda.empty_cache()
    
    # Compare models
    df = investigator.compare_models(results)
    
    # Generate report
    investigator.generate_report(df, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Performance Investigation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
