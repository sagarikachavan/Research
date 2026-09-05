"""
Ensemble evaluation for Stage 1 GNN.
Loads multiple trained models and averages their predictions.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, f1_score

from config import (
    INPUT_TEST_JSON, STEP_LABELS, MCP_LABELS, 
    TEXT_ENCODER_NAME, TEXT_EMB_DIM,
    GNN_HIDDEN, GNN_LAYERS, GNN_OUT_DIM, FUSION_HIDDEN,
    GNN_HEADS, GNN_DROPOUT, EDGE_ATTR_DIM
)
from data_utils import CONTEXT_COLUMNS, load_from_input_json, _embed_texts
from graph_encoder import Stage1Classifier
from stage1_gnn_train import Stage1Dataset, collate

# Ensemble configuration
ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]
ENSEMBLE_CKPT_DIR = "checkpoints/ensemble_stage1"

def load_ensemble_models(device):
    """Load all ensemble models."""
    models = []
    for seed in ENSEMBLE_SEEDS:
        ckpt_path = f"{ENSEMBLE_CKPT_DIR}/seed_{seed}.pt"
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found for seed {seed}: {ckpt_path}")
            continue
        
        print(f"Loading model with seed {seed}...")
        model = Stage1Classifier(edge_dim=EDGE_ATTR_DIM).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle different checkpoint structures
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            val_score = checkpoint.get('val_score', checkpoint.get('score', 'unknown'))
        else:
            model.load_state_dict(checkpoint)
            epoch = 'unknown'
            val_score = 'unknown'
        
        model.eval()
        models.append(model)
        print(f"  ✓ Loaded (epoch {epoch}, val_score {val_score})")
    
    return models

def ensemble_predict(models, loader, device):
    """Get ensemble predictions by averaging model outputs."""
    all_step_probs = []
    all_mcp_probs = []
    all_step_labels = []
    all_mcp_labels = []
    
    with torch.no_grad():
        for graphs, field_embs, step_idx, mcp_vec in loader:
            graphs = graphs.to(device)
            field_embs = field_embs.to(device)
            edge_attr = getattr(graphs, 'edge_attr', None)
            
            # Collect predictions from all models
            step_logits_list = []
            mcp_logits_list = []
            
            for model in models:
                step_logits, mcp_logits, _ = model(
                    graphs.x, graphs.edge_index, graphs.batch, field_embs,
                    edge_attr=edge_attr
                )
                step_logits_list.append(step_logits)
                mcp_logits_list.append(mcp_logits)
            
            # Average logits across models
            avg_step_logits = torch.stack(step_logits_list).mean(dim=0)
            avg_mcp_logits = torch.stack(mcp_logits_list).mean(dim=0)
            
            # Convert to probabilities
            step_probs = torch.softmax(avg_step_logits, dim=-1)
            mcp_probs = torch.sigmoid(avg_mcp_logits)
            
            all_step_probs.append(step_probs.cpu().numpy())
            all_mcp_probs.append(mcp_probs.cpu().numpy())
            all_step_labels.append(step_idx.cpu().numpy())
            all_mcp_labels.append(mcp_vec.cpu().numpy())
    
    return (
        np.concatenate(all_step_probs),
        np.concatenate(all_mcp_probs),
        np.concatenate(all_step_labels),
        np.concatenate(all_mcp_labels)
    )

def evaluate_ensemble(step_probs, mcp_probs, step_gold, mcp_gold, mcp_threshold=0.5):
    """Evaluate ensemble predictions."""
    # Step predictions
    step_preds = step_probs.argmax(axis=-1)
    step_acc = accuracy_score(step_gold, step_preds)
    step_macro_f1 = f1_score(step_gold, step_preds, average='macro', zero_division=0)
    step_weighted_f1 = f1_score(step_gold, step_preds, average='weighted', zero_division=0)
    
    # MCP predictions
    mcp_preds = (mcp_probs >= mcp_threshold).astype(int)
    mcp_subset_acc = accuracy_score(mcp_gold, mcp_preds)
    mcp_micro_f1 = f1_score(mcp_gold, mcp_preds, average='micro', zero_division=0)
    mcp_macro_f1 = f1_score(mcp_gold, mcp_preds, average='macro', zero_division=0)
    mcp_samples_f1 = f1_score(mcp_gold, mcp_preds, average='samples', zero_division=0)
    
    return {
        'step_accuracy': step_acc,
        'step_macro_f1': step_macro_f1,
        'step_weighted_f1': step_weighted_f1,
        'mcp_subset_accuracy': mcp_subset_acc,
        'mcp_micro_f1': mcp_micro_f1,
        'mcp_macro_f1': mcp_macro_f1,
        'mcp_samples_f1': mcp_samples_f1,
    }

def main():
    """Run ensemble evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test data using Stage1Dataset
    print(f"\nLoading test data from {INPUT_TEST_JSON}...")
    test_ds = Stage1Dataset(INPUT_TEST_JSON, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)
    print(f"Loaded {len(test_ds)} test examples")
    
    # Load ensemble models
    print(f"\nLoading ensemble models from {ENSEMBLE_CKPT_DIR}/...")
    models = load_ensemble_models(device)
    
    if len(models) == 0:
        print("✗ No models found. Train ensemble first with:")
        print("  python train_ensemble_stage1.py")
        return
    
    print(f"\nLoaded {len(models)} ensemble models")
    
    # Get ensemble predictions
    print("\nComputing ensemble predictions...")
    step_probs, mcp_probs, step_gold, mcp_gold = ensemble_predict(models, test_loader, device)
    
    # Evaluate
    print("\nEvaluating ensemble predictions...")
    metrics = evaluate_ensemble(step_probs, mcp_probs, step_gold, mcp_gold)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"ENSEMBLE EVALUATION RESULTS ({len(models)} models)")
    print(f"{'='*60}")
    print(f"\nSTEP CLASSIFICATION")
    print(f"  Accuracy      : {metrics['step_accuracy']:.4f}")
    print(f"  Macro F1      : {metrics['step_macro_f1']:.4f}")
    print(f"  Weighted F1   : {metrics['step_weighted_f1']:.4f}")
    
    print(f"\nMCP TOOL CLASSIFICATION")
    print(f"  Subset Accuracy : {metrics['mcp_subset_accuracy']:.4f}")
    print(f"  Micro F1       : {metrics['mcp_micro_f1']:.4f}")
    print(f"  Macro F1       : {metrics['mcp_macro_f1']:.4f}")
    print(f"  Samples F1     : {metrics['mcp_samples_f1']:.4f}")
    
    combined = metrics['step_accuracy'] * 0.5 + metrics['mcp_micro_f1'] * 0.5
    print(f"\nCombined Score : {combined:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
