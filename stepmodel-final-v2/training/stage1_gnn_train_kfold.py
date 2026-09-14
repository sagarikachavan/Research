"""Stage 1, k-fold ensemble variant.

Same model, same losses, same training loop as training/stage1_gnn_train.py
(it literally calls train_one_split() from that module for each fold) --
this script only changes *how many times* we train and *how we combine*
the results.

Why this exists
----------------
v2's single train/val machine split (149 train / 26 val machines, 239 val
rows) has two consequences visible directly in the v2 log:
  1. The per-epoch "best checkpoint" selection is noisy: val_step_acc
     swings between 0.45 and 0.81 across epochs with no clean monotonic
     trend, i.e. a lot of that variance is sampling noise from a 239-row
     val set, not real learning-curve signal.
  2. mcp_threshold_search.py's own logging shows *half the MCP classes*
     have fewer than 10 positive examples in that 239-row val set, so
     their thresholds fall back to an untuned 0.5 default -- not because
     the search is bad, but because there simply isn't enough val data.

Training K models on K different machine-level folds and averaging their
predicted probabilities at test time (a) gets every training machine used
for validation by exactly one fold, so the pooled out-of-fold (OOF)
predictions cover the *entire* training set (~1.5k rows) for threshold
search instead of 239, and (b) ensembling itself cancels out a good chunk
of the single-split variance in (1) -- this is the standard fix for
"my val set is too small to trust" and does not touch the model
architecture at all.

Usage:
    python training/stage1_gnn_train_kfold.py

Output:
    checkpoints/stage1_gnn_classifier_fold{0..K-1}.pt   -- one per fold
    checkpoints/stage1_gnn_classifier_ensemble_meta.pt  -- pooled OOF
                                                             thresholds +
                                                             per-fold paths
    output/stage1_kfold_ensemble.csv                    -- test predictions
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "core"), os.path.join(_ROOT, "data_prep"), os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    INPUT_TRAIN_JSON, INPUT_TEST_JSON, STAGE1_BATCH_SIZE, RANDOM_SEED,
    MCP_LABELS, STEP_LABELS, ROOT, CKPT_DIR, STAGE1_N_FOLDS,
)
from data_utils import load_from_input_json
from stage1_gnn_train import Stage1Dataset, collate, train_one_split
from mcp_threshold_search import search_per_class_thresholds
from sklearn.metrics import accuracy_score, f1_score


def make_folds(all_machines, n_folds, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_machines))
    shuffled = [all_machines[i] for i in perm]
    return [shuffled[i::n_folds] for i in range(n_folds)]


def ensemble_predict(models, loader, device, thresholds):
    """Average step softmax + MCP sigmoid across all fold models."""
    for m in models:
        m.eval()
    step_gold, mcp_gold = [], []
    step_prob_sum, mcp_prob_sum = None, None
    idx = 0
    with torch.no_grad():
        for graphs, step_idx, mcp_vec, sem_tokens, sem_mask in loader:
            graphs = graphs.to(device); step_idx = step_idx.to(device)
            mcp_vec = mcp_vec.to(device); sem_tokens = sem_tokens.to(device); sem_mask = sem_mask.to(device)
            edge_attr = getattr(graphs, "edge_attr", None)
            step_p_batch, mcp_p_batch = None, None
            for m in models:
                sl, ml, _ = m(graphs.x, graphs.edge_index, graphs.batch,
                              semantic_tokens=sem_tokens, semantic_mask=sem_mask, edge_attr=edge_attr)
                sp = torch.softmax(sl, dim=-1)
                mp = torch.sigmoid(ml)
                step_p_batch = sp if step_p_batch is None else step_p_batch + sp
                mcp_p_batch = mp if mcp_p_batch is None else mcp_p_batch + mp
            step_p_batch /= len(models)
            mcp_p_batch /= len(models)
            step_gold.append(step_idx.cpu().numpy())
            mcp_gold.append(mcp_vec.cpu().numpy())
            sp_np = step_p_batch.cpu().numpy()
            mp_np = mcp_p_batch.cpu().numpy()
            step_prob_sum = sp_np if step_prob_sum is None else np.concatenate([step_prob_sum, sp_np])
            mcp_prob_sum = mp_np if mcp_prob_sum is None else np.concatenate([mcp_prob_sum, mp_np])
    step_gold = np.concatenate(step_gold)
    mcp_gold = np.concatenate(mcp_gold)
    step_preds = step_prob_sum.argmax(-1)
    thr = np.array(thresholds, dtype=np.float32)
    mcp_preds = (mcp_prob_sum >= thr).astype(np.float32)
    metrics = {
        "step_accuracy": accuracy_score(step_gold, step_preds),
        "step_micro_f1": f1_score(step_gold, step_preds, average="micro", zero_division=0),
        "step_macro_f1": f1_score(step_gold, step_preds, average="macro", zero_division=0),
        "step_weighted_f1": f1_score(step_gold, step_preds, average="weighted", zero_division=0),
        "mcp_subset_accuracy": accuracy_score(mcp_gold, mcp_preds),
        "mcp_micro_f1": f1_score(mcp_gold, mcp_preds, average="micro", zero_division=0),
        "mcp_macro_f1": f1_score(mcp_gold, mcp_preds, average="macro", zero_division=0),
        "mcp_samples_f1": f1_score(mcp_gold, mcp_preds, average="samples", zero_division=0),
    }
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = STAGE1_N_FOLDS
    print(f"[Stage 1 k-fold] Device: {device}  Folds: {n_folds}")

    full_ds = Stage1Dataset(INPUT_TRAIN_JSON, split="train")
    examples = full_ds.examples
    all_machines = sorted(set(e["machine"] for e in examples))
    folds = make_folds(all_machines, n_folds, RANDOM_SEED + 7)

    test_pre = load_from_input_json(INPUT_TEST_JSON, "test")
    test_machines = set(e["machine"] for e in test_pre)
    for f in folds:
        assert not (set(f) & test_machines), "fold/TEST machine overlap detected"

    models = []
    oof_probs_parts, oof_gold_parts = [], []

    for k in range(n_folds):
        val_machines = set(folds[k])
        train_machines = set(all_machines) - val_machines
        train_idx = [i for i, e in enumerate(examples) if e["machine"] in train_machines]
        val_idx = [i for i, e in enumerate(examples) if e["machine"] in val_machines]
        ckpt_path = os.path.join(CKPT_DIR, f"stage1_gnn_classifier_fold{k}.pt")
        tag = f"[fold {k}/{n_folds - 1}]"
        print(f"{tag} train machines={len(train_machines)} ({len(train_idx)} rows)  "
              f"val machines={len(val_machines)} ({len(val_idx)} rows)")

        model, _mcp_w, _mcp_counts, val_probs, val_gold, val_metrics, _thr = train_one_split(
            full_ds, train_idx, val_idx, device, ckpt_path, tag=tag
        )
        models.append(model)
        oof_probs_parts.append(val_probs)
        oof_gold_parts.append(val_gold)
        print(f"{tag} val_step_acc={val_metrics['step_accuracy']:.4f}  "
              f"val_mcp_microF1={val_metrics['mcp_micro_f1']:.4f}")

    # Pooled out-of-fold probabilities cover every training row exactly
    # once (predicted by the one fold model that didn't train on it) --
    # this is the ~1.5k-row threshold search the single-split version
    # could never do with only 239 val rows.
    oof_probs = np.concatenate(oof_probs_parts, axis=0)
    oof_gold = np.concatenate(oof_gold_parts, axis=0)
    print(f"[Stage 1 k-fold] Pooled OOF rows for threshold search: {oof_probs.shape[0]}")
    rare = [i for i, c in enumerate(oof_gold.sum(axis=0)) if c < 15]
    thresholds = search_per_class_thresholds(oof_probs, oof_gold, rare_class_indices=rare)
    print(f"[Stage 1 k-fold] OOF-derived MCP thresholds: {[round(float(x), 2) for x in thresholds]}")

    torch.save({
        "fold_ckpts": [os.path.join(CKPT_DIR, f"stage1_gnn_classifier_fold{k}.pt") for k in range(n_folds)],
        "mcp_thresholds": [float(x) for x in thresholds],
        "n_folds": n_folds,
    }, os.path.join(CKPT_DIR, "stage1_gnn_classifier_ensemble_meta.pt"))

    test_ds = Stage1Dataset(INPUT_TEST_JSON, split="test")
    test_loader = DataLoader(test_ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_metrics = ensemble_predict(models, test_loader, device, thresholds)

    print(f"\n[Stage 1 k-fold] ===== ENSEMBLE TEST SET RESULTS ({n_folds} folds) =====")
    for name in ("step_accuracy", "step_micro_f1", "step_macro_f1", "step_weighted_f1",
                 "mcp_micro_f1", "mcp_macro_f1", "mcp_subset_accuracy", "mcp_samples_f1"):
        print(f"  {name:<20}: {test_metrics[name]:.4f}")
    combined = 0.5 * test_metrics["step_accuracy"] + 0.5 * test_metrics["mcp_micro_f1"]
    print(f"  {'combined_score':<20}: {combined:.4f}")


if __name__ == "__main__":
    main()
