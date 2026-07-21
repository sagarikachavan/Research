"""
MCP per-class threshold utilities.

predict_with_per_class_thresholds: applies a separate sigmoid threshold
per MCP label to a batch of probability matrices, returning a binary
prediction array. This is used by evaluate.py when the Stage-1 checkpoint
stores per-class calibrated thresholds alongside the model weights.

If no calibrated thresholds are available, the checkpoint loader falls back
to the uniform default (MCP_DECISION_THRESHOLD = 0.5 for all labels).
"""

import numpy as np


def predict_with_per_class_thresholds(
    probs: np.ndarray,
    thresholds: list[float],
) -> np.ndarray:
    """
    Apply per-class thresholds to MCP sigmoid probabilities.

    Args:
        probs:      (N, num_labels) float array of sigmoid probabilities.
        thresholds: list of length num_labels, one threshold per MCP label.

    Returns:
        (N, num_labels) float32 binary array.
    """
    thr = np.array(thresholds, dtype=np.float32)   # (num_labels,)
    return (probs >= thr).astype(np.float32)


def search_per_class_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    candidates: list[float] | None = None,
) -> list[float]:
    """
    Grid-search the F1-maximising threshold for each MCP label independently.

    Args:
        probs:      (N, num_labels) float array of sigmoid probabilities.
        targets:    (N, num_labels) float binary ground-truth array.
        candidates: threshold values to try (default: 0.1 … 0.9 in 0.05 steps).

    Returns:
        List of length num_labels with the best threshold per label.
    """
    from sklearn.metrics import f1_score

    if candidates is None:
        candidates = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    num_labels = probs.shape[1]
    best_thresholds = []

    for label_idx in range(num_labels):
        best_thr = 0.5
        best_f1 = -1.0
        for thr in candidates:
            preds = (probs[:, label_idx] >= thr).astype(int)
            score = f1_score(targets[:, label_idx], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thr = thr
        best_thresholds.append(best_thr)

    return best_thresholds
