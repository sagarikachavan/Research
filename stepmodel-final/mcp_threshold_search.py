"""
MCP per-class threshold utilities.

predict_with_per_class_thresholds: applies a separate sigmoid threshold
per MCP label to a batch of probability matrices, returning a binary
prediction array. This is used by evaluate.py when the Stage-1 checkpoint
stores per-class calibrated thresholds alongside the model weights.

If no calibrated thresholds are available, the checkpoint loader falls back
to the uniform default (MCP_DECISION_THRESHOLD = 0.5 for all labels).

--------------------------------------------------------------------------
FIX (see rationale in chat / run log): the previous search_per_class_thresholds
independently grid-searched an F1-maximizing threshold for EVERY class,
including ones with only 1-6 positive examples in the validation split
(e.g. SQLmap=2, Smb client=1, hydra=4, John-the-ripper=6 in a 239-row val
set). With that few positives, "F1-maximizing" just fits noise: a
threshold like 0.05 for SQLmap isn't a real calibration, it's overfitting
to one lucky validation row. That produced thresholds which, when applied
to the untouched test set, made MCP Subset Accuracy collapse to 4.5% and
even made overall Micro F1 WORSE than the untuned 0.5-for-all baseline
(0.4581 vs 0.6584) -- on the very validation set it was "optimized" on.

Two independent defenses are added:

  1. MIN-SUPPORT GATE. A class's threshold is only searched if it has at
     least `min_val_positives` positive examples in the validation split
     (default 10). Below that, the threshold stays at the safe default
     (0.5) instead of being fit to a handful of points.

  2. BOOTSTRAP-STABILIZED SEARCH + BOUNDED RANGE for classes that DO pass
     the gate: instead of one single-shot grid search on the raw
     validation set (which is still noisy even with 10-40 positives),
     the search is repeated over `n_bootstrap` bootstrap resamples of the
     validation set and the MEDIAN of the per-resample best thresholds is
     used. Candidates are also bounded to [0.15, 0.85] by default so the
     search can never pick a near-0/near-1 threshold that effectively
     always/never fires.

  3. SAFETY-NET CHECK (validate_thresholds_vs_baseline): after computing
     thresholds, compare their overall micro-F1 on the validation set
     against the untuned 0.5-for-all baseline. If the "optimized"
     thresholds are actually worse, this is now a visible warning instead
     of a silent checkpoint overwrite -- callers should heed it and (by
     default, if `search_per_class_thresholds(..., auto_fallback=True)`)
     the function will revert to 0.5-for-all automatically in that case.
--------------------------------------------------------------------------
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


def _grid_search_threshold(probs_col, targets_col, candidates):
    from sklearn.metrics import f1_score
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        preds = (probs_col >= thr).astype(int)
        score = f1_score(targets_col, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = score, thr
    return best_thr, best_f1


def search_per_class_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    candidates: list[float] | None = None,
    rare_class_indices: list[int] | None = None,
    min_val_positives: int = 10,
    candidate_floor: float = 0.15,
    candidate_ceil: float = 0.85,
    n_bootstrap: int = 25,
    bootstrap_seed: int = 42,
    auto_fallback: bool = True,
    verbose: bool = True,
) -> list[float]:
    """
    Grid-search a stabilized, support-gated F1-maximizing threshold for
    each MCP label independently.

    Args:
        probs:              (N, num_labels) sigmoid probabilities.
        targets:            (N, num_labels) binary ground truth.
        candidates:         threshold grid for well-supported classes
                             (default: candidate_floor..candidate_ceil in
                             0.05 steps). Values outside [floor, ceil] are
                             never tried, to avoid degenerate thresholds.
        rare_class_indices: classes explicitly known to be rare in the
                             TRAINING set. These still go through the same
                             min-support gate on the VALIDATION set (a
                             class can be common in train but still have
                             too few validation positives to trust), but
                             are allowed a slightly wider (still bounded)
                             candidate range: [max(0.05, floor-0.10), ceil].
        min_val_positives:  classes with fewer positive examples than this
                             in `targets` keep the 0.5 default -- their
                             threshold is not searched at all.
        candidate_floor/ceil: hard bounds on any searched threshold.
        n_bootstrap:        number of bootstrap resamples of the
                             validation set used to stabilize the search
                             for each class that passes the support gate.
                             The median of the per-resample best threshold
                             is used as the final value.
        auto_fallback:      if True, after computing all thresholds, checks
                             overall micro-F1 against the untuned 0.5
                             baseline on `probs`/`targets`. If the tuned
                             set is worse, reverts to 0.5-for-all and
                             prints a warning instead of silently keeping
                             a regression.
        verbose:            print per-class support / decision.

    Returns:
        List of length num_labels with the final threshold per label.
    """
    from sklearn.metrics import f1_score

    if candidates is None:
        candidates = [round(t, 2) for t in np.arange(candidate_floor, candidate_ceil + 1e-9, 0.05)]
    if rare_class_indices is None:
        rare_class_indices = []

    num_labels = probs.shape[1]
    n = probs.shape[0]
    rng = np.random.default_rng(bootstrap_seed)

    final_thresholds = []
    if verbose:
        print(f"[mcp_threshold_search] min_val_positives={min_val_positives}, "
              f"range=[{candidate_floor}, {candidate_ceil}], n_bootstrap={n_bootstrap}")

    for label_idx in range(num_labels):
        support = int(targets[:, label_idx].sum())

        if support < min_val_positives:
            final_thresholds.append(0.5)
            if verbose:
                print(f"  label {label_idx:2d}: val_positives={support:3d}  < min_val_positives "
                      f"-> keeping default 0.5 (not enough support to tune safely)")
            continue

        if label_idx in rare_class_indices:
            class_candidates = [round(t, 2) for t in
                                 np.arange(max(0.05, candidate_floor - 0.10), candidate_ceil + 1e-9, 0.02)]
        else:
            class_candidates = candidates

        # Bootstrap-stabilized search: repeat the grid search on resamples
        # of the validation set and take the median best threshold, instead
        # of trusting a single point estimate on possibly-noisy data.
        boot_thresholds = []
        col_probs, col_targets = probs[:, label_idx], targets[:, label_idx]
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            thr, _ = _grid_search_threshold(col_probs[idx], col_targets[idx], class_candidates)
            boot_thresholds.append(thr)

        final_thr = float(np.median(boot_thresholds))
        final_thresholds.append(final_thr)

        if verbose:
            direct_thr, direct_f1 = _grid_search_threshold(col_probs, col_targets, class_candidates)
            print(f"  label {label_idx:2d}: val_positives={support:3d}  "
                  f"bootstrap_median_thr={final_thr:.2f}  (single-shot grid search gave {direct_thr:.2f})")

    if auto_fallback:
        final_thresholds = validate_thresholds_vs_baseline(probs, targets, final_thresholds, verbose=verbose)

    return final_thresholds


def validate_thresholds_vs_baseline(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: list[float],
    baseline: float = 0.5,
    verbose: bool = True,
) -> list[float]:
    """
    Safety net: compares overall micro-F1 of `thresholds` against the
    untuned uniform `baseline` on the SAME probs/targets they were fit on.
    If the "optimized" thresholds are actually worse than just using 0.5
    for everything, this reverts to the uniform baseline and prints a
    warning -- so a regression like the one in the original bug report
    (optimized micro-F1 0.4581 vs baseline 0.6584) can never silently make
    it into a saved checkpoint again.
    """
    from sklearn.metrics import f1_score

    tuned_preds = predict_with_per_class_thresholds(probs, thresholds)
    tuned_f1 = f1_score(targets, tuned_preds, average="micro", zero_division=0)

    baseline_thresholds = [baseline] * probs.shape[1]
    baseline_preds = predict_with_per_class_thresholds(probs, baseline_thresholds)
    baseline_f1 = f1_score(targets, baseline_preds, average="micro", zero_division=0)

    if tuned_f1 < baseline_f1:
        if verbose:
            print(f"[mcp_threshold_search] ⚠ Per-class thresholds gave WORSE micro-F1 than uniform "
                  f"{baseline} ({tuned_f1:.4f} vs {baseline_f1:.4f}) on the validation set they were "
                  f"fit on -- reverting to uniform {baseline} for all classes.")
        return baseline_thresholds

    if verbose:
        print(f"[mcp_threshold_search] ✓ Per-class thresholds beat uniform {baseline} on validation "
              f"({tuned_f1:.4f} vs {baseline_f1:.4f}) -- keeping tuned thresholds.")
    return thresholds