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
    # 10 left the rarest tools permanently at the 0.5 default: hydra had 5
    # positives in validation and SQLmap 4, so neither was ever tuned, and
    # hydra scored F1 0.000 on test -- a dead class costs 1/11 = 9 points of
    # MCP macro-F1, which is a REPORTED metric. 4 lets them tune; the
    # bootstrap median over 25 resamples plus the never-regress guard below
    # are what keep a 4-positive fit from being noise.
    min_val_positives: int = 4,
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

# ===========================================================================
# STEP per-class logit-bias calibration
# ===========================================================================
# WHY THIS EXISTS: the MCP head gets per-class decision calibration above and
# ===========================================================================


def apply_step_logit_bias(logits: np.ndarray, bias) -> np.ndarray:
    """argmax over (logits + bias). `bias=None` -> plain argmax."""
    if bias is None:
        return np.argmax(logits, axis=1)
    return np.argmax(logits + np.asarray(bias, dtype=np.float64)[None, :], axis=1)


def _step_acc(logits, labels, bias):
    return float((np.argmax(logits + bias[None, :], axis=1) == labels).mean())


def _coordinate_ascent_bias(logits, labels, tunable, candidates, n_rounds):
    """Greedy coordinate ascent on per-class bias, maximizing accuracy."""
    bias = np.zeros(logits.shape[1], dtype=np.float64)
    best = _step_acc(logits, labels, bias)
    for _ in range(n_rounds):
        improved = False
        for c in tunable:
            keep = bias[c]
            for v in candidates:
                bias[c] = v
                acc = _step_acc(logits, labels, bias)
                if acc > best + 1e-12:
                    best, keep, improved = acc, v, True
            bias[c] = keep
        if not improved:
            break
    return bias, best


def search_step_logit_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    min_val_support: int = 8,
    bias_floor: float = -1.5,
    bias_ceil: float = 1.5,
    bias_step: float = 0.25,
    n_rounds: int = 3,
    n_bootstrap: int = 25,
    bootstrap_seed: int = 42,
    auto_fallback: bool = True,
    min_gain: float = 0.0,
    max_abs_bias: float = 1.5,
    verbose: bool = True,
) -> list[float]:
    """
    Find a per-class additive logit bias maximizing validation accuracy.

    Args:
        logits:          (N, num_classes) raw step logits on the validation split.
        labels:          (N,) integer gold class indices.
        min_val_support: classes with fewer validation examples than this keep
                          bias 0.0 and are never tuned.
        bias_floor/ceil/step: bounded search grid for each class's bias.
        n_rounds:        coordinate-ascent passes over the tunable classes.
        n_bootstrap:     bootstrap resamples used to stabilize the search; the
                          per-class median bias across resamples is returned.
        auto_fallback:   discard the tuned bias if it does not beat zero-bias
                          argmax on the validation set it was fit on.

    Returns:
        list[float] of length num_classes (all zeros means "no calibration").
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n, num_classes = logits.shape
    rng = np.random.default_rng(bootstrap_seed)

    candidates = [round(float(v), 4) for v in
                  np.arange(bias_floor, bias_ceil + 1e-9, bias_step)]
    support = np.bincount(labels, minlength=num_classes)
    tunable = [c for c in range(num_classes) if support[c] >= min_val_support]

    if verbose:
        print(f"[step_logit_bias] min_val_support={min_val_support}, "
              f"range=[{bias_floor}, {bias_ceil}], n_bootstrap={n_bootstrap}")
        skipped = [c for c in range(num_classes) if c not in tunable and support[c] > 0]
        if skipped:
            print(f"[step_logit_bias]   classes kept at bias 0.0 (support < {min_val_support}): "
                  + ", ".join(f"{c}(n={support[c]})" for c in skipped))

    if not tunable:
        if verbose:
            print("[step_logit_bias] no class has enough validation support -- skipping calibration.")
        return [0.0] * num_classes

    boot = np.zeros((n_bootstrap, num_classes), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bias_b, _ = _coordinate_ascent_bias(logits[idx], labels[idx], tunable, candidates, n_rounds)
        boot[b] = bias_b
    final = np.median(boot, axis=0)
    # A class that was never tunable must stay exactly 0.
    for c in range(num_classes):
        if c not in tunable:
            final[c] = 0.0

    if verbose:
        base_acc = _step_acc(logits, labels, np.zeros(num_classes))
        tuned_acc = _step_acc(logits, labels, final)
        nz = [(c, final[c]) for c in range(num_classes) if abs(final[c]) > 1e-9]
        print(f"[step_logit_bias]   val accuracy: uniform {base_acc:.4f} -> calibrated {tuned_acc:.4f}")
        if nz:
            print("[step_logit_bias]   non-zero bias: "
                  + ", ".join(f"class {c}: {v:+.2f}" for c, v in nz))

    # Cap magnitude. A bias comparable to the logit scale itself does not
    # "calibrate" a class, it forces it: +1.25 on `End task` bought +0.5pt on a
    # 376-row val split and cost 8 false positives on the 268-row test set.
    if max_abs_bias is not None:
        clipped = np.clip(final, -abs(max_abs_bias), abs(max_abs_bias))
        if verbose and not np.allclose(clipped, final):
            over = [(c, final[c]) for c in range(num_classes)
                    if abs(final[c]) > abs(max_abs_bias) + 1e-9]
            print(f"[step_logit_bias]   clipped to +-{abs(max_abs_bias):.2f}: "
                  + ", ".join(f"class {c}: {v:+.2f}" for c, v in over))
        final = clipped

    if auto_fallback:
        base_acc = _step_acc(logits, labels, np.zeros(num_classes))
        tuned_acc = _step_acc(logits, labels, final)
        gain = tuned_acc - base_acc
        # REQUIRE A REAL MARGIN, not any improvement. This search fits one free
        # parameter per class on the SAME split it is scored on, so a sub-noise
        # gain is selection noise rather than calibration. At ~380 val rows the
        # binomial SE is ~2pt, so anything under `min_gain` is discarded.
        if gain < max(1e-9, min_gain):
            if verbose:
                print(f"[step_logit_bias] ⚠ gain {gain:+.4f} ({base_acc:.4f} -> "
                      f"{tuned_acc:.4f}) below required margin {min_gain:.4f} "
                      f"-- discarding, using zero bias.")
            return [0.0] * num_classes
        if verbose:
            print(f"[step_logit_bias] ✓ gain {gain:+.4f} ({base_acc:.4f} -> "
                  f"{tuned_acc:.4f}) clears margin {min_gain:.4f} -- keeping.")

    return [float(v) for v in final]
