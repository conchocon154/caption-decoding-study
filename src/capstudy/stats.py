"""Is one decoding strategy actually better, or is the gap sampling noise?

Every strategy is scored on the same test images, so the comparison is paired.
A paired bootstrap resamples images (not captions) and recomputes the gap, which
respects the fact that a hard image is hard for every decoder.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(baseline: np.ndarray, contender: np.ndarray, *,
                     rounds: int = 5000, seed: int = 42) -> dict:
    """Bootstrap confidence interval for `contender - baseline`."""
    if len(baseline) != len(contender):
        raise ValueError("paired test needs one score per image from each system")
    rng = np.random.default_rng(seed)
    n = len(baseline)
    differences = np.empty(rounds, dtype=float)
    for round_index in range(rounds):
        picks = rng.integers(0, n, size=n)
        differences[round_index] = contender[picks].mean() - baseline[picks].mean()

    observed = float(contender.mean() - baseline.mean())
    low, high = np.percentile(differences, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often the resampled gap crosses zero.
    proportion = float(np.mean(differences <= 0)) if observed > 0 else float(np.mean(differences >= 0))
    return {
        "observed_difference": round(observed, 5),
        "ci_low": round(float(low), 5),
        "ci_high": round(float(high), 5),
        "p_value": round(min(1.0, 2 * proportion), 4),
        "significant": bool(low > 0 or high < 0),
    }


def win_rate(baseline: np.ndarray, contender: np.ndarray) -> dict:
    """On how many individual images does each system win?

    A mean can be moved by a handful of images.  If two systems tie on most
    images and the average differs, that is worth knowing before acting on it.
    """
    contender_wins = int(np.sum(contender > baseline))
    baseline_wins = int(np.sum(baseline > contender))
    ties = int(np.sum(np.isclose(baseline, contender)))
    total = len(baseline)
    return {
        "contender_wins": contender_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "contender_win_pct": round(100 * contender_wins / total, 1),
        "identical_output_pct": round(100 * ties / total, 1),
    }
