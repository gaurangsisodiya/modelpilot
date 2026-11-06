"""Paired bootstrap for comparing two strategies on the same queries."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Mean difference (a - b) and its percentile confidence interval."""

    diff: float
    low: float
    high: float


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> BootstrapResult:
    """Bootstrap the mean of a - b, resampling queries (pairs) with replacement.

    `a` and `b` are per-query scores of two strategies on the same queries, in the same order.
    Pairing keeps each query's difficulty in both arms, so the interval reflects only how
    the strategies differ, not which queries happened to be drawn.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length")
    diffs = a - b
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    resampled_means = diffs[samples].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(resampled_means, [tail, 100 - tail])
    return BootstrapResult(diff=float(diffs.mean()), low=float(low), high=float(high))
