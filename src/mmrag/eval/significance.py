"""Is the difference between two systems more than noise? Paired tests over queries.

Both systems are scored on the same queries, so the tests work on the per-query
differences: a randomization (sign-flip) test for the p-value and a bootstrap
over queries for the confidence interval.
"""

from __future__ import annotations

import numpy as np


def mean_ci(values: list[float], n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Mean with a 95% bootstrap confidence interval (resampling queries)."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n": len(x), "mean": float(x.mean()), "ci95": [float(low), float(high)]}


def paired_test(
    new: dict[str, float], old: dict[str, float], n_resamples: int = 10_000, seed: int = 0
) -> dict:
    """Compare two systems' per-query scores ({query_id: score}, same queries).

    Returns the mean difference new - old, its 95% bootstrap interval and the
    two-sided p-value of a paired randomization test: under "no difference" the
    sign of each query's difference is arbitrary, so signs are flipped at random and
    the p-value is the share of flips with a mean difference at least as large.
    """
    if set(new) != set(old):
        raise ValueError("The two runs were not scored on the same queries")
    ids = sorted(new)
    diff = np.array([new[q] - old[q] for q in ids], dtype=np.float64)
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(diff)))
    flipped = np.abs((signs * diff).mean(axis=1))
    # +1 in both: the observed assignment is one of the possible ones.
    p = (int((flipped >= abs(observed) - 1e-12).sum()) + 1) / (n_resamples + 1)
    return {**mean_ci(list(diff), n_resamples, seed), "mean": observed, "p_value": float(p)}
