"""Sufficient statistics for merging repeat-shards of the simulation.

The authors' `tools.summation(COV, SIZE)` (Main/tools.py:99) needs only three
things from a set of repeats:

    mar       = mean over all (repeat, test) entries
    size_std  = std over repeats of the per-repeat mean size      <- the "Std" column
    local_cov = mean_j | mean_repeat COV[repeat, j] - (1-alpha) |

All three are recoverable from per-repeat means plus a per-test-point sum over
repeats, so a shard never has to ship its full (repeats x testN) arrays.
"""

import numpy as np


def reduce_pair(cov, size, lo, hi):
    """cov/size have shape (..., repeats, testN); keep only repeats [lo, hi)."""
    cov = np.asarray(cov)[..., lo:hi, :]
    size = np.asarray(size)[..., lo:hi, :]
    return {
        "n_repeats": int(hi - lo),
        "cov_mean_per_repeat": np.round(cov.mean(axis=-1), 10).tolist(),
        "size_mean_per_repeat": np.round(size.mean(axis=-1), 10).tolist(),
        "cov_sum_over_repeats": np.round(cov.sum(axis=-2), 10).tolist(),
        "size_sum_over_repeats": np.round(size.sum(axis=-2), 10).tolist(),
    }


def merge(parts):
    """Concatenate per-repeat vectors and add the per-test-point sums."""
    total = sum(p["n_repeats"] for p in parts)
    cov_mean = np.concatenate([np.asarray(p["cov_mean_per_repeat"]) for p in parts], axis=-1)
    size_mean = np.concatenate([np.asarray(p["size_mean_per_repeat"]) for p in parts], axis=-1)
    cov_sum = sum(np.asarray(p["cov_sum_over_repeats"]) for p in parts)
    size_sum = sum(np.asarray(p["size_sum_over_repeats"]) for p in parts)
    return total, cov_mean, size_mean, cov_sum, size_sum


def summarise(parts, alpha=0.1):
    """Reproduce `tools.summation` outputs [mar, size, size_std, local_cov]."""
    total, cov_mean, size_mean, cov_sum, _ = merge(parts)
    mar = cov_mean.mean(axis=-1)
    size = size_mean.mean(axis=-1)
    size_std = size_mean.std(axis=-1)
    local_mar = cov_sum / total
    local_cov = np.abs(local_mar - (1 - alpha)).mean(axis=-1)
    return {
        "n_repeats": total,
        "marginal": np.asarray(mar).tolist(),
        "size": np.asarray(size).tolist(),
        "std": np.asarray(size_std).tolist(),
        "cond_miscoverage": np.asarray(local_cov).tolist(),
    }
