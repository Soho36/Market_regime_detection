"""Statistics that respect autocorrelation in weekly series.

- Circular-shift null: rotate one series against the other by a random offset (>= min_shift).
  Each series keeps its own autocorrelation; only their alignment is broken.
- Moving-block bootstrap: resample contiguous blocks so short-range dependence survives.
"""
import numpy as np


def pairs(x, y):
    return int((np.isfinite(x) & np.isfinite(y)).sum())


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]


def lagged(x, y, k):
    """Aligned pairs (x[t], y[t + k])."""
    n = len(x)
    if k >= 0:
        return x[:n - k], y[k:]
    return x[-k:], y[:n + k]


def cross_corr(x, y, lags):
    return np.array([corr(*lagged(x, y, k)) for k in lags])


def circular_shift_null(x, y, lags, draws, min_shift, seed):
    """Cross-correlations (draws x len(lags)) after rotating x against y."""
    rng = np.random.default_rng(seed)
    offsets = rng.integers(min_shift, len(x) - min_shift + 1, size=draws)
    return np.array([cross_corr(np.roll(x, s), y, lags) for s in offsets])


def p_two_sided(observed, null):
    null = np.abs(np.asarray(null))
    null = null[np.isfinite(null)]
    return (1 + np.sum(null >= abs(observed))) / (1 + len(null))


def block_bootstrap(x, y, stat, block, draws, seed):
    """stat(x*, y*) over moving-block resamples of the finite (x, y) pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(draws, -(-n // block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :n]
    return np.array([stat(x[i], y[i]) for i in idx])
