"""Synthetic price data.

Every number in this repo is produced on synthetic data so the results are
reproducible without any vendor feed. A zero-drift random walk is the honest
null: no strategy should have an edge on it, so any edge an audit finds there
is a bug, not alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
    n: int = 1500,
    seed: int = 0,
    drift: float = 0.0,
    vol: float = 0.01,
    start: str = "2018-01-01",
) -> pd.Series:
    """Geometric random walk of daily closes on a business-day index."""
    if n < 50:
        raise ValueError("n must be at least 50 bars")
    rng = np.random.default_rng(seed)
    log_r = rng.normal(drift, vol, n)
    log_r[0] = 0.0
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(log_r)), index=idx, name="close")


def replace_returns_from(
    prices: pd.Series, k: int, rng: np.random.Generator, shock_sign: float = 0.0
) -> pd.Series:
    """Copy of ``prices`` where every log return from bar ``k`` on is redrawn.

    Bars ``0..k-1`` are bit-for-bit identical. If ``shock_sign`` is non-zero,
    bar ``k`` additionally gets an 8-sigma move in that direction, so any
    signal that reads bar ``k`` is very likely to change.
    """
    log_p = np.log(prices.to_numpy(dtype=float))
    r = np.diff(log_p, prepend=log_p[0])
    sd = float(np.std(r[1:])) or 0.01
    new_r = r.copy()
    new_r[k:] = rng.normal(0.0, sd, len(r) - k)
    if shock_sign:
        new_r[k] = 8.0 * sd * np.sign(shock_sign)
    out = np.exp(log_p[0] + np.cumsum(new_r))
    out[:k] = prices.to_numpy(dtype=float)[:k]
    return pd.Series(out, index=prices.index, name=prices.name)
