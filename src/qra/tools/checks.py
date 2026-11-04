"""Behavioural leakage checks. Each one perturbs data and watches what moves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data import replace_returns_from
from .backtest import resolve_params, split_index
from .loader import Strategy


def check_signal_timing(
    strat: Strategy, prices: pd.Series, n_points: int = 12, seed: int = 0
) -> dict[str, Any]:
    """Redraw every bar from ``k`` on and see which signals change.

    * If ``signal[t]`` changes for some ``t < k``, the signal reads the future.
    * If ``signal[k]`` changes, the signal reads bar ``k`` itself. That is fine
      for a close-decided, next-bar-filled strategy and a leak otherwise.
    """
    rng = np.random.default_rng(seed)
    n = len(prices)
    split = split_index(n)
    params = resolve_params(strat, prices, split)
    base = strat.signal(prices, params).to_numpy()
    lo, hi = int(0.3 * n), n - 5
    ks = sorted(rng.choice(np.arange(lo, hi), size=min(n_points, hi - lo), replace=False).tolist())
    future_hits, current_hits = [], []
    for k in ks:
        sign = -np.sign(base[k]) if base[k] != 0 else 1.0
        for shock in (sign, -sign):
            pert = replace_returns_from(prices, k, rng, shock_sign=shock)
            s2 = strat.signal(pert, params).to_numpy()
            if not np.allclose(base[:k], s2[:k], atol=1e-9):
                future_hits.append(int(k))
                break
            if abs(base[k] - s2[k]) > 1e-9:
                current_hits.append(int(k))
                break
    reads_future = bool(future_hits)
    reads_current = bool(current_hits)
    d, f = strat.spec["decides_at"], strat.spec["fill"]
    issues = []
    if reads_future:
        issues.append("future_leak")
    if reads_current and d == "open":
        issues.append("current_bar_window")
    if reads_current and d == "close" and f == "same_bar":
        issues.append("same_bar_fill")
    return {
        "decides_at": d,
        "fill": f,
        "points_tested": len(ks),
        "reads_future_bars": reads_future,
        "future_leak_at": future_hits[:5],
        "reads_decision_bar": reads_current,
        "decision_bar_hits": len(current_hits),
        "issues": issues,
    }


def check_fit_window(strat: Strategy, prices: pd.Series, n_perturb: int = 10, seed: int = 0) -> dict[str, Any]:
    """Redraw the test period and refit. Train-only fitting must not move."""
    if not strat.has_fit:
        return {"applicable": False, "reason": "strategy has no fit()"}
    rng = np.random.default_rng(seed)
    split = split_index(len(prices))
    p0 = strat.fit(prices, split)
    changed: set[str] = set()
    for _ in range(n_perturb):
        p1 = strat.fit(replace_returns_from(prices, split, rng), split)
        for key in set(p0) | set(p1):
            a, b = p0.get(key), p1.get(key)
            try:
                same = bool(np.allclose(a, b, equal_nan=True))
            except TypeError:
                same = a == b
            if not same:
                changed.add(key)
    return {
        "applicable": True,
        "split": split,
        "params_on_original": {k: _num(v) for k, v in p0.items()},
        "params_moved_when_test_data_changed": sorted(changed),
        "issues": ["test_fitted_threshold"] if changed else [],
    }


def check_purge_embargo(strat: Strategy, n: int = 1500) -> dict[str, Any]:
    """Count training labels whose horizon overlaps a test fold."""
    if not strat.has_cv:
        return {"applicable": False, "reason": "strategy has no cv_splits()"}
    h = int(strat.spec.get("label_horizon", 1))
    emb = int(strat.spec.get("embargo", 0))
    folds = []
    total = 0
    for train, test in strat.cv_splits(n):
        train = np.asarray(train)
        test = np.asarray(test)
        t0, t1 = int(test.min()), int(test.max())
        # label at i spans bars i+1 .. i+h
        before = (train < t0) & (train + h >= t0)
        after = (train > t1) & (train <= t1 + max(h, 1))
        bad = int(before.sum() + after.sum())
        total += bad
        folds.append({"test": [t0, t1], "overlapping_train_labels": bad})
    return {
        "applicable": True,
        "label_horizon": h,
        "declared_embargo": emb,
        "folds": folds,
        "overlapping_train_labels_total": total,
        "required_purge": h,
        "issues": ["label_overlap"] if total > 0 else [],
    }


def _num(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    return round(v, 6) if isinstance(v, float) else v
