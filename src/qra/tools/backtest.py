"""Vectorised backtest, fill comparison and noise test.

Timing convention: ``signal[t]`` is the position decided at bar ``t``.

* ``fill="next_bar"``: the position is held over bar ``t+1``,
  so ``pnl[t] = signal[t-1] * r[t]``.
* ``fill="same_bar"``: the position earns bar ``t``'s own return,
  ``pnl[t] = signal[t] * r[t]``. That is only legitimate when the signal was
  decided at the open of bar ``t`` from data up to ``t-1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data import synthetic_prices
from .loader import Strategy
from .stats import max_drawdown, sharpe

SPLIT_FRAC = 0.6


def strategy_returns(prices: pd.Series, pos: pd.Series, fill: str) -> pd.Series:
    r = prices.pct_change().fillna(0.0)
    held = pos if fill == "same_bar" else pos.shift(1).fillna(0.0)
    return held * r


def split_index(n: int, frac: float = SPLIT_FRAC) -> int:
    return int(n * frac)


def resolve_params(strat: Strategy, prices: pd.Series, split: int) -> dict[str, Any]:
    """Params used for single-run checks: fitted if the strategy fits, else a grid point."""
    if strat.has_fit:
        return strat.fit(prices, split)
    grid = strat.param_grid
    return dict(grid[len(grid) // 2]) if grid else {}


def evaluate(strat: Strategy, prices: pd.Series, fill: str | None = None) -> dict[str, Any]:
    """Run the strategy the way its author reports it.

    If the module has a ``PARAM_GRID``, every grid point is run and the best
    test-period Sharpe is reported, which is exactly the habit the deflated
    Sharpe check is there to correct.
    """
    fill = fill or strat.spec["fill"]
    split = split_index(len(prices))
    grid = strat.param_grid
    if grid:
        trials = []
        for p in grid:
            ret = strategy_returns(prices, strat.signal(prices, p), fill).iloc[split:]
            trials.append((sharpe(ret), p, ret))
        best = max(trials, key=lambda t: t[0])
        params, test_ret = best[1], best[2]
        trial_sr = [s / np.sqrt(252) for s, _, _ in trials]
    else:
        params = strat.fit(prices, split) if strat.has_fit else {}
        pos = strat.signal(prices, params)
        test_ret = strategy_returns(prices, pos, fill).iloc[split:]
        trial_sr = None
    pos = strat.signal(prices, params)
    return {
        "params": params,
        "test_returns": test_ret,
        "trial_sharpes": trial_sr,
        "turnover": float(pos.diff().abs().iloc[split:].mean()),
        "split": split,
    }


def run_backtest(strat: Strategy, prices: pd.Series) -> dict[str, Any]:
    ev = evaluate(strat, prices)
    ret = ev["test_returns"]
    return {
        "strategy": strat.name,
        "fill": strat.spec["fill"],
        "decides_at": strat.spec["decides_at"],
        "n_bars_test": int(len(ret)),
        "params": _jsonable(ev["params"]),
        "n_trials_reported": strat.declared_trials,
        "sharpe_test": round(sharpe(ret), 4),
        "ann_return_test": round(float(ret.mean() * 252), 4),
        "max_drawdown_test": round(max_drawdown(ret), 4),
        "turnover": round(ev["turnover"], 4),
    }


def compare_fills(strat: Strategy, prices: pd.Series) -> dict[str, Any]:
    """Test-period Sharpe with same-bar vs next-bar fills, same params."""
    split = split_index(len(prices))
    params = resolve_params(strat, prices, split)
    pos = strat.signal(prices, params)
    same = sharpe(strategy_returns(prices, pos, "same_bar").iloc[split:])
    nxt = sharpe(strategy_returns(prices, pos, "next_bar").iloc[split:])
    return {
        "declared_fill": strat.spec["fill"],
        "sharpe_same_bar": round(same, 4),
        "sharpe_next_bar": round(nxt, 4),
        "gap": round(same - nxt, 4),
        "note": "A large positive gap means the edge lives in the bar the signal was computed on.",
    }


def noise_test(strat: Strategy, n_paths: int = 20, n_bars: int = 1500, seed: int = 1000) -> dict[str, Any]:
    """Run the full reporting procedure on zero-drift random walks.

    A correct, honestly reported strategy has mean Sharpe near zero here.
    """
    n_paths = int(min(max(n_paths, 5), 100))
    srs = []
    for i in range(n_paths):
        p = synthetic_prices(n=n_bars, seed=seed + i)
        srs.append(sharpe(evaluate(strat, p)["test_returns"]))
    a = np.asarray(srs)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.inf
    t = float(a.mean() / se) if se and np.isfinite(se) and se > 0 else 0.0
    return {
        "n_paths": n_paths,
        "mean_sharpe_on_noise": round(float(a.mean()), 4),
        "t_stat": round(t, 3),
        "flag": bool(a.mean() > 0.3 and t > 3.0),
    }


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = round(v, 6) if isinstance(v, float) else v
    return out
