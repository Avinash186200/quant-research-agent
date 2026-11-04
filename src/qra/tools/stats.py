"""Sharpe, probabilistic Sharpe and deflated Sharpe.

References: Bailey and Lopez de Prado, "The Sharpe Ratio Efficient Frontier"
(2012) for PSR, and "The Deflated Sharpe Ratio" (2014) for DSR. All Sharpe
ratios inside PSR/DSR are per-period (not annualised).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329
ANN = 252


def sharpe(returns, ann: int = ANN) -> float:
    x = np.asarray(returns, dtype=float)
    if x.size < 2:
        return 0.0
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 or not np.isfinite(sd) else float(x.mean() / sd * math.sqrt(ann))


def max_drawdown(returns) -> float:
    eq = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()) if eq.size else 0.0


def probabilistic_sharpe(returns, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > sr_benchmark) given the sample's skew and kurtosis."""
    x = np.asarray(returns, dtype=float)
    t = x.size
    if t < 3 or x.std(ddof=1) == 0:
        return 0.5
    sr = x.mean() / x.std(ddof=1)
    g3 = float(stats.skew(x))
    g4 = float(stats.kurtosis(x, fisher=False))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    denom = math.sqrt(max(denom, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period SR across n independent zero-skill trials."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(max(sr_variance, 0.0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns, n_trials: int, trial_sharpes=None) -> dict:
    """DSR of the selected strategy's returns.

    ``trial_sharpes`` are per-period SRs of every trial, if known; otherwise
    the variance of SR under the null (about 1/T) is used.
    """
    x = np.asarray(returns, dtype=float)
    t = max(x.size, 2)
    n_trials = max(int(n_trials), 1)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        var = 1.0 / t
    sr0 = expected_max_sharpe(n_trials, var)
    return {
        "n_trials": n_trials,
        "observed_sharpe_annual": round(sharpe(x), 4),
        "benchmark_sharpe_annual": round(sr0 * math.sqrt(ANN), 4),
        "psr": round(probabilistic_sharpe(x, 0.0), 4),
        "dsr": round(probabilistic_sharpe(x, sr0), 4),
    }
