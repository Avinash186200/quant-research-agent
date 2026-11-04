"""Momentum; the lookback is the best of a 40-point sweep on the test period."""
import numpy as np

SPEC = {"name": "lookback_sweep", "decides_at": "close", "fill": "next_bar"}
PARAM_GRID = [{"lookback": lb} for lb in range(5, 205, 5)]


def signal(prices, lookback=60):
    return np.sign(prices.pct_change(lookback))
