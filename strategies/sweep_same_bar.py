"""Fast momentum, best of a lookback sweep, credited with the decision bar's return."""
import numpy as np

SPEC = {"name": "sweep_same_bar", "decides_at": "close", "fill": "same_bar"}
PARAM_GRID = [{"lookback": lb} for lb in range(1, 21)]


def signal(prices, lookback=3):
    return np.sign(prices.pct_change(lookback))
