"""5-bar momentum, decided on the close and credited with that same bar's return."""
import numpy as np

SPEC = {"name": "momentum_same_bar", "decides_at": "close", "fill": "same_bar"}


def signal(prices, lookback=5):
    return np.sign(prices.pct_change(lookback))
