"""Z-score momentum, traded at the open.

The rolling mean and std are computed on a window that ends at today's
close, but the trade is supposed to happen at today's open.
"""
import numpy as np

SPEC = {"name": "zscore_open", "decides_at": "open", "fill": "same_bar"}


def signal(prices, window=20, entry=0.5):
    mu = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    z = (prices - mu) / sd
    return np.where(z > entry, 1.0, np.where(z < -entry, -1.0, 0.0))
