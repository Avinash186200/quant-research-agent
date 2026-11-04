"""Trend following: long when the 20-bar mean is above the 100-bar mean.

Decided on the close, filled on the next bar. The rolling windows include
the current close, which is fine because the trade happens one bar later.
"""
import numpy as np

SPEC = {"name": "sma_crossover", "decides_at": "close", "fill": "next_bar"}


def signal(prices, fast=20, slow=100):
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    return np.sign(f - s)
