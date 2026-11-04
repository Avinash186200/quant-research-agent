"""Breakout acted on at the open: yesterday's close above the prior 20-bar high.

Decided at the open of bar t from data up to t-1, so a same-bar fill is
legitimate here.
"""
import numpy as np
import pandas as pd

SPEC = {"name": "breakout_open", "decides_at": "open", "fill": "same_bar"}


def signal(prices, lookback=20):
    prev = prices.shift(1)
    prior_high = prev.shift(1).rolling(lookback).max()
    prior_low = prev.shift(1).rolling(lookback).min()
    pos = np.where(prev > prior_high, 1.0, np.where(prev < prior_low, -1.0, np.nan))
    return pd.Series(pos, index=prices.index).ffill().fillna(0.0)
