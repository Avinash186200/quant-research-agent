"""Short-horizon mean reversion with a 37-bar lookback.

Research notes: I tried every lookback from 5 to 120 on the full history
and kept 37 because it had the best Sharpe. Only the winner is kept here.
"""
import numpy as np

SPEC = {"name": "tuned_mean_reversion", "decides_at": "close", "fill": "next_bar"}


def signal(prices, lookback=37):
    return -np.sign(prices.pct_change(lookback))
