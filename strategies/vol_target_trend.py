"""60-bar momentum, sized to a 10% annual volatility target.

Decided on the close, filled on the next bar.
"""
import numpy as np

SPEC = {"name": "vol_target_trend", "decides_at": "close", "fill": "next_bar"}


def signal(prices, lookback=60, target=0.10):
    r = prices.pct_change()
    vol = r.rolling(20).std() * np.sqrt(252)
    return np.sign(prices.pct_change(lookback)) * (target / vol).clip(upper=1.0)
