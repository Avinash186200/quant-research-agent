"""Momentum with an entry threshold chosen on the training window only."""
import numpy as np
import pandas as pd

SPEC = {"name": "threshold_fit_train", "decides_at": "close", "fill": "next_bar"}
CANDIDATES = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04]


def signal(prices, threshold=0.01):
    m = prices.pct_change(10)
    return pd.Series(np.where(m > threshold, 1.0, np.where(m < -threshold, -1.0, 0.0)), index=prices.index)


def _score(prices, threshold):
    pos = signal(prices, threshold)
    ret = pos.shift(1).fillna(0) * prices.pct_change().fillna(0)
    sd = ret.std()
    return 0.0 if sd == 0 else ret.mean() / sd


def fit(prices, split):
    train = prices.iloc[:split]
    best = max(CANDIDATES, key=lambda t: _score(train, t))
    return {"threshold": best}
