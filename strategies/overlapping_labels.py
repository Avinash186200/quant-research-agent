"""10-bar-ahead direction model validated with plain contiguous k-fold."""
import numpy as np

SPEC = {"name": "overlapping_labels", "decides_at": "close", "fill": "next_bar",
        "label_horizon": 10}


def signal(prices, lookback=10):
    return np.sign(prices.pct_change(lookback))


def cv_splits(n, k=5):
    idx = np.arange(n)
    return [(np.setdiff1d(idx, test), test) for test in np.array_split(idx, k)]
