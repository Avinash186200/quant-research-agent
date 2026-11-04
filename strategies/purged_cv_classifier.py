"""10-bar-ahead direction model validated with purged, embargoed k-fold.

The model itself is a sign-of-momentum rule; what matters for the audit is
that cv_splits drops training labels that overlap each test fold.
"""
import numpy as np

SPEC = {"name": "purged_cv_classifier", "decides_at": "close", "fill": "next_bar",
        "label_horizon": 10, "embargo": 10}


def signal(prices, lookback=10):
    return np.sign(prices.pct_change(lookback))


def cv_splits(n, k=5):
    h, emb = SPEC["label_horizon"], SPEC["embargo"]
    idx = np.arange(n)
    out = []
    for test in np.array_split(idx, k):
        t0, t1 = test[0], test[-1]
        keep = (idx + h < t0) | (idx > t1 + max(h, emb))
        out.append((idx[keep], test))
    return out
