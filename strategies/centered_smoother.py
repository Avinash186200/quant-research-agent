"""Trade the slope of a 21-bar smoothed price."""
import numpy as np

SPEC = {"name": "centered_smoother", "decides_at": "close", "fill": "next_bar"}


def signal(prices, window=21):
    smooth = prices.rolling(window, center=True).mean()
    return np.sign(smooth.diff())
