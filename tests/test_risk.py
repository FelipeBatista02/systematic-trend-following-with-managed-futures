import numpy as np
import pandas as pd

from tf.risk.vol import average_true_range, ewma_vol, rolling_volatility


def _returns_frame() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    data = np.random.RandomState(0).normal(scale=0.01, size=(len(idx), 3))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


def test_ewma_vol_bounds() -> None:
    returns = _returns_frame()
    vol = ewma_vol(returns, lam=0.9, min_periods=10)
    assert vol.shape == returns.shape
    assert (vol >= 0).all().all()


def test_rolling_volatility() -> None:
    returns = _returns_frame()
    vol = rolling_volatility(returns, window=21, annualize=False)
    assert vol.notna().sum().sum() > 0


def test_average_true_range() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    base = np.linspace(100, 110, len(idx))
    high = pd.DataFrame([base + 1.5, base + 1.0, base + 0.5], index=["A", "B", "C"]).T
    low = pd.DataFrame([base - 1.5, base - 1.0, base - 0.5], index=["A", "B", "C"]).T
    close = pd.DataFrame([base, base + 0.2, base - 0.3], index=["A", "B", "C"]).T
    atr = average_true_range(high, low, close, window=5)
    assert atr.shape == high.shape
    assert (atr >= 0).all().all()
