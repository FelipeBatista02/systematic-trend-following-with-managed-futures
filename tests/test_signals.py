import numpy as np
import pandas as pd

from tf.signals.breakout import channel_breakout
from tf.signals.momentum import timeseries_momentum
from tf.signals.moving_average import moving_average_crossover, price_vs_sma
from tf.signals.normalizers import apply_guardrails, lag_signal, normalize_strength, rolling_zscore


def _price_frame(periods: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=periods, freq="B")
    base = np.linspace(100, 200, len(idx))
    prices = np.vstack([base + i for i in range(3)]).T
    return pd.DataFrame(prices, index=idx, columns=["A", "B", "C"])


def test_momentum_shape() -> None:
    df = _price_frame()
    sig = timeseries_momentum(df, lookbacks=(63, 126), skip_last_n=5)
    assert sig.shape == df.shape
    assert sig.abs().max().max() <= 1.0


def test_moving_average_crossover_direction() -> None:
    df = _price_frame()
    sig = moving_average_crossover(df, fast=20, slow=40)
    assert sig.index.equals(df.index)
    assert sig.columns.tolist() == df.columns.tolist()
    price_signal = price_vs_sma(df, fast=20, slow=60)
    assert price_signal.abs().max().max() <= 1.0


def test_breakout_bounds() -> None:
    df = _price_frame()
    sig = channel_breakout(df, window=50)
    assert np.isfinite(sig.to_numpy()).all()
    assert (sig.abs() <= 1.0 + 1e-6).all().all()


def test_normalizers() -> None:
    df = _price_frame(120).pct_change()
    z = rolling_zscore(df, window=20)
    assert z.shape == df.shape
    bounded = normalize_strength(z, clip=2.5, method="linear")
    assert bounded.max().max() <= 1.0
    lagged = lag_signal(bounded, periods=2)
    assert lagged.iloc[2:].equals(bounded.shift(2).iloc[2:])
    guard = apply_guardrails(z, clip=2.5)
    assert guard.isna().sum().sum() == 0
