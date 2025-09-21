"""Volatility estimators used for sizing and signal normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(
    returns: pd.DataFrame,
    lam: float = 0.94,
    min_periods: int = 20,
    *,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Exponentially weighted volatility estimate."""

    if not 0 < lam < 1:
        raise ValueError("lambda must be between 0 and 1")

    var = returns.ewm(alpha=(1 - lam), adjust=False, min_periods=min_periods).var(bias=False)
    vol = var.pow(0.5)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol.fillna(0.0)


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 63,
    *,
    min_periods: int | None = None,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rolling standard deviation of returns."""

    if window <= 1:
        raise ValueError("window must be greater than one")

    min_periods = min_periods or window
    vol = returns.rolling(window=window, min_periods=min_periods).std(ddof=0)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol.fillna(0.0)


def average_true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Average True Range (ATR) computed from OHLC bars."""

    if window <= 1:
        raise ValueError("window must be greater than one")

    prev_close = close.shift(1)
    range1 = (high - low).abs()
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()
    true_range = range1.combine(range2, np.maximum, fill_value=0.0)
    true_range = true_range.combine(range3, np.maximum, fill_value=0.0)

    min_periods = min_periods or window
    return true_range.rolling(window=window, min_periods=min_periods).mean().fillna(0.0)
