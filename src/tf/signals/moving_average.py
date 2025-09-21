"""Moving-average based signals."""

from __future__ import annotations

import pandas as pd

from .normalizers import apply_guardrails, lag_signal


def price_vs_sma(
    prices: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    strength_clip: float = 3.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Return price vs slow moving average strength."""

    if fast <= 0 or slow <= 0:
        raise ValueError("Moving average windows must be positive")
    if fast >= slow:
        raise ValueError("Fast window must be shorter than slow window")

    sma_fast = prices.rolling(window=fast, min_periods=fast).mean()
    sma_slow = prices.rolling(window=slow, min_periods=slow).mean()
    raw = (sma_fast - sma_slow) / (sma_slow.abs() + 1e-9)
    normalized = apply_guardrails(raw, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)


def moving_average_crossover(
    prices: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    strength_clip: float = 3.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Return SMA crossover signal scaled to [-1, 1]."""

    if fast <= 0 or slow <= 0:
        raise ValueError("Moving average windows must be positive")
    if fast >= slow:
        raise ValueError("Fast window must be shorter than slow window")

    ma_fast = prices.rolling(window=fast, min_periods=fast).mean()
    ma_slow = prices.rolling(window=slow, min_periods=slow).mean()
    raw = (ma_fast / (ma_slow + 1e-9)) - 1.0
    normalized = apply_guardrails(raw, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)
