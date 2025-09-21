"""Breakout style price signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalizers import apply_guardrails, lag_signal


def channel_breakout(
    prices: pd.DataFrame,
    window: int = 100,
    strength_clip: float = 3.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Return Donchian channel breakout signal."""

    if window <= 1:
        raise ValueError("Breakout window must be greater than 1")

    high = prices.rolling(window=window, min_periods=window).max()
    low = prices.rolling(window=window, min_periods=window).min()
    width = (high - low).replace(0, np.nan)
    mid = (high + low) / 2.0
    raw = (prices - mid) / (width.abs() + 1e-9)
    normalized = apply_guardrails(raw, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)
