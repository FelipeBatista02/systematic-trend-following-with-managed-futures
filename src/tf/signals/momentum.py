"""Momentum style signals."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .normalizers import apply_guardrails, lag_signal


def timeseries_momentum(
    prices: pd.DataFrame,
    lookbacks: Sequence[int] = (63, 126, 252),
    skip_last_n: int = 20,
    strength_clip: float = 3.0,
    lag: int = 1,
) -> pd.DataFrame:
    """Compute multi-horizon time-series momentum strength."""

    if prices.empty:
        raise ValueError("Price history is empty")

    if any(lb <= 0 for lb in lookbacks):
        raise ValueError("Lookbacks must be positive integers")

    prices = prices.sort_index()
    weights = np.array([1 / np.sqrt(lb) for lb in lookbacks], dtype=float)
    weights = weights / weights.sum()

    combined = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    shifted = prices.shift(skip_last_n)

    for weight, lookback in zip(weights, lookbacks):
        momentum = shifted / shifted.shift(lookback) - 1.0
        combined = combined.add(momentum * weight, fill_value=0.0)

    normalized = apply_guardrails(combined, clip=strength_clip)
    return lag_signal(normalized, lag).fillna(0.0)
