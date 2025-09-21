"""Common signal normalization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(
    frame: pd.DataFrame,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling z-score using the provided lookback window."""

    if window <= 1:
        raise ValueError("Window must be greater than one")

    min_periods = min_periods or window
    mean = frame.rolling(window=window, min_periods=min_periods).mean()
    std = frame.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return (frame - mean) / std.replace(0, np.nan)


def normalize_strength(
    frame: pd.DataFrame,
    *,
    clip: float | None = 3.0,
    method: str = "tanh",
) -> pd.DataFrame:
    """Scale signal strength to [-1, 1]."""

    if method not in {"tanh", "linear"}:
        raise ValueError("Unknown normalization method: %s" % method)

    if method == "tanh":
        scale = clip if clip and clip > 0 else 1.0
        return np.tanh(frame / scale)

    if clip is None or clip <= 0:
        raise ValueError("Linear normalization requires a positive clip value")
    return frame.clip(-clip, clip) / clip


def apply_guardrails(
    frame: pd.DataFrame,
    *,
    clip: float | None = 3.0,
    method: str = "tanh",
    fillna: float | None = 0.0,
) -> pd.DataFrame:
    """Normalize and sanitize a raw signal matrix."""

    normalized = normalize_strength(frame, clip=clip, method=method)
    if fillna is not None:
        normalized = normalized.fillna(fillna)
    return normalized


def lag_signal(signal: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Lag a signal to avoid lookahead bias."""

    if periods <= 0:
        return signal
    return signal.shift(periods)
