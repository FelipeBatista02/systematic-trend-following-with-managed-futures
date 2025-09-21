"""Signal library exports."""

from .breakout import channel_breakout
from .momentum import timeseries_momentum
from .moving_average import moving_average_crossover, price_vs_sma
from .normalizers import apply_guardrails, lag_signal, normalize_strength, rolling_zscore

__all__ = [
    "channel_breakout",
    "timeseries_momentum",
    "moving_average_crossover",
    "price_vs_sma",
    "apply_guardrails",
    "lag_signal",
    "normalize_strength",
    "rolling_zscore",
]
