"""Research utilities for robustness analysis and parameter exploration."""

from .monte_carlo import bootstrap_confidence_intervals
from .sensitivity import (
    compute_metric_sensitivity,
    lookback_sensitivity,
    vol_target_sensitivity,
)
from .walkforward import (
    WalkForwardResult,
    WalkForwardWindow,
    generate_walk_forward_windows,
)

__all__ = [
    "WalkForwardResult",
    "WalkForwardWindow",
    "bootstrap_confidence_intervals",
    "compute_metric_sensitivity",
    "generate_walk_forward_windows",
    "lookback_sensitivity",
    "vol_target_sensitivity",
]
