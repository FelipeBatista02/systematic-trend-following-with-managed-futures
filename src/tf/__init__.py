"""Package exports for the trend-following toolkit."""

__version__ = "0.9.0"

from .api import (  # noqa: F401
    BacktestContext,
    ParameterSweepResult,
    ParameterSweepScenario,
    export_sweep_metadata,
    load_config,
    load_universe,
    merge_overrides,
    prepare_backtester,
    run_backtest,
    run_parameter_sweep,
    run_walk_forward,
    serialise_sweep_scenarios,
)
from .cli import main  # noqa: F401

__all__ = [
    "__version__",
    "BacktestContext",
    "ParameterSweepResult",
    "ParameterSweepScenario",
    "export_sweep_metadata",
    "load_config",
    "load_universe",
    "main",
    "merge_overrides",
    "prepare_backtester",
    "run_backtest",
    "run_parameter_sweep",
    "run_walk_forward",
    "serialise_sweep_scenarios",
]
