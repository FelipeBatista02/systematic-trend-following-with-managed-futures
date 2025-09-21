"""Data layer exports."""

from .calendar import TradingCalendar
from .continuous import build_continuous_matrix, build_continuous_series
from .ingest import load_prices_or_generate
from .metadata import ContractMetadata, UniverseDefinition
from .validators import (
    SuspensionSpan,
    detect_limit_moves,
    detect_trading_suspensions,
    validate_price_data,
)

__all__ = [
    "ContractMetadata",
    "UniverseDefinition",
    "TradingCalendar",
    "build_continuous_series",
    "build_continuous_matrix",
    "load_prices_or_generate",
    "validate_price_data",
    "detect_trading_suspensions",
    "detect_limit_moves",
    "SuspensionSpan",
]