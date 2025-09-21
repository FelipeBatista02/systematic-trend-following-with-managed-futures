"""Trading calendar utilities for aligning daily futures data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_holidays(holidays: Optional[Iterable[pd.Timestamp]]) -> list[pd.Timestamp]:
    if not holidays:
        return []
    normalized: list[pd.Timestamp] = []
    for holiday in holidays:
        ts = pd.Timestamp(holiday).normalize()
        if ts not in normalized:
            normalized.append(ts)
    return normalized


@dataclass(slots=True)
class TradingCalendar:
    """Simple weekday trading calendar with optional holiday exclusions."""

    name: str = "weekday"
    holidays: Sequence[pd.Timestamp] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.holidays = tuple(_normalize_holidays(self.holidays))

    def sessions(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        """Return trading sessions between ``start`` and ``end`` (inclusive)."""

        sessions = pd.bdate_range(start=start, end=end, freq="C")
        if not self.holidays:
            return sessions
        mask = ~sessions.isin(self.holidays)
        return sessions[mask]

    def align(self, frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Reindex price data to the calendar and forward-fill gaps."""

        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("Expected price data indexed by pandas.DatetimeIndex")

        target_index = self.sessions(start, end)
        aligned = frame.reindex(target_index)
        missing = aligned.isna().sum().sum()
        if missing:
            logger.debug("Forward filling %s missing data points after calendar align", missing)
        return aligned.ffill()

    def validate(self, frame: pd.DataFrame) -> None:
        """Run basic sanity checks on a price dataframe."""

        index = frame.index
        if not isinstance(index, pd.DatetimeIndex):  # pragma: no cover - defensive
            raise TypeError("Price data must be indexed by pandas.DatetimeIndex")
        if not index.is_monotonic_increasing:
            raise ValueError("Price data index must be sorted in increasing order")
        if index.has_duplicates:
            duplicates = index[index.duplicated()].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Price data contains duplicate dates: {duplicates}")

