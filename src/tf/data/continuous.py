"""Utilities for building continuous futures series from individual contracts."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

RollSchedule = Sequence[tuple[pd.Timestamp | str, str]]


def _normalize_schedule(roll_schedule: RollSchedule) -> list[tuple[pd.Timestamp, str]]:
    if not roll_schedule:
        raise ValueError("Roll schedule must contain at least one contract")

    normalized: list[tuple[pd.Timestamp, str]] = []
    for start, contract in roll_schedule:
        timestamp = pd.Timestamp(start)
        normalized.append((timestamp.normalize() if isinstance(timestamp, pd.Timestamp) else timestamp, str(contract)))

    normalized.sort(key=lambda pair: pair[0])
    return normalized


def _segment_index(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DatetimeIndex:
    if end is None:
        return index[index >= start]
    mask = (index >= start) & (index < end)
    return index[mask]


def _back_adjusted(prices: pd.DataFrame, schedule: list[tuple[pd.Timestamp, str]]) -> pd.Series:
    index = prices.index
    continuous = pd.Series(index=index, dtype="float64")
    adjustment = 0.0

    for idx, (start, contract) in enumerate(schedule):
        end = schedule[idx + 1][0] if idx + 1 < len(schedule) else None
        segment_index = _segment_index(index, start, end)
        if segment_index.empty:
            continue
        try:
            contract_prices = prices.loc[segment_index, contract]
        except KeyError as exc:  # pragma: no cover - developer error
            raise KeyError(f"Missing contract column '{contract}' in price data") from exc

        continuous.loc[segment_index] = contract_prices + adjustment

        if end is not None:
            last_date = segment_index[-1]
            next_contract = schedule[idx + 1][1]
            current_cont_value = continuous.loc[last_date]
            next_price = prices.at[last_date, next_contract]
            adjustment = current_cont_value - next_price

    return continuous.dropna()


def _ratio_adjusted(prices: pd.DataFrame, schedule: list[tuple[pd.Timestamp, str]]) -> pd.Series:
    index = prices.index
    continuous = pd.Series(index=index, dtype="float64")
    factor = 1.0

    for idx, (start, contract) in enumerate(schedule):
        end = schedule[idx + 1][0] if idx + 1 < len(schedule) else None
        segment_index = _segment_index(index, start, end)
        if segment_index.empty:
            continue
        try:
            contract_prices = prices.loc[segment_index, contract]
        except KeyError as exc:  # pragma: no cover - developer error
            raise KeyError(f"Missing contract column '{contract}' in price data") from exc

        adjusted = contract_prices * factor
        continuous.loc[segment_index] = adjusted

        if end is not None:
            last_date = segment_index[-1]
            next_contract = schedule[idx + 1][1]
            current_cont_value = adjusted.loc[last_date]
            next_price = prices.at[last_date, next_contract]
            if next_price == 0:
                raise ValueError("Encountered zero price while computing ratio adjustment")
            factor = current_cont_value / next_price

    return continuous.dropna()


def _stitch_returns(prices: pd.DataFrame, schedule: list[tuple[pd.Timestamp, str]]) -> pd.Series:
    index = prices.index
    continuous = pd.Series(index=index, dtype="float64")
    prev_last_value: float | None = None

    for idx, (start, contract) in enumerate(schedule):
        end = schedule[idx + 1][0] if idx + 1 < len(schedule) else None
        segment_index = _segment_index(index, start, end)
        if segment_index.empty:
            continue
        try:
            contract_prices = prices.loc[segment_index, contract]
        except KeyError as exc:  # pragma: no cover - developer error
            raise KeyError(f"Missing contract column '{contract}' in price data") from exc

        if prev_last_value is None:
            scaled = contract_prices.copy()
        else:
            base = contract_prices.iloc[0]
            if base == 0:
                raise ValueError("Encountered zero price when stitching returns")
            scaled = contract_prices / base * prev_last_value

        continuous.loc[segment_index] = scaled
        prev_last_value = float(scaled.iloc[-1])

    return continuous.dropna()


_METHODS = {
    "back_adjusted": _back_adjusted,
    "ratio_adjusted": _ratio_adjusted,
    "stitched": _stitch_returns,
}


def build_continuous_series(
    prices: pd.DataFrame,
    roll_schedule: RollSchedule,
    method: str = "back_adjusted",
) -> pd.Series:
    """Return a single continuous series using the requested adjustment method."""

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price data must be indexed by a pandas.DatetimeIndex")

    schedule = _normalize_schedule(roll_schedule)
    try:
        builder = _METHODS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown continuous method '{method}'") from exc

    return builder(prices, schedule)


def build_continuous_matrix(
    prices: pd.DataFrame,
    schedules: Mapping[str, RollSchedule],
    method: str = "back_adjusted",
) -> pd.DataFrame:
    """Return a dataframe of continuous series for multiple root symbols."""

    data: dict[str, pd.Series] = {}
    for symbol, schedule in schedules.items():
        data[symbol] = build_continuous_series(prices, schedule, method=method)
    return pd.DataFrame(data)
