"""Attribution and analytics helpers for backtest evaluation."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def compute_pnl_contributions(
    prices: pd.DataFrame | None,
    positions: pd.DataFrame | None,
    point_values: Mapping[str, float] | None,
) -> pd.DataFrame:
    """Return daily PnL contributions by instrument."""

    if prices is None or positions is None or point_values is None:
        return pd.DataFrame()

    common_cols = prices.columns.intersection(positions.columns)
    if common_cols.empty:
        return pd.DataFrame(index=prices.index if prices is not None else None)

    pv = pd.Series(point_values, dtype=float)
    pv = pv.reindex(common_cols).fillna(1.0)

    price_changes = prices[common_cols].diff().fillna(0.0)
    lag_positions = positions[common_cols].shift(1).fillna(0.0)

    contributions = lag_positions.multiply(price_changes).multiply(pv, axis=1)
    contributions.index = prices.index
    return contributions


def compute_sector_contributions(
    contributions: pd.DataFrame,
    sector_map: Mapping[str, str] | None,
) -> pd.DataFrame:
    """Aggregate instrument contributions into sector contributions."""

    if contributions.empty:
        return pd.DataFrame(index=contributions.index)

    sector_map = sector_map or {}
    grouped: dict[str, pd.Series] = {}
    for symbol in contributions.columns:
        sector = sector_map.get(symbol, "Unknown")
        grouped.setdefault(sector, 0)
        grouped[sector] = grouped[sector] + contributions[symbol]

    if not grouped:
        return pd.DataFrame(index=contributions.index)

    sector_df = pd.DataFrame(grouped)
    sector_df.index = contributions.index
    return sector_df


def compute_exposures(
    prices: pd.DataFrame | None,
    positions: pd.DataFrame | None,
    point_values: Mapping[str, float] | None,
) -> pd.DataFrame:
    """Return notional exposures by instrument."""

    if prices is None or positions is None or point_values is None:
        return pd.DataFrame()

    common_cols = prices.columns.intersection(positions.columns)
    if common_cols.empty:
        return pd.DataFrame(index=prices.index if prices is not None else None)

    pv = pd.Series(point_values, dtype=float).reindex(common_cols).fillna(1.0)
    exposure = positions[common_cols].multiply(prices[common_cols]).multiply(pv, axis=1)
    exposure.index = prices.index
    return exposure


def compute_roll_cost_breakdown(
    trades: pd.DataFrame | None,
    costs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise trading vs roll costs from fills and ledger costs."""

    cost_components: dict[str, float] = {}

    if costs is not None and not costs.empty:
        totals = costs.sum()
        for key in ["trading", "roll", "total"]:
            if key in totals:
                cost_components[f"ledger_{key}"] = float(totals[key])

    if trades is not None and not trades.empty and "cost" in trades:
        grouped = trades.groupby(trades["reason"].fillna("unknown"))["cost"].sum()
        for reason, value in grouped.items():
            cost_components[f"trades_{reason}"] = float(value)

    if not cost_components:
        return pd.DataFrame()

    return pd.Series(cost_components, name="amount").to_frame()


def normalise_totals(series: pd.Series, top_n: int = 10) -> pd.Series:
    """Return ``top_n`` entries of ``series`` sorted by absolute contribution."""

    if series.empty:
        return series

    ordered = series.reindex(series.abs().sort_values(ascending=False).index)
    if len(ordered) <= top_n:
        return ordered

    top = ordered.iloc[: top_n - 1]
    remainder = ordered.iloc[top_n - 1 :].sum()
    return pd.concat([top, pd.Series({"Other": remainder})])
