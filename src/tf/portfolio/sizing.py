"""Portfolio sizing utilities."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..risk.vol import ewma_vol, rolling_volatility


def _resolve_contract_step(contract_rounding: float | Mapping[str, float], symbol: str) -> float:
    if isinstance(contract_rounding, Mapping):
        step = float(contract_rounding.get(symbol, 1.0))
    else:
        step = float(contract_rounding)
    return 1.0 if step <= 0 else step


def _round_series_to_contracts(
    series: pd.Series, contract_rounding: float | Mapping[str, float]
) -> pd.Series:
    rounded = pd.Series(index=series.index, dtype=float)
    for sym, value in series.items():
        step = _resolve_contract_step(contract_rounding, sym)
        if not np.isfinite(value):
            value = 0.0
        rounded[sym] = np.round(value / step) * step
    return rounded


def _scale_to_gross_limit(weights: pd.DataFrame, gross_limit: float | None) -> pd.DataFrame:
    if gross_limit is None or gross_limit <= 0:
        return weights
    gross = weights.abs().sum(axis=1)
    scale = pd.Series(1.0, index=weights.index)
    mask = gross > gross_limit
    if mask.any():
        scale.loc[mask] = gross_limit / gross.loc[mask]
    return weights.mul(scale, axis=0)


def _apply_sector_caps(
    budgets: pd.DataFrame,
    sector_map: Mapping[str, str] | None,
    sector_caps: Mapping[str, float] | None,
) -> pd.DataFrame:
    if not sector_map or not sector_caps:
        return budgets

    capped = budgets.copy()
    symbol_to_sector = {sym: sector_map[sym] for sym in capped.columns if sym in sector_map}
    for sector, cap in sector_caps.items():
        sector_symbols = [sym for sym, sec in symbol_to_sector.items() if sec == sector]
        if not sector_symbols:
            continue
        total = capped[sector_symbols].sum(axis=1)
        mask = total > cap
        if mask.any():
            scale = cap / total
            scale = scale.where(mask, 1.0)
            capped.loc[mask, sector_symbols] = capped.loc[mask, sector_symbols].mul(
                scale.loc[mask], axis=0
            )
    return capped


def _compute_volatility(
    prices: pd.DataFrame,
    *,
    vol_model: str,
    vol_lookback: int,
    ewma_lambda: float,
    min_vol_periods: int,
    volatility: pd.DataFrame | None,
) -> pd.DataFrame:
    if volatility is not None:
        vol = volatility.copy()
    else:
        returns = prices.pct_change().fillna(0.0)
        if vol_model == "ewma":
            vol = ewma_vol(returns, lam=ewma_lambda, min_periods=min_vol_periods)
        elif vol_model == "rolling":
            vol = rolling_volatility(
                returns, window=vol_lookback, min_periods=min_vol_periods
            )
        else:
            raise ValueError(f"Unknown volatility model: {vol_model}")
    vol = vol.reindex(index=prices.index, columns=prices.columns)
    return vol.replace(0.0, np.nan).ffill().bfill().fillna(0.0)


def _risk_budget(
    signals: pd.DataFrame,
    allocator: str,
) -> pd.DataFrame:
    abs_signals = signals.abs()
    if allocator == "erc":
        active = (abs_signals > 0).astype(float)
        row_sums = active.sum(axis=1).replace(0.0, np.nan)
        return active.div(row_sums, axis=0).fillna(0.0)
    if allocator != "proportional":
        raise ValueError(f"Unknown risk allocator: {allocator}")
    row_sums = abs_signals.sum(axis=1).replace(0.0, np.nan)
    return abs_signals.div(row_sums, axis=0).fillna(0.0)


def _apply_rebalance_threshold(
    desired_contracts: pd.DataFrame,
    contract_rounding: float | Mapping[str, float],
    threshold: float,
    prev_positions: pd.Series | Mapping[str, float] | None,
) -> pd.DataFrame:
    final = pd.DataFrame(0.0, index=desired_contracts.index, columns=desired_contracts.columns)
    if prev_positions is None:
        prev = pd.Series(0.0, index=desired_contracts.columns, dtype=float)
    else:
        if not isinstance(prev_positions, pd.Series):
            prev = pd.Series(prev_positions, dtype=float)
        else:
            prev = prev_positions.astype(float)
        prev = prev.reindex(desired_contracts.columns).fillna(0.0)

    threshold = max(float(threshold), 0.0)
    for ts in desired_contracts.index:
        target = desired_contracts.loc[ts].fillna(0.0).copy()
        delta = target - prev
        if threshold > 0:
            small = delta.abs() < threshold
            target.loc[small] = prev.loc[small]
        rounded = _round_series_to_contracts(target, contract_rounding)
        final.loc[ts] = rounded
        prev = rounded
    return final


def volatility_target_positions(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    point_values: Mapping[str, float],
    *,
    capital: float = 1_000_000.0,
    target_portfolio_vol: float = 0.15,
    gross_exposure_limit: float | None = 3.0,
    sector_map: Mapping[str, str] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    contract_rounding: float | Mapping[str, float] = 1.0,
    rebalance_threshold: float = 0.25,
    prev_positions: pd.Series | Mapping[str, float] | None = None,
    vol_model: str = "ewma",
    vol_lookback: int = 63,
    ewma_lambda: float = 0.94,
    min_vol_periods: int = 20,
    risk_allocator: str = "proportional",
    max_position_weight: float | None = None,
    volatility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Price history is empty")
    if set(prices.columns) != set(signals.columns):
        signals = signals.reindex(columns=prices.columns, fill_value=0.0)
    signals = signals.reindex(index=prices.index).fillna(0.0)
    prices = prices.sort_index()

    vol = _compute_volatility(
        prices,
        vol_model=vol_model,
        vol_lookback=vol_lookback,
        ewma_lambda=ewma_lambda,
        min_vol_periods=min_vol_periods,
        volatility=volatility,
    ).replace(0.0, np.nan)

    risk_budget = _risk_budget(signals, allocator=risk_allocator)
    risk_budget = _apply_sector_caps(risk_budget, sector_map, sector_caps)

    risk_target = risk_budget * float(target_portfolio_vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = risk_target.div(vol)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights * np.sign(signals)

    if max_position_weight is not None and max_position_weight > 0:
        weights = weights.clip(upper=max_position_weight, lower=-max_position_weight)

    weights = _scale_to_gross_limit(weights, gross_exposure_limit)

    capital = float(capital)
    notional = weights * capital
    contracts = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for sym in prices.columns:
        pv = float(point_values.get(sym, 1.0))
        denom = (prices[sym] * pv).replace(0.0, np.nan)
        contracts[sym] = notional[sym].div(denom)
    contracts = contracts.fillna(0.0)

    final_positions = _apply_rebalance_threshold(
        contracts,
        contract_rounding=contract_rounding,
        threshold=rebalance_threshold,
        prev_positions=prev_positions,
    )
    return final_positions
