"""Performance metrics and summary statistics for backtest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _annualise_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.mean() * TRADING_DAYS)


def _annualise_vol(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _cagr(nav: pd.Series) -> float:
    if len(nav) < 2 or (nav.iloc[0] == 0):
        return 0.0
    periods = len(nav) - 1
    years = periods / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def performance_summary(
    nav: pd.Series,
    *,
    pnl: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Return a dictionary of key performance metrics."""

    if nav.empty:
        return {
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Max Drawdown": 0.0,
            "Calmar": 0.0,
            "Skew": 0.0,
            "Kurtosis": 0.0,
            "Turnover": 0.0,
            "Hit Rate": 0.0,
        }

    returns = nav.pct_change().dropna()

    cagr = _cagr(nav)
    vol = _annualise_vol(returns)
    ann_return = _annualise_return(returns)
    sharpe = ann_return / vol if vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = np.sqrt((downside.pow(2).mean())) * np.sqrt(TRADING_DAYS) if not downside.empty else 0.0
    sortino = ann_return / downside_std if downside_std > 0 else 0.0

    maxdd = _max_drawdown(nav)
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0.0

    skew = float(returns.skew()) if not returns.empty else 0.0
    kurt = float(returns.kurtosis()) if not returns.empty else 0.0

    turnover = np.nan
    if trades is not None and not trades.empty and "notional" in trades and nav.mean() != 0:
        turnover = float(trades["notional"].abs().sum() / nav.mean())
    if not np.isfinite(turnover):
        turnover = 0.0

    hit_rate = float((returns > 0).mean()) if not returns.empty else 0.0

    return {
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": float(maxdd),
        "Calmar": float(calmar),
        "Skew": float(skew),
        "Kurtosis": float(kurt),
        "Turnover": float(turnover),
        "Hit Rate": float(hit_rate),
    }


def compute_rolling_metrics(nav: pd.Series, window: int = 126) -> pd.DataFrame:
    """Return rolling volatility and Sharpe statistics."""

    if nav.empty:
        return pd.DataFrame(columns=["rolling_vol", "rolling_sharpe"])

    returns = nav.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame(index=nav.index, columns=["rolling_vol", "rolling_sharpe"])

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    rolling_vol = rolling_std * np.sqrt(TRADING_DAYS)
    rolling_sharpe = rolling_mean * TRADING_DAYS / rolling_std.replace(0, np.nan)

    rolling = pd.DataFrame(
        {
            "rolling_vol": rolling_vol,
            "rolling_sharpe": rolling_sharpe,
        }
    )
    return rolling.dropna(how="all")
