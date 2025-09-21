import pandas as pd
import pytest

from tf.eval.analytics import (
    compute_exposures,
    compute_pnl_contributions,
    compute_roll_cost_breakdown,
    compute_sector_contributions,
    normalise_totals,
)
from tf.eval.metrics import compute_rolling_metrics, performance_summary


def test_performance_summary_and_turnover():
    idx = pd.bdate_range("2020-01-01", periods=6)
    nav = pd.Series([1_000_000, 1_010_000, 1_020_000, 1_015_000, 1_025_000, 1_040_000], index=idx)
    trades = pd.DataFrame(
        {
            "notional": [200_000, 150_000, 50_000],
            "reason": ["rebalance", "rebalance", "roll"],
        }
    )

    summary = performance_summary(nav, trades=trades)

    assert summary["CAGR"] > 0
    assert summary["Volatility"] > 0
    assert summary["Sharpe"] > 0
    assert summary["Max Drawdown"] <= 0
    expected_turnover = trades["notional"].abs().sum() / nav.mean()
    assert summary["Turnover"] == pytest.approx(expected_turnover)
    assert 0 <= summary["Hit Rate"] <= 1

    rolling = compute_rolling_metrics(nav, window=3)
    assert set(["rolling_vol", "rolling_sharpe"]).issubset(rolling.columns)


def test_attribution_and_roll_costs():
    idx = pd.bdate_range("2021-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 52.0, 51.0]}, index=idx)
    positions = pd.DataFrame({"A": [0.0, 1.0, 1.0], "B": [0.0, -1.0, -1.0]}, index=idx)
    point_values = {"A": 1.0, "B": 2.0}
    sector_map = {"A": "Alpha", "B": "Beta"}

    contrib = compute_pnl_contributions(prices, positions, point_values)
    assert contrib.loc[idx[-1], "A"] == pytest.approx(1.0)
    assert contrib.loc[idx[-1], "B"] == pytest.approx(2.0)

    sector = compute_sector_contributions(contrib, sector_map)
    assert set(sector.columns) == {"Alpha", "Beta"}
    assert sector.loc[idx[-1], "Alpha"] == pytest.approx(1.0)
    assert sector.loc[idx[-1], "Beta"] == pytest.approx(2.0)

    exposures = compute_exposures(prices, positions, point_values)
    assert exposures.loc[idx[1], "A"] == pytest.approx(101.0)
    assert exposures.loc[idx[1], "B"] == pytest.approx(-104.0)

    costs = pd.DataFrame({"trading": [1.0, 2.0], "roll": [0.5, 0.5], "total": [1.5, 2.5]})
    trades = pd.DataFrame({"reason": ["rebalance", "roll"], "cost": [3.0, 2.0]})
    roll_costs = compute_roll_cost_breakdown(trades, costs)
    assert {"ledger_trading", "ledger_roll", "trades_rebalance", "trades_roll"} <= set(roll_costs.index)

    totals = pd.Series({"A": 10.0, "B": -5.0, "C": 1.0, "D": 0.5})
    normalised = normalise_totals(totals, top_n=3)
    assert "Other" in normalised.index
