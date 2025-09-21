import yaml
import pandas as pd
import pytest

from tf.data.ingest import load_prices_or_generate
from tf.engine.backtester import Backtester, BacktestResults

def test_backtester_runs(tmp_path):
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        """
universe: {assets_file: configs/universe.yaml}
backtest: {start: "2015-01-01", end: "2018-12-31", results_dir: "results"}
signals: {momentum: {lookbacks: [63,126], skip_last_n: 10}}
risk: {target_portfolio_vol: 0.15, max_instrument_vol_weight: 0.05}
execution: {impact: {k: 0.05, alpha: 0.5}, commission_per_contract: 2.5, tick_value: 10.0}
"""
    )
    cfg = yaml.safe_load(config_path.read_text())
    # Minimal universe
    uni = [{'symbol':'ES','sector':'Equities','point_value':50},{'symbol':'CL','sector':'Commodities','point_value':1000}]
    prices = load_prices_or_generate(uni, "2015-01-01", "2018-12-31", seed=1)
    bt = Backtester(prices, uni, yaml.safe_load((tmp_path/'base.yaml').read_text()))
    res = bt.run()
    assert len(res.nav) > 100


def test_execution_partial_fills_and_pnl(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "ES": [100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=idx,
    )

    raw_targets = pd.DataFrame(
        {
            "ES": [6.0, 6.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )

    def fake_volatility_target_positions(*args, **kwargs):
        return raw_targets.copy()

    monkeypatch.setattr(
        "tf.engine.backtester.volatility_target_positions",
        fake_volatility_target_positions,
    )

    cfg = {
        "backtest": {
            "start": str(idx[0].date()),
            "end": str(idx[-1].date()),
            "results_dir": "results",
            "starting_nav": 1_000_000.0,
        },
        "signals": {"momentum": {"lookbacks": [1], "skip_last_n": 0}},
        "risk": {},
        "execution": {
            "adv_limit_pct": 1.0,
            "adv_contracts": {"ES": 3},
            "impact": {"k": 0.0, "alpha": 1.0},
            "commission_per_contract": 0.0,
            "tick_value": 1.0,
            "min_slippage_ticks": 0.0,
        },
    }
    universe = [{"symbol": "ES", "sector": "Equities", "point_value": 1.0}]
    bt = Backtester(prices, universe, cfg)
    res = bt.run()

    trades = res.trades
    assert trades["qty"].tolist() == [3.0, 3.0, -3.0, -3.0]
    assert trades["partial"].tolist() == [True, False, True, False]

    expected_nav_end = 1_000_000.0 + (3.0 + 6.0 + 3.0)
    assert res.nav.iloc[0] == pytest.approx(1_000_000.0)
    assert res.nav.iloc[-1] == pytest.approx(expected_nav_end)
    assert res.positions.iloc[-1]["ES"] == pytest.approx(0.0)
    assert res.costs["total"].sum() == pytest.approx(0.0)

    assert res.cash is not None
    assert res.cash.iloc[0] == pytest.approx(1_000_000.0)
    assert res.cash.iloc[-1] == pytest.approx(expected_nav_end)
    assert res.ledger is not None
    assert set(["nav", "cash", "asset_value", "pnl", "trading_cost", "roll_cost", "total_cost"]) <= set(res.ledger.columns)
    assert (res.ledger["nav"] - res.ledger["cash"] - res.ledger["asset_value"]).abs().max() <= 1e-6


def test_roll_orders_book_separate_costs(monkeypatch):
    idx = pd.bdate_range("2024-02-01", periods=5)
    prices = pd.DataFrame({"ES": [100, 101, 102, 103, 104]}, index=idx, dtype=float)
    raw_targets = pd.DataFrame({"ES": [2.0, 2.0, 2.0, 0.0, 0.0]}, index=idx)

    def fake_volatility_target_positions(*args, **kwargs):
        return raw_targets.copy()

    monkeypatch.setattr(
        "tf.engine.backtester.volatility_target_positions",
        fake_volatility_target_positions,
    )

    cfg = {
        "backtest": {
            "start": str(idx[0].date()),
            "end": str(idx[-1].date()),
            "results_dir": "results",
            "starting_nav": 1_000_000.0,
        },
        "signals": {"momentum": {"lookbacks": [1], "skip_last_n": 0}},
        "risk": {},
        "execution": {
            "adv_limit_pct": 1.0,
            "adv_contracts": {"ES": 10},
            "impact": {"k": 0.0, "alpha": 1.0},
            "commission_per_contract": 1.0,
            "tick_value": 1.0,
            "min_slippage_ticks": 0.0,
            "roll_schedule": {"ES": [idx[2]]},
        },
    }
    universe = [{"symbol": "ES", "sector": "Equities", "point_value": 1.0}]
    bt = Backtester(prices, universe, cfg)
    res = bt.run()

    roll_trades = res.trades[res.trades["reason"] == "roll"]
    assert len(roll_trades) == 2
    assert sorted(roll_trades["qty"].tolist()) == [-2.0, 2.0]

    fill_day = idx[3]
    assert res.costs.loc[fill_day, "roll"] == pytest.approx(4.0)
    assert res.costs.loc[fill_day, "trading"] == pytest.approx(0.0)
    assert (res.orders["reason"] == "roll").any()


def test_parameter_grid_and_walk_forward():
    idx = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame({"ES": range(100, 108)}, index=idx, dtype=float)

    cfg = {
        "backtest": {
            "start": str(idx[0].date()),
            "end": str(idx[-1].date()),
            "results_dir": "results",
            "starting_nav": 1_000_000.0,
        },
        "signals": {"momentum": {"lookbacks": [2], "skip_last_n": 0}},
        "risk": {"target_portfolio_vol": 0.1, "max_instrument_vol_weight": 0.05},
        "execution": {"adv_limit_pct": 1.0, "adv_contracts": {"ES": 100}},
    }
    universe = [{"symbol": "ES", "sector": "Equities", "point_value": 1.0}]
    bt = Backtester(prices, universe, cfg)

    grid = {
        "signals.momentum.lookbacks": [[2], [3]],
        "risk.target_portfolio_vol": [0.05, 0.1],
    }
    splits = [
        {"start": str(idx[0].date()), "end": str(idx[3].date())},
        {"start": str(idx[3].date()), "end": str(idx[-1].date())},
    ]

    results = bt.run_parameter_grid(grid, walk_forward_splits=splits, seeds=[7])
    expected_runs = len(grid["signals.momentum.lookbacks"]) * len(grid["risk.target_portfolio_vol"]) * len(splits)
    assert len(results) == expected_runs
    hashes = {res.config_hash for res in results}
    assert len(hashes) == expected_runs

    split_bounds = [
        (pd.Timestamp(s["start"]), pd.Timestamp(s["end"])) for s in splits
    ]
    for res in results:
        start = res.nav.index.min()
        end = res.nav.index.max()
        assert any(start >= s and end <= e for s, e in split_bounds)


def test_reproducibility_guard(monkeypatch):
    idx = pd.bdate_range("2024-03-01", periods=4)
    prices = pd.DataFrame({"ES": [100.0, 101.0, 102.0, 103.0]}, index=idx)
    cfg = {
        "backtest": {
            "start": str(idx[0].date()),
            "end": str(idx[-1].date()),
            "results_dir": "results",
            "starting_nav": 1_000_000.0,
        },
        "signals": {"momentum": {"lookbacks": [1], "skip_last_n": 0}},
        "risk": {},
        "execution": {"adv_limit_pct": 1.0, "adv_contracts": {"ES": 10}},
    }
    universe = [{"symbol": "ES", "sector": "Equities", "point_value": 1.0}]
    bt = Backtester(prices, universe, cfg)

    res = bt.run(check_reproducibility=True)
    assert isinstance(res, BacktestResults)

    call_count = {"n": 0}
    original = Backtester._simulate_once

    def fake_sim(self, cfg, *, config_hash):
        if call_count["n"] == 0:
            call_count["n"] += 1
            return original(self, cfg, config_hash=config_hash)
        altered = original(self, cfg, config_hash=config_hash)
        altered.nav = altered.nav + 1.0
        return altered

    monkeypatch.setattr(Backtester, "_simulate_once", fake_sim)
    with pytest.raises(RuntimeError):
        bt.run(check_reproducibility=True)
