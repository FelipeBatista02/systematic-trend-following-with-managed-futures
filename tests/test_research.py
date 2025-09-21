import pandas as pd
import pytest

from tf.data.ingest import load_prices_or_generate
from tf.engine.backtester import Backtester
from tf.research.monte_carlo import bootstrap_confidence_intervals
from tf.research.sensitivity import (
    compute_metric_sensitivity,
    lookback_sensitivity,
    vol_target_sensitivity,
)
from tf.research.walkforward import generate_walk_forward_windows


@pytest.fixture
def small_backtester():
    idx = pd.bdate_range("2020-01-01", periods=40)
    universe = [{"symbol": "ES", "sector": "Equities", "point_value": 50}]
    prices = load_prices_or_generate(universe, str(idx[0].date()), str(idx[-1].date()), seed=5)
    cfg = {
        "backtest": {
            "start": str(idx[0].date()),
            "end": str(idx[-1].date()),
            "results_dir": "results",
            "starting_nav": 1_000_000.0,
        },
        "signals": {"momentum": {"lookbacks": [5], "skip_last_n": 0}},
        "risk": {"target_portfolio_vol": 0.12, "max_instrument_vol_weight": 0.05},
        "execution": {"adv_limit_pct": 1.0, "adv_contracts": {"ES": 1000}},
    }
    return Backtester(prices, universe, cfg)


def test_generate_walk_forward_windows():
    idx = pd.bdate_range("2022-01-03", periods=30)
    windows = generate_walk_forward_windows(idx, insample=10, oos=5, anchored=False)
    assert windows
    assert windows[0].insample_end < windows[0].oos_start
    # ensure anchored version shares the same oos window but earlier start
    anchored = generate_walk_forward_windows(idx, insample=10, oos=5, anchored=True)
    assert anchored[0].insample_start == idx[0]
    assert anchored[0].oos_start == windows[0].oos_start


def test_walk_forward_execution(small_backtester):
    results = small_backtester.run_walk_forward(insample=15, oos=5, anchored=False)
    assert results
    for wf in results:
        assert not wf.insample_nav.empty
        assert not wf.oos_nav.empty
        assert wf.oos_nav.index.min() >= wf.window.oos_start
        oos_summary = wf.oos_summary
        assert "Sharpe" in oos_summary


def test_parameter_grid_parallel(small_backtester):
    grid = {"signals.momentum.lookbacks": [[4], [6]]}
    seq_results = small_backtester.run_parameter_grid(grid, seeds=[11])
    par_results = small_backtester.run_parameter_grid(grid, seeds=[11], n_jobs=2, prefer="threads")
    assert len(seq_results) == len(par_results) == len(grid["signals.momentum.lookbacks"])
    for seq, par in zip(seq_results, par_results, strict=True):
        pd.testing.assert_series_equal(seq.nav, par.nav)


def test_bootstrap_confidence_intervals():
    returns = pd.Series([0.01, -0.005, 0.004, 0.012, -0.002])
    estimates = bootstrap_confidence_intervals(returns, n_samples=200, block_size=2, seed=7)
    assert set(estimates) == {"sharpe", "max_drawdown"}
    sharpe_interval = estimates["sharpe"]
    assert sharpe_interval.lower <= sharpe_interval.upper


def test_sensitivity_helpers(small_backtester):
    lookbacks = [4, 8]
    look_df = lookback_sensitivity(small_backtester, lookbacks)
    assert list(look_df.index) == lookbacks

    vol_targets = [0.08, 0.12, 0.16]
    vol_df = vol_target_sensitivity(small_backtester, vol_targets)
    assert list(vol_df.index) == vol_targets

    general = compute_metric_sensitivity(
        small_backtester,
        "signals.momentum.lookbacks",
        lookbacks,
        base_overrides={"signals": {"momentum": {"skip_last_n": 2}}},
        value_adapter=lambda x: [x],
    )
    assert list(general.index) == lookbacks
