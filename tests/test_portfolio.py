import numpy as np
import pandas as pd

from tf.portfolio.sizing import volatility_target_positions


def _constant_vol(prices: pd.DataFrame, value: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(value, index=prices.index, columns=prices.columns, dtype=float)


def test_sector_caps_and_gross_limit() -> None:
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=(len(idx), 3))
    prices = pd.DataFrame(100.0 * np.exp(returns.cumsum(axis=0)), index=idx, columns=["ES", "NQ", "ZN"])
    signals = pd.DataFrame(1.0, index=idx, columns=prices.columns)

    point_values = {"ES": 1.0, "NQ": 1.0, "ZN": 1.0}
    sector_map = {"ES": "Equities", "NQ": "Equities", "ZN": "Rates"}
    sector_caps = {"Equities": 0.5, "Rates": 0.6}

    targets = volatility_target_positions(
        prices=prices,
        signals=signals,
        point_values=point_values,
        capital=1_000_000.0,
        target_portfolio_vol=0.2,
        gross_exposure_limit=0.6,
        sector_map=sector_map,
        sector_caps=sector_caps,
        contract_rounding=0.01,
        rebalance_threshold=0.0,
        volatility=_constant_vol(prices, 0.2),
    )

    last_idx = targets.index[-1]
    last_pos = targets.loc[last_idx]
    prices_last = prices.loc[last_idx]
    weights = (last_pos * prices_last * pd.Series(point_values)).div(1_000_000.0)

    assert weights.abs().sum() <= 0.6 + 1e-6
    equities_weight = weights[["ES", "NQ"]].abs().sum()
    rates_weight = weights[["ZN"]].abs().sum()
    assert equities_weight <= 0.5 + 1e-6
    assert rates_weight <= 0.6 + 1e-6


def test_rounding_and_threshold() -> None:
    idx = pd.date_range("2021-01-01", periods=4, freq="B")
    prices = pd.DataFrame(100.0, index=idx, columns=["ES", "CL"])
    signals = pd.DataFrame(
        [[0.0, 0.0], [1.0, 1.0], [1.0, 0.9], [1.0, 0.1]],
        index=idx,
        columns=["ES", "CL"],
    )

    targets = volatility_target_positions(
        prices=prices,
        signals=signals,
        point_values={"ES": 1.0, "CL": 1.0},
        capital=10_000.0,
        target_portfolio_vol=0.02,
        gross_exposure_limit=1.0,
        sector_map={"ES": "Equities", "CL": "Commodities"},
        contract_rounding=1.0,
        rebalance_threshold=0.5,
        volatility=_constant_vol(prices, 0.2),
    )

    # All positions should be rounded to whole contracts
    assert (targets % 1.0 == 0).all().all()
    # Small changes should not trigger a trade
    assert targets.iloc[2].equals(targets.iloc[1])
    # Larger changes should result in updated positions after rounding
    assert targets.iloc[3]["ES"] == 9
    assert targets.iloc[3]["CL"] == 1
