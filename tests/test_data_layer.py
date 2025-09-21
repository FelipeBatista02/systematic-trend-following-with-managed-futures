"""Unit tests for the upgraded data layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf.data.calendar import TradingCalendar
from tf.data.continuous import build_continuous_series
from tf.data.ingest import load_prices_or_generate
from tf.data.metadata import ContractMetadata, UniverseDefinition
from tf.data.validators import (
    detect_limit_moves,
    detect_trading_suspensions,
    validate_price_data,
)


def _mini_universe() -> list[dict[str, object]]:
    return [
        {"symbol": "ES", "sector": "Equities", "point_value": 50},
        {"symbol": "CL", "sector": "Commodities", "point_value": 1000},
    ]


def test_contract_metadata_default_vendor_symbol() -> None:
    contract = ContractMetadata(symbol="ES", sector="Equities", point_value=50)
    assert contract.vendor_symbol == "ES=F"


def test_universe_definition_detects_duplicates() -> None:
    payload = _mini_universe()
    payload.append({"symbol": "ES", "sector": "Dup", "point_value": 10})
    with pytest.raises(ValueError):
        UniverseDefinition.from_payload(payload)


def test_trading_calendar_aligns_data() -> None:
    cal = TradingCalendar()
    index = pd.to_datetime(["2024-01-01", "2024-01-03"])  # missing Jan 2nd
    frame = pd.DataFrame({"ES": [100, 101]}, index=index)
    aligned = cal.align(frame, "2024-01-01", "2024-01-03")
    assert len(aligned) == 3  # business days between start and end
    # Forward fill fills the missing second day
    assert aligned.loc["2024-01-02", "ES"] == 100


def test_trading_calendar_respects_holidays() -> None:
    holiday = pd.Timestamp("2024-01-02")
    cal = TradingCalendar(holidays=[holiday])
    index = pd.to_datetime(["2024-01-01", "2024-01-03"])
    frame = pd.DataFrame({"ES": [100, 101]}, index=index)
    aligned = cal.align(frame, "2024-01-01", "2024-01-03")
    assert list(aligned.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]


def test_load_prices_or_generate_falls_back_to_synthetic(monkeypatch) -> None:
    from tf.data import ingest

    calls: dict[str, int] = {"count": 0}

    def boom(*args, **kwargs):
        calls["count"] += 1
        raise RuntimeError("network down")

    monkeypatch.setattr(ingest, "_load_from_vendor", boom)

    prices = load_prices_or_generate(
        _mini_universe(), "2020-01-01", "2020-01-10", seed=1
    )
    assert calls["count"] == 1
    assert list(prices.columns) == ["ES", "CL"]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.shape[0] >= 5


def test_validate_price_data_detects_negative() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    frame = pd.DataFrame({"ES": [100, 101, -2, 103, 104]}, index=index)
    with pytest.raises(ValueError):
        validate_price_data(frame)


def test_validate_price_data_detects_suspension() -> None:
    index = pd.date_range("2024-01-01", periods=7, freq="B")
    frame = pd.DataFrame(
        {
            "ES": [100.0, np.nan, np.nan, np.nan, np.nan, 105.0, 106.0],
            "CL": [50.0, 51.0, np.nan, 53.0, 54.0, 55.0, 56.0],
        },
        index=index,
    )

    with pytest.raises(ValueError):
        validate_price_data(frame, max_consecutive_missing=2)

    suspensions = detect_trading_suspensions(frame, min_gap=2)
    assert set(suspensions.keys()) == {"ES"}
    span = suspensions["ES"][0]
    assert span.length == 4
    assert span.start == index[1]
    assert span.end == index[4]


def test_detect_limit_moves_flags_large_returns() -> None:
    index = pd.date_range("2024-02-01", periods=5, freq="B")
    frame = pd.DataFrame(
        {
            "ES": [100.0, 105.0, 160.0, 161.0, 162.0],
            "CL": [50.0, 50.5, 50.6, 50.7, 50.8],
        },
        index=index,
    )

    flagged = detect_limit_moves(frame, threshold=0.4)
    assert list(flagged.index) == [index[2]]
    assert bool(flagged.loc[index[2], "ES"]) is True
    assert bool(flagged.loc[index[2], "CL"]) is False


def test_continuous_builder_back_adjusted() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame(
        {
            "ES_H": [100, 101, 102, 103, 104, 105],
            "ES_M": [110, 111, 112, 113, 114, 115],
            "ES_U": [120, 121, 122, 123, 124, 125],
        },
        index=index,
    )
    schedule = [
        (index[0], "ES_H"),
        (index[3], "ES_M"),
        (index[5], "ES_U"),
    ]
    continuous = build_continuous_series(prices, schedule, method="back_adjusted")
    assert continuous.loc[index[2]] == pytest.approx(102)
    assert continuous.loc[index[3]] == pytest.approx(103)
    assert continuous.loc[index[-1]] == pytest.approx(105)


def test_load_prices_from_csv(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    frame = pd.DataFrame({"date": dates, "close": [100, 101, 102, 103, 104]})
    path = tmp_path / "ES.csv"
    frame.to_csv(path, index=False)

    universe = [
        {
            "symbol": "ES",
            "sector": "Equities",
            "point_value": 50,
            "data_source": "csv",
            "data_symbol": str(path),
        }
    ]

    prices = load_prices_or_generate(universe, "2024-01-01", "2024-01-07", prefer="auto")
    assert list(prices.columns) == ["ES"]
    assert prices.loc["2024-01-03", "ES"] == pytest.approx(102)
