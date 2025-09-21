# Adding a new instrument

Follow the checklist below to onboard an additional futures contract into the
backtester.

## 1. Capture contract metadata

Each instrument entry in `configs/universe.yaml` maps directly to a
`ContractMetadata` object.  At a minimum you should populate:

* `symbol`: internal identifier used across the system.
* `sector`: optional grouping for exposure/risk analytics.
* `point_value`: dollar value per point of price movement.
* `contract_step`: minimum contract increment (used for exposure rounding).
* `data_source`: one of `yahoo`, `csv`, `parquet` or `synthetic`.
* `data_symbol`: vendor specific identifier (file path for local sources).

Refer to the existing sample instruments in `configs/universe.yaml` for the
canonical field ordering.  When sourcing data from CSV/Parquet files the
`data_symbol` must be an absolute or repo-relative path.

## 2. Validate data availability

Run the loader in isolation to confirm that historical prices resolve correctly:

```python
from tf import api
from tf.data.ingest import load_prices_or_generate
from tf.data.validators import validate_price_data

cfg = api.load_config("configs/base.yaml")
universe = api.load_universe(cfg)
prices = load_prices_or_generate(universe, start="2015-01-01", end="2020-12-31")
validate_price_data(prices)
```

The helper raises immediately if the series contains duplicate dates, zero or
negative prices, or suspensions longer than the configured threshold.

## 3. Check roll alignment (optional)

If the instrument has a bespoke roll pattern, extend the `roll_schedule` section
under `execution` in `configs/base.yaml`.  The
[`RollEngine`](../src/tf/engine/execution.py) consumes this mapping to emit roll
orders on the requested dates.

## 4. Smoke test the strategy

Execute a local backtest to ensure the new instrument integrates end-to-end:

```bash
tf run --config configs/base.yaml --run-id add-instrument-check
tf report --config configs/base.yaml --run-id add-instrument-check
```

Inspect the generated report to confirm the instrument contributes expected
exposures and that trading costs remain within tolerance.
