# How rolling works in `tf-trend`

Contract rolling is handled in two layers: the data layer builds continuous
series for research, while the execution layer submits roll orders during the
backtest.  Understanding both is key when extending or debugging the strategy.

## Data layer – continuous prices

The [`build_continuous_series`](../src/tf/data/continuous.py) helper stitches
individual contract months together based on a roll schedule.  The default
configuration in `configs/base.yaml` uses a back-adjusted series, but you can
switch to ratio adjustment or return-stitching via the `data.continuous`
settings.  When you edit the schedule make sure it stays monotonic and matches
the actual contract expiry cadence.

To generate a bespoke series inside a notebook:

```python
import pandas as pd
from tf.data.continuous import build_continuous_series

# prices: dataframe with individual contract columns (e.g. ES_H, ES_M)
# schedule: list of (timestamp, front_contract) pairs
continuous = build_continuous_series(prices, schedule, method="back_adjusted")
```

## Execution layer – roll orders

During a live backtest the engine does **not** rely on the continuous series for
trading.  Instead it looks at the discrete contract positions and follows the
`execution.roll_schedule` mapping defined in `configs/base.yaml`.  The
[`RollEngine`](../src/tf/engine/execution.py) converts the schedule into pairs of
exit/entry orders.  Costs are tracked separately from regular rebalances so you
can isolate slippage attributable to rolling.

A minimal configuration snippet looks like:

```yaml
execution:
  roll_schedule:
    ES:
      - 2024-03-12
      - 2024-06-11
```

When the clock reaches `2024-03-12` the backtester submits two orders for `ES`:
close the current position in the front month and immediately open the same
quantity in the next month.  If no position is held the roll is skipped.

## Analytics and reporting

Roll costs are accumulated in the `result.costs['roll']` column and exposed in
both the HTML report and `tests/test_engine.py`.  Use the generated
`roll_costs.csv` artefact to diagnose large spikes or to reconcile with broker
statements.
