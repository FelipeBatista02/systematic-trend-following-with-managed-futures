# Cost model calibration

Transaction costs materially impact trend-following performance.  The framework
splits them into explicit commissions, slippage from market impact and roll
friction.  Use the following workflow to calibrate parameters per instrument or
sector.

## 1. Start with broker schedules

Populate `execution.commission_per_contract` and `execution.tick_value` in
`configs/base.yaml` using published exchange + broker fee schedules.  If the
broker charges different fees by product, override them via the
`execution.commissions` mapping (symbol to commission).

## 2. Set ADV participation caps

The `execution.adv_limit_pct` guard prevents the simulator from trading more
than a fixed percentage of the instrument's average daily volume.  Supply
contract-specific values through `execution.adv_contracts` to reflect liquidity
variance (e.g. trade fewer contracts in illiquid grains than in S&P futures).

## 3. Calibrate impact coefficients

The [`participation_slippage`](../src/tf/engine/execution.py) model uses two
parameters:

* `impact.k` – base tick cost when trading 100% of ADV.
* `impact.alpha` – curvature of the impact function.

A pragmatic approach is to gather historical fills, compute realised execution
vs. mid prices, and regress the residual against `abs(qty) / ADV`.  Iterate until
simulated costs align with realised ones.  If you lack live data, start with
`k=0.05`, `alpha=0.5` for liquid contracts and double `k` for thin markets.

## 4. Validate with attribution outputs

Run a representative backtest and inspect `results/*/roll_costs.csv` along with
`return_histogram.png`.  High slippage relative to turn-over may point to an
aggressive `k` value or overly restrictive ADV caps.  The HTML report also breaks
out roll vs. trading costs to help isolate issues.

## 5. Keep the changelog updated

Whenever cost assumptions change, note the rationale in `CHANGELOG.md` so future
investigations know which parameters shifted between releases.
