# System Design Spec (Condensed)

Maintained by Engineer Investor ([@egr_investor](https://x.com/egr_investor)).

- Multi-asset futures support with implemented roll/continuous construction helpers.
- Signals: momentum, moving-average crossover, and breakout libraries with guardrails.
- Volatility targeting & sector caps configurable via the risk module.
- Execution: next-open market fills with commissions, slippage, ADV caps, and roll handling.
- Backtest engine: deterministic daily loop producing full NAV/position/trade artefacts.
- Reporting: HTML bundle with equity, drawdown, contributions, exposures, and rolling stats.
- CLI: `tf run`, `tf report`, `tf sweep`, and `tf walkforward` for core workflows.
- Tests: unit and integration coverage across data, signals, portfolio, engine, and CLI layers.
