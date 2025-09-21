# Changelog

## [Unreleased]

### Added
- MIT license file and attribution to Engineer Investor ([@egr_investor](https://x.com/egr_investor)).

### Changed
- Updated project metadata, documentation, and spec to reflect full signal coverage and current maintainer contacts.

## [0.9.0] - 2024-05-07

### Added
- Data quality audits covering trading suspensions, holiday gaps and limit-up/down days.
- User guides for onboarding instruments, understanding roll mechanics and calibrating the cost model.
- Version constant exported via `tf.__version__` for downstream tooling.

### Changed
- README expanded with CLI/API guidance, QA workflow and release process.
- Dependencies pinned in `pyproject.toml` to guarantee reproducible installations.

## [0.8.0] - 2024-04-15

### Added
- End-to-end CLI (`run`, `report`, `sweep`, `walkforward`).
- Python API helpers (`run_backtest`, `run_parameter_sweep`, `run_walk_forward`).
- Example notebooks for quick start, parameter studies and attribution analysis.
