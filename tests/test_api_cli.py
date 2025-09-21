import os
from pathlib import Path

import pandas as pd
import yaml

from tf import api
from tf.cli import main as cli_main


def _make_test_config(tmp_path: Path) -> Path:
    base_cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    base_cfg["signals"]["momentum"]["lookbacks"] = [20, 40, 80]
    base_cfg["signals"]["momentum"]["skip_last_n"] = 5
    base_cfg["backtest"]["start"] = "2020-01-01"
    base_cfg["backtest"]["end"] = "2020-12-31"
    base_cfg["backtest"]["results_dir"] = str(tmp_path / "results")
    universe_path = Path("configs/universe.yaml").resolve()
    base_cfg["universe"]["assets_file"] = os.path.relpath(universe_path, start=tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base_cfg))
    return config_path


def test_api_run_backtest(tmp_path):
    config_path = _make_test_config(tmp_path)
    result, context = api.run_backtest(config_path, price_seed=99)
    assert context.config_path == config_path
    assert context.price_seed == 99
    assert not result.nav.empty
    assert isinstance(result.positions, pd.DataFrame)


def test_api_parameter_sweep(tmp_path):
    config_path = _make_test_config(tmp_path)
    sweep_results, context = api.run_parameter_sweep(
        config_path,
        parameter_grid={"signals.momentum.lookbacks": [[10], [20]]},
        price_seed=5,
    )
    assert len(sweep_results) == 2
    assert sweep_results[0].scenario.grid_overrides["signals"]["momentum"]["lookbacks"] == [10]
    assert sweep_results[1].scenario.grid_overrides["signals"]["momentum"]["lookbacks"] == [20]
    assert context.config_path == config_path


def test_api_walk_forward(tmp_path):
    config_path = _make_test_config(tmp_path)
    wf_results, _ = api.run_walk_forward(
        config_path,
        insample=60,
        oos=20,
        step=20,
        anchored=False,
        price_seed=7,
    )
    assert wf_results
    for item in wf_results:
        assert not item.oos_nav.empty


def test_cli_run_and_report(tmp_path, capsys):
    config_path = _make_test_config(tmp_path)
    run_id = "cli-run"
    cli_main(["run", "--config", str(config_path), "--run-id", run_id, "--price-seed", "3"])
    run_dir = tmp_path / "results" / run_id
    assert (run_dir / "summary.json").exists()

    cli_main(["report", "--config", str(config_path), "--run-id", run_id])
    captured = capsys.readouterr()
    assert "report.html" in captured.out


def test_cli_sweep(tmp_path):
    config_path = _make_test_config(tmp_path)
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump({"signals.momentum.lookbacks": [[10], [20]]}))

    sweep_id = "cli-sweep"
    cli_main([
        "sweep",
        "--config",
        str(config_path),
        "--run-id",
        sweep_id,
        "--grid",
        str(grid_path),
        "--price-seed",
        "4",
    ])

    sweep_dir = tmp_path / "results" / sweep_id
    assert (sweep_dir / "scenarios.json").exists()
    summary_csv = sweep_dir / "sweep_summary.csv"
    assert summary_csv.exists()
    data = pd.read_csv(summary_csv)
    assert len(data) == 2


def test_cli_walkforward(tmp_path):
    config_path = _make_test_config(tmp_path)
    walk_id = "cli-wf"
    cli_main([
        "walkforward",
        "--config",
        str(config_path),
        "--run-id",
        walk_id,
        "--insample",
        "60",
        "--oos",
        "20",
        "--step",
        "20",
        "--rolling",
        "--price-seed",
        "6",
    ])

    walk_dir = tmp_path / "results" / walk_id
    summary_csv = walk_dir / "walkforward_summary.csv"
    assert summary_csv.exists()
    # Walk-forward does not emit scenarios.json; ensure folds created
    assert any(p.name.startswith("fold-") for p in walk_dir.iterdir() if p.is_dir())
