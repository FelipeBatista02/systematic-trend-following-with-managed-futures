# Quick demo: generate synthetic data and run a backtest programmatically
import yaml
from tf.data.ingest import load_prices_or_generate
from tf.engine.backtester import Backtester
from tf.report.plots import plot_equity, plot_drawdown
from tf.eval.metrics import performance_summary
from pathlib import Path

cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
uni = yaml.safe_load(Path(cfg['universe']['assets_file']).read_text())['symbols']
prices = load_prices_or_generate(uni, cfg['backtest']['start'], cfg['backtest']['end'], seed=cfg['backtest'].get('seed',42))
bt = Backtester(prices, uni, cfg)
res = bt.run()
outdir = Path(cfg['backtest']['results_dir'])/"example"
outdir.mkdir(parents=True, exist_ok=True)
plot_equity(res.nav, outdir/"equity.png")
plot_drawdown(res.nav, outdir/"drawdown.png")
print(performance_summary(res.nav))
print(f"Artifacts in {outdir}")
