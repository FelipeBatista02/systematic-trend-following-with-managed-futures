import numpy as np
import pandas as pd

def generate_synthetic_prices(symbols, start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, freq='C')
    data = {}
    # Create mild trending random walks with regime switches
    for i, sym in enumerate(symbols):
        mu_daily = 0.0002 + 0.0001 * np.sin(i)  # slight symbol variation
        vol_daily = 0.01 + 0.002 * (i % 3)      # heteroskedastic
        shocks = rng.normal(mu_daily, vol_daily, size=len(dates))
        # inject a few trend regimes
        for k in range(50, len(shocks), 500):
            shocks[k:k+100] += 0.0008 * np.sign(np.sin(k+i))
        levels = 100*np.exp(np.cumsum(shocks))
        data[sym] = levels
    prices = pd.DataFrame(data, index=dates)
    return prices
