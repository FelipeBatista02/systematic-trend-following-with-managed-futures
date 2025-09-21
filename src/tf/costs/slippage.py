import numpy as np
import pandas as pd

def simple_slippage(trade_notional: pd.Series, k=0.05, alpha=0.5, tick_value=10.0, min_ticks=0.5):
    impact = k * (trade_notional.abs() ** alpha)
    min_cost = min_ticks * tick_value
    return impact.clip(lower=min_cost)
