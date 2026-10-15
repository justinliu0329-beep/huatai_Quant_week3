import numpy as np
import pandas as pd

import config as C
from src.factor_test import weekly_rebalance_dates
from src.composite import generate_targets
from src.backtest import run_backtest
from src.volatility import exposure_schedule


WINDOWS = {
    "2020疫情暴跌": ("2020-01-20", "2020-03-31"),
    "2022熊市底部": ("2022-03-01", "2022-05-31"),
}


def run_window(prices: pd.DataFrame, score_wide: pd.DataFrame, industry: dict, name: str,
               market_returns: pd.Series = None, dynamic: bool = False,
               exposure_fn=None, n_stocks: int = C.MAX_HOLD, stop_loss: float = -0.10,
               take_profit: float = 0.20) -> dict:

    s, e = pd.Timestamp(WINDOWS[name][0]), pd.Timestamp(WINDOWS[name][1])
    px = prices.loc[(prices.index >= s) & (prices.index <= e)]
    if len(px) < 5:
        return None
    reb = weekly_rebalance_dates(px.index)
    if px.index[0] not in reb:
        reb = [px.index[0]] + list(reb)
    sc = score_wide.reindex(px.index)
    ex = None
    if exposure_fn is not None:
        ex = exposure_fn(reb)
    elif dynamic and market_returns is not None:
        ex = exposure_schedule(market_returns, reb)
    tg = generate_targets(px, sc, industry, reb, n_stocks, stop_loss, take_profit, exposure=ex)
    res = run_backtest(px, tg)
    res["window"], res["exposure"] = name, ex
    return res


def factor_ic_in_window(fp: pd.DataFrame, name: str, factors) -> dict:

    s, e = pd.Timestamp(WINDOWS[name][0]), pd.Timestamp(WINDOWS[name][1])
    sub = fp[(fp["date"] >= s) & (fp["date"] <= e)]
    out = {}
    for f in factors:
        zc = f + "_z"
        if zc not in sub.columns:
            continue
        ics = []
        for _, g in sub.groupby("date"):
            g = g.dropna(subset=[zc, "fwd_ret"])
            if len(g) >= 30:
                ics.append(g[zc].corr(g["fwd_ret"], method="spearman"))
        out[f] = round(float(np.mean(ics)), 4) if ics else np.nan
    return out
