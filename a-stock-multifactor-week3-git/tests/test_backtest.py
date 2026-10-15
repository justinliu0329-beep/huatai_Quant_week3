import numpy as np
import pandas as pd

from src.composite import generate_targets
from src.backtest import run_backtest


def _mk(n=20, k=4):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    px = pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (n, k)), axis=0), index=idx,
                      columns=[f"{i:06d}" for i in range(k)])
    sc = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    sc.loc[idx[0]] = list(range(k, 0, -1))
    sc.loc[idx[n // 2]] = list(range(1, k + 1))
    return px, sc, [px.index[0], px.index[n // 2]]


def test_run_backtest_invariants():
    px, sc, reb = _mk()
    tg = generate_targets(px, sc, {}, reb, n_stocks=4)
    res = run_backtest(px, tg)
    assert (res["net_nav"] <= res["gross_nav"] + 1e-6).all()
    assert res["total_cost"] >= 0
    assert (res["net_nav"] > 0).all()


def test_run_backtest_cost_positive_when_trading():
    px, sc, reb = _mk()
    tg = generate_targets(px, sc, {}, reb, n_stocks=4)
    res = run_backtest(px, tg)
    assert res["total_cost"] > 0
