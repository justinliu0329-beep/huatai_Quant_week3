import numpy as np
import pandas as pd

from src.factor_test import (weekly_rebalance_dates, compute_ic,
                             factor_correlation, layered_backtest)


def test_weekly_rebalance_last_of_each_week():
    dates = pd.date_range("2024-01-01", "2024-03-29", freq="B")
    wk = weekly_rebalance_dates(dates)
    iso = lambda d: (d.isocalendar()[0], d.isocalendar()[1])
    weeks = {iso(d) for d in dates}
    assert len(wk) == len(weeks)
    for d in wk:
        assert d == max(x for x in dates if iso(x) == iso(d))
    assert all(d.dayofweek == 4 for d in wk)


def test_compute_ic_perfect_rank():
    d = pd.Timestamp("2024-01-05")
    fp = pd.DataFrame({"date": d, "code": [f"{i:06d}" for i in range(30)],
                       "x_z": [float(i) for i in range(30)],
                       "fwd_ret": [float(i) for i in range(30)]})
    ic = compute_ic(fp, "x", [d])
    assert abs(ic.iloc[0] - 1.0) < 1e-9


def test_factor_correlation_symmetric_no_nan():

    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2024-01-01", periods=40, freq="B"):
        for i in range(50):
            rows.append({"date": d, "code": f"{i:06d}",
                         "a_z": rng.normal(),
                         "b_z": (rng.normal() if i % 2 == 0 else np.nan)})
    corr = factor_correlation(pd.DataFrame(rows), ["a", "b"])
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr.values), 1.0)
    assert not corr.isna().any().any()
    assert np.allclose(corr.values, corr.values.T)


def test_layered_backtest_groups():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    rows = []
    for d in dates:
        for i in range(60):
            rows.append({"date": d, "code": f"{i:06d}",
                         "x_z": rng.normal(), "fwd_ret": rng.normal() * 0.01})
    grp, cum, ls = layered_backtest(pd.DataFrame(rows), "x", dates)
    assert list(cum.columns) == ["G1", "G2", "G3", "G4", "G5"]
