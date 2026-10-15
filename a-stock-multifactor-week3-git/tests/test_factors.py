import numpy as np
import pandas as pd

from src.factors import compute_price_factors


def _panel(n=60, codes=("000001", "000002")):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    rows = []
    for c in codes:
        p = 100 + np.cumsum(rng.normal(0, 1, n))
        for i, d in enumerate(dates):
            rows.append({"date": d, "code": c, "close": p[i], "turnover": 0.01,
                         "outstanding_share": 1e9, "volume": 1e6, "amount": 1e7})
    return pd.DataFrame(rows)


def test_price_factor_columns():
    df = compute_price_factors(_panel())
    for c in ["momentum_20", "reversal_5", "volatility_20", "turnover_20"]:
        assert c in df.columns


def test_no_lookahead_price_factor():

    panel = _panel(n=60)
    t = pd.Timestamp("2024-02-01")
    f1 = compute_price_factors(panel)
    base = f1[(f1["code"] == "000001") & (f1["date"] == t)]["momentum_20"].iloc[0]

    panel2 = panel.copy()
    panel2.loc[(panel2["code"] == "000001") & (panel2["date"] > t), "close"] *= 2.0
    f2 = compute_price_factors(panel2)
    after = f2[(f2["code"] == "000001") & (f2["date"] == t)]["momentum_20"].iloc[0]

    assert np.isclose(base, after)
