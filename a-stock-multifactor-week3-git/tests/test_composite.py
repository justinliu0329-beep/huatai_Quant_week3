import numpy as np
import pandas as pd

import config as C
from src.composite import build_score, select_weights, generate_targets


def _scores(n=100):
    codes = [f"{i:06d}" for i in range(n)]
    return pd.Series(np.linspace(1, 0, n), index=codes), codes


def test_build_score_linearity():
    fp = pd.DataFrame({"A_z": [1.0, -2.0, 0.5, np.nan], "B_z": [0.0, 1.0, -1.0, 2.0]})
    got = build_score(fp, {"A": 0.2, "B": -0.1}).values
    exp = np.array([(0.2*1 - 0.1*0)/0.3, (0.2*-2 - 0.1*1)/0.3,
                    (0.2*0.5 - 0.1*-1)/0.3, (0 - 0.1*2)/0.3])
    assert np.allclose(got, exp)


def test_build_score_missing_factor_neutral():
    fp = pd.DataFrame({"A_z": [1.0, 2.0]})
    got = build_score(fp, {"A": 1.0, "B": 1.0}).values
    assert np.allclose(got, [1.0, 2.0])


def test_select_weights_count_and_stock_cap():
    s, _ = _scores()
    w = select_weights(s, {}, n_stocks=20)
    assert len(w) == 20
    assert max(w.values()) <= C.MAX_WEIGHT_PER_STOCK + 1e-9


def test_select_weights_industry_cap():
    s, codes = _scores()
    ind = {c: "X" for c in codes[:20]}
    ind.update({c: "Y" for c in codes[20:]})
    w = select_weights(s, ind, n_stocks=20)
    xsum = sum(v for c, v in w.items() if ind.get(c) == "X")
    assert xsum <= C.MAX_WEIGHT_PER_INDUSTRY + 1e-6


def test_select_weights_small_universe_sums_to_one():
    s = pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"])
    w = select_weights(s, {}, n_stocks=20)
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_generate_targets_rebalance_nan_stop():
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    px = pd.DataFrame(100.0, index=idx, columns=["000001", "000002", "000003"])
    sc = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    sc.loc[idx[0]] = [3.0, 2.0, 1.0]
    px.loc[idx[5]:, "000001"] = 87.0
    tg = generate_targets(px, sc, {}, [idx[0]], n_stocks=3, stop_loss=-0.10, take_profit=0.20)
    assert abs(tg.loc[idx[0]].dropna().sum() - 1.0) < 1e-6
    assert tg.loc[idx[3]].isna().all()
    assert tg.loc[idx[5]].get("000001") == 0.0


def test_generate_targets_take_profit():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    px = pd.DataFrame(100.0, index=idx, columns=["000001", "000002"])
    sc = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    sc.loc[idx[0]] = [2.0, 1.0]
    px.loc[idx[3]:, "000001"] = 125.0
    tg = generate_targets(px, sc, {}, [idx[0]], n_stocks=2, stop_loss=-0.10, take_profit=0.20)
    assert tg.loc[idx[3]].get("000001") == 0.0
