import pandas as pd

from src.data_clean import financial_available_date, apply_pool_rules


def test_financial_available_date_mapping():
    assert financial_available_date(pd.Timestamp("2023-03-31")) == pd.Timestamp("2023-04-30")
    assert financial_available_date(pd.Timestamp("2023-06-30")) == pd.Timestamp("2023-08-31")
    assert financial_available_date(pd.Timestamp("2023-09-30")) == pd.Timestamp("2023-10-31")
    assert financial_available_date(pd.Timestamp("2023-12-31")) == pd.Timestamp("2024-04-30")


def _panel():
    dates = pd.date_range("2023-01-02", periods=40, freq="B")
    rows = []
    for c in ["000001", "000002", "000003"]:
        for d in dates:
            rows.append({"date": d, "code": c, "close": 10.0, "volume": 1e6, "amount": 1e8})
    return pd.DataFrame(rows)


def _meta(list3):
    return pd.DataFrame({"code": ["000001", "000002", "000003"],
                         "name": ["正常股", "ST某某", "新股"],
                         "list_date": [pd.Timestamp("2010-01-01")] * 2 + [list3],
                         "is_st_name": [False, True, False]})


def test_apply_pool_rules_removes_st_and_new_stock():
    panel = _panel()
    out = apply_pool_rules(panel, _meta(panel["date"].max()), verbose=False)
    codes = set(out["code"].unique())
    assert "000002" not in codes
    assert "000003" not in codes
    assert "000001" in codes


def test_apply_pool_rules_removes_illiquid():
    panel = _panel()
    panel.loc[panel["code"] == "000002", "amount"] = 1e6
    out = apply_pool_rules(panel, _meta(pd.Timestamp("2010-01-01")), verbose=False)
    assert "000002" not in set(out["code"].unique())
