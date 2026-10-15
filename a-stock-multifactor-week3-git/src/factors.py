import numpy as np
import pandas as pd

import config as C
from src.data_clean import financial_available_date
from src.utils import save_parquet, log
import os


TECH_FACTORS = ["momentum_20", "reversal_5", "volatility_20", "turnover_20"]
FUND_FACTORS = ["roe", "gross_margin", "pb", "growth"]


EXTRA_FACTORS = ["size"]
ALL_FACTORS = TECH_FACTORS + FUND_FACTORS

FACTOR_CN = {
    "momentum_20": "动量(20日)", "reversal_5": "短期反转(5日)",
    "volatility_20": "波动率(20日)", "turnover_20": "流动性/换手(20日)",
    "roe": "净资产收益率ROE", "gross_margin": "销售毛利率", "pb": "市净率PB",
    "size": "市值(对数)", "growth": "净利润增长率",
}


def compute_price_factors(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["code", "date"]).copy()
    g = df.groupby("code", sort=False)

    df["ret"] = g["close"].pct_change()
    df["momentum_20"] = df["close"] / g["close"].shift(20) - 1
    df["reversal_5"] = df["close"] / g["close"].shift(5) - 1
    df["volatility_20"] = g["ret"].transform(lambda s: s.rolling(20, min_periods=15).std())
    df["turnover_20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).mean())


    if "outstanding_share" in df.columns:
        df["size"] = np.log(df["close"] * df["outstanding_share"] + 1)
    else:
        df["size"] = np.nan
    return df


def _extract_financial_features(fin: pd.DataFrame) -> pd.DataFrame:

    f = fin.copy()
    f["avail_date"] = f["日期"].map(financial_available_date)
    for c in ["roe", "gross_margin", "bvps", "eps", "growth"]:
        if c not in f.columns:
            f[c] = np.nan
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f[["code", "avail_date", "roe", "gross_margin", "bvps", "eps", "growth"]]
    f = f.dropna(subset=["avail_date"]).sort_values(["code", "avail_date"])
    return f


def merge_fundamentals(price_factors: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:

    feats = _extract_financial_features(fin)
    left = price_factors.sort_values("date").copy()

    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    right = feats.rename(columns={"avail_date": "date"}).sort_values("date").copy()
    right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(left, right, on="date", by="code", direction="backward")
    merged["pb"] = merged["close"] / merged["bvps"].replace(0, np.nan)
    return merged


def _mad_winsorize(s: pd.Series, n: float = 5.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    upper, lower = med + n * 1.4826 * mad, med - n * 1.4826 * mad
    return s.clip(lower, upper)


def standardize(df: pd.DataFrame, factors) -> pd.DataFrame:
    df = df.copy()
    for c in factors:
        df[c] = df.groupby("date")[c].transform(_mad_winsorize)
        g = df.groupby("date")[c]
        df[c + "_z"] = (df[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return df


def add_forward_return(df: pd.DataFrame, horizon: int = C.FORWARD_PERIOD) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).copy()
    g = df.groupby("code", sort=False)
    df["fwd_ret"] = g["close"].shift(-horizon) / df["close"] - 1
    return df


def build_factor_panel(clean_panel: pd.DataFrame, fin_panel: pd.DataFrame,
                       save=True) -> pd.DataFrame:
    log("计算价量因子 ...")
    pf = compute_price_factors(clean_panel)
    log("对齐财务因子（防未来函数）...")
    fp = merge_fundamentals(pf, fin_panel)
    log("横截面去极值 + 标准化 ...")
    fp = standardize(fp, TECH_FACTORS + FUND_FACTORS)
    log("计算前瞻收益 ...")
    fp = add_forward_return(fp)
    if save:
        save_parquet(fp, os.path.join(C.DATA_DIR, "factor_panel.parquet"))
    log(f"✓ 因子面板：{len(fp):,} 行，{len(ALL_FACTORS)} 个因子")
    return fp
