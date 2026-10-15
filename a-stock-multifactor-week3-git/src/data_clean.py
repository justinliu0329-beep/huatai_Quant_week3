import os
import glob

import numpy as np
import pandas as pd

import config as C
from src.utils import save_parquet, load_parquet, log


def build_price_panel(save=True) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(C.DAILY_CACHE, "*.parquet")))
    if not files:
        raise RuntimeError("日线缓存为空，请先运行 scripts/run_data.py")
    frames = [pd.read_parquet(f) for f in files]
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel = panel.drop_duplicates(["date", "code"]).sort_values(["code", "date"]).reset_index(drop=True)
    if save:
        save_parquet(panel, os.path.join(C.DATA_DIR, "price_panel.parquet"))
    log(f"✓ 价格面板：{len(panel):,} 行，{panel['code'].nunique()} 只，"
        f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")
    return panel


def build_financial_panel(save=True) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(C.FIN_CACHE, "*.parquet")))
    if not files:
        log("⚠ 财务缓存为空")
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    fin = pd.concat(frames, ignore_index=True)
    fin["code"] = fin["code"].astype(str).str.zfill(6)
    fin["日期"] = pd.to_datetime(fin["日期"])
    if save:
        save_parquet(fin, os.path.join(C.DATA_DIR, "financial_panel.parquet"))
    log(f"✓ 财务面板：{len(fin):,} 行，{fin['code'].nunique()} 只")
    return fin


def financial_available_date(report_date: pd.Timestamp) -> pd.Timestamp:

    y, m = report_date.year, report_date.month
    if m == 3:
        return pd.Timestamp(year=y, month=4, day=30)
    if m == 6:
        return pd.Timestamp(year=y, month=8, day=31)
    if m == 9:
        return pd.Timestamp(year=y, month=10, day=31)
    if m == 12:
        return pd.Timestamp(year=y + 1, month=4, day=30)

    return report_date + pd.Timedelta(days=90)


def apply_pool_rules(panel: pd.DataFrame, meta: pd.DataFrame,
                     verbose=True) -> pd.DataFrame:


    df = panel.copy()
    n0, s0 = len(df), df["code"].nunique()


    df = df[(df["volume"] > 0) & df["close"].notna()]
    nA = n0 - len(df)


    m = meta[["code", "name", "list_date", "is_st_name"]].drop_duplicates("code")
    df = df.drop(columns=[c for c in ["name", "list_date", "is_st_name"] if c in df.columns])
    df = df.merge(m, on="code", how="left")


    st_mask = df["is_st_name"].fillna(False)
    nB = int(st_mask.sum())
    df = df[~st_mask]


    df["_listed_days"] = (df["date"] - pd.to_datetime(df["list_date"])).dt.days
    new_mask = df["_listed_days"].isna() | (df["_listed_days"] < C.MIN_LISTED_DAYS)
    nC = int(new_mask.sum())
    df = df[~new_mask]


    avg_amt = df.groupby("code")["amount"].mean()
    illiquid = set(avg_amt[avg_amt < C.MIN_AVG_AMOUNT].index)
    nD = int(df["code"].isin(illiquid).sum())
    df = df[~df["code"].isin(illiquid)]

    df["eligible"] = True
    df = df.drop(columns=["_listed_days"]).reset_index(drop=True)

    if verbose:
        log("── 股票池剔除统计 ──")
        log(f"   初始: {n0:,} 行 / {s0} 只")
        log(f"   A 停牌无成交: -{nA:,} 行")
        log(f"   B ST/*ST : -{nB:,} 行")
        log(f"   C 次新股 <120日: -{nC:,} 行")
        log(f"   D 流动性<5000万: -{nD:,} 行（剔除 {len(illiquid)} 只）")
        log(f"   最终: {len(df):,} 行 / {df['code'].nunique()} 只")
    return df


def split_panel(panel: pd.DataFrame):

    out = {}
    for k, (s, e) in C.SPLITS.items():
        sub = panel[(panel["date"] >= s) & (panel["date"] <= e)]
        out[k] = sub.reset_index(drop=True)
        log(f"   {k:5s}: {len(sub):,} 行 / {sub['code'].nunique()} 只  ({s}~{e})")
    return out


def load_clean_panel():

    p = os.path.join(C.DATA_DIR, "price_panel_clean.parquet")
    cached = load_parquet(p)
    if cached is not None:
        return cached
    raw = build_price_panel(save=False)
    meta = pd.read_csv(os.path.join(C.DATA_DIR, "stock_pool_raw.csv"))
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    clean = apply_pool_rules(raw, meta)
    save_parquet(clean, p)
    return clean


if __name__ == "__main__":
    raw = build_price_panel()
    meta = pd.read_csv(os.path.join(C.DATA_DIR, "stock_pool_raw.csv"))
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    clean = apply_pool_rules(raw, meta)
    save_parquet(clean, os.path.join(C.DATA_DIR, "price_panel_clean.parquet"))
    build_financial_panel()
    log("切分：")
    split_panel(clean)
