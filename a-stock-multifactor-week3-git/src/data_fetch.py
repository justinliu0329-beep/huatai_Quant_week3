import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import akshare as ak
import pandas as pd

import config as C
from src.utils import retry, save_parquet, load_parquet, log


def to_sina_code(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith("6"):
        return "sh" + c
    if c.startswith(("0", "3")):
        return "sz" + c
    if c.startswith(("4", "8", "9")):
        return "bj" + c
    return "sz" + c


@retry(times=C.MAX_RETRY, sleep=C.RETRY_SLEEP)
def _stock_list():
    return ak.stock_info_a_code_name()


@retry(times=C.MAX_RETRY, sleep=C.RETRY_SLEEP)
def _sh_list():
    return ak.stock_info_sh_name_code(symbol="主板A股")


@retry(times=C.MAX_RETRY, sleep=C.RETRY_SLEEP)
def _sz_list():
    return ak.stock_info_sz_name_code(symbol="A股列表")


def _board_of(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith("688"):
        return "科创板"
    if c.startswith(("300", "301")):
        return "创业板"
    if c.startswith(("8", "4", "920")):
        return "北交所"
    if c.startswith("6"):
        return "沪主板"
    if c.startswith(("000", "001", "002", "003")):
        return "深主板"
    return "其他"


def fetch_stock_list(save=True) -> pd.DataFrame:

    base = _stock_list()[["code", "name"]].copy()
    base["code"] = base["code"].astype(str).str.zfill(6)

    frames = []
    try:
        sh = _sh_list().rename(columns={"证券代码": "code", "上市日期": "list_date"})
        sh["code"] = sh["code"].astype(str).str.zfill(6)
        sh["exchange"] = "SH"
        frames.append(sh[["code", "list_date", "exchange"]])
    except Exception as e:
        log(f"⚠ 沪市列表获取失败：{e}")
    try:
        sz = _sz_list().rename(columns={"A股代码": "code", "A股上市日期": "list_date",
                                        "所属行业": "industry"})
        sz["code"] = sz["code"].astype(str).str.zfill(6)
        sz["exchange"] = "SZ"
        frames.append(sz[["code", "list_date", "exchange", "industry"]])
    except Exception as e:
        log(f"⚠ 深市列表获取失败：{e}")

    if frames:
        info = pd.concat(frames, ignore_index=True)
        if "industry" not in info.columns:
            info["industry"] = None
        df = base.merge(info, on="code", how="left")
    else:
        df = base.copy()
        df["list_date"] = None
        df["exchange"] = None
        df["industry"] = None

    df["board"] = df["code"].map(_board_of)
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")

    df["is_st_name"] = df["name"].astype(str).str.upper().str.replace(" ", "").str.contains("ST")
    if save:
        df.to_csv(os.path.join(C.DATA_DIR, "stock_pool_raw.csv"), index=False)
    log(f"✓ 股票列表：{len(df)} 只（含上市日期 {df['list_date'].notna().sum()} 只）")
    return df


@retry(times=C.MAX_RETRY, sleep=C.RETRY_SLEEP)
def _daily_sina(sina_code, start, end):
    return ak.stock_zh_a_daily(symbol=sina_code,
                               start_date=start.replace("-", ""),
                               end_date=end.replace("-", ""),
                               adjust="qfq")


def fetch_daily(code, start=C.DATA_START, end=C.DATA_END, use_cache=True):

    path = os.path.join(C.DAILY_CACHE, f"{code}.parquet")
    if use_cache:
        cached = load_parquet(path)
        if cached is not None:
            return cached
    try:
        df = _daily_sina(to_sina_code(code), start, end)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = str(code).zfill(6)
    df = df.rename(columns={"turnover": "turnover"})
    df = df.sort_values("date").reset_index(drop=True)

    keep = ["date", "code", "open", "high", "low", "close", "volume",
            "amount", "turnover", "outstanding_share"]
    df = df[[c for c in keep if c in df.columns]]
    save_parquet(df, path)
    return df


def _parse_cn_num(x):

    if x is None or x is False:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in ("", "False", "None", "--", "nan", "NaN"):
        return np.nan
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        mul = 1.0
        if s.endswith("亿"):
            mul, s = 1e8, s[:-1]
        elif s.endswith("万"):
            mul, s = 1e4, s[:-1]
        return float(s) * mul
    except ValueError:
        return np.nan


@retry(times=C.MAX_RETRY, sleep=C.RETRY_SLEEP)
def _financial_ths(code):
    return ak.stock_financial_abstract_ths(symbol=str(code).zfill(6), indicator="按报告期")


def fetch_financial(code, use_cache=True):

    path = os.path.join(C.FIN_CACHE, f"{code}.parquet")
    if use_cache:
        cached = load_parquet(path)
        if cached is not None:
            return cached
    try:
        df = _financial_ths(code)
    except Exception:
        return None
    if df is None or df.empty or "报告期" not in df.columns:
        return None

    def col(name):
        return df[name].map(_parse_cn_num) if name in df.columns else np.nan

    out = pd.DataFrame({
        "code": str(code).zfill(6),
        "日期": pd.to_datetime(df["报告期"], errors="coerce"),
        "roe": col("净资产收益率"),
        "gross_margin": col("销售毛利率"),
        "bvps": col("每股净资产"),
        "eps": col("基本每股收益"),
        "growth": col("净利润同比增长率"),
        "debt_ratio": col("资产负债率"),
    }).dropna(subset=["日期"]).sort_values("日期")
    save_parquet(out, path)
    return out


def batch_download(codes, kind="daily", use_cache=True):

    assert kind in ("daily", "fin")
    worker = fetch_daily if kind == "daily" else fetch_financial

    ok, fail, done = 0, 0, 0
    total = len(codes)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=C.N_WORKERS) as ex:
        futs = {ex.submit(worker, c, use_cache=use_cache): c for c in codes}
        for f in as_completed(futs):
            done += 1
            try:
                if f.result() is not None:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if done % 250 == 0 or done == total:
                el = time.time() - t0
                eta = el / done * (total - done) if done else 0
                log(f"  {kind} 进度 {done}/{total}  成功{ok} 失败{fail}  "
                    f"用时{el:.0f}s  预计剩余{eta:.0f}s")
    log(f"✓ {kind} 下载完成：成功 {ok}，失败 {fail}，总用时 {time.time()-t0:.0f}s")
    return ok, fail


@retry(times=C.MAX_RETRY, sleep=C.RETRY_SLEEP)
def _benchmark():
    return ak.stock_zh_index_daily(symbol=C.BENCHMARK)


def fetch_benchmark(save=True):
    df = _benchmark().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= C.DATA_START) & (df["date"] <= C.DATA_END)]
    df = df.sort_values("date").reset_index(drop=True)
    if save:
        df.to_csv(os.path.join(C.DATA_DIR, "benchmark.csv"), index=False)
    log(f"✓ 基准 {C.BENCHMARK}：{len(df)} 个交易日")
    return df
