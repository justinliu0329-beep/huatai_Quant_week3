import os
import time
import functools
import datetime as dt

import pandas as pd


def retry(times=4, sleep=1.5):

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for i in range(times):
                try:
                    r = fn(*args, **kwargs)
                    if r is not None:
                        return r
                except Exception as e:
                    last = e
                time.sleep(sleep * (i + 1))
            if last is not None:
                raise last
            return None
        return wrapper
    return deco


def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def ymd_to_iso(s: str) -> str:

    s = str(s)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def to_date(x) -> dt.date:

    if isinstance(x, dt.date):
        return x
    return pd.to_datetime(str(x)).date()


def days_between(d1, d2) -> int:
    return (to_date(d2) - to_date(d1)).days


def log(msg: str):
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
