import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config as C
from src.utils import load_parquet, log
from src.composite import (build_score, factor_weights_from_ic,
                           select_weights, EFFECTIVE_FACTORS)


try:
    from xtquant import xtconstant
    from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
    from xtquant.xttype import StockAccount
    _XTQUANT_AVAILABLE = True
except Exception:
    _XTQUANT_AVAILABLE = False


def normalize_symbol(code: str) -> str:

    c = str(code).zfill(6)
    if c.startswith("6"):
        return c + ".SH"
    if c.startswith(("0", "3")):
        return c + ".SZ"
    if c.startswith(("4", "8", "9")):
        return c + ".BJ"
    return c


def latest_target(as_of: str = None) -> pd.Series:

    fp = load_parquet(os.path.join(C.DATA_DIR, "factor_panel_final.parquet"))
    meta = pd.read_csv(os.path.join(C.DATA_DIR, "stock_pool_raw.csv"))
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    industry = dict(zip(meta["code"], meta["industry"]))

    fw = factor_weights_from_ic(os.path.join(C.TAB_DIR, "ic_summary.csv"))
    d = pd.Timestamp(as_of) if as_of else fp["date"].max()
    day = fp[fp["date"] == d]
    score = build_score(day, fw)
    scores = day.assign(_score=score).set_index("code")["_score"]
    w = select_weights(scores, industry, C.MAX_HOLD)
    log(f"调仓日 {d.date()}：目标持仓 {len(w)} 只，有效因子权重 "
        f"{ {k: round(v,4) for k,v in fw.items()} }")
    return pd.Series(w)


def make_orders(target: pd.Series, current: dict = None) -> pd.DataFrame:

    current = current or {}
    codes = sorted(set(target.index) | set(current))
    rows = []
    for c in codes:
        tgt_w, cur_w = target.get(c, 0.0), current.get(c, 0.0)
        if tgt_w > cur_w + 1e-6:
            side = "BUY"
        elif tgt_w < cur_w - 1e-6:
            side = "SELL"
        else:
            continue
        rows.append({"证券代码": normalize_symbol(c), "方向": side,
                     "目标权重": round(tgt_w, 4), "当前权重": round(cur_w, 4),
                     "变动": round(tgt_w - cur_w, 4)})
    return pd.DataFrame(rows)


def execute_on_qmt(account_id: str, qmt_path: str, orders: pd.DataFrame,
                   total_capital: float = C.INIT_CAPITAL):

    if not _XTQUANT_AVAILABLE:
        log("⚠ 未检测到 xtquant，跳过下单（本地仿真模式）")
        return
    trader = XtQuantTrader(qmt_path, int(pd.Timestamp.now().timestamp()))
    trader.start()
    if trader.connect() != 0:
        log("✗ QMT 连接失败"); return
    acc = StockAccount(account_id)
    trader.subscribe(acc)

    for _, r in orders.iterrows():

        log(f"  {r['方向']} {r['证券代码']}  目标权重 {r['目标权重']:.2%}")
        side = (xtconstant.STOCK_BUY if r["方向"] == "BUY" else xtconstant.STOCK_SELL)

    trader.stop()
    log("✓ 信号已下发（骨架示意）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None, help="调仓日 YYYY-MM-DD（默认最新）")
    ap.add_argument("--execute", action="store_true", help="连接 QMT 下单（需 QMT 环境）")
    ap.add_argument("--account", default="", help="QMT 资金账号")
    ap.add_argument("--qmt-path", default=r"C:\\QMT\\userdata_mini", help="QMT userdata_mini 路径")
    args = ap.parse_args()

    target = latest_target(args.as_of)
    orders = make_orders(target)

    out = os.path.join(C.TAB_DIR, "qmt_weekly_signals.csv")
    orders.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n=== 本周调仓信号 ===")
    print(orders.to_string(index=False) if len(orders) else "（无变动）")
    print(f"\n目标持仓明细（等权，单票 {1/len(target):.1%}）：")
    for c, w in target.items():
        print(f"  {normalize_symbol(c):12s}  {w:.2%}")
    log(f"信号已保存 → {out}")

    if args.execute:
        execute_on_qmt(args.account, args.qmt_path, orders)


if __name__ == "__main__":
    main()
