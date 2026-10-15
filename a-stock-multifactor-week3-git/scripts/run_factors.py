import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config as C
from src.data_clean import build_price_panel, apply_pool_rules, build_financial_panel
from src.factors import build_factor_panel
from src.utils import save_parquet, log


def main():
    log("① 汇总价格面板 ...")
    raw = build_price_panel()
    meta = pd.read_csv(os.path.join(C.DATA_DIR, "stock_pool_raw.csv"))
    meta["code"] = meta["code"].astype(str).str.zfill(6)

    log("② 应用股票池剔除规则（ST / 次新股 / 流动性）...")
    clean = apply_pool_rules(raw, meta)
    save_parquet(clean, os.path.join(C.DATA_DIR, "price_panel_clean.parquet"))

    log("③ 汇总财务面板 ...")
    fin = build_financial_panel()

    log("④ 构建因子面板（价量 + 基本面，防未来函数）...")
    fp = build_factor_panel(clean, fin)

    log("✅ 因子面板构建完成 → data/factor_panel.parquet")


if __name__ == "__main__":
    main()
