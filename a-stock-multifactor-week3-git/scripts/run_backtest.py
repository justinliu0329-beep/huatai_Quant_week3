import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config as C
from src.utils import load_parquet, save_parquet, log
from src.factor_test import weekly_rebalance_dates
from src.composite import (build_score, factor_weights_from_ic,
                           generate_targets, select_weights, EFFECTIVE_FACTORS)
from src.backtest import run_backtest
from src.performance import (performance_metrics, plot_nav_vs_benchmark,
                             plot_drawdown, plot_monthly_heatmap)


def _load_benchmark():
    b = pd.read_csv(os.path.join(C.DATA_DIR, "benchmark.csv"))
    b["date"] = pd.to_datetime(b["date"])
    return b.set_index("date")["close"].sort_index()


def main():
    log("① 载入清洗面板 / 因子面板 / 基准 ...")
    clean = load_parquet(os.path.join(C.DATA_DIR, "price_panel_clean.parquet"))
    fp = load_parquet(os.path.join(C.DATA_DIR, "factor_panel_final.parquet"))
    meta = pd.read_csv(os.path.join(C.DATA_DIR, "stock_pool_raw.csv"))
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    bench = _load_benchmark()
    industry = dict(zip(meta["code"], meta["industry"]))


    fw = factor_weights_from_ic(os.path.join(C.TAB_DIR, "ic_summary.csv"))
    log(f"② IC 加权（有效因子）：{ {k: round(v,4) for k,v in fw.items()} }")


    prices = clean.pivot_table(index="date", columns="code", values="close")
    score = build_score(fp, fw)
    score_wide = fp.assign(_score=score).pivot_table(index="date", columns="code", values="_score")


    reb = [d for d in weekly_rebalance_dates(fp[fp["date"] >= C.TRAIN_START]["date"])
           if d in prices.index]
    log(f"③ 周度调仓日：{len(reb)} 个（{C.TRAIN_START}~）")


    log("④ 生成目标权重（选股+仓位约束+止盈止损）...")
    targets = generate_targets(prices, score_wide, industry, reb,
                               n_stocks=C.MAX_HOLD, stop_loss=-0.10, take_profit=0.20)


    log("⑤ VectorBT 回测（撮合/净值）+ 自研成本扣减 ...")
    res = run_backtest(prices, targets)
    log(f"   累计交易成本：{res['total_cost']:,.0f} 元（初始 {C.INIT_CAPITAL:,.0f}）")


    net = res["net_nav"]
    full = performance_metrics(net, bench)
    test = performance_metrics(net[(net.index >= C.TEST_START) & (net.index <= C.TEST_END)],
                              bench)
    valid = performance_metrics(net[(net.index >= C.VALID_START) & (net.index <= C.VALID_END)],
                               bench)
    tbl = pd.DataFrame({"全样本": full, "验证集2025": valid, "测试集2026": test})
    tbl.to_csv(os.path.join(C.TAB_DIR, "static_backtest_metrics.csv"), encoding="utf-8-sig")
    log("\n【静态多因子策略 绩效】\n" + tbl.to_string())


    plot_nav_vs_benchmark(net, bench, os.path.join(C.FIG_DIR, "static_nav_vs_benchmark.png"))
    plot_drawdown(net, os.path.join(C.FIG_DIR, "static_drawdown.png"))
    plot_monthly_heatmap(net, os.path.join(C.FIG_DIR, "static_monthly_returns.png"))


    res["orders"].to_csv(os.path.join(C.TAB_DIR, "static_orders.csv"),
                         index=False, encoding="utf-8-sig")


    _write_report(fw, tbl, res)
    log("\n✅ 静态多因子回测完成：output/figures/static_*.png，"
        "output/tables/static_*，reports/week2_static_backtest.md")


def _write_report(fw, tbl, res):
    L = ["# Week2 静态多因子基准策略 回测报告\n",
         f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n",
         "## 一、策略设计\n",
         f"- **有效因子**（Week1 筛选，|t|>2）：{', '.join(EFFECTIVE_FACTORS)}",
         f"- **合成方式**：IC 加权，score = Σ(IC_f · z_f)/Σ|IC_f|。"
         f"权重：{ {k: round(v,4) for k,v in fw.items()} }",
         "- **选择理由**：各因子 IC 量级差异大（波动率 -0.075 显著强于反转 -0.032），"
         "等权会稀释强因子；IC 加权按预测力配比，信息利用更有效。",
         f"- **调仓**：周度（每周最后交易日打分，次日起生效，符合 T+1）",
         f"- **持仓约束**：{C.MIN_HOLD}~{C.MAX_HOLD} 只等权；单票 ≤{C.MAX_WEIGHT_PER_STOCK:.0%}；"
         f"单行业 ≤{C.MAX_WEIGHT_PER_INDUSTRY:.0%}",
         f"- **成交成本**：佣金 {C.COMMISSION:.2%}(双边) + 印花税 {C.STAMP_TAX:.2%}(卖出) + "
         f"滑点 {C.SLIPPAGE:.2%}，逐笔按成交额核算",
         f"- **止盈止损**：单票 -10% 止损 / +20% 止盈（持有期每日检查）",
         f"- **回测框架**：VectorBT（撮合/净值） + 自研调仓与成本模型",
         f"- **累计交易成本**：{res['total_cost']:,.0f} 元\n",
         "## 二、绩效指标\n", tbl.to_markdown(), "\n",
         "## 三、交付图件\n",
         "- `static_nav_vs_benchmark.png`：净值 vs 沪深300",
         "- `static_drawdown.png`：回撤曲线",
         "- `static_monthly_returns.png`：月度收益热力图",
         "- `static_orders.csv`：逐笔成交明细（含成本）",
         "\n---\n> 免责声明：本项目仅用于实习量化研究能力考核与学习，不构成任何投资建议。"]
    with open(os.path.join(C.REPORT_DIR, "week2_static_backtest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
