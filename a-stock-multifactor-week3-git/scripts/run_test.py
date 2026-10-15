import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config as C
from src.utils import load_parquet, save_parquet, log
from src import factor_test as ft
from src.factors import ALL_FACTORS, FACTOR_CN, TECH_FACTORS, FUND_FACTORS


def main():
    fp = load_parquet(os.path.join(C.DATA_DIR, "factor_panel.parquet"))
    if fp is None:
        raise SystemExit("请先运行 scripts/run_factors.py 生成因子面板")
    factors = ALL_FACTORS
    log(f"因子面板 {len(fp):,} 行；因子 {len(factors)} 个")


    train = fp[(fp["date"] >= C.TRAIN_START) & (fp["date"] <= C.TRAIN_END)]
    reb_train = ft.weekly_rebalance_dates(train["date"])
    log(f"① 训练集 IC/IR（调仓日 {len(reb_train)} 个）...")
    summary, ic_series = ft.run_ic_all(train, factors, reb_train)
    summary.insert(1, "因子中文", summary["因子"].map(FACTOR_CN))
    summary["类别"] = summary["因子"].map(
        lambda f: "技术" if f in TECH_FACTORS else ("基本面" if f in FUND_FACTORS else "扩展"))
    summary.to_csv(os.path.join(C.TAB_DIR, "ic_summary.csv"), index=False, encoding="utf-8-sig")
    log("\n" + summary.to_string(index=False))


    log("\n② 全样本 5 组分层回测 ...")
    reb_all = ft.weekly_rebalance_dates(fp["date"])
    mono_rows = []
    for f in factors:
        grp_ret, cum, ls_cum = ft.layered_backtest(fp, f, reb_all)
        grp_ret.to_csv(os.path.join(C.TAB_DIR, f"layer_{f}.csv"), encoding="utf-8-sig")
        if cum.shape[1] >= 2:
            ft.plot_layered(cum, f, os.path.join(C.FIG_DIR, f"layer_{f}.png"))
        mono = ft.monotonicity(cum)

        ls_days = len(ls_cum)
        ls_final = float(ls_cum.iloc[-1]) if ls_days else np.nan
        ls_ann = ls_final ** (52 / max(ls_days, 1)) - 1 if ls_days else np.nan
        mono_rows.append({"因子": f, "因子中文": FACTOR_CN[f],
                          "分层单调性(Spearman)": round(mono, 3) if mono == mono else np.nan,
                          "多空组合累计": round(ls_final, 3),
                          "多空组合年化": round(ls_ann, 3),
                          "G5累计": round(float(cum.iloc[-1, -1]), 3) if cum.shape[1] >= 2 else np.nan,
                          "G1累计": round(float(cum.iloc[-1, 0]), 3) if cum.shape[1] >= 2 else np.nan})
    mono_df = pd.DataFrame(mono_rows)
    mono_df.to_csv(os.path.join(C.TAB_DIR, "layered_monotonicity.csv"),
                   index=False, encoding="utf-8-sig")
    log("\n" + mono_df.to_string(index=False))


    log("\n③ 因子相关性 & 正交化 ...")
    corr = ft.factor_correlation(fp, factors)
    corr.index = [FACTOR_CN.get(i, i) for i in corr.index]
    corr.columns = [FACTOR_CN.get(c, c) for c in corr.columns]
    corr.to_csv(os.path.join(C.TAB_DIR, "factor_corr.csv"), encoding="utf-8-sig")
    ft.plot_correlation(corr, os.path.join(C.FIG_DIR, "factor_corr.png"))

    fp2, kept, ortho, dropped = ft.orthogonalize(fp, factors, summary)
    log(f"   保留因子：{kept}")
    log(f"   正交化因子：{ortho}")
    log(f"   剔除冗余：{dropped}")


    ft.plot_ic_bar(summary, os.path.join(C.FIG_DIR, "ic_bar.png"))
    ft.plot_ic_ts(ic_series, os.path.join(C.FIG_DIR, "ic_ts.png"))


    keep_cols = ["date", "code", "close"] +\
                [f + "_z" for f in factors] + [c for c in fp2.columns if c.endswith("_orth")] + ["fwd_ret"]
    save_parquet(fp2[[c for c in keep_cols if c in fp2.columns]],
                 os.path.join(C.DATA_DIR, "factor_panel_final.parquet"))


    _write_report(summary, mono_df, corr, fp)
    log("\n✅ 因子检验完成，图表见 output/，报告见 reports/week1_factor_report.md")


FACTOR_LOGIC = {
    "momentum_20": ("技术", "过去20日收益率（动量/反转效应）", "A股短周期动量偏反转"),
    "reversal_5": ("技术", "过去5日收益率（短线反转效应）", "预期负IC（超跌反弹）"),
    "volatility_20": ("技术", "过去20日日收益标准差（已实现波动率）", "预期负IC（低波溢价）"),
    "turnover_20": ("技术", "过去20日平均换手率（流动性/交易热度）", "预期负IC（低换手溢价）"),
    "roe": ("基本面", "净资产收益率（盈利质量）", "预期正IC"),
    "gross_margin": ("基本面", "销售毛利率（盈利能力）", "预期正IC"),
    "pb": ("基本面", "市净率=收盘价/每股净资产（估值）", "预期负IC（价值效应）"),
    "growth": ("基本面", "净利润同比增长率（成长性）", "预期正IC"),
    "size": ("扩展", "对数市值（规模）", "预期负IC（小市值效应）"),
}


def _write_report(summary, mono_df, corr, fp):
    L = []
    L.append("# Week1 单因子有效性检验报告\n")
    L.append(f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")


    L.append("## 一、数据治理\n")
    L.append(f"- **股票池**：全 A 股，按硬性约束剔除 ST/*ST/退市、上市不足 120 自然日次新股、"
             f"全样本日均成交额 < 5000 万（流动性不足）标的。")
    L.append(f"- **样本区间**：训练 2020-01-01~2024-12-31｜验证 2025 全年｜测试 2026-01-01~2026-08-01（严格隔离）。")
    L.append(f"- **数据源**：日线行情＝AKShare 新浪源（前复权 OHLCV）；财务＝同花顺摘要。")
    L.append(f"- **防未来函数**：价量因子用 shift/rolling 仅含历史；财务因子按**法定披露截止日**"
             f"（Q1/年报→4-30，半年报→8-31，Q3→10-31）之后才生效。")
    L.append(f"- **清洗后面板规模**：{len(fp):,} 行 / {fp['code'].nunique()} 只 / {len(ALL_FACTORS)} 个因子。\n")


    L.append("## 二、因子体系\n")
    L.append("| 类别 | 因子 | 逻辑 | 方向预期 |")
    L.append("|---|---|---|---|")
    for f in ALL_FACTORS:
        cat, logic, direction = FACTOR_LOGIC.get(f, ("", "", ""))
        L.append(f"| {cat} | `{f}` | {logic} | {direction} |")
    L.append("\n")


    L.append("## 三、因子 IC / IR（训练集，周度 Rank IC）\n")
    L.append(summary.to_markdown(index=False))
    L.append("\n> IC_IR(年化)=单期 IC_IR ×√52；一般 |IC均值|>0.03、|t|>2、IC_IR(年化)>0.5 视为预测力较强。\n")


    L.append("\n## 四、全样本 5 组分层回测\n")
    L.append(mono_df.to_markdown(index=False))
    L.append("\n> 分层单调性（Spearman）越接近 ±1，因子对收益的排序能力越强；多空组合=G5−G1。\n")


    L.append("\n## 五、因子相关性与正交化\n")
    L.append(corr.to_markdown())
    L.append(f"\n> 相关阈值 |r|>{C.CORR_THRESHOLD} 视为高共线；高共线因子经 Gram-Schmidt 正交化，剔除冗余。\n")


    L.append("\n## 六、因子有效性结论与无效因子剔除\n")
    valid, weak = [], []
    for _, r in summary.iterrows():
        t = abs(r["t值"]) if pd.notna(r["t值"]) else 0
        icir = abs(r["IC_IR(年化)"]) if pd.notna(r["IC_IR(年化)"]) else 0
        (valid if (t > 2 and icir > 0.5) else weak).append((r["因子"], r["因子中文"], r["IC均值"], r["t值"]))
    L.append(f"**有效因子（|t|>2 且 |IC_IR年化|>0.5）**：")
    for f, cn, ic, t in valid:
        L.append(f"- `{f}`（{cn}）：IC均值={ic}，t={t} → 预测力显著，纳入多因子模型。")
    L.append(f"\n**弱/无效因子（暂剔除）**：")
    for f, cn, ic, t in weak:
        L.append(f"- `{f}`（{cn}）：IC均值={ic}，t={t} → 预测力不足或方向不稳定，暂不纳入，"
                 f"后续可结合行业中性化、更长窗口复检。")


    L.append("\n## 七、交付图件（output/figures/）\n")
    for f in ALL_FACTORS:
        L.append(f"- `layer_{f}.png`：{FACTOR_CN.get(f, f)} 5 组分层累计收益")
    L.append("- `ic_bar.png`：各因子 IC 均值对比")
    L.append("- `ic_ts.png`：各因子 IC 累计曲线")
    L.append("- `factor_corr.png`：因子相关矩阵热力图")

    L.append("\n---\n> 免责声明：本项目仅用于实习量化研究能力考核与学习，所有策略与结论不构成任何投资建议。")

    with open(os.path.join(C.REPORT_DIR, "week1_factor_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
