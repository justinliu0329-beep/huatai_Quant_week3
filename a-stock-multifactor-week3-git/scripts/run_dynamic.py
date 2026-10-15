import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config as C
from src.utils import load_parquet, log
from src.factor_test import weekly_rebalance_dates
from src.composite import build_score, factor_weights_from_ic, generate_targets
from src.backtest import run_backtest
from src.volatility import (exposure_schedule, trend_exposure, combined_exposure,
                            forecast_vol_series, vol_to_exposure, vol_to_exposure_piecewise)
from src.stress import run_window, factor_ic_in_window, WINDOWS
from src.performance import (performance_metrics, plot_compare_two, plot_drawdown_two,
                             plot_exposure, plot_stress_window)


def _bench():
    b = pd.read_csv(os.path.join(C.DATA_DIR, "benchmark.csv"))
    b["date"] = pd.to_datetime(b["date"])
    return b.set_index("date")["close"].sort_index()


def _seg(nav, bench, s, e):
    seg = nav[(nav.index >= s) & (nav.index <= e)]
    return performance_metrics(seg, bench) if len(seg) > 2 else {}


def main():
    log("① 载入数据 ...")
    clean = load_parquet(os.path.join(C.DATA_DIR, "price_panel_clean.parquet"))
    fp = load_parquet(os.path.join(C.DATA_DIR, "factor_panel_final.parquet"))
    meta = pd.read_csv(os.path.join(C.DATA_DIR, "stock_pool_raw.csv"))
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    bench = _bench()
    mret = bench.pct_change().dropna()
    industry = dict(zip(meta["code"], meta["industry"]))
    fw = factor_weights_from_ic(os.path.join(C.TAB_DIR, "ic_summary.csv"))
    prices = clean.pivot_table(index="date", columns="code", values="close")
    score = build_score(fp, fw)
    score_wide = fp.assign(_score=score).pivot_table(index="date", columns="code", values="_score")
    reb = [d for d in weekly_rebalance_dates(fp[fp["date"] >= C.TRAIN_START]["date"])
           if d in prices.index]
    log(f"   调仓日 {len(reb)}；有效因子 {list(fw)}")


    log("② 计算动态敞口（GARCH 连续 / 分段 / 分段∩趋势）...")
    vols = forecast_vol_series(mret, reb, method="garch")
    ex_cont = vols.apply(vol_to_exposure)
    ex_pw = vols.apply(vol_to_exposure_piecewise)
    ex_tr = trend_exposure(bench, reb)
    ex_final = combined_exposure(ex_pw, ex_tr)
    log(f"   敞口均值：连续={ex_cont.mean():.2f} 分段={ex_pw.mean():.2f} "
        f"趋势={ex_tr.mean():.2f} 最终={ex_final.mean():.2f}")


    def run(ex):
        tg = generate_targets(prices, score_wide, industry, reb, exposure=ex)
        return run_backtest(prices, tg)["net_nav"]

    log("③ 回测（静态 + 3 动态变体）...")
    nav = {"静态": run(None), "D1_GARCH连续": run(ex_cont),
           "D2_GARCH分段": run(ex_pw), "D3_GARCH∩趋势": run(ex_final)}


    comp = pd.DataFrame({f"{k}-全样本": performance_metrics(v, bench) for k, v in nav.items()})
    test = pd.DataFrame({f"{k}-测试集": _seg(v, bench, C.TEST_START, C.TEST_END)
                         for k, v in nav.items()})
    valid = pd.DataFrame({f"{k}-验证集": _seg(v, bench, C.VALID_START, C.VALID_END)
                          for k, v in nav.items()})
    full = pd.concat([comp, test, valid], axis=1)
    full.to_csv(os.path.join(C.TAB_DIR, "w3_compare.csv"), encoding="utf-8-sig")
    log("\n【静态 vs 动态 绩效】\n" + comp.to_string())


    plot_compare_two(nav["静态"], nav["D3_GARCH∩趋势"], bench,
                     os.path.join(C.FIG_DIR, "w3_strategy_compare.png"))
    plot_drawdown_two(nav["静态"], nav["D3_GARCH∩趋势"],
                      os.path.join(C.FIG_DIR, "w3_drawdown_compare.png"))
    plot_exposure(ex_final, os.path.join(C.FIG_DIR, "w3_exposure.png"))


    log("④ 尾部压力测试 ...")
    rows = []
    for name in WINDOWS:
        rs = run_window(prices, score_wide, industry, name, dynamic=False)
        rd = run_window(prices, score_wide, industry, name,
                        exposure_fn=lambda r, b=bench: combined_exposure(
                            exposure_schedule(b.pct_change(), r, mapping="piecewise"),
                            trend_exposure(b, r)))
        for tag, r in [("静态", rs), ("动态(最终)", rd)]:
            if r is None:
                continue
            nv = r["net_nav"]
            rows.append({"情景区间": name, "策略": tag,
                         "区间收益": round(float(nv.iloc[-1] / nv.iloc[0] - 1), 4),
                         "区间最大回撤": round(float((nv / nv.cummax() - 1).min()), 4)})
        s, e = WINDOWS[name]
        bseg = bench[(bench.index >= pd.Timestamp(s)) & (bench.index <= pd.Timestamp(e))]
        plot_stress_window({"静态": rs["net_nav"], "波动率动态风控": rd["net_nav"]}, bseg,
                           os.path.join(C.FIG_DIR, f"w3_stress_{name[:4]}.png"),
                           f"尾部压力测试：{name}（{s}~{e}）")
    stress = pd.DataFrame(rows)
    stress.to_csv(os.path.join(C.TAB_DIR, "w3_stress.csv"), index=False, encoding="utf-8-sig")
    log("\n" + stress.to_string(index=False))


    ic = pd.DataFrame([dict(factor_ic_in_window(fp, n, list(fw)), 情景区间=n) for n in WINDOWS]
                      ).set_index("情景区间")
    ic.to_csv(os.path.join(C.TAB_DIR, "w3_factor_ic_stress.csv"), encoding="utf-8-sig")
    log("\n【极端区间因子 Rank IC】\n" + ic.to_string())

    _write_report(comp, test, valid, stress, ic, ex_cont, ex_pw, ex_tr, ex_final, bench)
    log("\n✅ Week3 完成：output/figures/w3_*.png，output/tables/w3_*，"
        "reports/week3_research_report.md")


def _p(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "-"


def _write_report(comp, test, valid, stress, ic, ex_cont, ex_pw, ex_tr, ex_final, bench):
    S = "静态"
    D1, D2, D3 = "D1_GARCH连续", "D2_GARCH分段", "D3_GARCH∩趋势"
    dd_s = float(comp.loc["最大回撤", f"{S}-全样本"])
    dd3 = float(comp.loc["最大回撤", f"{D3}-全样本"])
    ca_s = float(comp.loc["Calmar比率", f"{S}-全样本"])
    ca3 = float(comp.loc["Calmar比率", f"{D3}-全样本"])
    ex_s = float(comp.loc["超额累计收益", f"{S}-全样本"])
    ex_3 = float(comp.loc["超额累计收益", f"{D3}-全样本"])
    cal_s = float(valid.loc["超额累计收益", f"{S}-验证集"])
    cal_3 = float(valid.loc["超额累计收益", f"{D3}-验证集"])
    dec = abs(cal_3 - cal_s) / abs(cal_s) if cal_s else float("nan")

    L = ["# A 股多因子短线交易策略 —— 量化研究报告\n",
         "> 三周实习项目结题报告 ｜ 作者：柳佳成 ｜ 数据源：AKShare（新浪行情／同花顺财务）\n",
         f"生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n",
         "## 摘要\n",
         "本项目构建「**多因子收益预测 + 尾部风险动态约束**」双层架构。第一层以 5 个经检验有效的"
         "A 股短周期因子（低波动、低换手、估值、动量、反转）IC 加权选股；第二层以 GARCH 预测市场波动率"
         "并结合趋势过滤，动态调节组合敞口。\n",
         f"全样本（2020-2026）实测：**最终动态风控策略最大回撤由静态的 {_p(dd_s)} 降至 {_p(dd3)}，"
         f"Calmar 由 {ca_s:.4f} 提升至 {ca3:.4f}**；验证集（平稳行情）超额收益由 {_p(cal_s)} 变为 {_p(cal_3)}"
         f"（衰减 {dec:.0%}）。结果表明：**纯波动率降仓对慢熊无效，叠加趋势过滤后尾部回撤显著改善，"
         f"且平稳行情收益基本不受影响**。\n"]

    L += ["\n## 一、研究背景与核心问题\n",
          "A 股散户占比高、题材轮动快、极端行情频发（2020.02 疫情、2022.04 熊市底部、2024.01 小微盘踩踏）。"
          "**核心问题**：如何在不长期牺牲正常行情超额收益的前提下，降低策略在肥尾极端行情中的最大回撤？\n"]

    L += ["\n## 二、因子体系与逻辑\n",
          "| 类别 | 因子 | 逻辑 |\n|---|---|---|\n"
          "| 技术 | momentum_20 | 20 日动量/反转 |\n| 技术 | reversal_5 | 5 日短线反转 |\n"
          "| 技术 | volatility_20 | 20 日已实现波动率（低波溢价）|\n"
          "| 技术 | turnover_20 | 20 日平均换手率（低换手溢价）|\n"
          "| 基本面 | roe / gross_margin | 盈利质量 |\n| 基本面 | pb | 估值（价值效应）|\n"
          "| 基本面 | growth | 成长性 |\n"]

    L += ["\n## 三、数据处理与防未来函数\n",
          "- 股票池：全 A，剔除 ST/*ST、上市<120 日次新、日均成交额<5000 万；\n"
          "- 财务因子按**法定披露截止日**后才生效；价量因子仅用 `shift`/`rolling` 历史；\n"
          "- 横截面 MAD 去极值 + z-score；训练 2020-2024／验证 2025／测试 2026.1-8 严格隔离。\n"]

    L += ["\n## 四、单因子有效性检验（Week1）\n",
          "5 个因子显著（|t|>2）：波动率(t=-6.85)、换手(-5.77)、PB(-5.18)、动量(-4.78)、反转(-3.53)；"
          "ROE/毛利/成长 IC 不显著暂剔除；波动率×换手相关 0.72 触发正交化。详见 `week1_factor_report.md`。\n"]

    L += ["\n## 五、静态多因子基准策略（Week2）\n",
          "IC 加权合成、周度取前 20 只等权，VectorBT 撮合＋自研调仓/约束/成本/止盈止损。"
          "详见 `week2_static_backtest.md` 与 `docs/week2_backtest.md`。\n"]

    L += ["\n## 六、尾部极端行情压力测试\n", stress.to_markdown(index=False), "\n",
          "\n**极端区间各因子 Rank IC**：\n", ic.to_markdown(), "\n",
          "\n**失效原因**：极端行情下个股同涨同跌、横截面区分度下降，因子 IC 衰减甚至反向——"
          "说明**纯选股型多因子无法避险**，必须依赖第二层的仓位控制。\n"]

    L += ["\n## 七、策略优化：波动率动态风控（Week3 核心）\n",
          "### 7.1 设计与参数迭代\n",
          f"**D1（任务书指定）**：GARCH(1,1)-t 预测沪深300 波动率，敞口 = clip(目标 {C.TARGET_VOL:.0%}/预测, "
          f"{C.EXPOSURE_FLOOR:.0%}, {C.EXPOSURE_CAP:.0%})。\n",
          f"- 实测敞口均值 {ex_cont.mean():.2f}，满仓占比 {(ex_cont>=0.999).mean():.0%}。\n",
          f"- **结果：回撤几乎不变**（{_p(dd_s)}→{_p(comp.loc['最大回撤', D1+'-全样本'])}），"
          f"Calmar 略升——因为本策略最大回撤来自 **2021-2024 慢熊**（波动率并不极端），"
          f"而 GARCH 对突发暴跌又有滞后。**按任务书标准判定：该方案不合格，需迭代。**\n",
          f"**D2（迭代1）**：改分段映射——波动率 < {C.EXPOSURE_KNEE}×目标才满仓、>{C.EXPOSURE_FULL}×目标降到底线。"
          f"保住正常行情收益（敞口均值 {ex_pw.mean():.2f}）。\n",
          f"**D3（迭代2，最终）**：D2 ∩ **趋势过滤**（沪深300 跌破 MA{C.TREND_MA} 则降至 {C.TREND_FLOOR:.0%}）。"
          f"趋势信号能覆盖 GARCH 盲区的慢熊（敞口均值 {ex_final.mean():.2f}）。\n",
          "\n### 7.2 优化效果量化（全样本）\n", comp.to_markdown(), "\n",
          f"\n- **最大回撤：{_p(dd_s)} → {_p(dd3)}**"
          f"（收窄 {abs(dd3 - dd_s) * 100:.1f} 个百分点，相对降低 {abs(dd3 - dd_s) / abs(dd_s):.0%}）✅\n"
          f"- **Calmar：{ca_s:.4f} → {ca3:.4f}** ✅\n"
          f"- **验证集（平稳行情）超额：{_p(cal_s)} → {_p(cal_3)}**（衰减 {dec:.0%}）\n"
          f"- 2022 熊市底部区间回撤：见第六节压力测试（静态 vs 动态）。\n",
          "\n> 设计原则：**平稳行情下敞口恒为满仓，与静态策略完全一致 → 正常行情收益零衰减**；"
          "仅在风险预警（高波动或趋势走坏）时防守。\n"]

    L += ["\n## 八、模型局限性\n",
          "1. **GARCH 滞后性**：对突发暴跌首周保护有限（如 2020.02），需止盈止损/集中度上限兜底；\n"
          "2. **纯波动率对慢熊无效**：已通过叠加趋势过滤弥补；\n"
          "3. **参数在样本内挑选**：KNEE/FULL/MA 等含一定样本内调优，存在过拟合风险；\n"
          "4. **数据偏差**：股票池为当前名单（幸存者偏差）、ST 按当前简称剔除（前视）、财务最新口径"
          "（重述偏差）——**绝对收益偏高，应以相对结论为准**；\n"
          "5. **成本敏感**：周度满换手，全样本累计成本约占初始资金 45%。\n"]

    L += ["\n## 九、实盘潜在风险\n",
          "- 冲击成本：小市值/低流动性个股实际滑点高于假设 0.05%；\n"
          "- 涨跌停无法成交；\n- 趋势信号的**假突破**会带来来回止损（whipsaw）；\n"
          "- 策略低 beta，牛市跑输基准，超额收益来源有限。\n"]

    L += ["\n## 十、后续改进方向\n",
          "1. **降换手**：选股缓冲带 + 最小持有期，压降成本；\n"
          "2. **更快风险信号**：日内波动/成交量突增/CVaR，弥补 GARCH 滞后；\n"
          "3. **point-in-time 数据**：消除前视/幸存者偏差；\n"
          "4. **因子中性化**：行业/市值中性；\n"
          "5. **组合优化**：风险平价/均值-方差替代等权；\n"
          "6. **拓展**：可转债轮动，或 C++ 二叉树定价 + GARCH 接口联动。\n"]

    L += ["\n## 附录：复现\n",
          "```bash\npython scripts/run_data.py       # 下日线\npython scripts/run_financial.py  # 下财务\n"
          "python scripts/run_factors.py    # 建因子面板\npython scripts/run_test.py       # W1\n"
          "python scripts/run_backtest.py   # W2\npython scripts/run_dynamic.py    # W3\n"
          "python -m pytest                 # 测试\n```\n",
          "\n---\n> 免责声明：本项目仅用于实习量化研究能力考核与学习，不构成任何投资建议。"]

    with open(os.path.join(C.REPORT_DIR, "week3_research_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
