import os
import numpy as np
import pandas as pd

import config as C


def _ann(days, freq=252):
    return freq / max(days, 1)


def performance_metrics(nav: pd.Series, benchmark: pd.Series = None) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (nav.iloc[-1] / nav.iloc[0]) ** _ann(len(nav)) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = ann / vol if vol else np.nan
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = ann / abs(max_dd) if max_dd else np.nan
    out = {"累计收益率": round(float(total), 4), "年化收益率": round(float(ann), 4),
           "年化波动率": round(float(vol), 4), "夏普比率": round(float(sharpe), 4),
           "最大回撤": round(float(max_dd), 4), "Calmar比率": round(float(calmar), 4)}
    if benchmark is not None:
        b = benchmark.reindex(nav.index).ffill()
        common = nav.index.intersection(b.index)
        if len(common) > 2:
            ex = ((nav.loc[common] / nav.loc[common].iloc[0]) /
                  (b.loc[common] / b.loc[common].iloc[0]))
            exr = ex.pct_change().dropna()
            out["超额累计收益"] = round(float(ex.iloc[-1] - 1), 4)
            out["超额年化收益"] = round(float((ex.iloc[-1]) ** _ann(len(common)) - 1), 4)
            out["信息比率"] = round(float(exr.mean() / exr.std() * np.sqrt(252))
                                    if exr.std() else np.nan, 4)
            out["超额最大回撤"] = round(float((ex / ex.cummax() - 1).min()), 4)
    m = nav.resample("ME").last().pct_change().dropna()
    if len(m):
        out["月度胜率"] = round(float((m > 0).mean()), 4)
        out["月度收益均值"] = round(float(m.mean()), 4)
    return out


def monthly_returns(nav: pd.Series) -> pd.DataFrame:
    m = nav.resample("ME").last().pct_change().dropna()
    df = pd.DataFrame({"ret": m})
    df["year"], df["month"] = df.index.year, df.index.month
    return df.pivot(index="year", columns="month", values="ret")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for p in ["/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc"]:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_nav_vs_benchmark(nav, benchmark, path):
    plt = _plt()
    plt.figure(figsize=(11, 6))
    plt.plot(nav.index, nav / nav.iloc[0], label="静态多因子策略", lw=1.6)
    if benchmark is not None:
        b = benchmark.reindex(nav.index).ffill()
        plt.plot(b.index, b / b.iloc[0], label="沪深300", lw=1.2, alpha=0.85)
    plt.title("静态多因子策略 净值 vs 沪深300")
    plt.xlabel("日期"); plt.ylabel("累计净值"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_drawdown(nav, path):
    plt = _plt()
    dd = nav / nav.cummax() - 1
    plt.figure(figsize=(11, 4))
    plt.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    plt.plot(dd.index, dd.values, color="#d62728", lw=0.8)
    plt.title("策略回撤曲线"); plt.xlabel("日期"); plt.ylabel("回撤")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_monthly_heatmap(nav, path):
    plt = _plt()
    try:
        import seaborn as sns
        mr = monthly_returns(nav)
        plt.figure(figsize=(11, 5))
        sns.heatmap(mr, annot=True, fmt=".1%", cmap="RdYlGn", center=0,
                    cbar_kws={"shrink": .7})
        plt.title("策略月度收益热力图（含分布）")
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    except Exception:
        pass


def plot_compare_two(nav_s, nav_d, benchmark, path, title="静态 vs 波动率动态风控"):
    plt = _plt()
    plt.figure(figsize=(11, 6))
    plt.plot(nav_s.index, nav_s / nav_s.iloc[0], label="静态多因子（对照）", lw=1.5)
    plt.plot(nav_d.index, nav_d / nav_d.iloc[0], label="波动率动态风控", lw=1.5)
    if benchmark is not None:
        b = benchmark.reindex(nav_s.index).ffill()
        plt.plot(b.index, b / b.iloc[0], label="沪深300", lw=1.1, alpha=0.7)
    plt.title(title); plt.xlabel("日期"); plt.ylabel("累计净值")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def plot_drawdown_two(nav_s, nav_d, path):
    plt = _plt()
    plt.figure(figsize=(11, 4.5))
    for nav, lab in [(nav_s, "静态多因子"), (nav_d, "波动率动态风控")]:
        dd = nav / nav.cummax() - 1
        plt.plot(dd.index, dd.values, lw=1.2, label=lab)
    plt.fill_between((nav_d / nav_d.cummax() - 1).index,
                     (nav_d / nav_d.cummax() - 1).values, 0, alpha=0.15)
    plt.title("回撤对比：静态 vs 动态风控"); plt.xlabel("日期"); plt.ylabel("回撤")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def plot_exposure(exposure: pd.Series, path):
    plt = _plt()
    plt.figure(figsize=(11, 4))
    plt.plot(exposure.index, exposure.values, color="#1f77b4", lw=1.3)
    plt.fill_between(exposure.index, exposure.values, 0, alpha=0.2, color="#1f77b4")
    plt.axhline(1.0, color="gray", ls="--", lw=0.8)
    plt.axhline(0.3, color="#d62728", ls=":", lw=0.8)
    plt.title("波动率驱动的动态仓位敞口（1.0=满仓, 0.3=底线）")
    plt.xlabel("日期"); plt.ylabel("目标敞口")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_stress_window(nav_map: dict, benchmark, path, title):
    plt = _plt()
    plt.figure(figsize=(10, 5))
    for lab, nav in nav_map.items():
        if nav is None or len(nav) < 2:
            continue
        plt.plot(nav.index, nav / nav.iloc[0], label=lab, lw=1.6)
    if benchmark is not None and len(benchmark) > 2:
        b = benchmark / benchmark.iloc[0]
        plt.plot(b.index, b.values, label="沪深300", lw=1.2, alpha=0.75)
    plt.title(title); plt.xlabel("日期"); plt.ylabel("区间累计净值")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
