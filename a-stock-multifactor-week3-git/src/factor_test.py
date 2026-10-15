import os

import numpy as np
import pandas as pd

import config as C
from src.utils import save_parquet, log


def weekly_rebalance_dates(dates) -> pd.DatetimeIndex:

    idx = pd.DatetimeIndex(sorted(pd.unique(pd.Series(dates))))
    s = pd.Series(idx)
    iso = idx.isocalendar()
    key = (iso.year.astype(str) + "W" + iso.week.astype(str)).values
    last = s.groupby(key).max()
    return pd.DatetimeIndex(sorted(last.values))


def compute_ic(fp: pd.DataFrame, factor: str, reb_dates, method=C.IC_METHOD) -> pd.Series:
    zcol = factor + "_z"
    sub = fp[fp["date"].isin(reb_dates)][["date", zcol, "fwd_ret"]].dropna()
    ics = {}
    for d, g in sub.groupby("date"):
        if len(g) >= 10:
            ics[d] = g[zcol].corr(g["fwd_ret"], method=method)
    ic = pd.Series(ics)
    ic.name = factor
    return ic.dropna()


def ic_summary(ic: pd.Series, periods_per_year: int = 52) -> dict:
    ic = pd.Series(ic).dropna()
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    n = int(len(ic))
    ir = mean / std if (std and not np.isnan(std)) else np.nan
    t = ir * np.sqrt(n) if (std and not np.isnan(std)) else np.nan
    return {
        "因子": ic.name,
        "IC均值": round(mean, 4),
        "IC标准差": round(std, 4),
        "IC_IR": round(ir, 4),
        "IC_IR(年化)": round(ir * np.sqrt(periods_per_year), 4),
        "t值": round(t, 2),
        "IC>0占比": round((ic > 0).mean(), 4),
        "|IC|>0.02占比": round((ic.abs() > 0.02).mean(), 4),
        "样本数": n,
    }


def run_ic_all(fp, factors, reb_dates):
    rows, series = [], {}
    for f in factors:
        ic = compute_ic(fp, f, reb_dates)
        series[f] = ic
        rows.append(ic_summary(ic))
    summary = pd.DataFrame(rows).sort_values("IC_IR(年化)", key=lambda s: s.abs(),
                                             ascending=False).reset_index(drop=True)
    return summary, series


def layered_backtest(fp, factor, reb_dates, n=C.N_QUANTILE):

    zcol = factor + "_z"
    sub = fp[fp["date"].isin(reb_dates)][["date", zcol, "fwd_ret"]].dropna().copy()

    grps = []
    for d, g in sub.groupby("date"):
        if len(g) >= n * 3:
            g = g.assign(grp=pd.qcut(g[zcol].rank(method="first"), n, labels=False))
        else:
            g = g.assign(grp=np.nan)
        grps.append(g)
    sub = pd.concat(grps, ignore_index=True) if grps else sub.assign(grp=np.nan)

    grp_ret = sub.dropna(subset=["grp"]).groupby(["date", "grp"])["fwd_ret"].mean().unstack()
    grp_ret.columns = [f"G{int(c)+1}" for c in grp_ret.columns]
    cum = (1 + grp_ret.fillna(0)).cumprod()

    if grp_ret.empty or grp_ret.shape[1] < 2:
        return grp_ret, cum, pd.Series([1.0], dtype=float)
    ls = (grp_ret.iloc[:, -1] - grp_ret.iloc[:, 0]).fillna(0)
    ls_cum = (1 + ls).cumprod()
    return grp_ret, cum, ls_cum


def monotonicity(cum: pd.DataFrame) -> float:

    if cum.empty:
        return np.nan
    final = cum.iloc[-1].values
    rank_group = np.arange(len(final))
    return pd.Series(final).corr(pd.Series(rank_group), method="spearman")


def factor_correlation(fp, factors) -> pd.DataFrame:

    zcols = [f + "_z" for f in factors]
    mats = []
    for _, g in fp.groupby("date"):
        if len(g) > 30:
            mats.append(g[zcols].corr().values)
    if not mats:
        return pd.DataFrame(index=factors, columns=factors)
    arr = np.nanmean(np.stack(mats), axis=0)
    return pd.DataFrame(arr, index=factors, columns=factors)


def orthogonalize(fp, factors, ic_summary_df, threshold=C.CORR_THRESHOLD):


    df = fp.reset_index(drop=True).copy()
    strength = ic_summary_df.set_index("因子")["IC_IR(年化)"].abs().to_dict()
    order = sorted(factors, key=lambda f: (strength.get(f, 0) if strength.get(f, 0) == strength.get(f, 0) else -1),
                   reverse=True)
    avg_corr = factor_correlation(fp, order)

    kept_orig, kept_cols, ortho, dropped = [], [], [], []
    for f in order:
        if df[f + "_z"].notna().sum() == 0:
            dropped.append(f)
            continue
        if not kept_orig:
            kept_orig.append(f); kept_cols.append(f + "_z"); continue

        max_r = avg_corr.loc[f, kept_orig].abs().max() if kept_orig else np.nan
        if pd.isna(max_r) or max_r <= threshold:
            kept_orig.append(f); kept_cols.append(f + "_z")
        else:

            others = list(kept_cols)
            sub = df[["date", f + "_z"] + others].dropna()
            resid = np.full(len(df), np.nan)
            for _, g in sub.groupby("date"):
                y = g[f + "_z"].values
                X = np.column_stack([g[c].values for c in others] + [np.ones(len(g))])
                try:
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    resid[g.index.values] = y - X @ beta
                except Exception:
                    resid[g.index.values] = y
            df[f + "_orth"] = resid
            kept_orig.append(f); kept_cols.append(f + "_orth")
            ortho.append(f)
    return df, kept_cols, ortho, dropped


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fpth in ["/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/STHeiti Light.ttc",
                 "/Library/Fonts/Arial Unicode.ttf"]:
        if os.path.exists(fpth):
            try:
                font_manager.fontManager.addfont(fpth)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=fpth).get_name()
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_layered(cum, factor, path):
    plt = _plt()
    plt.figure(figsize=(10, 6))
    for c in cum.columns:
        plt.plot(cum.index, cum[c], label=c, linewidth=1.5)
    plt.title(f"{factor} 分层累计收益（5组）")
    plt.xlabel("日期"); plt.ylabel("累计净值"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_ic_bar(summary, path):
    plt = _plt()
    plt.figure(figsize=(10, 5))
    s = summary.sort_values("IC均值")
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in s["IC均值"]]
    plt.barh(s["因子"], s["IC均值"], color=colors)
    plt.title("各因子 IC 均值"); plt.xlabel("Rank IC 均值"); plt.grid(alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_ic_ts(ic_series, path):
    plt = _plt()
    plt.figure(figsize=(11, 6))
    for f, ic in ic_series.items():
        plt.plot(ic.index, ic.cumsum(), label=f, linewidth=1.3)
    plt.title("各因子 Rank IC 累计曲线"); plt.xlabel("日期"); plt.ylabel("IC 累计值")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_correlation(corr, path, title="因子相关矩阵"):
    plt = _plt()
    try:
        import seaborn as sns
        plt.figure(figsize=(8, 6.5))
        sns.heatmap(corr.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
        plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    except Exception:
        plt.figure(figsize=(8, 6.5)); plt.imshow(corr.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(); plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr)), corr.index); plt.title(title)
        plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
