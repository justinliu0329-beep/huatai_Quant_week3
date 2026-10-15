import numpy as np
import pandas as pd

import config as C


EFFECTIVE_FACTORS = ["volatility_20", "turnover_20", "pb", "momentum_20", "reversal_5"]


def build_score(fp: pd.DataFrame, factor_weights: dict) -> pd.Series:

    score, wsum = None, 0.0
    for f, w in factor_weights.items():
        col = f + "_z"
        if col not in fp.columns or not np.isfinite(w):
            continue
        term = w * fp[col].fillna(0.0)
        score = term if score is None else score + term
        wsum += abs(w)
    if score is None:
        return pd.Series(0.0, index=fp.index)
    return score / (wsum if wsum else 1.0)


def factor_weights_from_ic(summary_path: str) -> dict:

    s = pd.read_csv(summary_path)
    w = {}
    for _, r in s.iterrows():
        if r["因子"] in EFFECTIVE_FACTORS and pd.notna(r["IC均值"]):
            w[r["因子"]] = float(r["IC均值"])
    return w


def select_weights(day_scores: pd.Series, industry: dict, n_stocks: int = C.MAX_HOLD,
                   max_stock_w: float = C.MAX_WEIGHT_PER_STOCK,
                   max_ind_w: float = C.MAX_WEIGHT_PER_INDUSTRY) -> dict:

    s = day_scores.dropna().sort_values(ascending=False)
    if s.empty:
        return {}
    n = min(n_stocks, len(s))
    picked = list(s.head(n).index)
    w = {c: 1.0 / n for c in picked}

    ind_w = {}
    for c in picked:
        ind = industry.get(c)
        if not ind:
            continue
        ind_w[ind] = ind_w.get(ind, 0.0) + w[c]
    for ind, tot in ind_w.items():
        if tot > max_ind_w:
            scale = max_ind_w / tot
            for c in picked:
                if industry.get(c) == ind:
                    w[c] *= scale
    return w


def _expo_at(exposure, d) -> float:

    if exposure is None:
        return 1.0
    try:
        v = float(exposure.get(d, 1.0))
    except Exception:
        return 1.0
    return v if np.isfinite(v) else 1.0


def generate_targets(prices: pd.DataFrame, score_wide: pd.DataFrame, industry: dict,
                     reb_dates, n_stocks: int = C.MAX_HOLD,
                     stop_loss: float = -0.10, take_profit: float = 0.20,
                     exposure: pd.Series = None) -> pd.DataFrame:


    all_days = prices.index
    reb_set = set(reb_dates)
    targets = pd.DataFrame(np.nan, index=all_days, columns=prices.columns)

    holdings, wt, entry = [], {}, {}
    cash_frac = 1.0

    def _val(c, d):
        if c in prices.columns and c in entry and not np.isnan(prices.at[d, c]):
            return wt[c] * prices.at[d, c] / entry[c]
        return None

    for d in all_days:
        if d in reb_set:
            scores = score_wide.loc[d] if d in score_wide.index else pd.Series(dtype=float)
            tw = select_weights(scores, industry, n_stocks)
            expo = _expo_at(exposure, d)
            holdings = list(tw.keys())
            wt = {c: w * expo for c, w in tw.items()}
            entry = {c: prices.at[d, c] for c in holdings
                     if c in prices.columns and not np.isnan(prices.at[d, c])}
            cash_frac = 1.0 - sum(wt.values())
            row = pd.Series(0.0, index=prices.columns)
            for c, w in wt.items():
                row[c] = w
            targets.loc[d] = row
            continue


        stopped = []
        for c in list(holdings):
            v = _val(c, d)
            if v is None:
                continue
            cum = prices.at[d, c] / entry[c] - 1
            if cum <= stop_loss or cum >= take_profit:
                stopped.append(c)
        if stopped:
            vals = {c: _val(c, d) for c in holdings}
            total = sum(v for v in vals.values() if v is not None) + cash_frac
            row = pd.Series(0.0, index=prices.columns)
            for c in holdings:
                if c in stopped or vals.get(c) is None:
                    continue
                row[c] = vals[c] / total
            targets.loc[d] = row

            for c in stopped:
                cash_frac += (vals.get(c) or 0.0) / total
            holdings = [c for c in holdings if c not in stopped]

    return targets
