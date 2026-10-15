import numpy as np
import pandas as pd

import config as C


def ewma_vol(returns: pd.Series, span: int = 32, ann: int = 252) -> float:

    r = returns.dropna()
    if len(r) < 20:
        return np.nan
    return float(r.ewm(span=span, min_periods=20).std().iloc[-1] * np.sqrt(ann))


def garch_forecast_vol(returns: pd.Series, ann: int = 252) -> float:

    r = returns.dropna() * 100.0
    if len(r) < 60 or not np.isfinite(r.std()) or r.std() == 0:
        return ewma_vol(returns)
    try:
        import warnings
        from arch import arch_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(r, vol="Garch", p=1, q=1, dist="t")
            res = am.fit(disp="off", show_warning=False)
            var = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
        if not np.isfinite(var) or var <= 0:
            return ewma_vol(returns)
        return float(np.sqrt(var) / 100.0 * np.sqrt(ann))
    except Exception:
        return ewma_vol(returns)


def vol_to_exposure(pred_vol: float, target: float = C.TARGET_VOL,
                    floor: float = C.EXPOSURE_FLOOR, cap: float = C.EXPOSURE_CAP) -> float:

    if not np.isfinite(pred_vol) or pred_vol <= 0:
        return cap
    return float(np.clip(target / pred_vol, floor, cap))


def vol_to_exposure_piecewise(pred_vol: float, target: float = C.TARGET_VOL,
                              knee: float = C.EXPOSURE_KNEE, full: float = C.EXPOSURE_FULL,
                              floor: float = C.EXPOSURE_FLOOR, cap: float = C.EXPOSURE_CAP) -> float:


    if not np.isfinite(pred_vol) or pred_vol <= 0:
        return cap
    x = np.clip((pred_vol - knee * target) / ((full - knee) * target), 0.0, 1.0)
    return float(cap + x * (floor - cap))


def forecast_vol_series(market_returns: pd.Series, reb_dates, method: str = "garch",
                        window: int = C.VOL_WINDOW) -> pd.Series:

    rets = market_returns.dropna().sort_index()
    out = {}
    for d in pd.DatetimeIndex(sorted(set(reb_dates))):
        hist = rets[rets.index < d].tail(window)
        out[d] = garch_forecast_vol(hist) if method == "garch" else ewma_vol(hist)
    return pd.Series(out).sort_index()


def exposure_schedule(market_returns: pd.Series, reb_dates, method: str = "garch",
                      mapping: str = "continuous", target: float = C.TARGET_VOL,
                      window: int = C.VOL_WINDOW,
                      floor: float = C.EXPOSURE_FLOOR, cap: float = C.EXPOSURE_CAP) -> pd.Series:

    vols = forecast_vol_series(market_returns, reb_dates, method, window)
    fn = vol_to_exposure if mapping == "continuous" else vol_to_exposure_piecewise
    return vols.apply(lambda v: fn(v, target, floor=floor, cap=cap))


def trend_exposure(benchmark: pd.Series, reb_dates, ma: int = C.TREND_MA,
                   floor: float = C.TREND_FLOOR, cap: float = C.EXPOSURE_CAP) -> pd.Series:

    b = benchmark.dropna().sort_index()
    m = b.rolling(ma, min_periods=max(40, ma // 2)).mean()
    out = {}
    for d in pd.DatetimeIndex(sorted(set(reb_dates))):
        hist = b[b.index <= d]
        if len(hist) == 0:
            out[d] = cap
            continue
        px, mv = hist.iloc[-1], m.reindex([hist.index[-1]]).iloc[0]
        out[d] = cap if (np.isfinite(mv) and px > mv) else floor
    return pd.Series(out).sort_index()


def combined_exposure(*schedules) -> pd.Series:

    parts = [s for s in schedules if s is not None and len(s)]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).min(axis=1)
