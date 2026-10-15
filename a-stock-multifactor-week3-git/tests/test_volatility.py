import numpy as np
import pandas as pd

import config as C
from src.volatility import (vol_to_exposure, vol_to_exposure_piecewise,
                            trend_exposure, combined_exposure)


def test_vol_to_exposure_bounds_and_monotone():
    assert vol_to_exposure(0.01) == C.EXPOSURE_CAP
    assert vol_to_exposure(9.9) == C.EXPOSURE_FLOOR
    vs = [vol_to_exposure(v) for v in np.linspace(0.05, 1.0, 20)]
    assert all(C.EXPOSURE_FLOOR <= x <= C.EXPOSURE_CAP for x in vs)
    assert all(vs[i] >= vs[i + 1] - 1e-12 for i in range(len(vs) - 1))


def test_piecewise_keeps_full_in_normal():
    t = C.TARGET_VOL
    assert np.isclose(vol_to_exposure_piecewise(t), C.EXPOSURE_CAP)
    assert np.isclose(vol_to_exposure_piecewise(C.EXPOSURE_KNEE * t), C.EXPOSURE_CAP)
    assert np.isclose(vol_to_exposure_piecewise(C.EXPOSURE_FULL * t), C.EXPOSURE_FLOOR)
    assert np.isclose(vol_to_exposure_piecewise(10.0), C.EXPOSURE_FLOOR)


def test_trend_exposure_above_below_ma():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    up = pd.Series(np.linspace(100, 200, 300), index=idx)
    dn = pd.Series(np.linspace(200, 100, 300), index=idx)
    assert trend_exposure(up, [idx[-1]]).iloc[0] == C.EXPOSURE_CAP
    assert trend_exposure(dn, [idx[-1]]).iloc[0] == C.TREND_FLOOR


def test_combined_exposure_takes_min():
    a = pd.Series([1.0, 0.5, 0.8], index=pd.date_range("2024-01-01", periods=3))
    b = pd.Series([0.9, 0.9, 0.3], index=a.index)
    c = combined_exposure(a, b)
    assert list(np.round(c.values, 4)) == [0.9, 0.5, 0.3]
