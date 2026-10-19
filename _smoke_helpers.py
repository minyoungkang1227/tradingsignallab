"""
스모크테스트 공용 헬퍼 (합성 데이터 — 출력된 수익률·t값은 성과가 아니다).
make_cfg / make_symbol 을 테스트마다 복사해 쓰지 않도록 한 곳에 둔다.
"""
import types

import numpy as np
import pandas as pd

from advisor.evaluator import MIN_OWN_EVENTS


def make_cfg(**overrides):
    cfg = types.SimpleNamespace(
        id_vol_mode="multiple", id_vol_mult=2.0, id_z_threshold=1.96,
        big_sales_len=7, vp_bins=20, vp_freq="W",
        mom_filter=False, mom_lookback=126, mom_window=252, mom_threshold=0.7,
        liq_filter=False, liq_window=20, liq_percentile_window=252, liq_threshold=0.9,
        entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
        target_horizon=5, t_threshold=3.0, min_own_events=MIN_OWN_EVENTS,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_symbol(seed, n=400, force_event_at_end=True):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.uniform(1000, 5000, n)
    if force_event_at_end:
        # 마지막 며칠은 거래량을 확 키워서 institutional_displacement의
        # high_vol 조건(거래량 이상치)이 마지막 봉에서 뜨도록 유도한다.
        volume[-3:] *= 20
        close[-2] = close[-3] * 1.05  # 강한 양봉으로 몸통비율 조건도 만족
        close[-1] = close[-2] * 1.05
        high[-1] = close[-1] * 1.02
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": volume}, index=dates)
