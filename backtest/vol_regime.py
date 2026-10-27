"""
변동성 국면(고변동/저변동) 분류 — 종목 자체 가격만 쓰고, 그 봉까지의 과거 정보만 쓴다.

[무엇을 하나]
  1) 실현변동성: 최근 vol_len봉 일간수익률의 표본표준편차(ddof=1).
  2) 그 봉의 변동성이 '지금까지 본 변동성 분포'에서 어느 위치(분위)인지 계산한다.
     분포는 처음부터 그 봉까지의 값 전부(확장 창)다. 전 기간 분위수를 한 번에 계산하면
     미래의 변동성이 기준선에 섞이므로 쓰지 않는다.
  3) 히스테리시스: 저변동에서 분위가 hi_q 이상이 되어야 고변동으로 바뀌고, 고변동에서
     분위가 lo_q 이하가 되어야 저변동으로 돌아온다. 그 사이(lo_q~hi_q)에서는 직전
     상태를 유지한다 — 경계 근처에서 라벨이 매일 뒤집혀 전환 비용만 키우는 걸 막는다.

[라벨은 장 마감 시점에 알 수 있다]
t봉의 라벨은 t봉 종가까지의 수익률과 t봉까지의 변동성 분포만 쓴다. 따라서 t봉에
뜬 신호를 t봉의 라벨로 나눠도 룩어헤드가 아니다(진입이 t봉 종가 이후인 경우).

[정직한 한계]
- 분위는 '이 종목의 과거 대비' 높고 낮음이다. 변동성이 구조적으로 올라간 종목은 확장 창이
  그 변화를 천천히 따라가므로 '고변동'이 오래 지속되는 것처럼 보일 수 있다.
- 처음 min_history봉은 분포가 얇아 분류하지 않는다(NaN). 짧은 종목은 표본이 줄어든다.
- 기본값(vol_len=20, hi_q=0.6, lo_q=0.4, min_history=252)은 관행적으로 고른 값이지
  최적화한 값이 아니다. 결과가 이 값에 민감한지는 바꿔 가며 확인해야 한다.
- 분류가 있다고 해서 '국면에 맞춰 전략을 바꾸면 나아진다'가 증명되는 것은 아니다. 이
  모듈은 신호 성과가 국면 간에 실제로 다른지 보는 첫 단계의 도구다.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

VOL_REGIMES = ("high_vol", "low_vol")
VOL_REGIME_KR = {"high_vol": "고변동", "low_vol": "저변동"}


def realized_vol(df: pd.DataFrame, vol_len: int = 20, close_col: str = "close") -> pd.Series:
    """최근 vol_len개 일간수익률의 표본표준편차(ddof=1). 앞의 vol_len봉은 NaN."""
    c = df[close_col].astype(float)
    return c.pct_change(fill_method=None).rolling(vol_len, min_periods=vol_len).std(ddof=1)


def classify_from_vol(vol: pd.Series, hi_q: float = 0.6, lo_q: float = 0.4,
                      min_history: int = 252) -> pd.Series:
    """
    변동성 시계열에서 'high_vol'/'low_vol' 라벨을 만든다. 분류 불가 구간은 NaN.

    분위 p_t = (vol_0..vol_t 중 vol_t 이하인 값의 개수) / (유효한 값의 개수).  자기 자신을
    포함하므로 p_t는 (0, 1]이다.
    첫 분류 봉(유효한 값이 min_history개 쌓인 봉)은 p_t >= 0.5면 고변동, 아니면 저변동이다.
    이후는 히스테리시스 규칙을 따른다.
    """
    if not (0.0 < lo_q < hi_q < 1.0):
        raise ValueError(f"0 < lo_q < hi_q < 1 이어야 합니다 (lo_q={lo_q}, hi_q={hi_q})")
    if min_history < 1:
        raise ValueError("min_history는 1 이상이어야 합니다")

    out = pd.Series(np.nan, index=vol.index, dtype=object, name="regime")
    seen: list[float] = []          # 정렬된 과거 변동성(확장 창)
    state = None
    for i, v in enumerate(vol.to_numpy(dtype=float)):
        if not np.isfinite(v):
            continue                # 결측은 분포에 넣지 않고, 상태는 유지한다
        bisect.insort(seen, v)
        n = len(seen)
        if n < min_history:
            continue
        if seen[0] == seen[-1]:
            continue                # 변동성이 한 값뿐(예: 가격이 평탄)이면 '높고 낮음'이 정의되지 않는다
        p = bisect.bisect_right(seen, v) / n        # v 이하인 값의 개수 / n
        if state is None:
            state = "high_vol" if p >= 0.5 else "low_vol"
        elif state == "low_vol" and p >= hi_q:
            state = "high_vol"
        elif state == "high_vol" and p <= lo_q:
            state = "low_vol"
        out.iloc[i] = state
    return out


def classify_vol_regime(df: pd.DataFrame, vol_len: int = 20, hi_q: float = 0.6, lo_q: float = 0.4,
                        min_history: int = 252, close_col: str = "close") -> pd.Series:
    """종가 데이터프레임에서 바로 고변동/저변동 라벨을 만든다."""
    return classify_from_vol(realized_vol(df, vol_len, close_col), hi_q, lo_q, min_history)


def regime_switches(regimes: pd.Series) -> int:
    """분류된 구간에서 라벨이 바뀐 횟수 — 히스테리시스가 전환을 얼마나 줄이는지 볼 때 쓴다."""
    r = regimes.dropna()
    if len(r) < 2:
        return 0
    return int((r != r.shift()).iloc[1:].sum())
