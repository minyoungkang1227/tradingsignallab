"""
backtest.vol_regime 스모크테스트 — 기대값은 손계산. 합성 데이터 결과는 실제 시장 성과가 아님.

(1) 실현변동성 손계산  (2) 확장 분위수 + 히스테리시스 상태기계 손계산  (3) 독립 구현과 일치
(4) 인과성(미래를 잘라도 과거 라벨 불변)  (5) 변동성이 바뀌면 라벨이 따라가는가
(6) regime_study(labels=VOL_REGIMES) 연결, 기본값(추세 3분류) 보존
"""
import math

import numpy as np
import pandas as pd

from backtest.vol_regime import (VOL_REGIMES, classify_from_vol, classify_vol_regime, realized_vol,
                                 regime_switches)


def labels(vals, **kw):
    out = classify_from_vol(pd.Series(vals, dtype=float), **kw)
    return [None if pd.isna(x) else {"high_vol": "H", "low_vol": "L"}[x] for x in out]


# --- 1) 실현변동성 손계산 ----------------------------------------------------------------
# 종가 100, 110, 99, 108.9 -> 수익률 +0.1, -0.1, +0.1 (99/110 = 0.9, 108.9/99 = 1.1).
# window=2: t=2 에서 [0.1, -0.1] 의 표본표준편차(ddof=1) = sqrt(((0.1)^2 + (0.1)^2) / 1) = sqrt(0.02) = 0.141421
df = pd.DataFrame({"close": [100.0, 110.0, 99.0, 108.9]}, index=pd.bdate_range("2024-01-01", periods=4))
v = realized_vol(df, vol_len=2)
assert v.iloc[:2].isna().all(), v
assert abs(v.iloc[2] - math.sqrt(0.02)) < 1e-12 and abs(v.iloc[3] - math.sqrt(0.02)) < 1e-12, v
print(f"[PASS] 1) 실현변동성 손계산: t=2,3 모두 sqrt(0.02)={math.sqrt(0.02):.6f}, 앞 2봉 NaN")

# --- 2) 확장 분위수 상태기계 손계산 ---------------------------------------------------------
# vol = [5,3,4,6,2,7,1,8], hi=0.6, lo=0.4, min_history=3.   분위 p = (이하인 개수)/(개수)
#   t=2 : [5,3,4]            v=4 -> 2/3=.667 >= .5 첫 분류=고변동
#   t=3 : [5,3,4,6]          v=6 -> 4/4=1.0  고변동 유지
#   t=4 : [5,3,4,6,2]        v=2 -> 1/5=.2 <= .4 -> 저변동
#   t=5 : (+7)               v=7 -> 6/6=1.0 >= .6 -> 고변동
#   t=6 : (+1)               v=1 -> 1/7=.143 -> 저변동
#   t=7 : (+8)               v=8 -> 8/8=1.0 -> 고변동
got = labels([5, 3, 4, 6, 2, 7, 1, 8], hi_q=0.6, lo_q=0.4, min_history=3)
assert got == [None, None, "H", "H", "L", "H", "L", "H"], got
print("[PASS] 2) 상태기계 손계산: [NaN,NaN,고,고,저,고,저,고] 일치")

# --- 3) 히스테리시스: 문턱 사이에서는 직전 상태 유지 ---------------------------------------------
# vol = [1,2,3,4,5,2.5,1.5,2.2], hi=0.7, lo=0.3, min_history=3
#   t=2..4: v=3,4,5 모두 p=1.0 -> 고변동
#   t=5 : v=2.5 [1,2,3,4,5,2.5] 이하 {1,2,2.5}=3/6=.5, 0.3<.5<0.7 -> 고변동 유지
#   t=6 : v=1.5 이하 {1,1.5}=2/7=.2857 <= .3 -> 저변동
#   t=7 : v=2.2 이하 {1,1.5,2,2.2}=4/8=.5, 0.3<.5<0.7 -> 저변동 유지 (같은 p=.5인데 직전 상태가 다르면 라벨이 다르다)
got = labels([1, 2, 3, 4, 5, 2.5, 1.5, 2.2], hi_q=0.7, lo_q=0.3, min_history=3)
assert got == [None, None, "H", "H", "H", "H", "L", "L"], got
print("[PASS] 3) 히스테리시스 손계산: p=0.5가 t=5에선 고변동 유지, t=7에선 저변동 유지")

# --- 4) 결측·평탄·잘못된 인자 ----------------------------------------------------------------
# [1,2,3,NaN,4], min_history=3 -> t=2 고(p=1), t=3 결측은 라벨 NaN(분포에 넣지 않음), t=4 v=4 p=4/4 -> 고
assert labels([1, 2, 3, np.nan, 4], hi_q=0.6, lo_q=0.4, min_history=3) == [None, None, "H", None, "H"]
# 변동성이 한 값뿐이면 높고 낮음이 정의되지 않으므로 분류하지 않는다
assert labels([0.0] * 10, min_history=3) == [None] * 10
flat = pd.DataFrame({"close": [100.0] * 40}, index=pd.bdate_range("2024-01-01", periods=40))
assert classify_vol_regime(flat, vol_len=5, min_history=5).isna().all()
for bad in (dict(hi_q=0.4, lo_q=0.6), dict(hi_q=1.0, lo_q=0.4), dict(hi_q=0.6, lo_q=0.0)):
    try:
        classify_from_vol(pd.Series([1.0, 2.0, 3.0]), **bad)
        raise AssertionError(f"잘못된 문턱이 통과: {bad}")
    except ValueError:
        pass
print("[PASS] 4) 결측은 NaN 라벨(분포 제외), 평탄 가격은 분류 안 함, 문턱 순서 위반은 ValueError")

# --- 5) 독립 구현(O(n^2) 순수 루프)과 일치 ---------------------------------------------------
rng = np.random.default_rng(11)
vol = np.abs(rng.normal(0.02, 0.01, 400))
vol[[50, 51, 200]] = np.nan
H, L, MH = 0.65, 0.35, 30
ref, hist, st = [], [], None
for x in vol:
    if np.isnan(x):
        ref.append(None); continue
    hist.append(x)
    if len(hist) < MH:
        ref.append(None); continue
    p = sum(1 for y in hist if y <= x) / len(hist)
    if st is None:
        st = "H" if p >= 0.5 else "L"
    elif st == "L" and p >= H:
        st = "H"
    elif st == "H" and p <= L:
        st = "L"
    ref.append(st)
assert labels(vol, hi_q=H, lo_q=L, min_history=MH) == ref
print("[PASS] 5) bisect 구현 = 순수 파이썬 루프 구현 (400봉, 결측 3개 포함, 불일치 0)")

# --- 6) 인과성 ------------------------------------------------------------------------------
cc = 100 * np.cumprod(1 + rng.normal(0, 0.015, 600))
d6 = pd.DataFrame({"close": cc}, index=pd.bdate_range("2021-01-01", periods=600))
full = classify_vol_regime(d6, vol_len=20, min_history=100)
for k in (150, 300, 450, 599):
    a, b = full.iloc[:k], classify_vol_regime(d6.iloc[:k], vol_len=20, min_history=100)
    assert ((a.isna() & b.isna()) | (a == b)).all(), k
print("[PASS] 6) 인과성: 150/300/450/599봉에서 자른 라벨이 전체 계산의 앞부분과 정확히 일치(미래정보 없음)")

# --- 7) 변동성이 바뀌면 라벨이 따라간다 -------------------------------------------------------
# 400봉 σ=1%, 이어서 300봉 σ=3%. 고변동 구간의 변동성은 이전 분포의 거의 모든 값보다 크다.
# 고변동 구간에서 분위의 하한은 400/700=0.571 > lo_q 이므로 한 번 고변동이 되면 저변동으로 돌아올 수 없다.
r = np.concatenate([rng.normal(0, 0.01, 400), rng.normal(0, 0.03, 300)])
c7 = 100 * np.cumprod(1 + r)
d7 = pd.DataFrame({"close": c7}, index=pd.bdate_range("2020-01-01", periods=700))
lab7 = classify_vol_regime(d7, vol_len=20, hi_q=0.6, lo_q=0.4, min_history=100)
assert (lab7.iloc[-150:] == "high_vol").all(), lab7.iloc[-150:].value_counts()
print("[PASS] 7) 변동성 3배 구간의 마지막 150봉이 전부 고변동으로 분류")

# --- 8) 히스테리시스가 전환 횟수를 줄인다(고정 시드의 성질 확인) ----------------------------------
noisy = np.abs(rng.normal(0.02, 0.005, 800))
sw_wide = regime_switches(classify_from_vol(pd.Series(noisy), hi_q=0.7, lo_q=0.3, min_history=50))
sw_thin = regime_switches(classify_from_vol(pd.Series(noisy), hi_q=0.51, lo_q=0.49, min_history=50))
assert sw_wide < sw_thin, (sw_wide, sw_thin)
print(f"[PASS] 8) 라벨 전환 횟수: 문턱 폭 넓게(0.3/0.7) {sw_wide}회 < 좁게(0.49/0.51) {sw_thin}회")

# --- 9) regime_study에 고변동/저변동 라벨 연결, 기본값 보존 -----------------------------------------
from backtest.regime import RegimeResult, format_regime_report, regime_share, regime_study, REGIMES
from backtest.report_common import regime_header

N = 600
EVT = np.arange(12, N - 5, 8)
r9 = np.random.default_rng(3).normal(0, 0.005, N)
reg = pd.Series(["high_vol" if i < 300 else "low_vol" for i in range(N)],
                index=pd.bdate_range("2021-01-01", periods=N), dtype=object)
for e in EVT:
    if e < 300:
        r9[e + 1] += 0.015                                     # 우위는 고변동 국면에만 심는다
cl = 10000 * np.cumprod(1 + r9)
d = pd.DataFrame({"open": cl, "high": cl, "low": cl, "close": cl, "volume": 1e5}, index=reg.index)
d["crossover_confirmed"] = False
d.iloc[EVT, d.columns.get_loc("crossover_confirmed")] = True
d["crossunder_confirmed"] = False
CK = dict(horizons=(1, 3, 5), max_curve_horizon=5, entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
          newey_west=False, drop_overlapping=False, fdr=0.10, deflate=True, dsr_threshold=0.95, min_events=20,
          extra_trials=0, barrier=False)
res = regime_study({"delta_trading": d}, reg, CK, labels=VOL_REGIMES)
s = res.summary
assert set(s["regime"]) == set(VOL_REGIMES) and res.labels == VOL_REGIMES
g = lambda rg: s[(s["regime"] == rg) & (s["signal"] == "델타트레이딩-골든크로스") & (s["horizon"] == 3)].iloc[0]
assert g("high_vol")["n_events"] == int((EVT < 300).sum()) and g("low_vol")["n_events"] == int((EVT >= 300).sum())
assert g("high_vol")["mean_return_pct"] > 1.0 and abs(g("low_vol")["mean_return_pct"]) < 0.8
# 국면이 2개이므로 DSR 시도 횟수 가산 = (2-1) * 12 * 3 = 36  (추세 3분류였다면 72)
assert res.params["extra_trials_added"] == 1 * 12 * 3 == 36, res.params
assert abs(res.share["high_vol"] - 0.5) < 1e-9 and set(res.share.index) == set(VOL_REGIMES)
rep = format_regime_report(res, 3)
assert "고변동" in rep and "저변동" in rep and "상승" not in rep.split("국면 비중")[1].split("\n")[0], rep[:300]
hdr = regime_header(res, CK, 3)
assert "high_vol" in hdr and "low_vol" in hdr and "bull" not in hdr, hdr
# 기본값 보존: labels 인자를 안 주면 이전과 같은 3분류
base = regime_study({"delta_trading": d}, reg.map({"high_vol": "bull", "low_vol": "bear"}), CK)
assert base.labels == REGIMES and set(base.summary["regime"]) == set(REGIMES)
assert base.params["extra_trials_added"] == 72
print("[PASS] 9) regime_study(labels=고변동/저변동): 이벤트 분할·DSR 가산 36·보고서/헤더 라벨, 기본은 추세 3분류(가산 72) 유지")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
