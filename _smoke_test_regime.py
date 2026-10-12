"""
backtest.regime 스모크테스트 — 기대값은 손계산. 합성 데이터 결과는 실제 시장 성과가 아님.
"""
import math, statistics
import numpy as np
import pandas as pd

from backtest.regime import (REGIMES, classify_regime, regime_share, regime_study, regime_consistency,
                             format_regime_report)

# --- 1) 손계산 라벨: ma=3, ret=2, vol=2, z_thr=0.5 ------------------------------
c = [100, 102, 101, 105, 110, 104, 98, 99, 100]
df = pd.DataFrame({"close": c}, index=pd.bdate_range("2024-01-01", periods=len(c)))
lab = classify_regime(df, ma_len=3, ret_len=2, vol_len=2, z_thr=0.5)
expected = [None, None, "sideways", "bull", "bull", "sideways", "bear", "bear", "bull"]
# 손계산 근거(일부): t=3  z=0.02941/(0.034935*sqrt2)=+0.595 > 0.5, 종가 105 > MA 102.67 -> bull
#                   t=5  z=-0.0932 (|z|<0.5) -> sideways          t=7  z=-0.708 < -0.5, 종가 99 < MA 100.33 -> bear
#                   t=2  종가 101 == MA 101.0 (>도 <도 아님) -> sideways
for i, (got, exp) in enumerate(zip(lab, expected)):
    assert (pd.isna(got) and exp is None) or got == exp, (i, got, exp)
print("[PASS] 1) 손계산 라벨 9봉: [NaN, NaN, 횡보, 상승, 상승, 횡보, 하락, 하락, 상승] 일치")

# --- 2) 독립 구현(파이썬 루프)과 pandas 구현이 같은가 --------------------------------
rng = np.random.default_rng(1)
cc = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, 500))
d2 = pd.DataFrame({"close": cc}, index=pd.bdate_range("2021-01-01", periods=500))
M, R, V, Z = 50, 20, 20, 0.5
got = classify_regime(d2, M, R, V, Z)
ref = []
for t in range(len(cc)):
    if t < max(M - 1, R, V):
        ref.append(None); continue
    ma = sum(cc[t - M + 1: t + 1]) / M
    ret = cc[t] / cc[t - R] - 1
    pcts = [cc[k] / cc[k - 1] - 1 for k in range(t - V + 1, t + 1)]
    z = ret / (statistics.stdev(pcts) * math.sqrt(R))
    ref.append("bull" if (cc[t] > ma and z > Z) else "bear" if (cc[t] < ma and z < -Z) else "sideways")
bad = [t for t in range(len(cc)) if not ((pd.isna(got.iloc[t]) and ref[t] is None) or got.iloc[t] == ref[t])]
assert not bad, bad[:5]
print(f"[PASS] 2) pandas 구현 = 순수 파이썬 루프 구현 (500봉 중 불일치 0), 분류 비중 {regime_share(got).round(2).to_dict()}")

# --- 3) 인과성: 미래 데이터를 잘라내도 과거 라벨이 변하지 않는다 ----------------------------
full = classify_regime(d2, M, R, V, Z)
for k in (80, 200, 333, 499):
    part = classify_regime(d2.iloc[:k], M, R, V, Z)
    a, b = full.iloc[:k], part
    assert ((a.isna() & b.isna()) | (a == b)).all(), k
print("[PASS] 3) 인과성: 80/200/333/499봉에서 자른 라벨이 전체 계산의 앞부분과 정확히 일치(미래정보 없음)")

# --- 4) 변동성 0이면 분류하지 않는다(0으로 나누기 방지) -----------------------------------
flat = pd.DataFrame({"close": [100.0] * 30}, index=pd.bdate_range("2024-01-01", periods=30))
assert classify_regime(flat, 5, 5, 5, 0.5).isna().all()
print("[PASS] 4) 가격이 완전히 평탄하면 라벨 NaN(0 나누기 방지)")

# --- 5) 국면별 연구 + 전체 BH + DSR 시도 횟수 -----------------------------------------
N = 600
EVT = np.arange(12, N - 5, 8)
rng = np.random.default_rng(3)
r = rng.normal(0, 0.005, N)
reg = pd.Series(["bull" if i < 300 else "bear" for i in range(N)],
                index=pd.bdate_range("2021-01-01", periods=N), dtype=object)     # 앞 절반 bull, 뒤 절반 bear
for e in EVT:
    if e < 300:
        r[e + 1] += 0.015                                        # 우위는 bull 국면에만
cl = 10000 * np.cumprod(1 + r)
d = pd.DataFrame({"open": cl, "high": cl, "low": cl, "close": cl, "volume": 1e5}, index=reg.index)
d["crossover_confirmed"] = False
d.iloc[EVT, d.columns.get_loc("crossover_confirmed")] = True
d["crossunder_confirmed"] = False
CK = dict(horizons=(1, 3, 5), max_curve_horizon=5, entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
          newey_west=False, drop_overlapping=False, fdr=0.10, deflate=True, dsr_threshold=0.95, min_events=20,
          extra_trials=0, barrier=False)
res = regime_study({"delta_trading": d}, reg, CK)
s = res.summary
g = lambda rg: s[(s["regime"] == rg) & (s["signal"] == "델타트레이딩-골든크로스") & (s["horizon"] == 3)].iloc[0]
exp_bull, exp_bear = int((EVT < 300).sum()), int((EVT >= 300).sum())
assert g("bull")["n_events"] == exp_bull and g("bear")["n_events"] == exp_bear      # 이벤트가 국면대로 갈린다
assert (s[s["regime"] == "sideways"]["n_events"] == 0).all()                      # 이벤트 없는 국면은 n_events=0 행으로만 남음
assert g("bull")["mean_return_pct"] > 1.0 and abs(g("bear")["mean_return_pct"]) < 0.8
from backtest.engine import benjamini_hochberg
# 전체 BH는 모든 국면의 p값을 한 번에 놓고 계산한 값과 정확히 같아야 한다(정의 그대로)
assert (s["significant_bh_global"].values == benjamini_hochberg(s["p_value"].values, fdr=0.10)).all()
assert bool(g("bull")["significant_bh_global"])                      # 심은 우위는 통과
# (참고) BH는 거짓발견비율을 제어하는 절차라 진짜 발견이 여럿이면 우연히 p가 작은 항목이 함께 통과할 수 있다
# -> 우위가 없는 국면이 통과하지 않는다고 단정하는 테스트는 만들지 않는다.
# DSR 시도 횟수 = (판정 가능한 행 수) + (n_reg-1) * (SIGNAL_SPECS 12 x 보유기간 3) = elig + 72
assert res.params["extra_trials_added"] == 2 * 12 * 3 == 72
assert all(v >= 72 for v in res.trials_by_regime.values()), res.trials_by_regime
assert abs(res.share["bull"] - 0.5) < 1e-9 and res.n_unclassified == 0
print(f"[PASS] 5) 국면별 연구: 이벤트 상승 {exp_bull}/하락 {exp_bear}건 손계산 일치, 심은 우위는 상승 국면에서 통과, "
      f"DSR 시도 횟수 {res.trials_by_regime}(>=72)")

# --- 6) 시험이 늘면 BH 기준이 엄격해진다(국면을 나누면 시험 수가 늘어나는 이유) ---------------------
from backtest.engine import benjamini_hochberg
p3 = np.array([0.04, 0.05, 0.06])
assert benjamini_hochberg(p3, 0.10).sum() == 3        # m=3: 기준 .0333 .0667 .1 -> 3번째 p=.06 <= .1 이므로 가장 큰 k=3, 셋 다 통과
p6 = np.concatenate([p3, [0.3, 0.4, 0.5]])            # 같은 세 개 + 우연 수준 시험 3개를 더 하면
assert benjamini_hochberg(p6, 0.10).sum() == 0        # m=6: 기준 .0167 .0333 .05 .0667 .0833 .1 -> 어느 것도 통과 못함
print("[PASS] 6) 같은 p값 세 개가 시험 3개일 땐 3개 통과, 시험을 6개로 늘리면 0개 통과(손계산)")

# --- 7) 국면 간 일관성 손계산 ----------------------------------------------------------
cs = pd.DataFrame({
    "signal": ["A"] * 3 + ["B"] * 3 + ["C"] * 3, "horizon": 5,
    "regime": list(REGIMES) * 3,
    "n_obs": [10, 10, 10, 10, 10, 10, 10, 2, 2],
    "mean_return_pct": [1.0, 2.0, -1.0, 1.0, 2.0, 3.0, 1.0, 5.0, 5.0]})
cons = regime_consistency(cs, min_n=5).set_index("signal")
assert cons.loc["A", "sign_flip"] and not cons.loc["A", "consistent_positive"]      # +,+,- -> 뒤집힘
assert cons.loc["B", "consistent_positive"] and not cons.loc["B", "sign_flip"]      # +,+,+
assert cons.loc["C", "n_regimes"] == 1 and not cons.loc["C", "consistent_positive"]  # 평가 가능 국면 1개뿐 -> 판단 안 함
print("[PASS] 7) 일관성: A(+,+,-)=뒤집힘, B(+,+,+)=전부 양수, C(표본 충분한 국면 1개)=판단 안 함")

rep = format_regime_report(res, 3)
for token in ["시장 국면별 검증", "국면 비중", "significant_bh_global", "국면 간 일관성"]:
    assert token in rep, token
print("[PASS] 8) 보고서에 국면 비중·전체 BH·일관성이 출력")

# --- 9) lo/hi 구간과 blocked 반영 -------------------------------------------------------
res_w = regime_study({"delta_trading": d}, reg, CK, lo=reg.index[100], hi=reg.index[250], blocked=[reg.index[116]])
gw = res_w.summary[(res_w.summary["regime"] == "bull") & (res_w.summary["signal"] == "델타트레이딩-골든크로스") &
                    (res_w.summary["horizon"] == 3)].iloc[0]
exp_w = int(((EVT >= 100) & (EVT <= 250) & (EVT != 116)).sum())   # 116 = 12 + 8*13 은 이벤트
assert 116 in EVT and gw["n_events"] == exp_w, (gw["n_events"], exp_w)
print(f"[PASS] 9) 구간 [100,250] + 차단(116): 이벤트 {exp_w}건")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
