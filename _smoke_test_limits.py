"""
risk/limits.py 스모크테스트 — 한도 도출·적용·비교. 기대값은 손계산 또는 독립 계산. 합성 데이터, 실제 시장 성과 아님.
"""
import numpy as np
import pandas as pd

from risk.limits import (DerivedLimits, apply_controls, bootstrap_mdd_quantile, compare_controlled, derive_limits,
                         max_drawdown)
from risk.positions import constant_weights, portfolio_returns
from risk.var import compute_var, ewma_vol

# --- 1) 최대낙폭 손계산 -----------------------------------------------------------------------
# +10%, -20%, +5% -> 자산 1.1, 0.88, 0.924 ; 고점 1.1 ; 낙폭 최대 = 1 - 0.88/1.1 = 0.2
assert abs(max_drawdown([0.1, -0.2, 0.05]) - 0.2) < 1e-12
assert max_drawdown([0.01, 0.02]) == 0.0 and np.isnan(max_drawdown([]))
print("[PASS] 1) MDD 손계산 0.2, 상승만이면 0, 빈 입력 NaN")

# --- 2) 블록 부트스트랩 MDD 분위 ---------------------------------------------------------------
rng = np.random.default_rng(1)
r = pd.Series(rng.normal(0.0003, 0.012, 1500))
q95, mdd = bootstrap_mdd_quantile(r, horizon=252, block_len=20, n_sims=500, quantile=0.95, seed=3)
assert len(mdd) == 500 and 0 < q95 < 1 and abs(q95 - np.quantile(mdd, 0.95)) < 1e-12
q95b, _ = bootstrap_mdd_quantile(r, horizon=252, block_len=20, n_sims=500, quantile=0.95, seed=3)
assert q95 == q95b                                                   # 시드 고정 → 재현
q50, _ = bootstrap_mdd_quantile(r, n_sims=500, quantile=0.5, seed=3)
assert q50 < q95
qn, mn = bootstrap_mdd_quantile(pd.Series(rng.normal(0, 0.01, 30)), block_len=20)
assert np.isnan(qn) and len(mn) == 0                                 # 표본 부족
# 수익률이 전부 0이면 MDD 0
qz, _ = bootstrap_mdd_quantile(pd.Series(np.zeros(500)), n_sims=100)
assert qz == 0.0
print(f"[PASS] 2) 부트스트랩: 재현 가능, 50%<95% 분위({q50:.3f}<{q95:.3f}), 표본 부족 NaN, 0 수익률→MDD 0")

# --- 3) 한도 도출: 출처가 기록되고 값이 정의와 일치 -------------------------------------------------
ix = pd.bdate_range("2018-01-01", periods=1500)
A = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.012, 1500)), index=ix)
B = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.025, 1500)), index=ix)
closes = {"A": A, "B": B}
w = constant_weights({"A": 0.5, "B": 0.5}, ix)
rp = portfolio_returns(w, closes)
lim = derive_limits(rp, alpha=0.975, window=250, method="fhs", seed=0)
es_series = compute_var(rp, "fhs", 0.975, 250)["es"].dropna()
assert abs(lim.es_target - es_series.median()) < 1e-15
assert abs(lim.name_budget - 0.5 * lim.es_target) < 1e-15
assert 0 < lim.dd_limit < 1 and "부트스트랩" in lim.provenance["dd_limit"] and "분위" in lim.provenance["es_target"]
print(f"[PASS] 3) 한도 도출: ES 목표=ES 중앙값 {lim.es_target:.3%}, 종목 예산=절반, 낙폭 한도 {lim.dd_limit:.1%} (출처 기록)")

# --- 4) 통제 적용 — 각 한도를 독립적으로 확인 ----------------------------------------------------
W = 250
# 4a) 한도를 아주 느슨하게 → 워밍업(미정의) 뒤에는 원 비중 그대로, 워밍업은 0
loose = DerivedLimits(es_target=10.0, name_budget=10.0, dd_limit=np.nan, alpha=0.975)
res = apply_controls(w, closes, loose, window=W, method="fhs")
und = res.log["undefined"]
assert und.iloc[:W + 1].all() or und.iloc[:W].all(), und.iloc[:W + 2].tolist()
assert (res.weights[~und] == w.reindex(res.weights.index)[~und]).all().all()
assert (res.weights[und] == 0).all().all()
print(f"[PASS] 4a) 느슨한 한도: 미정의 {int(und.sum())}일은 비중 0, 나머지는 원 비중 그대로")

# 4b) ES 한도만: scale = es_target / ES[t] (ES[t] > target 인 날), 그 외 1. 독립 계산으로 대조
es_t = compute_var(rp, "fhs", 0.975, W)["es"]
tight_es = DerivedLimits(es_target=float(es_t.dropna().quantile(0.3)), name_budget=10.0, dd_limit=np.nan, alpha=0.975)
res_es = apply_controls(w, closes, tight_es, window=W, method="fhs")
ok = ~res_es.log["undefined"]
exp_scale = np.minimum(1.0, tight_es.es_target / es_t[ok])
assert np.allclose(res_es.log.loc[ok, "scale_es"], exp_scale)
assert np.allclose(res_es.weights.loc[ok, "A"], 0.5 * exp_scale)
n_scaled = int((exp_scale < 1).sum())
assert 0.6 * ok.sum() < n_scaled < 0.8 * ok.sum(), n_scaled      # 30% 분위를 목표로 하면 약 70%의 날에 축소
print(f"[PASS] 4b) ES 한도: 축소 배율 = 목표/ES[t] 독립 계산과 일치, 축소 발동 {n_scaled}/{int(ok.sum())}일")

# 4c) 종목 예산만: 변동성 큰 B가 잡힌다. 잡힌 날 |w_B| σ_B == 예산
sigB = ewma_vol(B.pct_change()); sigA = ewma_vol(A.pct_change())
budget = float((0.5 * sigB.dropna()).quantile(0.5))              # B의 절반이 걸리는 수준
tight_name = DerivedLimits(es_target=10.0, name_budget=budget, dd_limit=np.nan, alpha=0.975)
res_nm = apply_controls(w, closes, tight_name, window=W, method="fhs")
ok = ~res_nm.log["undefined"]
capped = ok & (res_nm.log["n_name_capped"] > 0)
assert capped.sum() > 0
riskB = (res_nm.weights["B"].abs() * sigB)[capped]
assert np.allclose(riskB, budget), riskB.describe()
assert (res_nm.weights.loc[capped, "B"] < 0.5 - 1e-12).all()
assert (res_nm.weights.loc[ok & ~capped, "B"] == 0.5).all()
assert (res_nm.weights.loc[ok, "A"] <= 0.5 + 1e-15).all()
print(f"[PASS] 4c) 종목 예산: 잡힌 {int(capped.sum())}일에 |w_B|σ_B = 예산, 안 잡힌 날은 원 비중")

# 4d) 일일 손실 한도: t일 손실 > ES[t] → t+1일 비중 증가 금지(축소·유지만)
C = A.copy(); t_crash = 600
C.iloc[t_crash] = C.iloc[t_crash - 1] * 0.85                     # -15% 하루
C.iloc[t_crash + 1:] = C.iloc[t_crash + 1:] * (C.iloc[t_crash] / A.iloc[t_crash])
closes_c = {"A": C, "B": B}
w_step = constant_weights({"A": 0.3, "B": 0.3}, ix)
w_step.iloc[t_crash + 1:] = [0.6, 0.6]                           # 다음 날 비중을 두 배로 올리려는 원 비중
res_h = apply_controls(w_step, closes_c, loose, window=W, method="fhs")
assert bool(res_h.log["halt_next"].iloc[t_crash])
assert (res_h.weights.iloc[t_crash + 1].to_numpy() == res_h.weights.iloc[t_crash].to_numpy()).all()   # 증가 차단
assert (res_h.weights.iloc[t_crash + 2].to_numpy() == [0.6, 0.6]).all()                              # 하루만 차단
rate = res_h.log.loc[~res_h.log["undefined"], "halt_next"].mean()
# 정규분포에서 P(L > ES_97.5%) = P(Z > 2.338) ≈ 0.97%. 합성 수익률이 정규이므로 이 근방이어야 한다.
assert 0.004 < rate < 0.025, rate
print(f"[PASS] 4d) 일일 손실: 폭락 다음 날 비중 증가 차단(그 다음 날 해제), 발동 빈도 {rate:.2%} (정규 이론값 ≈0.97%)")

# 4e) 낙폭 서킷브레이커(히스테리시스): 한도 초과 → 절반, 한도/2 아래로 회복 → 해제
# 가격이 완전히 평탄하면 σ=0 → ES 미정의 → 비중 0이 되어 낙폭 자체가 생기지 않는다(의도된 동작).
# 그래서 작은 노이즈를 깔고 그 위에 -30% 하락 → 바닥 유지 → 전고점 회복을 곱한다.
base = 100 * np.cumprod(1 + rng.normal(0, 0.002, 1500))
mult = np.ones(1500); mult[300:320] = np.linspace(1, 0.70, 20); mult[320:400] = 0.70
D = pd.Series(base * mult, index=ix)
D.iloc[400:] = D.iloc[:300].max() * 1.02                          # 전고점 위로 회복(평탄)
D.iloc[401:] = D.iloc[400] * np.cumprod(1 + rng.normal(0, 0.002, 1099))
E_ = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.001, 1500)), index=ix)
dd_lim = DerivedLimits(es_target=10.0, name_budget=10.0, dd_limit=0.20, alpha=0.975)
res_dd = apply_controls(constant_weights({"D": 1.0, "E": 0.0}, ix), {"D": D, "E": E_}, dd_lim, window=W, method="fhs")
lg = res_dd.log
assert not lg["undefined"].iloc[300], "낙폭 구간이 워밍업 안에 있음"
assert not lg["dd_active"].iloc[302] and lg["dd"].iloc[319] > 0.20 and lg["dd_active"].iloc[319], (lg["dd"].iloc[319], lg["dd_active"].iloc[319])
assert abs(res_dd.weights["D"].iloc[350] - 0.5) < 1e-12           # 바닥에서 절반
assert lg["dd_active"].iloc[399]                                  # 통제 후 자산도 10%(한도/2) 넘게 빠져 있음 → 유지
assert not lg["dd_active"].iloc[420] and res_dd.weights["D"].iloc[420] == 1.0, (lg["dd"].iloc[420], lg["dd_active"].iloc[420])   # 회복 → 해제
print("[PASS] 4e) 낙폭: 20% 넘자 노출 절반, 바닥 유지 중 계속, 10% 아래로 회복하면 해제")

# 4f) 룩어헤드: t+1 이후 가격을 바꿔도 t의 통제 비중은 변하지 않는다
closes_fut = {"A": A.copy(), "B": B.copy()}
closes_fut["A"].iloc[900:] *= 0.5
r1 = apply_controls(w, closes, tight_es, window=W, method="fhs").weights.iloc[:900]
r2 = apply_controls(w, closes_fut, tight_es, window=W, method="fhs").weights.iloc[:900]
assert np.allclose(r1.to_numpy(), r2.to_numpy())
print("[PASS] 4f) 미래 가격을 바꿔도 과거 통제 비중 불변(룩어헤드 없음)")

# --- 5) 비교표와 자기 점검: 통제 후 ES 초과율 ---------------------------------------------------------
res_full = apply_controls(w, closes, lim, window=W, method="fhs")
cmp = compare_controlled(res_full)
assert list(cmp["구분"]) == ["통제 전", "통제 후"] and cmp["평균총노출"].iloc[1] < cmp["평균총노출"].iloc[0]
assert cmp["연환산변동성"].iloc[1] < cmp["연환산변동성"].iloc[0]
c = res_full.counts()
assert c["es_scaled"] > 0 and c["undefined"] >= W
print(f"[PASS] 5) 비교표: 통제 후 노출·변동성 감소, 발동 횟수 {c}")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
