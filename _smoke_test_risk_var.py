"""
risk/positions.py · risk/var.py 스모크테스트 — 기대값은 손계산. 합성 데이터 결과는 실제 시장 성과가 아님.
"""
import math

import numpy as np
import pandas as pd

from backtest.stats import norm_ppf
from risk.positions import constant_weights, gross_exposure, portfolio_returns
from risk.var import (compute_var, ewma_vol, filtered_historical_var, historical_var, parametric_var,
                      rolling_vol, _tail_k)

IX = pd.bdate_range("2024-01-01", periods=5)

# --- 1) 포트폴리오 수익률: t-1 비중 × t 수익률 (룩어헤드 없음) --------------------------------
# A: 100,110,99,108.9,108.9  -> r = .1, -.1, .1, 0
# B: 50, 50, 55, 55,  49.5   -> r = 0, .1, 0, -.1
# 비중 A=0.5,B=0.5 고정 -> r_p = .05, 0, .05, -.05 (첫날 NaN)
closes = {"A": pd.Series([100, 110, 99, 108.9, 108.9], index=IX, dtype=float),
          "B": pd.Series([50, 50, 55, 55, 49.5], index=IX, dtype=float)}
w = constant_weights({"A": 0.5, "B": 0.5}, IX)
rp = portfolio_returns(w, closes)
assert np.isnan(rp.iloc[0])
assert np.allclose(rp.iloc[1:].to_numpy(), [0.05, 0.0, 0.05, -0.05]), rp.to_numpy()
# 비중이 날마다 바뀌면 전날 비중이 적용돼야 한다: t=1에 A=1.0으로 바꾸면 t=2 수익률만 r_A(t=2) = -0.1
w2 = w.copy(); w2.iloc[1] = [1.0, 0.0]
rp2 = portfolio_returns(w2, closes)
assert abs(rp2.iloc[2] - (-0.1)) < 1e-12 and abs(rp2.iloc[1] - 0.05) < 1e-12, rp2.to_numpy()
# 숏과 레버리지: A=-0.5, B=1.5 -> t=1: -0.5*0.1 + 1.5*0 = -0.05 ; 총노출 2.0
w3 = constant_weights({"A": -0.5, "B": 1.5}, IX)
assert abs(portfolio_returns(w3, closes).iloc[1] + 0.05) < 1e-12 and (gross_exposure(w3) == 2.0).all()
# 결측 종가는 0 수익으로 메우지 않고 그 날을 NaN으로
c_miss = {k: v.copy() for k, v in closes.items()}; c_miss["B"].iloc[2] = np.nan
rpm = portfolio_returns(w, c_miss)
assert np.isnan(rpm.iloc[2]) and np.isnan(rpm.iloc[3])            # t=2 (B NaN), t=3 (B pct_change도 NaN)
print("[PASS] 1) 포트폴리오 수익률: 전날 비중 적용, 숏·레버리지, 결측은 NaN")

# --- 2) 꼬리 표본 수 k = ceil((1-alpha) n) ------------------------------------------------
assert _tail_k(0.99, 250) == 3 and _tail_k(0.99, 100) == 1 and _tail_k(0.95, 100) == 5 and _tail_k(0.975, 40) == 1
print("[PASS] 2) k: (0.99,250)=3, (0.99,100)=1, (0.95,100)=5, (0.975,40)=1")

# --- 3) 역사적 VaR/ES 손계산 ------------------------------------------------------------------
# 수익률 10개: -5,-3,-1,0,1,2,3,4,5,6 (%) 를 window=10, alpha=0.8 -> k=2: 손실 상위 2개 = 5,3 -> VaR=3%, ES=4%
# 11번째 날의 VaR가 앞 10개로 계산된다.
r = pd.Series([-.05, -.03, -.01, 0, .01, .02, .03, .04, .05, .06, .10], index=pd.bdate_range("2024-01-01", periods=11))
hv = historical_var(r, alpha=0.8, window=10)
assert hv["var"].iloc[:10].isna().all()
assert abs(hv["var"].iloc[10] - 0.03) < 1e-12 and abs(hv["es"].iloc[10] - 0.04) < 1e-12, hv.iloc[10]
# alpha=0.95, window=10 -> k=1: VaR=ES=5%
hv2 = historical_var(r, alpha=0.95, window=10)
assert abs(hv2["var"].iloc[10] - 0.05) < 1e-12 and abs(hv2["es"].iloc[10] - 0.05) < 1e-12
print("[PASS] 3) 역사적 VaR: alpha .8 -> VaR 3%/ES 4%, alpha .95 -> 5%/5%, 앞 window일 NaN")

# --- 4) EWMA 변동성 손계산 -------------------------------------------------------------------
# r = [0.01, 0.02, -0.01], λ=0.5, min_periods=1.  ewm(adjust=False) 재귀: v0 = r0² = 1e-4;
# v1 = .5*1e-4 + .5*4e-4 = 2.5e-4 ; v2 = .5*2.5e-4 + .5*1e-4 = 1.75e-4.  시프트 1 -> σ[1]=sqrt(1e-4)=0.01, σ[2]=sqrt(2.5e-4)
rr = pd.Series([0.01, 0.02, -0.01, 0.0], index=pd.bdate_range("2024-01-01", periods=4))
ev = ewma_vol(rr, lam=0.5, min_periods=1)
assert np.isnan(ev.iloc[0]) and abs(ev.iloc[1] - 0.01) < 1e-12 and abs(ev.iloc[2] - math.sqrt(2.5e-4)) < 1e-12 \
    and abs(ev.iloc[3] - math.sqrt(1.75e-4)) < 1e-12, ev.to_numpy()
rv = rolling_vol(rr, window=2)
assert abs(rv.iloc[2] - np.std([0.01, 0.02], ddof=1)) < 1e-12 and np.isnan(rv.iloc[1])
print("[PASS] 4) EWMA σ 재귀 손계산 일치(시프트 1), 롤링 σ ddof=1")

# --- 5) 파라메트릭 VaR = z σ, ES = σ φ(z)/(1-α) --------------------------------------------
sig = pd.Series([np.nan, 0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=3))
pv = parametric_var(pd.Series([0, 0, 0.0], index=sig.index), alpha=0.99, vol=sig)
z = norm_ppf(0.99)
assert abs(z - 2.326348) < 1e-6
assert abs(pv["var"].iloc[1] - 0.01 * z) < 1e-12 and abs(pv["var"].iloc[2] - 0.02 * z) < 1e-12
es_ratio = math.exp(-z * z / 2) / math.sqrt(2 * math.pi) / 0.01        # φ(z)/(1-α) ≈ 2.6652
assert abs(es_ratio - 2.6652) < 1e-3, es_ratio
assert abs(pv["es"].iloc[1] - 0.01 * es_ratio) < 1e-12
print(f"[PASS] 5) 파라메트릭: z(.99)={z:.6f}, ES/σ={es_ratio:.4f}(정규 ES 상수 2.665 일치)")

# --- 6) FHS: 표준화 손실의 분위 × 현재 σ ------------------------------------------------------
# σ가 상수 s이면 FHS = 역사적 VaR 와 정확히 같다(표준화와 재스케일이 상쇄). 상수 σ를 만들려면 ewma에 λ=1은
# 불가(분산이 첫 값에 고정) -> 대신 직접 비교: FHS 결과 / σ[t] 가 표준화 손실의 k번째 값인지 확인.
rng = np.random.default_rng(5)
rs = pd.Series(rng.normal(0, 0.01, 400) * np.r_[np.ones(200), 3 * np.ones(200)],
               index=pd.bdate_range("2022-01-01", periods=400))
fh = filtered_historical_var(rs, alpha=0.95, window=100)
sg = ewma_vol(rs)
t = 350
stdl = (-rs / sg).to_numpy()[t - 100:t]
k = _tail_k(0.95, 100)
q = np.sort(stdl)[-k]
assert abs(fh["var"].iloc[t] - q * sg.iloc[t]) < 1e-12
# 변동성 3배 구간 진입 후 FHS VaR는 롤링 역사적보다 빨리 커져야 한다. 역사적(100창)은 t=260에서 아직
# 저변동 60일이 섞여 있고, FHS는 σ[t]가 이미 고변동을 반영한다.
hv100 = historical_var(rs, 0.95, 100)
assert fh["var"].iloc[260] > hv100["var"].iloc[260], (fh["var"].iloc[260], hv100["var"].iloc[260])
print("[PASS] 6) FHS = 표준화 손실 k번째 × σ[t]; 변동성 급등 직후 FHS > 롤링 역사적(반응 속도)")

# --- 7) compute_var 분기·attrs·잘못된 입력 --------------------------------------------------
for m in ("historical", "parametric", "fhs"):
    out = compute_var(rs, m, 0.99, 100)
    assert set(out.columns) == {"var", "es"} and out.attrs["method"] == m and (out["es"].dropna() >= out["var"].dropna() - 1e-15).all()
for bad in (dict(alpha=0.3), dict(alpha=1.0)):
    try:
        compute_var(rs, "fhs", **bad); raise AssertionError(bad)
    except ValueError:
        pass
try:
    compute_var(rs, "montecarlo"); raise AssertionError("unknown method passed")
except ValueError:
    pass
print("[PASS] 7) compute_var: 세 방법 모두 ES >= VaR, attrs 기록, 잘못된 alpha·method는 ValueError")

# --- 8) 예측 시점: VaR[t]는 r[t]를 모른다 ---------------------------------------------------
# r[t]를 크게 바꿔도 VaR[t]는 그대로여야 하고 VaR[t+1]만 달라진다.
rs2 = rs.copy(); rs2.iloc[300] = -0.5
for m in ("historical", "parametric", "fhs"):
    a, b = compute_var(rs, m, 0.99, 100), compute_var(rs2, m, 0.99, 100)
    assert abs(a["var"].iloc[300] - b["var"].iloc[300]) < 1e-15, m
    assert b["var"].iloc[301] > a["var"].iloc[301], m
print("[PASS] 8) 세 방법 모두 VaR[t]는 r[t]에 의존하지 않음(룩어헤드 없음), VaR[t+1]은 반응")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
