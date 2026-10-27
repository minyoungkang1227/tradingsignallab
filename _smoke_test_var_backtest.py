"""
risk/var_backtest.py 스모크테스트 — Kupiec·Christoffersen·바젤 신호등 손계산. 합성 데이터, 실제 시장 성과가 아님.
"""
import math

import numpy as np
import pandas as pd

from backtest.stats import chi2_sf
from risk.var import compute_var
from risk.var_backtest import (FAIL, FIT, PASS, UNFIT, UNKNOWN, basel_traffic_light,
                               christoffersen_independence, format_var_backtest, kupiec_pof, var_backtest)

# --- 1) Kupiec 손계산: n=250, x=5, p=0.01 ---------------------------------------------------
# LR = -2[ 245 ln(.99) + 5 ln(.01) - 245 ln(1-.02) - 5 ln(.02) ]
lr_hand = -2 * (245 * math.log(0.99) + 5 * math.log(0.01) - 245 * math.log(0.98) - 5 * math.log(0.02))
k = kupiec_pof(250, 5, 0.01)
assert abs(k["lr"] - lr_hand) < 1e-12 and abs(lr_hand - 1.9568) < 1e-3, (k["lr"], lr_hand)
assert abs(k["p"] - chi2_sf(lr_hand, 1)) < 1e-15 and k["status"] == PASS            # p≈0.127: 초과 2배인데도 통과 — 검정력이 약하다
# x=10, n=250: LR≈ 15.6? 직접 계산해서 기각인지
k10 = kupiec_pof(250, 10, 0.01)
assert k10["status"] == FAIL and k10["p"] < 0.01, k10
# 기대치와 정확히 같으면 LR=0 (x/n == p)  : n=100, x=1, p=0.01
k0 = kupiec_pof(100, 1, 0.01)
assert abs(k0["lr"]) < 1e-12 and k0["status"] == PASS
# 초과 0회도 정의된다(과대추정 쪽): n=1000, x=0, p=0.05 -> LR = -2*1000*ln(.95) = 102.6 -> 기각
kz = kupiec_pof(1000, 0, 0.05)
assert abs(kz["lr"] + 2 * 1000 * math.log(0.95)) < 1e-9 and kz["status"] == FAIL
print(f"[PASS] 1) Kupiec: (250,5,.01) LR={lr_hand:.4f} p={k['p']:.3f} 통과(검정력 약함 확인), (250,10) 기각, "
      f"정확 일치 LR=0, 초과 0회 과대추정 기각")

# --- 2) Christoffersen 손계산 ----------------------------------------------------------------
# 초과가 뭉친 경우: hits = 0,0,0,0,1,1,1,0,0,0  (전이: 00=5, 01=1, 10=1, 11=2)
# π01 = 1/6, π11 = 2/3, π = 3/9
h = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
c = christoffersen_independence(h)
assert (c["n00"], c["n01"], c["n10"], c["n11"]) == (5, 1, 1, 2), c
pi01, pi11, pi = 1 / 6, 2 / 3, 3 / 9
ll0 = 6 * math.log(1 - pi) + 3 * math.log(pi)
ll1 = 5 * math.log(1 - pi01) + 1 * math.log(pi01) + 1 * math.log(1 - pi11) + 2 * math.log(pi11)
assert abs(c["lr"] - (-2 * (ll0 - ll1))) < 1e-12, (c["lr"], -2 * (ll0 - ll1))
# 초과가 규칙적으로 흩어진 경우: 0,1,0,0,1,0,0,1,0,0 -> 전이 (3,3,3,0), π01=.5, π11=0, π=1/3
# 주의: 이 검정은 '뭉침'만 잡는 게 아니라 '너무 규칙적(초과 뒤엔 절대 초과 없음)'도 독립 위반으로 본다 —
# 그래서 이 작은 표본에선 흩어진 쪽 LR(3.14)이 뭉친 쪽(2.23)보다 크다. 크기 비교가 아니라 수식 일치를 확인한다.
c2 = christoffersen_independence(np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0]))
assert (c2["n00"], c2["n01"], c2["n10"], c2["n11"]) == (3, 3, 3, 0)
ll0_2 = 6 * math.log(2 / 3) + 3 * math.log(1 / 3)
ll1_2 = 6 * math.log(0.5)                       # 3 ln(1-.5) + 3 ln(.5) + 3 ln(1-0) + 0·ln(0)
assert abs(c2["lr"] - (-2 * (ll0_2 - ll1_2))) < 1e-12, c2
# 초과 없음 -> 판정불가
c3 = christoffersen_independence(np.zeros(50, dtype=int))
assert c3["status"] == UNKNOWN
# 초과가 마지막 날 한 번뿐 -> 초과 뒤 전이가 없어(n10+n11=0) 판정불가
c4 = christoffersen_independence(np.r_[np.zeros(20, dtype=int), 1])
assert c4["status"] == UNKNOWN and c4["n01"] == 1
print(f"[PASS] 2) Christoffersen: 전이표 (5,1,1,2) LR={c['lr']:.4f}, (3,3,3,0) LR={c2['lr']:.4f} 둘 다 손계산 일치, 초과 없음/끝에만 있음=판정불가")

# --- 3) 바젤 신호등 ------------------------------------------------------------------------
assert basel_traffic_light(250, 4, 0.99)["zone"] == "녹색" and basel_traffic_light(250, 5, 0.99)["zone"] == "황색"
assert basel_traffic_light(250, 9, 0.99)["zone"] == "황색" and basel_traffic_light(250, 10, 0.99)["status"] == FAIL
assert basel_traffic_light(500, 4, 0.99)["zone"] == "해당없음" and basel_traffic_light(250, 4, 0.95)["zone"] == "해당없음"
print("[PASS] 3) 바젤: 4 녹색 / 5·9 황색 / 10 적색, n≠250 또는 α≠.99는 해당없음")

# --- 4) var_backtest 통합: 손실·VaR 비교, 날짜 정렬, 종합 판정 ----------------------------------
ix = pd.bdate_range("2023-01-01", periods=250)
ret = pd.Series(0.0, index=ix)
var = pd.Series(0.02, index=ix)
ret.iloc[[10, 50, 90, 130, 170]] = -0.03               # 5회 초과, 흩어짐
res = var_backtest(ret, var, alpha=0.99)
assert res.n == 250 and res.n_exceed == 5 and abs(res.expected_exceed - 2.5) < 1e-12
assert res.kupiec["status"] == PASS and res.basel["zone"] == "황색" and res.independence["status"] == PASS
assert res.verdict == FIT and len(res.exceed_dates) == 5 and res.exceed_dates[0] == ix[10]
# 손실이 VaR와 정확히 같으면 초과가 아님(> 비교)
ret_eq = ret.copy(); ret_eq.iloc[200] = -0.02
assert var_backtest(ret_eq, var, 0.99).n_exceed == 5
# 뭉친 초과 10회 -> Kupiec 기각 + 바젤 적색 -> 부적합
ret_c = pd.Series(0.0, index=ix); ret_c.iloc[100:110] = -0.03
res_c = var_backtest(ret_c, var, 0.99)
assert res_c.n_exceed == 10 and res_c.verdict == UNFIT and res_c.basel["zone"] == "적색"
assert res_c.independence["status"] == FAIL              # 10연속 초과: π11=9/9 vs π01=1/240 -> 독립성 기각
# NaN은 비교에서 제외
var_nan = var.copy(); var_nan.iloc[:50] = np.nan
assert var_backtest(ret, var_nan, 0.99).n == 200
# 초과 0회 250일 -> Kupiec LR = -2*250*ln(.99) = 5.025 -> p=.025 기각(과대추정) ... 독립성은 판정불가지만 기각이 우선 -> 부적합
res0 = var_backtest(pd.Series(0.0, index=ix), var, 0.99)
assert res0.kupiec["status"] == FAIL and res0.verdict == UNFIT, (res0.kupiec, res0.verdict)
rep = format_var_backtest(res, "fhs")
for tok in ("VaR 백테스트", "Kupiec", "Christoffersen", "바젤", "판정: 적합"):
    assert tok in rep, tok
print("[PASS] 4) 통합: 흩어진 5회=적합(황색), 뭉친 10회=부적합(독립성·바젤 기각), 등호는 초과 아님, NaN 제외, 0회는 과대추정 기각")

# --- 5) 모델 자체 검증: 정규분포 데이터에서 파라메트릭 VaR는 적합해야 하고, 변동성 급변 데이터에서 롤링 역사적은 떨어지기 쉽다
rng = np.random.default_rng(42)
n = 3000
r_norm = pd.Series(rng.normal(0, 0.01, n), index=pd.bdate_range("2010-01-01", periods=n))
from risk.var import parametric_var
pv = parametric_var(r_norm, 0.99, window=250, use_ewma=False)        # 롤링 250일 σ
rb = var_backtest(r_norm, pv["var"], 0.99)
assert rb.verdict == FIT, rb.summary_row()
# 같은 정규 데이터인데 EWMA(λ=.94, 유효 창 ≈ 33일) σ로 만든 파라메트릭 VaR는 초과가 기대보다 많다.
# σ 추정 잡음 때문에 r/σ̂ 가 정규보다 꼬리가 두꺼워지기 때문(자유도 작은 t분포에 가깝다). 모델이 정규를
# 가정해도 σ를 짧은 창으로 추정하면 99% VaR를 과소추정한다 — 기본값으로 EWMA 파라메트릭을 쓰지 않는 이유.
pv_e = compute_var(r_norm, "parametric", 0.99, 250)
rb_e = var_backtest(r_norm, pv_e["var"], 0.99)
assert rb_e.n_exceed > rb_e.expected_exceed * 1.2, rb_e.summary_row()
# 변동성이 매 250일마다 1%↔4%로 바뀌는 데이터: 롤링 역사적(250)은 전환 직후 초과가 뭉친다 -> 독립성 기각 기대
sig = np.where((np.arange(n) // 250) % 2 == 0, 0.01, 0.04)
r_sw = pd.Series(rng.normal(0, 1, n) * sig, index=r_norm.index)
hv = compute_var(r_sw, "historical", 0.99, 250)
fh = compute_var(r_sw, "fhs", 0.99, 250)
rb_h = var_backtest(r_sw, hv["var"], 0.99)
rb_f = var_backtest(r_sw, fh["var"], 0.99)
assert rb_h.independence["status"] == FAIL, rb_h.summary_row()
assert rb_f.independence["p"] > rb_h.independence["p"], (rb_f.summary_row(), rb_h.summary_row())
print(f"[PASS] 5) 정규 데이터: 롤링σ 파라메트릭 적합(초과 {rb.n_exceed}/{rb.n}), EWMAσ는 초과 {rb_e.n_exceed}회(기대 {rb_e.expected_exceed:.0f}) 과소추정; 변동성 전환 데이터에서 롤링 역사적 독립성 기각"
      f"(p={rb_h.independence['p']:.4f}), FHS는 덜 뭉침(p={rb_f.independence['p']:.3f})")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
