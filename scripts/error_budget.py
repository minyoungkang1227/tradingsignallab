import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""오차 예산: 결과의 불확실성이 '난수'에서 오는지 '가정'에서 오는지 나눠 본다.

실행: PYTHONPATH=. python scripts/error_budget.py
결론(2026-10 기준): 몬테카를로 오차는 ±0.6%p, 주식 위험프리미엄 1σ만으로 ±7%p.
→ 계산 횟수를 늘려 얻을 것은 거의 없고, 가정의 폭을 함께 보여주는 것이 맞다.
"""
import numpy as np
from dataclasses import replace
from retire_sim.config import Person, Household, SimConfig, CareMarkov
from retire_sim.economy_v2 import EconomyV2
from retire_sim.tax import TaxConfig
from retire_sim import engine_tax

ECO = EconomyV2.enhanced(); TC = TaxConfig(overseas_share=0.2)
HH = Household(members=[Person(60, "M", nps_monthly=110, private_pension_annual=600, private_pension_start=60),
                        Person(58, "F", nps_monthly=50)],
               liquid_assets=50_000, stock_weight=0.4, annual_spending=3600,
               essential=3000, lifestyle=600, legacy_target=10_000)
KEY = "기본생활 유지 확률"


def P(eco=None, n=10_000, seed=42):
    r = engine_tax.run(HH, SimConfig(n_paths=n, seed=seed, care=CareMarkov()), e=eco or ECO, tc=TC)
    return engine_tax.summarize(r)[KEY]


print("① 몬테카를로 오차 — 모형은 그대로 두고 난수만 바꿨을 때")
for n in (2_500, 10_000, 40_000):
    v = np.array([P(n=n, seed=s) for s in range(6)])
    print(f"   경로 {n:6,d}개: 평균 {v.mean()*100:5.1f}%, 95% 오차 ±{1.96*v.std(ddof=1)*100:.2f}%p")

print("\n② 가정의 불확실성 — 추정치를 ±1 표준오차만큼 움직였을 때")
base = P()
for lab, kw, sd in [("주식 위험프리미엄", "erp", 0.014), ("장기 실질금리", "q_theta", 0.0038), ("장기 물가", "pi_theta", 0.0052)]:
    lo, hi = P(replace(ECO, **{kw: getattr(ECO, kw) - sd})), P(replace(ECO, **{kw: getattr(ECO, kw) + sd}))
    print(f"   {lab:10s} ±1σ: {lo*100:5.1f}% ~ {hi*100:5.1f}%  (폭 {abs(hi-lo)*100:4.1f}%p)")
print(f"\n기준값 {base*100:.1f}%. 난수를 아무리 늘려도 ②는 줄지 않는다 — 결과는 범위로 읽어야 한다.")
