"""해석해 대조: 변동성·세금·사망·간병을 모두 끈 결정론적 극한에서
   W_{t+1} = (W_t − 지출_t) × (1+r) 와 정확히 일치하는지 확인."""
import numpy as np
from retire_sim.config import Person, Household, SimConfig, CareMarkov, NPSRules
from retire_sim.economy_v2 import EconomyV2
from retire_sim.tax import TaxConfig, HouseConfig
from retire_sim import engine_tax

R = 0.03                                  # 결정론적 명목 로그수익률 (실효 = e^R − 1)
eco = EconomyV2(r0=R, r_theta=R, r_sigma=0.0, q0=R, q_theta=R, q_sigma=0.0,
                pi0=0.0, pi_theta=0.0, pi_sigma=0.0, erp=0.0, s_sigma=0.0,
                fee=0.0, jump_lambda=0.0, erp_sd=0.0, pi_theta_sd=0.0, q_theta_sd=0.0,
                antithetic=False,term_premium=0.0,
                fisher=True, house_real_sd=0.0, house_sigma=0.0)
qx = {"M": np.zeros(111), "F": np.zeros(111)}      # 아무도 죽지 않음
qx["M"][100] = 1.0; qx["F"][100] = 1.0
SPEND = 3000.0
hh = Household(members=[Person(60, "M")], liquid_assets=50_000, stock_weight=1.0,
               annual_spending=SPEND, essential=SPEND, lifestyle=0.0, legacy_target=0.0,
               survivor_spending_ratio=1.0, floor_method="fixed95")
cfg = SimConfig(n_paths=64, seed=1, care=CareMarkov(enabled=False), mortality_improvement=False,
                longevity_sigma_level=0.0, longevity_sigma_trend=0.0, widow_excess=(0.0, 0.0),
                nps=NPSRules(basic_on=False, survivor_on=False))
res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), house=HouseConfig(), qx_table=qx)

W = res["W_real"][0]                               # 물가 0 → 실질=명목
w, rows = float(hh.liquid_assets), []
for t in range(12):
    w = (w - SPEND) * np.exp(R)
    rows.append((t + 1, W[t + 1], w, W[t + 1] - w))
print(f"{'연차':>4} {'모형':>12} {'닫힌식':>12} {'차이':>10}")
for t, a, b, d in rows:
    print(f"{t:>4} {a:12,.1f} {b:12,.1f} {d:10.4f}")
print("최대 오차:", max(abs(d) for *_, d in rows))
