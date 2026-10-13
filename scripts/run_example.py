import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""예시 실행: 60세 부부, 금융자산 5억, 생활비 월 300만원."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from retire_sim.config import Person, Household, SimConfig, CareMarkov
from retire_sim.economy_v2 import EconomyV2
from retire_sim import engine_tax, mortality

hh = Household(
    members=[Person(age=60, sex="M", nps_monthly=110, nps_start_age=65,
                    private_pension_annual=600, private_pension_start=60),
             Person(age=58, sex="F", nps_monthly=50, nps_start_age=65)],
    liquid_assets=50_000, stock_weight=0.4, annual_spending=3600)
cfg = SimConfig(care=CareMarkov())

q = mortality.default_qx()
print(f"[사망률 체크] 60세 남 기대여명 {mortality.life_expectancy(q['M'], 60):.1f}년, "
      f"58세 여 {mortality.life_expectancy(q['F'], 58):.1f}년")

res = engine_tax.run(hh, cfg, e=EconomyV2.enhanced())
for k, v in engine_tax.summarize(res).items():
    print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

# 시나리오 비교: 국민연금 연기수령
for start in (60, 65, 70):
    for m in hh.members: m.nps_start_age = start
    p = engine_tax.summarize(engine_tax.run(hh, cfg, e=EconomyV2.enhanced()))["고갈확률"]
    print(f"국민연금 {start}세 개시 → 고갈확률 {p:.1%}")
for m in hh.members: m.nps_start_age = 65

plt.style.use("dark_background")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
# 생존 가구 기준 실질자산 분위수 (그림 전용이라 라이브러리 대신 여기서 계산)
W, alive = res["W_real"], res["hh_alive"]
ages = np.arange(W.shape[1]) + res["youngest_age"]
F = np.array([np.percentile(W[alive[:, t], t], [5, 25, 50, 75, 95]) if alive[:, t].sum() > 100
              else [np.nan] * 5 for t in range(W.shape[1])])
ax[0].fill_between(ages, F[:, 0], F[:, 4], alpha=.2, color="#4fc3f7", label="5–95%")
ax[0].fill_between(ages, F[:, 1], F[:, 3], alpha=.4, color="#4fc3f7", label="25–75%")
ax[0].plot(ages, F[:, 2], color="#4fc3f7", label="median")
ax[0].set(title="Real wealth (surviving households, 10k KRW)", xlabel="Age (youngest)")
ax[0].legend()
d = res["depleted_at"]
a = ages; c = np.array([((d >= 0) & (d <= t)).mean() for t in range(W.shape[1])])
ax[1].plot(a, c * 100, color="#ff8a65")
ax[1].set(title="Cumulative depletion probability (%)", xlabel="Age (youngest)")
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "..", "docs", "images", "example_output.png"), dpi=130)
print("saved example_output.png")
