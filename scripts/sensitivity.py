import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, copy, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from retire_sim.config import Person, Household, SimConfig, CareMarkov, CareShock
from retire_sim.economy_v2 import EconomyV2
from retire_sim import engine_tax, mortality

def hh(spend=3600, w=0.4):
    return Household(members=[Person(60,"M",nps_monthly=110,private_pension_annual=600,private_pension_start=60),
                              Person(58,"F",nps_monthly=50)], liquid_assets=50_000, stock_weight=w, annual_spending=spend)
q0 = mortality.default_qx()
def run(h=None, e=None, cfg=None, q=None):
    r = engine_tax.run(h or hh(), cfg or SimConfig(care=CareMarkov()), qx_table=q or q0, e=e or EconomyV2.enhanced())
    return engine_tax.summarize(r)["고갈확률"]*100
base = run()
qlow = {k: v*0.8 for k,v in q0.items()}; qhigh = {k: np.minimum(v*1.2,1) for k,v in q0.items()}
for d in (qlow,qhigh): d["M"][-1]=d["F"][-1]=1
tests = [
 ("주식 위험프리미엄 3%↔5%", run(e=EconomyV2(erp=0.03)), run(e=EconomyV2(erp=0.05))),
 ("주식 변동성 15%↔21%", run(e=EconomyV2(s_sigma=0.15)), run(e=EconomyV2(s_sigma=0.21))),
 ("장기 금리 2%↔4%", run(e=EconomyV2(r_theta=0.02)), run(e=EconomyV2(r_theta=0.04))),
 ("장기 물가 1.5%↔2.5%", run(e=EconomyV2(pi_theta=0.015)), run(e=EconomyV2(pi_theta=0.025))),
 ("생활비 −10%↔+10%", run(h=hh(3240)), run(h=hh(3960))),
 ("주식 비중 20%↔60%", run(h=hh(w=0.2)), run(h=hh(w=0.6))),
 ("사망률 ×1.2↔×0.8 (수명↓↑)", run(q=qhigh), run(q=qlow)),
 ("간병 발생률 1.5%↔6%", run(cfg=SimConfig(care=CareShock(lam=0.015))), run(cfg=SimConfig(care=CareShock(lam=0.06)))),
]
print(f"기준 고갈확률 {base:.1f}%")
for n,a,b in tests: print(f"{n}: {a:.1f}% / {b:.1f}%  (폭 {abs(b-a):.1f}%p)")
tests.sort(key=lambda t: abs(t[2]-t[1]))
plt.style.use("dark_background"); fig, ax = plt.subplots(figsize=(9,5))
for i,(n,a,b) in enumerate(tests):
    ax.barh(i, a-base, left=base, color="#4fc3f7"); ax.barh(i, b-base, left=base, color="#ff8a65")
ax.axvline(base, color="w", lw=.8)
ax.set_yticks(range(len(tests))); en={"주식 위험프리미엄 3%↔5%":"Equity premium 3%/5%","주식 변동성 15%↔21%":"Equity vol 15%/21%","장기 금리 2%↔4%":"Long-run rate 2%/4%","장기 물가 1.5%↔2.5%":"Long-run inflation 1.5%/2.5%","생활비 −10%↔+10%":"Spending -10%/+10%","주식 비중 20%↔60%":"Equity weight 20%/60%","사망률 ×1.2↔×0.8 (수명↓↑)":"Mortality x1.2/x0.8","간병 발생률 1.5%↔6%":"Care incidence 1.5%/6%"}
ax.set_yticklabels([en[t[0]] for t in tests])
ax.set(title=f"Sensitivity of depletion probability (base {base:.1f}%)", xlabel="Depletion probability (%)")
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "..", "docs", "images", "sensitivity.png"), dpi=120)
print("order(bottom->top):", [t[0] for t in tests])
