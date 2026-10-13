import sys, numpy as np
from dataclasses import replace
sys.path.insert(0,'tests')
import test_engine as T
from retire_sim import engine_tax, metrics
from retire_sim.config import SimConfig, CareMarkov, NPSRules
from retire_sim.tax import TaxConfig

ECO = T.ECO; TC = TaxConfig(overseas_share=0.2)
def run(hh, eco=None, cfg=None):
    return engine_tax.run(hh, cfg or SimConfig(care=CareMarkov()), e=eco or ECO, tc=TC)

ABL = [
 ("폭락 점프",            dict(eco=dict(jump_lambda=0.0))),
 ("파라미터 불확실성",     dict(eco=dict(erp_sd=0.0, pi_theta_sd=0.0, q_theta_sd=0.0))),
 ("기간프리미엄",          dict(eco=dict(term_premium=0.0))),
 ("운용보수 0.5%",        dict(eco=dict(fee=0.0))),
 ("금리-주식 상관",        dict(eco=dict(rho_rs=0.0))),
 ("사망률 개선(코호트)",   dict(cfg=dict(mortality_improvement=False))),
 ("장수 불확실성(v30)",    dict(cfg=dict(longevity_sigma_level=0.0, longevity_sigma_trend=0.0))),
 ("사별 효과(v30)",       dict(cfg=dict(widow_excess=(0.0,0.0)))),
 ("유족연금(v28)",         dict(nps=dict(survivor_on=False))),
 ("기초연금(v28)",         dict(nps=dict(basic_on=False))),
 ("간병(마르코프)",        dict(care_off=True)),
]
for name, fn in [("60대 부부", T.couple_60), ("30대 부부", T.couple_30)]:
    hh = fn(); base = run(hh); b = engine_tax.summarize(base)["기본생활 유지 확률"]
    print(f"\n== {name}  기준 {b*100:.1f}%")
    rows=[]
    for lab, spec in ABL:
        eco = replace(ECO, **spec["eco"]) if "eco" in spec else ECO
        ckw = dict(spec.get("cfg", {}))
        if "nps" in spec: ckw["nps"] = NPSRules(**spec["nps"])
        cfg = SimConfig(care=None if spec.get("care_off") else CareMarkov(), **ckw)
        if spec.get("care_off"):
            from retire_sim.config import CareShock
            cfg = SimConfig(care=CareShock(), **ckw)
        r = run(hh, eco, cfg); s = engine_tax.summarize(r)
        d = metrics.compare(r, base)
        rows.append((abs(d["차이"]), lab, s["기본생활 유지 확률"]-b, d["±"], s["CTE90 부족액(실질)"]))
    for _, lab, dd, ci, cte in sorted(rows, reverse=True):
        mark = "●" if abs(dd) > ci else "○"
        print(f"  {mark} {lab:20s} {dd*100:+6.2f}%p (±{ci*100:.2f})  CTE90 {cte:,.0f}")
