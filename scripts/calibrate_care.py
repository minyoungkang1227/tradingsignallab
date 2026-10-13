import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""간병 발생률·진행률 보정 (v17).

목표 연령별 유병률(장기요양 인정자 / 인구):
  인정자 1,235,045명(2025, 건강보험공단) × 연령 분포(2022 장기요양실태조사: 65~69 4.5%, 70~74 7.9%,
  75~79 14.3%, 80~84 26.2%, 85~89 26.7%, 90+ 17.5%) ÷ 연령별 인구(2025 고령자 통계: 65~69 약 367만,
  70~74 약 253만, 75세 이상 430만 — 75세 이상 세부는 생존 구조로 나눈 추정치)
  (60~64세는 65세 미만 인정자 2.8%를 고려한 추정 0.3%)
  → 65~69 1.5%, 70~74 3.9%, 75~79 9.9%, 80~84 24.0%, 85~89 42.3%, 90+ 55.4%
목표 중증 비율: 1~2등급 12.7% (2025)
방법: 60세 남녀 코호트를 시뮬레이션해 연령대별 유병률과 중증 비율이 목표에 맞도록 발생률·진행률을 반복 조정.
"""
import numpy as np
from dataclasses import replace
from retire_sim.config import CareMarkov
from retire_sim import mortality

groups = [(60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 100)]
target = [0.003, 0.015, 0.039, 0.099, 0.240, 0.423, 0.554]
target_sev = 0.127
q = mortality.default_qx()

def simulate(c, n=60000):
    prev = np.zeros(len(groups)); sev_num = sev_den = 0; dur = []
    for sex in ("M", "F"):
        al, ic, sv = mortality.simulate_life(q[sex], 60, 45, n, np.random.default_rng(1 if sex == "M" else 2), c, return_severe=True)
        for g, (lo, hi) in enumerate(groups):
            cols = slice(lo - 60, hi - 60)
            prev[g] += ic[:, cols].sum() / max(al[:, cols].sum(), 1) / 2
        sev_num += sv[:, 5:].sum(); sev_den += ic[:, 5:].sum()
        ever = ic.any(1); dur.append(ic.sum(1)[ever].mean())
    return prev, sev_num / sev_den, float(np.mean(dur))

c = CareMarkov()
inc = [v for _, v in c.incidence]
for it in range(25):
    c = replace(c, incidence=tuple(zip([65, 70, 75, 80, 85, 90, 999], inc)))
    prev, sev, dur = simulate(c)
    inc = [min(max(v * (tg / max(pv, 1e-6)) ** 0.7, 1e-4), 0.6) for v, tg, pv in zip(inc, target, prev)]
    c = replace(c, progress=float(np.clip(c.progress * (target_sev / sev) ** 0.8, 0.005, 0.5)))
print("발생률:", [round(v, 4) for v in inc])
print("진행률:", round(c.progress, 4))
print("유병률 모델 vs 목표:", [(round(a, 3), b) for a, b in zip(prev, target)])
print(f"중증 비율 {sev:.3f} (목표 {target_sev}), 간병 경험자 평균 기간 {dur:.1f}년")
