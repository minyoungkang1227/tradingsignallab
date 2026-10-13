import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""v11 검증: ① 적분 금리 결합공분산, ③ 사망률 보존, ⑤ OU 최우추정 복원."""
import numpy as np
from retire_sim.economy_v2 import integrated_cov
from retire_sim.config import CareMarkov
from retire_sim import mortality
from retire_sim.calibrate import ou_mle

k1, s1, k2, s2, c12, c1s, c2s = 0.2, 0.008, 0.4, 0.01, 0.3, -0.2, -0.1
M = integrated_cov(k1, s1, k2, s2, c12, c1s, c2s)
rng = np.random.default_rng(0); m, steps = 200000, 400; dt = 1 / steps
L = np.linalg.cholesky(np.array([[1, c12, c1s], [c12, 1, c2s], [c1s, c2s, 1]]))
x1 = np.zeros(m); x2 = np.zeros(m); I1 = np.zeros(m); I2 = np.zeros(m); W = np.zeros(m)
for _ in range(steps):
    dW = (rng.standard_normal((m, 3)) @ L.T) * np.sqrt(dt)
    I1 += x1 * dt; I2 += x2 * dt
    x1 += -k1 * x1 * dt + s1 * dW[:, 0]; x2 += -k2 * x2 * dt + s2 * dW[:, 1]; W += dW[:, 2]
emp = np.cov(np.vstack([x1, I1, x2, I2, W]))
cA = M / np.sqrt(np.outer(np.diag(M), np.diag(M))); cE = emp / np.sqrt(np.outer(np.diag(emp), np.diag(emp)))
print(f"[①] 5×5 결합공분산: 해석해 vs 세밀 시뮬 상관 최대 차이 {np.abs(cA - cE).max():.4f}, "
      f"분산 상대오차 최대 {np.max(np.abs(np.diag(M) - np.diag(emp)) / np.diag(M)):.4f}")

q = mortality.default_qx(); n = 100000
a0 = mortality.simulate_alive(q["M"], 60, 50, n, np.random.default_rng(0))
a1, ic = mortality.simulate_life(q["M"], 60, 50, n, np.random.default_rng(0), CareMarkov())
print(f"[③] 60세 남 기대여명: 생명표 {a0.sum(1).mean() - .5:.2f}년 / 간병 마르코프 {a1.sum(1).mean() - .5:.2f}년")

rng = np.random.default_rng(3); k, th, s, dt = 0.4, 0.02, 0.01, 1 / 12; est = []
for _ in range(300):
    x = [0.03]
    for _ in range(360):
        e = np.exp(-k * dt); x.append(th + (x[-1] - th) * e + s * np.sqrt((1 - e * e) / (2 * k)) * rng.standard_normal())
    est.append(ou_mle(np.array(x), dt))
K = np.array([r["kappa"] for r in est]); TH = np.array([r["theta"] for r in est]); S = np.array([r["sigma"] for r in est])
print(f"[⑤] OU MLE 300회(30년 월별): 참값 κ={k}, θ={th}, σ={s} → 평균 κ={K.mean():.3f}, θ={TH.mean():.4f}, σ={S.mean():.4f} "
      f"(κ는 소표본 상향 편의)")
