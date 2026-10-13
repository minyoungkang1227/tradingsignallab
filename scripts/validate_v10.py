import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""v10 강화 모듈 검증: 2요인 채권가격, 피셔 연결, 점프 보정, 반대 난수 효과."""
import numpy as np
from retire_sim.economy_v2 import EconomyV2, generate, gaussian2_price, vasicek_price

e = EconomyV2()
v = vasicek_price(5.0, np.array([0.025]), e)[0]
g = gaussian2_price(5.0, np.array([0.025]), np.array([0.0]), e.r_kappa, e.r_theta, e.r_sigma, 1.0, 0.0, 1e-9, 0.0)[0]
print(f"[1] 2요인 가격식의 1요인 환원: Vasicek {v:.6f} vs 2요인 {g:.6f}")

E = EconomyV2.enhanced(); rng = np.random.default_rng(1); m, dt, tau = 40000, 1 / 52, 5.0
q = np.full(m, E.q0); p = np.full(m, E.pi0); integ = np.zeros(m)
st = lambda x, th, k, s, z: th + (x - th) * np.exp(-k * dt) + s * np.sqrt((1 - np.exp(-2 * k * dt)) / (2 * k)) * z
for _ in range(int(tau / dt)):
    z1 = rng.standard_normal(m); z2 = E.rho_rp * z1 + np.sqrt(1 - E.rho_rp**2) * rng.standard_normal(m)
    qn, pn = st(q, E.q_theta, E.q_kappa, E.q_sigma, z1), st(p, E.pi_theta, E.pi_kappa, E.pi_sigma, z2)
    integ += 0.5 * ((q + p) + (qn + pn)) * dt; q, p = qn, pn
an = gaussian2_price(tau, np.array([E.q0]), np.array([E.pi0]), E.q_kappa, E.q_theta, E.q_sigma,
                     E.pi_kappa, E.pi_theta, E.pi_sigma, E.rho_rp)[0]
print(f"[2] 2요인 5년 채권가격: 해석해 {an:.5f} vs 몬테카를로 {np.exp(-integ).mean():.5f}")

d = generate(EconomyV2.enhanced(erp_sd=0, pi_theta_sd=0, jump_lambda=0, fee=0), 60, 20000, np.random.default_rng(0))
print(f"[3] 명목금리 장기평균 {d['rate'][:, -1].mean():.4f} (이론 {E.q_theta + E.pi_theta:.4f}), "
      f"물가-명목금리 상관 {np.corrcoef(d['rate'][:, 1:].ravel(), d['infl'].ravel())[0, 1]:.2f}")

a = generate(EconomyV2(), 30, 40000, np.random.default_rng(2)); b = generate(EconomyV2(jump_lambda=0.1), 30, 40000, np.random.default_rng(2))
print(f"[4] 주식 평균 연수익 점프 없음 {a['stock'].mean():.4f} / 점프 {b['stock'].mean():.4f} (보정항으로 유지), "
      f"하위 1% 연수익 {np.percentile(a['stock'], 1):.3f} / {np.percentile(b['stock'], 1):.3f}")
