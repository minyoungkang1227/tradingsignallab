import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from retire_sim.economy_v2 import EconomyV2, generate, vasicek_price, _ou_step
rng = np.random.default_rng(0)
e = EconomyV2()
n, T = 20000, 60
eco = generate(e, T, n, rng)

print("[검증1] OU 정확 이산화: 장기 평균·표준편차 이론값 vs 시뮬")
r_end = eco["rate"][:, -1]; pi_end = eco["infl"][:, -1]
print(f"  금리 평균 {r_end.mean():.4f} (이론 {e.r_theta})  표준편차 {r_end.std():.4f} (이론 {e.r_sigma/np.sqrt(2*e.r_kappa):.4f})")
print(f"  물가 평균 {pi_end.mean():.4f} (이론 {e.pi_theta})  표준편차 {pi_end.std():.4f} (이론 {e.pi_sigma/np.sqrt(2*e.pi_kappa):.4f})")

print("[검증2] Vasicek 채권가격 해석해 vs 몬테카를로 E[exp(-∫r dt)]")
dt, tau, m = 1/52, 5.0, 40000
r = np.full(m, e.r0); integ = np.zeros(m)
for _ in range(int(tau/dt)):
    ek = np.exp(-e.r_kappa*dt)
    r_new = e.r_theta + (r-e.r_theta)*ek + e.r_sigma*np.sqrt((1-ek**2)/(2*e.r_kappa))*rng.standard_normal(m)
    integ += 0.5*(r+r_new)*dt; r = r_new
print(f"  해석해 {vasicek_price(tau, e.r0, e):.5f}  MC {np.exp(-integ).mean():.5f}")

print("[검증3] 자산 수익률 통계 (연)")
for k in ["stock","bond"]:
    x = eco[k]; print(f"  {k}: 평균 {x.mean():.4f}  표준편차 {x.std():.4f}")
print(f"  음의 금리 발생 비율: {(eco['rate']<0).mean():.3%}")
print(f"  채권-주식 수익 상관: {np.corrcoef(eco['bond'].ravel(), eco['stock'].ravel())[0,1]:.3f}")
