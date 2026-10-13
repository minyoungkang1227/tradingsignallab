"""경제 시나리오 v2(+v10 강화): 금리·물가·주식·채권, 정확 이산화.

기본 모드(fisher=False, v2와 동일)
- 명목 단기금리 r (Vasicek):  dr = k_r(th_r - r)dt + s_r dW_r
- 물가상승률 pi (OU):           dpi = k_p(th_p - pi)dt + s_p dW_p

피셔 모드(fisher=True, v10) — 금리와 물가를 구조적으로 연결
- 실질금리 q (OU):  dq = k_q(th_q - q)dt + s_q dW_q
- 명목금리 r = q + pi   → 물가가 오르면 명목금리가 자동으로 따라 오름
- 채권 가격: 2요인 가우시안 해석해 P = exp(-E[∫r] + Var[∫r]/2)

공통
- 주식 로그수익 = r + ERP - s_s²/2 + s_s·Z_s  (+ 선택: Merton 점프, 보정항으로 평균 유지)
- 채권: 만기 D년 무이표채 롤링 펀드 (위험프리미엄 0 가정)
- 운용보수 fee: 주식·채권 수익률에서 매년 차감
- 파라미터 불확실성: 경로마다 ERP·장기 물가를 한 번 뽑아 그 경로 전체에 사용
- 반대 난수(antithetic): 경로를 절반씩 +Z / -Z 로 짝지어 표본오차 감소
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class EconomyV2:
    r0: float = 0.025
    r_theta: float = 0.03
    r_kappa: float = 0.15
    r_sigma: float = 0.010
    pi0: float = 0.02
    pi_theta: float = 0.02
    pi_kappa: float = 0.40
    pi_sigma: float = 0.010
    erp: float = 0.04          # 주식 위험프리미엄
    s_sigma: float = 0.18
    bond_duration: float = 5.0
    rho_rp: float = 0.5        # (기본) 금리-물가 / (피셔) 실질금리-물가 충격 상관
    rho_rs: float = -0.2       # 금리-주식 충격 상관
    rho_ps: float = -0.1       # 물가-주식 충격 상관
    stock_shocks: dict | None = None  # 쇼크 테스트: {연차: 주식수익률}, 예 {0: -0.40}
    infl_shocks: dict | None = None   # 쇼크 테스트: {연차: 물가상승률}
    # ── v10 강화 옵션 (기본값은 v2와 동일하게 꺼져 있음) ──
    fisher: bool = False       # 명목금리 = 실질금리 + 물가
    q0: float = 0.005          # 실질금리 초기값 (피셔 모드)
    q_theta: float = 0.01      # 실질금리 장기평균 → 명목 장기평균 = q_theta + pi_theta
    q_kappa: float = 0.20
    q_sigma: float = 0.008
    fee: float = 0.0           # 연 운용보수 (예: 0.005 = 0.5%)
    erp_sd: float = 0.0        # ERP 추정 불확실성 (경로별 추출 표준편차)
    pi_theta_sd: float = 0.0   # 장기 물가 추정 불확실성
    q_theta_sd: float = 0.0    # 장기 실질금리 추정 불확실성 (피셔 모드)
    jump_lambda: float = 0.0   # 주가 점프 연평균 횟수 (Merton)
    jump_mu: float = -0.15     # 점프 크기 로그평균
    jump_sigma: float = 0.10   # 점프 크기 로그표준편차
    antithetic: bool = False   # 반대 난수 사용
    # ── v11 수리적 보완 ──
    term_premium: float = 0.0      # 위험의 시장가격: 가격용(Q) 장기평균 = θ + term_premium → 채권 기간 프리미엄
    # ── v26 집값 ──
    house_real_growth: float = 0.0   # 물가 대비 실질 상승률(연). 0이면 집값 = 물가만큼만 상승(v25까지)
    house_real_sd: float = 0.0       # 실질 상승률 추정 불확실성(경로별)
    house_sigma: float = 0.0         # 집값 로그수익 변동성(연, 개별 주택)
    rho_hs: float = 0.1              # 집값-주식 충격 상관

    @classmethod
    def enhanced(cls, calibrated: bool = True, **kw):
        """권장 설정: 피셔 연결 + 운용보수 0.5% + 파라미터 불확실성 + 폭락 점프 + 반대 난수 + 적분 금리 + 기간 프리미엄.
        calibrated=True(기본): 물가·실질금리(v14, ECOS)와 주식(v18, KOSPI) 파라미터를 실데이터 추정치로 사용.
        calibrated=False: v13 이전 임시값."""
        base = dict(fisher=True, rho_rp=0.0, fee=0.005, erp_sd=0.01, pi_theta_sd=0.005,
                    jump_lambda=0.1, antithetic=True, term_premium=0.005)
        if calibrated:
            base.update(CALIBRATED_2026_08)
            base.update(CALIBRATED_EQUITY_2026_08)
            base.update(CALIBRATED_HOUSE_2026_08)
        base.update(kw)
        return cls(**base)


# ECOS 실데이터 추정 (2000-01~2026-08 월별, 소비자물가지수 901Y009 · 국고채 3년 721Y001/5020000)
# 물가 = 전년동월비, 실질금리 = 국고채 3년 − 물가(사후). OU 정확 MLE 후 회귀속도는 AR(1) 소표본 편의 보정
# (Kendall·Marriott-Pope: b + (1+3b)/n). 장기평균의 표준오차는 경로별 파라미터 불확실성으로 사용.
CALIBRATED_2026_08 = dict(
    pi_theta=0.0247, pi_kappa=0.35, pi_sigma=0.0133, pi_theta_sd=0.0052, pi0=0.0309,
    q_theta=0.0073, q_kappa=0.56, q_sigma=0.0137, q_theta_sd=0.0038, q0=0.0070,
)


def _ou_step(x, theta, kappa, sigma, z):
    """OU 정확 이산화 (dt=1)."""
    e = np.exp(-kappa)
    sd = sigma * np.sqrt((1 - e**2) / (2 * kappa))
    return theta + (x - theta) * e + sd * z


def vasicek_price(tau, r, e: EconomyV2, theta=None):
    k, s = e.r_kappa, e.r_sigma
    th = e.r_theta if theta is None else theta
    B = (1 - np.exp(-k * tau)) / k
    A = (th - s**2 / (2 * k**2)) * (B - tau) - s**2 * B**2 / (4 * k)
    return np.exp(A - B * r)


def _B(k, tau):
    return (1 - np.exp(-k * tau)) / k


def gaussian2_price(tau, x, y, kx, thx, sx, ky, thy, sy, rho):
    """r = x + y (두 상관 OU) 일 때 무이표채 가격 exp(-E[∫r] + Var[∫r]/2)."""
    if tau <= 0:
        return np.ones_like(x)
    Bx, By = _B(kx, tau), _B(ky, tau)
    mean = thx * tau + (x - thx) * Bx + thy * tau + (y - thy) * By
    vx = sx**2 / kx**2 * (tau - 2 * Bx + _B(2 * kx, tau))
    vy = sy**2 / ky**2 * (tau - 2 * By + _B(2 * ky, tau))
    cxy = rho * sx * sy / (kx * ky) * (tau - Bx - By + _B(kx + ky, tau))
    return np.exp(-mean + 0.5 * (vx + vy + 2 * cxy))


def _E(a):
    return (1 - np.exp(-a)) / a


def _safe_chol(M):
    """반양정부호 행렬의 촐레스키. 분산이 0인 요인이 있어 수치적으로 음수가 나오면 고윳값을 잘라 쓴다."""
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        wv, V = np.linalg.eigh(M)
        return V @ np.diag(np.sqrt(np.maximum(wv, 0)))


def integrated_cov(k1, s1, k2, s2, c12, c1s, c2s):
    """1년 동안 (X1_{t+1} 잡음, ∫X1 잡음, X2_{t+1} 잡음, ∫X2 잡음, W_s(1))의 공분산 행렬.
    OU 해 X_{t+1} = θ + (X_t-θ)e^{-κ} + σ∫e^{-κ(1-s)}dW,  ∫X = θ + (X_t-θ)E(κ) + (σ/κ)∫(1-e^{-κ(1-s)})dW
    에 이토 등거리를 적용해 계산."""
    K, S = [k1, k2], [s1, s2]
    C = np.array([[1, c12], [c12, 1]]); Cs = [c1s, c2s]
    M = np.zeros((5, 5))
    for i in range(2):
        for j in range(2):
            ki, kj, si, sj, c = K[i], K[j], S[i], S[j], C[i, j]
            M[2*i, 2*j] = c * si * sj * _E(ki + kj)                                   # (끝값_i, 끝값_j)
            M[2*i, 2*j+1] = c * si * sj / kj * (_E(ki) - _E(ki + kj))                  # (끝값_i, 적분_j)
            M[2*i+1, 2*j] = c * si * sj / ki * (_E(kj) - _E(ki + kj))                  # (적분_i, 끝값_j)
            M[2*i+1, 2*j+1] = c * si * sj / (ki * kj) * (1 - _E(ki) - _E(kj) + _E(ki + kj))
        M[2*i, 4] = M[4, 2*i] = Cs[i] * S[i] * _E(K[i])
        M[2*i+1, 4] = M[4, 2*i+1] = Cs[i] * S[i] / K[i] * (1 - _E(K[i]))
    M[4, 4] = 1.0
    return (M + M.T) / 2


def _house_index(e: EconomyV2, stock, T, n, rng, u0):
    """집값 실질 지수(오늘 대비). 명목 집값 = 이 지수 × CPI.

    로그 실질수익 = g + σ_h(ρ Z_s + √(1−ρ²) Z_h) − σ_h²/2,  Z_s는 주식의 표준화 충격.
    물가 연동은 "명목 = 실질 × CPI" 구조로 반영되므로 물가와의 상관을 따로 넣지 않는다.
    g는 경로별로 N(house_real_growth, house_real_sd²)에서 추출(추정 불확실성).
    """
    if e.house_sigma <= 0 and e.house_real_growth == 0 and e.house_real_sd == 0:
        return np.ones((n, T + 1))                       # v25까지와 동일: 집값 = 물가만큼
    g = e.house_real_growth + (e.house_real_sd * u0 if e.house_real_sd > 0 else 0.0)
    ls = np.log1p(stock)
    zs = (ls - ls.mean(0, keepdims=True)) / np.maximum(ls.std(0, keepdims=True), 1e-9)
    if e.antithetic:
        h = (n + 1) // 2
        zh_half = rng.standard_normal((h, T)); zh = np.concatenate([zh_half, -zh_half])[:n]
    else:
        zh = rng.standard_normal((n, T))
    rho = e.rho_hs
    shock = rho * zs + np.sqrt(max(1 - rho ** 2, 0)) * zh
    lr = np.log1p(g)[:, None] - 0.5 * e.house_sigma ** 2 + e.house_sigma * shock
    return np.concatenate([np.ones((n, 1)), np.cumprod(np.exp(lr), axis=1)], axis=1)


def generate(e: EconomyV2, T: int, n: int, rng) -> dict:
    """(금리 요인의 연말값, 1년 적분)을 결합정규로 정확히 뽑아 주식은 ∫r, 채권은 가격식을 사용.

    v31: 연초 금리를 쓰던 간이 생성기를 없애고 이 방식 하나로 통일했다. 두 방식의 결과 차이는
    몬테카를로 오차 안에 있었고(대표 가구 ±1%p), 주식과 채권이 같은 금리를 쓰는 이 쪽이 일관된다."""
    if e.fisher:
        k1, th1, s1, x0 = e.q_kappa, None, e.q_sigma, e.q0
    else:
        k1, th1, s1, x0 = e.r_kappa, e.r_theta, e.r_sigma, e.r0
    k2, s2 = e.pi_kappa, e.pi_sigma
    M = integrated_cov(k1, s1, k2, s2, e.rho_rp, e.rho_rs, e.rho_ps)
    d = np.diag(M).copy()                             # (v31) 지터는 요인별 분산에 비례 — 변동성 0인 요인엔 가짜 충격이 안 섞인다
    L = _safe_chol(M + 1e-12 * np.diag(d))
    need_u = e.erp_sd > 0 or e.pi_theta_sd > 0 or e.q_theta_sd > 0
    if e.antithetic:
        h = (n + 1) // 2
        Zh = rng.standard_normal((h, T, 5)); Z = np.concatenate([Zh, -Zh])[:n]
        u = rng.standard_normal((h, 3)) if need_u else np.zeros((h, 3)); U = np.concatenate([u, -u])[:n]
    else:
        Z = rng.standard_normal((n, T, 5)); U = rng.standard_normal((n, 3)) if need_u else np.zeros((n, 3))
    X = Z @ L.T
    erp = e.erp + e.erp_sd * U[:, 0]
    pth = e.pi_theta + e.pi_theta_sd * U[:, 1]
    qth = e.q_theta + e.q_theta_sd * U[:, 2]
    if th1 is None:
        th1 = qth                                         # 피셔 모드: 경로별 실질금리 장기평균
    f1 = np.empty((n, T + 1)); f1[:, 0] = x0
    pi = np.empty((n, T + 1)); pi[:, 0] = e.pi0
    r = np.empty((n, T + 1)); w2 = 1.0 if e.fisher else 0.0
    r[:, 0] = f1[:, 0] + w2 * pi[:, 0]
    stock = np.empty((n, T)); bond = np.empty((n, T)); D = e.bond_duration
    k_j = np.exp(e.jump_mu + 0.5 * e.jump_sigma**2) - 1
    ek1, ek2 = np.exp(-k1), np.exp(-k2)
    for t in range(T):
        I1 = th1 + (f1[:, t] - th1) * _E(k1) + X[:, t, 1]
        I2 = pth + (pi[:, t] - pth) * _E(k2) + X[:, t, 3]
        f1[:, t + 1] = th1 + (f1[:, t] - th1) * ek1 + X[:, t, 0]
        pi[:, t + 1] = pth + (pi[:, t] - pth) * ek2 + X[:, t, 2]
        r[:, t + 1] = f1[:, t + 1] + w2 * pi[:, t + 1]
        if e.fisher:
            g = lambda tau, qq, pp: gaussian2_price(tau, qq, pp, k1, th1 + e.term_premium, s1, k2, pth, s2, e.rho_rp)
            bond[:, t] = g(D - 1, f1[:, t + 1], pi[:, t + 1]) / g(D, f1[:, t], pi[:, t]) - 1
        else:
            th_q = e.r_theta + e.term_premium
            bond[:, t] = vasicek_price(D - 1, r[:, t + 1], e, th_q) / vasicek_price(D, r[:, t], e, th_q) - 1
        Iint = I1 + w2 * I2
        logret = Iint + erp - 0.5 * e.s_sigma**2 + e.s_sigma * X[:, t, 4]
        if e.jump_lambda > 0:
            nj = rng.poisson(e.jump_lambda, n)
            logret += nj * e.jump_mu + np.sqrt(nj) * e.jump_sigma * rng.standard_normal(n) - e.jump_lambda * k_j
        stock[:, t] = np.exp(logret) - 1
    for t, v in (e.stock_shocks or {}).items():
        if t < T: stock[:, t] = v
    if e.fee > 0:
        stock = (1 + stock) * (1 - e.fee) - 1
        bond = (1 + bond) * (1 - e.fee) - 1
    infl = pi[:, 1:].copy()
    for t, v in (e.infl_shocks or {}).items():
        if t < T: infl[:, t] = v
    cpi = np.concatenate([np.ones((n, 1)), np.cumprod(1 + infl, axis=1)], axis=1)
    return {"stock": stock, "bond": bond, "infl": infl, "cpi": cpi, "rate": r,
            "house": _house_index(e, stock, T, n, rng, U[:, 0]),
            "antithetic": e.antithetic, "erp_path": erp}


# KOSPI 실데이터 추정 (v18, 2000-02~2026-08 월별 319개월, FinanceDataReader 'KS11' + ECOS 국고채 3년)
# - 변동성·폭락 점프: 월 수익률 연환산 변동성 23.9%, 초과첨도 2.43, 12개월 수익률 −20% 이하 9.4%·−30% 이하 2.6%에
#   맞춰 확산 21% + 점프(연 0.05회, 로그 −20%±10%)로 보정 → 연 로그수익 표준편차 21.6%, 꼬리 빈도 9.6%·2.9%
# - 위험프리미엄: 데이터(가격 초과수익 3.92%±9.09%p + 배당 1.8% 가정 + ½σ²) 8.6%±4.6%p를
#   사전분포(성숙시장 ERP 4.2%[Damodaran 2026] + 한국 국가위험 약 0.7%p = 4.9%±1.5%p)와 결합한 사후값
# - 금리 상관: 월 수익률과 국고채 3년 변화의 상관 +0.09
CALIBRATED_EQUITY_2026_08 = dict(
    s_sigma=0.21, jump_lambda=0.05, jump_mu=-0.20, jump_sigma=0.10,
    erp=0.052, erp_sd=0.014, rho_rs=0.09, rho_ps=0.0,
)

# 주택가격 실데이터 추정 (v26, ECOS 한국부동산원 전국주택가격동향조사 월간, 2013-01~ — 작성기관 변경 이후만 사용)
# - 변동성: 월별 지수는 평활되어(1개월 자기상관 0.82) 변동성을 2.5%로 과소평가 → 연도별 수익률로 재추정 6.6%.
#   전국 지수보다 개별 주택 한 채가 더 흔들리므로 고유 위험 7%를 더해 총 ≈ 10% 사용
# - 실질 상승률: 데이터 3.25%(표준오차 1.89%p)와 사전분포 N(0%, 1.5%)의 베이즈 결합 → 1.25% ± 1.18%p.
#   2013~2026 표본은 상승기가 길어 그대로 쓰면 과대평가되고, 장기적으로 실질 집값이 소득보다 빨리
#   오르기 어렵다는 점(인구 감소 포함)을 사전분포로 반영
# - 상관: 월별 주식 0.05·물가 0.07·금리변화 0.18 (평활로 0 쪽 편의) → 주식 상관 0.1 사용.
#   물가 연동은 "명목 집값 = 실질 지수 × CPI" 구조로 반영
CALIBRATED_HOUSE_2026_08 = dict(house_real_growth=0.0125, house_real_sd=0.0118, house_sigma=0.10, rho_hs=0.1)
