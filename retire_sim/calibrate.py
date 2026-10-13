"""파라미터 추정 (v11): OU 과정 최우추정(MLE)과 ECOS 데이터 연결.

OU 정확 이산화 X_{t+Δ} = θ + (X_t−θ)e^{−κΔ} + ε,  ε ~ N(0, σ²(1−e^{−2κΔ})/(2κ))
는 AR(1) X_{t+Δ} = a + b·X_t + ε 과 같으므로, 정확 MLE는 회귀계수로 닫힌 해를 가진다.
  b = e^{−κΔ} → κ = −ln b / Δ,  θ = a / (1−b),  σ² = s²·2κ / (1−b²)
표준오차는 (a, b, s²)의 점근 분산에 델타 방법을 적용.
"""
import numpy as np


def ou_mle(x, dt: float) -> dict:
    x = np.asarray(x, float); x0, x1 = x[:-1], x[1:]; n = len(x0)
    X = np.column_stack([np.ones(n), x0])
    coef, *_ = np.linalg.lstsq(X, x1, rcond=None); a, b = coef
    res = x1 - X @ coef; s2 = res @ res / n
    if not (0 < b < 1):
        raise ValueError(f"평균회귀가 추정되지 않음 (b={b:.3f}). 기간·데이터를 확인하세요.")
    kappa = -np.log(b) / dt; theta = a / (1 - b); sigma = np.sqrt(s2 * 2 * kappa / (1 - b**2))
    cov_ab = s2 * np.linalg.inv(X.T @ X)
    # 델타 방법: θ = a/(1−b)
    g_theta = np.array([1 / (1 - b), a / (1 - b) ** 2])
    se_theta = float(np.sqrt(g_theta @ cov_ab @ g_theta))
    se_kappa = float(np.sqrt(cov_ab[1, 1]) / (b * dt))
    return {"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma),
            "se_theta": se_theta, "se_kappa": se_kappa, "n": n}


def fetch_ecos(api_key: str, stat_code: str, cycle: str, start: str, end: str, item_code: str) -> list:
    """ECOS StatisticSearch 호출 → [(시점, 값)] (Colab 등 인터넷 환경에서 실행).
    예) 소비자물가지수: stat_code='901Y009', cycle='M', item_code='0'
        국고채 금리: 통계표·항목 코드는 ECOS에서 확인 후 입력 (StatisticItemList로 조회 가능)"""
    import requests
    url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/"
           f"{stat_code}/{cycle}/{start}/{end}/{item_code}")
    rows = requests.get(url, timeout=30).json()["StatisticSearch"]["row"]
    return [(r["TIME"], float(r["DATA_VALUE"])) for r in rows]


def calibrate_from_series(rate_pct, cpi_index, dt: float = 1 / 12) -> dict:
    """월별 명목금리(%)와 CPI 지수로 피셔 모드 파라미터 추정.
    물가 = 전년동월비, 실질금리 = 명목금리 − 물가(사후적 근사; 기대물가를 쓰려면 칼만 필터 필요)."""
    r = np.asarray(rate_pct, float) / 100
    cpi = np.asarray(cpi_index, float)
    infl = cpi[12:] / cpi[:-12] - 1
    r = r[12:]
    q = r - infl
    pi_est, q_est = ou_mle(infl, dt), ou_mle(q, dt)
    return {"pi": pi_est, "q": q_est,
            "economy_kwargs": dict(fisher=True, pi_theta=pi_est["theta"], pi_kappa=pi_est["kappa"],
                                   pi_sigma=pi_est["sigma"], pi_theta_sd=pi_est["se_theta"],
                                   q_theta=q_est["theta"], q_kappa=q_est["kappa"], q_sigma=q_est["sigma"],
                                   pi0=float(infl[-1]), q0=float(q[-1]))}
