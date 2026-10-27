"""
VaR(Value at Risk)·ES(Expected Shortfall) — 포트폴리오 수익률 시계열에서 '내일 잃을 수 있는 금액'을 추정한다.

[관례]
  - 손실 L = -r. VaR·ES는 양수(손실 크기)로 돌려준다. VaR 0.03 = "자산의 3%".
  - 신뢰수준 alpha(기본 0.99): 손실이 VaR를 넘는 날이 (1-alpha)×100% ≈ 1%여야 한다.
  - 예측 시점: VaR[t]는 t-1일까지의 수익률만으로 계산한 't일 손실'의 추정치다. 그래서 t일 실현 손실과
    VaR[t]를 비교하는 것이 백테스트(risk/var_backtest.py)다. 이 시프트를 여기서 처리하므로 백테스트는
    두 시계열을 그대로 비교하면 된다.

[세 가지 방법 — 가정이 다르다]
  1. historical  : 최근 window일 손실의 경험분포에서 k번째 큰 손실. k = ceil((1-alpha)×window).
                   분포 가정이 없다. 대신 변동성이 바뀌면 window만큼 늦게 반응하고, window 안에 없던 크기의
                   손실은 절대 예측하지 못한다.
  2. parametric  : 정규분포 가정. VaR = z_alpha × σ[t]. σ는 롤링 표준편차 또는 EWMA(RiskMetrics λ=0.94).
                   평균은 0으로 둔다(일간 평균수익률은 추정오차가 더 크다). 꼬리가 두꺼운 실제 수익률에서는
                   99% VaR를 과소추정하는 것이 알려진 한계다.
  3. fhs         : 필터드 역사적 시뮬레이션(Filtered Historical Simulation). 과거 손실을 그 시점의 EWMA σ로
                   나눠 표준화한 뒤 경험분위를 구하고, 현재 σ[t]를 다시 곱한다. 분포 모양은 데이터에서(1의 장점),
                   변동성 수준은 현재 상태를 따른다(2의 장점). 이 저장소의 기본 권장 방법이다.

[ES]
  VaR는 "넘을 확률"만 말하고 "넘으면 얼마나"는 말하지 않는다. ES = VaR를 넘는 손실의 평균. 바젤 FRTB가
  VaR를 ES(97.5%)로 교체한 이유다. historical·fhs는 꼬리 표본 평균, parametric은 정규분포 공식
  σ × φ(z)/(1-alpha).

[정직한 한계]
  - 모두 과거 수익률이 미래 분포를 대표한다는 가정 위에 있다. 전례 없는 사건은 어느 방법도 모른다.
  - 1일 VaR다. 청산에 며칠 걸리는 포지션(유동성)은 √h 규칙이 과소추정이며, 아직 반영하지 않았다.
  - 어떤 방법이 맞는지는 이 모듈이 아니라 var_backtest.py의 초과 횟수·독립성 검정이 판정한다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.stats import norm_ppf

METHODS = ("historical", "parametric", "fhs")
EWMA_LAMBDA = 0.94          # RiskMetrics(1996) 일간 감쇠계수


def _check_alpha(alpha: float) -> None:
    if not (0.5 < alpha < 1.0):
        raise ValueError(f"alpha는 0.5 < alpha < 1 이어야 합니다 (alpha={alpha})")


def ewma_vol(returns: pd.Series, lam: float = EWMA_LAMBDA, min_periods: int = 20) -> pd.Series:
    """
    EWMA 변동성 σ[t] = sqrt(λ σ²[t-1] + (1-λ) r²[t-1])  — t-1까지의 정보로 t일 변동성을 예측한다.
    pandas ewm(alpha=1-λ, adjust=False)로 r²의 EWMA를 구한 뒤 한 칸 시프트한다. 처음 min_periods 전은 NaN.
    평균을 빼지 않는다(RiskMetrics 관례, 일간 평균≈0).
    """
    r2 = returns.astype(float) ** 2
    var = r2.ewm(alpha=1.0 - lam, adjust=False, min_periods=min_periods).mean()
    return np.sqrt(var).shift(1).rename("ewma_vol")


def rolling_vol(returns: pd.Series, window: int = 250) -> pd.Series:
    """롤링 표준편차(ddof=1), 한 칸 시프트 — t-1까지의 window일로 t일 변동성 예측."""
    return returns.astype(float).rolling(window, min_periods=window).std(ddof=1).shift(1).rename("rolling_vol")


def _tail_k(alpha: float, n: int) -> int:
    """경험분포에서 쓸 꼬리 표본 수 k = ceil((1-alpha) n), 최소 1."""
    return max(1, int(math.ceil((1.0 - alpha) * n - 1e-12)))


def historical_var(returns: pd.Series, alpha: float = 0.99, window: int = 250) -> pd.DataFrame:
    """
    롤링 역사적 VaR·ES. 각 t에 대해 (t-window .. t-1)의 손실 중 k번째 큰 값이 VaR, 상위 k개 평균이 ES.
    반환: DataFrame(var, es), 앞 window일은 NaN.
    """
    _check_alpha(alpha)
    r = returns.astype(float)
    loss = -r
    k = _tail_k(alpha, window)
    n = len(loss)
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    vals = loss.to_numpy()
    for t in range(window, n):
        win = vals[t - window:t]
        if np.isnan(win).any():
            continue
        top = np.partition(win, len(win) - k)[len(win) - k:]     # 가장 큰 k개
        var[t] = top.min()
        es[t] = top.mean()
    return pd.DataFrame({"var": var, "es": es}, index=r.index)


def parametric_var(returns: pd.Series, alpha: float = 0.99, vol: pd.Series | None = None,
                   window: int = 250, use_ewma: bool = True) -> pd.DataFrame:
    """
    정규분포 VaR = z_alpha σ[t], ES = σ[t] φ(z_alpha)/(1-alpha). σ는 vol 인자가 없으면 EWMA(기본) 또는 롤링.
    """
    _check_alpha(alpha)
    r = returns.astype(float)
    if vol is None:
        vol = ewma_vol(r) if use_ewma else rolling_vol(r, window)
    z = norm_ppf(alpha)
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    var = z * vol
    es = vol * phi / (1.0 - alpha)
    return pd.DataFrame({"var": var.to_numpy(), "es": es.to_numpy()}, index=r.index)


def filtered_historical_var(returns: pd.Series, alpha: float = 0.99, window: int = 250,
                            lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """
    FHS: 표준화 손실 z_s = L[s]/σ[s] (σ[s]는 s-1까지의 EWMA) 의 최근 window개에서 k번째 큰 값 q,
    VaR[t] = q × σ[t], ES[t] = (상위 k개 평균) × σ[t].
    """
    _check_alpha(alpha)
    r = returns.astype(float)
    sig = ewma_vol(r, lam)
    std_loss = (-r / sig)
    k = _tail_k(alpha, window)
    vals = std_loss.to_numpy()
    sigv = sig.to_numpy()
    n = len(r)
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    for t in range(1, n):
        lo = t - window
        if lo < 0 or not np.isfinite(sigv[t]):
            continue
        win = vals[lo:t]
        if np.isnan(win).any():
            continue
        top = np.partition(win, len(win) - k)[len(win) - k:]
        var[t] = top.min() * sigv[t]
        es[t] = top.mean() * sigv[t]
    return pd.DataFrame({"var": var, "es": es}, index=r.index)


def compute_var(returns: pd.Series, method: str = "fhs", alpha: float = 0.99, window: int = 250,
                lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """메서드 이름으로 분기하는 단일 진입점. 반환 DataFrame(var, es), attrs에 method/alpha/window."""
    if method not in METHODS:
        raise ValueError(f"method는 {METHODS} 중 하나여야 합니다 (method={method})")
    if method == "historical":
        out = historical_var(returns, alpha, window)
    elif method == "parametric":
        out = parametric_var(returns, alpha, window=window)
    else:
        out = filtered_historical_var(returns, alpha, window, lam)
    out.attrs.update(method=method, alpha=alpha, window=window, lam=lam)
    return out
