"""
디플레이티드 샤프 비율(Deflated Sharpe Ratio, DSR) — Bailey & López de Prado(2014).

[왜 필요한가 — 지금 engine.py의 유의성 판정에 뚫려 있는 구멍]

현재 compare_all()은 t-stat을 p_value_from_t()로 p-value로 바꾼 뒤 BH보정을
적용한다. BH 자체는 올바르지만, 그 입력인 p-value가 "수익률이 정규분포를
따른다"는 가정 위에서 계산된다는 점이 문제다. 이벤트 스터디의 순방향수익률은
실제로는 거의 항상

  - 왜도(skew)가 0이 아니고(급락이 급등보다 가파름),
  - 첨도(kurtosis)가 3보다 크다(팻테일 — 극단값이 정규분포보다 자주 나옴).

팻테일 구간에서 정규분포 근사는 분산을 과소평가하므로 t-stat이 부풀려지고,
그 결과 p-value가 실제보다 작게 나온다. 즉 "유의하다"고 잘못 판정하는 쪽으로
일관되게 치우친다. 잘못된 p-value에 BH를 적용해봐야 여전히 잘못된 판정이다.

DSR은 이 세 가지를 한꺼번에 보정한다.

  1) 비정규성 — 샤프비율 추정량의 표준오차에 왜도·첨도를 직접 넣는다(Lo 2002).
  2) 다중검정 선택편의 — N번 시도해서 고른 "최고 성과"는 우연만으로도 어느
     정도 높게 나온다. 그 기대 최대치를 귀무가설 기준선으로 삼는다.
  3) 표본 길이 — 관측 수가 적을수록 추정이 불안정하다는 점을 반영한다.

BH보정이 "여러 번 검정했으니 임계값을 높이자"는 접근이라면, DSR은 "여러 번
시도했을 때 우연히 나올 수 있는 최고 성과가 이 정도이니, 그걸 넘는지 보자"는
접근이다. 둘은 대체재가 아니라 보완재이고, DSR 쪽이 비정규성까지 잡아준다.

[이벤트 스터디에 샤프비율을 적용하는 근거]

DSR은 원래 "수익률 시계열의 샤프비율"에 대한 이론이지만, 여기서는 이벤트
수익률 r_1..r_n을 "거래 1건당 수익률의 표본"으로 보고 거래당 샤프비율
SR = mean(r)/std(r)를 쓴다. 이건 임의의 적용이 아니다 — 기존 engine.py의
t-stat이 정확히 t = mean/(std/sqrt(n)) = SR*sqrt(n) 구조이기 때문에, 같은
대상을 샤프비율 언어로 바꿔 말한 것뿐이다.

다만 engine.py의 t-stat은 Newey-West나 클러스터 강건 표준오차를 쓸 수 있어서
그 경우 t != SR*sqrt(n)이 된다. 그래서 이 모듈은 t-stat에서 SR을 역산하지
않고, 항상 원시 이벤트 수익률에서 mean/std/skew/kurt를 직접 계산한다.

[연율화하지 않는다]

SR과 관측 수 n의 단위가 서로 맞아야 공식이 성립한다. 여기서 관측 1개는
"거래 1건"이고 n은 이벤트 수이므로, 거래당 샤프비율을 그대로 쓴다. 연율화해서
넣으면 n과 단위가 어긋나 완전히 틀린 값이 나온다(√252를 곱하는 흔한 실수).

[의존성]

engine.py와 동일하게 scipy를 쓰지 않는다. 정규분포 CDF는 math.erfc로,
역CDF(분위수)는 Acklam 유리근사 + erfc 기반 Halley 보정 1회로 계산한다.
"""
from __future__ import annotations

import math

import numpy as np

# 오일러-마스케로니 상수. E[최대 SR] 공식에서 두 분위수 항의 가중치로 쓰인다.
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# 정규분포 CDF / 역CDF (scipy 없이)
# ---------------------------------------------------------------------------

def norm_cdf(z: float) -> float:
    """표준정규분포 누적분포함수 P(Z <= z)."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# Acklam(2003) 유리근사 계수. 상대오차 약 1.15e-9.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """
    표준정규분포 역누적분포함수(분위수) Phi^-1(p). scipy.stats.norm.ppf 대체.

    Acklam 유리근사로 초기값을 잡고 Halley 보정을 1회 적용해 배정밀도 수준까지
    정확도를 올린다. p가 0 또는 1이면 각각 -inf/+inf를 반환한다.

    E[최대 SR] 공식이 Phi^-1(1 - 1/N)처럼 꼬리 쪽 분위수를 요구하는데, 시행
    횟수 N이 커질수록 1에 매우 가까운 p가 들어온다(N=1000이면 p=0.999).
    그래서 꼬리 영역 정확도가 중요해 보정 단계를 넣었다.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
             / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
             / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
              / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))

    # Halley 보정 1회: e = Phi(x) - p 를 쓰되, 2차항까지 반영해 수렴을 가속한다.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return x


# ---------------------------------------------------------------------------
# PSR / E[최대 SR] / DSR
# ---------------------------------------------------------------------------

def sharpe_std_error(sr: float, n_obs: int, skew: float = 0.0,
                      kurt: float = 3.0) -> float:
    """
    샤프비율 추정량의 표준오차(Lo 2002, Mertens 보정형).

        Var(SR_hat) = (1 - skew*SR + ((kurt - 1)/4) * SR^2) / (n - 1)

    정규분포(skew=0, kurt=3)면 (1 + SR^2/2)/(n-1)로 줄어든다. 팻테일(kurt>3)이면
    분자가 커져 표준오차가 커지고, 그만큼 유의성 판정이 보수적으로 바뀐다.
    여기서 kurt는 초과첨도가 아니라 원첨도(정규분포=3)다 — 초과첨도를 그대로
    넣으면 분자가 작아져 유의성을 과대평가하게 되므로 주의.

    [주의: 왜도항과 첨도항은 SR에 대한 차수가 다르다]

        왜도항 = -skew * SR        (SR의 1차)
        첨도항 = (kurt-1)/4 * SR^2 (SR의 2차)

    그래서 거래당 SR이 작으면(이벤트 스터디에서는 0.1~0.3이 흔하다) 왜도가
    첨도를 압도한다. 예를 들어 SR=0.12에서 첨도가 2.9->15로 다섯 배 뛰어도
    첨도항은 0.006->0.058만큼만 늘지만, 왜도가 +0.77이면 왜도항 하나로
    -0.092가 움직여 첨도 효과를 통째로 뒤집는다.

    실무적 함의: "팻테일이라 DSR이 낮게 나오겠지"라고 단정하면 안 된다.
    양의 왜도(가끔 크게 벌고 자주 조금 잃음)는 오히려 DSR을 올리고, 음의
    왜도(자주 조금 벌고 가끔 크게 잃음 — 손절 없는 전략의 전형)는 떨어뜨린다.
    두 효과의 크기는 SR 수준에 따라 역전되므로 항상 둘을 같이 봐야 한다.

    n_obs <= 1이거나 분산 추정이 음수면 NaN.
    """
    if n_obs is None or n_obs <= 1:
        return float("nan")
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n_obs - 1)
    if var <= 0 or math.isnan(var):
        return float("nan")
    return math.sqrt(var)


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                                skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    확률적 샤프비율(PSR) = P(진짜 SR > sr_benchmark).

        PSR = Phi( (SR - SR_b) / SE(SR_hat) )

    관측된 SR이 기준선을 넘었다는 사실 자체보다, "표본 노이즈·비정규성을 감안해도
    여전히 넘는다고 볼 확률"을 준다. 0.95 이상이면 통상적인 5% 유의수준에 해당.
    """
    se = sharpe_std_error(sr, n_obs, skew=skew, kurt=kurt)
    if math.isnan(se):
        return float("nan")
    if se == 0:
        return 1.0 if sr > sr_benchmark else 0.0
    return norm_cdf((sr - sr_benchmark) / se)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """
    귀무가설(진짜 실력은 0) 하에서 N번 시도했을 때 기대되는 "최대 샤프비율".

        E[max SR] = sr_std * [ (1 - gamma) * Phi^-1(1 - 1/N)
                               + gamma * Phi^-1(1 - 1/(N*e)) ]

    gamma는 오일러-마스케로니 상수. sr_std는 시도들 간 샤프비율의 표준편차로,
    "이 탐색 공간이 얼마나 넓게 흩어진 결과를 내는가"를 나타낸다.

    직관: 동전 던지기를 100번 시도해 그중 제일 잘 나온 걸 고르면, 실력이 0이어도
    꽤 높은 성과가 나온다. 그 "우연히 나올 수 있는 최고치"가 바로 넘어야 할
    진짜 기준선이다. N이 커질수록, 시도들이 넓게 흩어질수록 기준선이 올라간다.

    n_trials <= 1이면 비교할 다른 시도가 없으므로 0.0(기준선 없음)을 반환한다.
    """
    if n_trials is None or n_trials <= 1:
        return 0.0
    if sr_std is None or math.isnan(sr_std) or sr_std <= 0:
        return 0.0
    q1 = norm_ppf(1.0 - 1.0 / n_trials)
    q2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2)


def deflated_sharpe_ratio(returns, n_trials: int, sr_std: float) -> dict:
    """
    원시 이벤트 수익률에서 DSR을 계산한다.

    returns: 이벤트별 수익률 array-like(거래 1건 = 관측 1개). engine.py의
        event_returns()가 반환하는 Series를 그대로 넣으면 된다.
    n_trials: 이 결과를 고르기까지 시도한 전체 검정 횟수(신호 x 방향 x 보유기간
        조합 수 등). 많이 시도했을수록 기준선이 높아진다.
    sr_std: 그 시도들 간 샤프비율의 표준편차.

    반환 dict:
        sharpe        — 거래당 샤프비율 mean/std (연율화 안 함)
        n_obs         — 이벤트 수
        skew, kurt    — 표본 왜도 / 원첨도(정규분포=3)
        sr_benchmark  — E[최대 SR] (넘어야 할 기준선)
        dsr           — 디플레이티드 샤프비율(= 확률값 0~1)

    표본이 2개 미만이거나 표준편차가 0이면 전부 NaN(판정 불가)으로 반환한다 —
    "유의하다"로 잘못 흘러가지 않게 하기 위함.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)

    out = {"sharpe": float("nan"), "n_obs": n, "skew": float("nan"),
           "kurt": float("nan"), "sr_benchmark": float("nan"), "dsr": float("nan")}
    if n < 2:
        return out

    mean = float(r.mean())
    # 표본표준편차(ddof=1) — t-stat과 동일한 분모 관례를 쓴다.
    std = float(r.std(ddof=1))
    if std <= 0 or math.isnan(std):
        return out

    sr = mean / std
    # 왜도·첨도는 모멘트 추정량(편향보정 없음). 표본이 작을 때 첨도가 과소
    # 추정되는 경향이 있는데, 그 방향은 유의성을 과대평가하는 쪽이므로
    # n_obs가 작으면 결과를 보수적으로 해석해야 한다.
    z = (r - mean) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))

    sr_bench = expected_max_sharpe(n_trials, sr_std)
    dsr = probabilistic_sharpe_ratio(sr, sr_bench, n, skew=skew, kurt=kurt)

    out.update(sharpe=sr, skew=skew, kurt=kurt, sr_benchmark=sr_bench, dsr=dsr)
    return out


def sharpe_dispersion(sharpes) -> float:
    """
    여러 시도의 샤프비율 "실측 분산"으로 sr_std를 추정한다.

    [언제 쓰면 안 되는가 — 실제로 당한 함정]

    이 추정량은 모든 시도의 표본 수(n)가 비슷할 때만 타당하다. 표본 수가
    제각각이면 n이 작은 시도의 추정 노이즈가 분산을 통째로 오염시킨다.

    진짜 SR이 0일 때 SR 추정치의 표준편차는 대략 1/sqrt(n-1)인데,
    n=2면 그 값이 발산한다(몬테카를로 2만회 실측: n=2에서 169, n=4에서 0.84,
    n=21에서 0.23, n=76에서 0.12). 그래서 이벤트가 2~4개뿐인 신호가 섞여
    있으면 sr_std가 실제보다 몇 배 부풀고, expected_max_sharpe()의 기준선이
    비현실적으로 높아져서 "진짜 신호는 떨어뜨리고 표본 적은 노이즈만
    통과시키는" 정반대 결과가 나온다.

    실제로 trading_signal_lab의 12개 신호는 빈도 차이가 극심해서(빅세일 0~8건
    vs VP박스 70건대) 이 함정에 그대로 걸렸다. 그래서 compare_all()의 기본
    추정 방식은 아래 null_sharpe_dispersion()(이론값)이고, 이 함수는 표본
    수가 균질하다고 확신할 때만 쓴다.

    유효값이 2개 미만이면 NaN.
    """
    s = np.asarray(sharpes, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1))


def null_sharpe_dispersion(n_obs_list) -> float:
    """
    귀무가설(진짜 SR=0) 하에서 샤프비율 추정치가 우연만으로 흩어지는 정도.

        sd ~ 1 / sqrt(n_typical - 1)

    진짜 실력이 0인 전략을 n건 관측하면 SR 추정치는 0 주변에서 이만큼
    요동친다(정규 수익률 가정에서 Var(SR_hat) = (1 + SR^2/2)/(n-1), SR=0이면
    1/(n-1)). 즉 "우연이 만들어내는 흩어짐"을 표본 수만으로 직접 계산한 값이다.

    n_obs_list: 각 시도의 관측(이벤트) 수. 대표값으로 중앙값을 쓴다 — 평균을
        쓰면 표본이 유난히 많은 시도 하나에 끌려가고, 최소값을 쓰면 가장 빈약한
        시도 기준이 되어 지나치게 보수적이 된다.

    실측 분산(sharpe_dispersion)보다 이쪽을 기본으로 삼는 이유는 표본 수
    차이에 오염되지 않기 때문이다. 다만 시도들이 "서로 다른 진짜 실력"을
    가질 수 있는 경우(파라미터 탐색이 아니라 완전히 다른 전략들을 비교하는
    경우)에는 이 값이 실제 흩어짐을 과소평가할 수 있다.

    유효한 n이 없거나 전부 2 미만이면 NaN.
    """
    n = np.asarray(list(n_obs_list), dtype=float)
    n = n[np.isfinite(n) & (n >= 2)]
    if len(n) == 0:
        return float("nan")
    n_typ = float(np.median(n))
    if n_typ < 2:
        return float("nan")
    return 1.0 / math.sqrt(n_typ - 1.0)
