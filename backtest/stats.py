"""
정규분포·카이제곱 꼬리확률 — 이 저장소 전체가 공유하는 단일 구현.

[왜 따로 모았나]
engine.py(p_value_from_t), deflated.py(DSR), walkforward.py(합산 z→p)가 각자
`0.5 * math.erfc(...)`를 따로 들고 있었다. 수식이 같아 결과는 같았지만, 나중에
정규분포 근사를 t분포나 부트스트랩 경험분포로 바꿀 때 세 곳을 전부 고쳐야 하고
한 곳을 빠뜨리면 모듈마다 같은 t값에서 다른 p값이 나온다. 검정 기준이 모듈마다
갈리는 건 통계적 결론을 믿을 수 없게 만들므로 진입점을 하나로 둔다.

[왜 scipy를 안 쓰나]
이 저장소는 scipy 의존성 없이 돌아가는 것을 원칙으로 한다(GitHub Actions 알림
러너와 Colab 양쪽에서 설치 부담을 줄이기 위해). math.erfc는 표준 라이브러리이고
배정밀도 한계까지 정확하므로 근사 오차 문제도 없다.

[한계]
표준정규분포이지 t분포가 아니다. 표본이 작으면(대략 30건 미만) 실제 t분포보다
꼬리가 얇아 p-value를 과소평가한다 — 즉 유의하다고 판정하기 쉬워진다. 각 호출부
docstring에 이 점을 명시해 두었다.
"""
from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def norm_sf(z: float) -> float:
    """표준정규분포 생존함수 P(Z > z). 단측 p-value로 바로 쓴다."""
    return 0.5 * math.erfc(float(z) / _SQRT2)


def norm_cdf(z: float) -> float:
    """표준정규분포 누적분포함수 P(Z <= z). norm_sf(-z)와 같다."""
    return 0.5 * math.erfc(-float(z) / _SQRT2)


def chi2_sf(x: float, df: int) -> float:
    """카이제곱 꼬리확률 P(X > x). scipy 없이 정규화된 상한 불완전 감마 Q(df/2, x/2)로 계산한다."""
    if df <= 0 or x is None or not math.isfinite(x):
        return float("nan")
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    lg = math.lgamma(a)
    if xx < a + 1.0:                                  # 급수: P(a,x), Q = 1 - P
        term = total = 1.0 / a
        n = a
        for _ in range(500):
            n += 1.0
            term *= xx / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-xx + a * math.log(xx) - lg))
    tiny = 1e-300                                     # 연분수(Lentz): Q(a,x)
    b = xx + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-xx + a * math.log(xx) - lg) * h)


def norm_ppf(p: float) -> float:
    """표준정규분포 분위함수(역 CDF). Acklam(2003) 유리근사 + 뉴턴 보정 1회로 배정밀도 수준.

    VaR의 z 승수(예: 99% -> 2.326348)에 쓴다. 0 < p < 1 밖이면 NaN."""
    p = float(p)
    if not (0.0 < p < 1.0):
        return float("nan")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    # 뉴턴 1회: 잔차 e = Φ(x) - p 를 밀도로 나눠 보정
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)
