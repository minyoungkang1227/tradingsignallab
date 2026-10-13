"""결과 지표 — 선택지 비교용 짝지은 신뢰구간.

v31: 가구별 요약은 engine_tax.summarize 하나로 통일하고 여기서는 비교만 다룬다."""
import numpy as np


def prob_se(flags, antithetic=False):
    """고갈 여부(0/1) 평균의 표준오차. 반대 난수면 짝(i, i+h)의 평균으로 계산."""
    x = np.asarray(flags, float); n = len(x)
    if antithetic and n >= 4:
        h = n // 2
        pair = (x[:h] + x[h:2 * h]) / 2
        return float(pair.std(ddof=1) / np.sqrt(h))
    p = x.mean()
    return float(np.sqrt(p * (1 - p) / n))


def compare(res_a: dict, res_b: dict) -> dict:
    """두 선택지의 고갈확률 차이와 '짝지은' 95% 신뢰구간 (공통 난수 전제).

    경로별 d_i = 1[A 고갈] − 1[B 고갈] 의 평균과 표준오차를 쓴다.
    같은 시나리오를 공유하므로 Var(A−B) = VarA + VarB − 2Cov(A,B) 가 각각의 분산 합보다 훨씬 작다.
    """
    a = (np.asarray(res_a["depleted_at"]) >= 0).astype(float)
    b = (np.asarray(res_b["depleted_at"]) >= 0).astype(float)
    d = a - b; n = len(d)
    if res_a.get("antithetic") and res_b.get("antithetic") and n >= 4:
        h = n // 2; x = (d[:h] + d[h:2 * h]) / 2; se = x.std(ddof=1) / np.sqrt(h)
    else:
        se = d.std(ddof=1) / np.sqrt(n)
    naive = np.sqrt(a.mean() * (1 - a.mean()) / n + b.mean() * (1 - b.mean()) / n)
    diff = d.mean()
    return {"차이": float(diff), "±": float(1.96 * se), "독립 가정 시 ±": float(1.96 * naive),
            "유의": bool(abs(diff) > 1.96 * se)}
