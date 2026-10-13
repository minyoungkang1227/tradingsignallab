"""사망률 모듈.

기본값(v13): 통계청 「생명표」 2024 완전생명표(1세별, KOSIS DT_1B42)의 성별 사망확률.
100세 이상은 90~99세 사망력에 Gompertz(로그-선형)를 맞춰 109세까지 외삽하고 110세에서 종료.
CSV 형식: age,qx_M,qx_F (retire_sim/data/life_table_2024.csv). gompertz_qx()는 이전 임시값(비교용).
"""
import os
import numpy as np
import csv

# 임시 Gompertz 파라미터 mu(x) = a*exp(b*x) — 한국 대략 수준에 맞춘 값, 실데이터로 교체 필요
_GOMPERTZ = {"M": (1.65e-5, 0.100), "F": (4.28e-6, 0.108)}  # 60세 기대여명 남 23.5 / 여 29.5년에 보정


def gompertz_qx(max_age: int = 110) -> dict:
    ages = np.arange(max_age + 1)
    out = {}
    for sex, (a, b) in _GOMPERTZ.items():
        H = a / b * (np.exp(b * (ages + 1)) - np.exp(b * ages))  # 1년 누적위험
        q = 1 - np.exp(-H)
        q[-1] = 1.0
        out[sex] = np.clip(q, 0, 1)
    return out


_DEFAULT_TABLE = os.path.join(os.path.dirname(__file__), "data", "life_table_2024.csv")


# ── v27 사망률 개선(코호트) ──
# 연령별 연간 개선율: q(x, 연도) = q_2024(x) × (1 − k_x)^(연도 − 2024)
# 미국 SOA Scale MP와 같은 역할. 통계청 생명표 추이(2013→2024 기대수명 남 +2.3년, 여 +1.5년,
# 연 +0.21·+0.14년)와 맞도록 연령대별로 설정한 근사값. 실제 다년도 생명표로 재추정하려면
# scripts/calibrate_mortality.py 참고.
IMPROVEMENT = ((65, 0.025), (80, 0.020), (90, 0.013), (100, 0.007), (999, 0.003))
BASE_YEAR = 2024


def improvement_rate(age: int) -> float:
    for lim, v in IMPROVEMENT:
        if age < lim:
            return v
    return IMPROVEMENT[-1][1]


def project_qx(qx: np.ndarray, years_ahead: float, scale=None) -> np.ndarray:
    """기준연도 사망률을 years_ahead년 뒤 수준으로 투영(연령별 개선율 적용)."""
    if years_ahead <= 0:
        return qx
    k = np.array([improvement_rate(x) for x in range(len(qx))])
    out = qx * (1 - k) ** years_ahead
    out[-1] = 1.0
    return np.clip(out, 0, 1)


def cohort_qx(qx: np.ndarray, start_age: int, T: int, calendar_offset: float = 0.0) -> np.ndarray:
    """코호트 사망률 행렬 (T+1, len(qx)): t년 뒤에 적용할 사망률 벡터.
    start_age에서 t년 뒤면 달력연도도 t년 흐르므로 개선율을 t+offset년치 적용."""
    return np.stack([project_qx(qx, t + calendar_offset) for t in range(T + 1)])


def default_qx(max_age: int = 110) -> dict:
    """기본 사망률: 통계청 2024 완전생명표(0~99세) + 100~109세 Gompertz 외삽. 파일이 없으면 임시 Gompertz."""
    if os.path.exists(_DEFAULT_TABLE):
        return load_life_table(_DEFAULT_TABLE, max_age)
    return gompertz_qx(max_age)


def load_life_table(path: str, max_age: int = 110) -> dict:
    qM, qF = np.ones(max_age + 1), np.ones(max_age + 1)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = int(row["age"])
            if a <= max_age:
                qM[a], qF[a] = float(row["qx_M"]), float(row["qx_F"])
    return {"M": qM, "F": qF}


def life_expectancy(qx: np.ndarray, age: int, improve: bool = False) -> float:
    if improve:
        p = 1.0; tot = 0.0
        for t in range(len(qx) - age):
            q = project_qx(qx, t)[min(age + t, len(qx) - 1)]
            p *= (1 - q); tot += p
        return tot + 0.5
    return _life_expectancy_period(qx, age)


def _life_expectancy_period(qx: np.ndarray, age: int) -> float:
    surv = np.cumprod(1 - qx[age:])
    return float(surv.sum() + 0.5)


def _qx_at(qx, t, improve: bool, offset: float = 0.0):
    """t년 뒤에 쓸 사망률 벡터 (개선 반영 여부)."""
    return project_qx(qx, t + offset) if improve else qx


def simulate_alive(qx: np.ndarray, start_age: int, T: int, n: int, rng, improve: bool = False, offset: float = 0.0,
                   qmult=None) -> np.ndarray:
    """alive[p, t] : t년 시작 시점 생존 여부 (t=0은 현재)."""
    ages = np.minimum(start_age + np.arange(T), len(qx) - 1)
    qq = qx[ages][None, :] * (1.0 if qmult is None else qmult)
    die = rng.random((n, T)) < qq
    alive = np.ones((n, T + 1), dtype=bool)
    alive[:, 1:] = np.cumprod(~die, axis=1).astype(bool)
    return alive


def simulate_life(qx: np.ndarray, start_age: int, T: int, n: int, rng, care, return_severe: bool = False,
                  improve: bool = False, offset: float = 0.0, qmult=None) -> tuple:
    """다중상태 경로. 반환: alive (n, T+1), in_care (n, T+1) [, severe (n, T+1)].

    단일 상태(legacy): q_H = q_x / (1 + (m−1)·p), p ≈ i·D, D = 1/(m·q + rec)
    2단계(v17): q_H = q_x / (1 + (m_M−1)·p(1−s) + (m_S−1)·p·s), p·s는 보정 목표 유병률·중증 비율
    """
    alive = np.ones((n, T + 1), bool); incare = np.zeros((n, T + 1), bool); severe = np.zeros((n, T + 1), bool)
    two = getattr(care, "two_level", False)
    m, rec = care.mort_mult, care.recovery
    for t in range(T):
        q_t = _qx_at(qx, t, improve, offset)
        x = min(start_age + t, len(qx) - 1)
        q = q_t[x] * (1.0 if qmult is None else qmult[:, t]); i = care.inc(x)
        if two:
            p = care.prev(x); sh = care.severe_share; mS = care.mort_mult_severe
            qH = np.minimum(q / (1 + (m - 1) * p * (1 - sh) + (mS - 1) * p * sh), 1.0)
        else:
            D = np.minimum(1.0 / np.maximum(m * q + rec, 1e-9), 10.0); p = np.minimum(i * D, 0.5)
            qH = np.minimum(q / (1 + (m - 1) * p), 1.0); mS = m
        a, c, sv = alive[:, t], incare[:, t], severe[:, t]
        u = rng.random(n); v = rng.random(n)
        qdie = np.where(sv, np.minimum(mS * qH, 1.0), np.where(c, np.minimum(m * qH, 1.0), qH))
        die = u < qdie
        to_care = ~c & ~die & (v < i)
        recover = c & ~die & (v < rec)
        alive[:, t + 1] = a & ~die
        incare[:, t + 1] = alive[:, t + 1] & ((c & ~recover) | to_care)
        if two:
            w = rng.random(n)
            new_sev = to_care & (w < care.severe_entry)
            prog = c & ~sv & ~die & ~recover & (w < care.progress)
            severe[:, t + 1] = incare[:, t + 1] & ((sv & ~recover) | new_sev | prog)
    return (alive, incare, severe) if return_severe else (alive, incare)



# ── v30 장수위험: 확률적 사망률 수준 · 사별 효과 ──
def longevity_multiplier(n: int, T: int, sigma_level: float, sigma_trend: float, seed) -> np.ndarray:
    """경로별 사망률 배수 (n, T). 두 배우자에게 공통(체계적 위험).
    m[p,t] = exp(η_p − ε_p·t − 분산보정),  η~N(0,σ_level²) : 수준 불확실성,  ε~N(0,σ_trend²) : 사망률 개선 속도 불확실성.
    별도 난수 흐름을 쓰므로 끄면 기존 결과와 동일하고, 켜도 다른 난수(경제·간병)는 그대로(공통 난수 유지)."""
    if sigma_level <= 0 and sigma_trend <= 0:
        return None
    r = np.random.default_rng([int(seed) if seed is not None else 0, 30])
    eta = sigma_level * r.standard_normal(n); eps = sigma_trend * r.standard_normal(n)
    t = np.arange(T)[None, :]
    var = sigma_level ** 2 + (sigma_trend * t) ** 2
    return np.exp(eta[:, None] - eps[:, None] * t - 0.5 * var)


def apply_widowhood(alive, incare, severe, qx_list, ages, sexes, excess, tau, improve, seed, qmult=None, spike=0.0):
    """사별 후 생존 배우자의 사망 위험 상승(애도·돌봄 상실 등). 2인 가구에만 적용.
    alive: (2, n, T+1). 사별 s년 뒤 추가 사망률 = q_t × excess × (1 + spike·exp(−s/τ)).
    먼저 사망한 사람은 영향 없음(순서 보존).
    sexes: 구성원별 "M"/"F", excess: (남성, 여성) 초과 위험 — 구성원의 성별로 고른다."""
    if alive.shape[0] != 2 or (excess[0] <= 0 and excess[1] <= 0):
        return alive, incare, severe
    r = np.random.default_rng([int(seed) if seed is not None else 0, 31])
    n, Tp1 = alive.shape[1], alive.shape[2]
    out = alive.copy()
    for i in (0, 1):
        j = 1 - i; ex = excess[0] if sexes[i] == "M" else excess[1]
        if ex <= 0: continue
        since = np.cumsum(~alive[j], axis=1)                       # j 사망 후 경과 (사망 시점 포함해 1부터)
        for t in range(Tp1 - 1):
            x = min(ages[i] + t, len(qx_list[i]) - 1)
            q = project_qx(qx_list[i], t)[x] if improve else qx_list[i][x]
            if qmult is not None: q = q * qmult[:, t]
            s_ = np.maximum(since[:, t] - 1, 0)
            qe = np.minimum(q * ex * (1.0 + spike * np.exp(-s_ / tau)), 1.0) * (since[:, t] > 0)
            d = r.random(n) < qe
            out[i, d, t + 1:] = False
    inc2 = incare & out if incare is not None else None
    sev2 = severe & out if severe is not None else None
    return out, inc2, sev2
