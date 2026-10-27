"""
통제 계층 — 한도를 VaR·ES에서 도출하고, 비중에 적용하고, 그 비용을 측정한다.

[왜 만들었나]
반전매매 프로젝트의 4단계(손절 -10%, 일일한도 -25%, 보유 10일, 서킷 5%)는 숫자에 근거가 없었다. 여기서는
모든 한도를 (1) 리스크 모델(risk/var.py의 ES)과 (2) 수익률 시계열의 재표본(블록 부트스트랩)에서 도출한다.
한도의 출처(provenance)를 함께 돌려주므로 "왜 이 숫자인가"에 답할 수 있다.

[네 한도 — 각각 다른 실패를 막고, 행동도 다르다]
  1. 포트폴리오 ES 한도 (사전, 비중 축소)
       예측 ES[t] > es_target 이면 모든 비중에 es_target / ES[t] 를 곱한다(변동성 목표의 ES 버전).
       ES[t]는 t-1까지의 정보. 비중을 늘리는 쪽(ES[t] < target일 때 레버리지)은 기본 꺼짐(max_scale=1.0).
  2. 종목 리스크 예산 (사전, 종목 비중 축소)
       |w_i| × σ_i[t] > name_budget 이면 그 종목만 name_budget / (|w_i| σ_i) 배로 줄인다. σ_i는 종목별 EWMA.
       집중도 한도를 '금액'이 아니라 '위험 기여'로 둔 것. 예산은 ES 목표의 일정 비율(기본 50%)로 정한다.
  3. 일일 손실 한도 (사후, 다음 날 신규 진입 중단)
       당일 손실 L[t] > ES[t] 이면 "모델이 예상한 꼬리 밖 사건"으로 보고, t+1일에 비중이 '늘어나는' 변화를 막는다
       (기존 포지션 유지·축소는 허용). 한도가 ES이므로 모델이 맞다면 발동 빈도는 P(L > ES_α)다 — 정규분포에서
       α=97.5%면 ES=2.338σ라 약 1.0%(VaR 97.5%의 초과율 2.5%보다 작다). 실현 빈도가 이와 크게 다르면 리스크
       모델이 틀렸다는 뜻이므로 자기 점검이 된다.
  4. 낙폭 서킷브레이커 (사후, 노출 절반)
       고점 대비 낙폭 > dd_limit 이면 비중을 절반으로 줄이고, 낙폭이 dd_limit/2 아래로 회복될 때까지 유지한다
       (히스테리시스). dd_limit은 통제 전 수익률의 블록 부트스트랩으로 만든 '1년 최대낙폭 분포'의 dd_quantile
       (기본 95%) 분위다. 유일하게 시뮬레이션이 필요한 한도이고, 그래서 가장 불확실하다. 회복 기준이 dd_limit/2라
       추세가 없는 포트폴리오에선 한 번 발동하면 오래 유지될 수 있다 — 이것은 결함이 아니라 '깊은 낙폭에서 회복될
       때까지 노출을 줄인다'는 선택이며, 그 비용은 compare_controlled()가 보여준다.

[미정의 처리]
ES[t]·σ_i[t]가 NaN인 날(히스토리 부족)은 비중을 0으로 둔다. 리스크를 재지 못하는 날에는 포지션을 들지 않는다.
공통 판정 게이트의 '판정불가는 통과가 아니다'와 같은 원칙이다. 워밍업이 길어지는 대신 '검사 안 한 것을
통과시킨' 자리는 없다.

[적용 순서와 룩어헤드]
t일 비중 w[t](t일 마감 후 보유)는 아래 순서로 조정된다. 모두 t일 마감까지의 정보만 쓴다.
  원 비중 → (2) 종목 예산 → (1) ES 한도 → (3) 일일손실(t일 손실이 ES[t]를 넘었으면 t+1 비중의 증가 차단)
  → (4) 낙폭(t일 마감 자산 기준). 그 결과가 t+1일 수익률에 적용된다(risk/positions.py).

[평가 — 무엇이 '성공'인가]
통제의 목적은 수익률 극대화가 아니라 "실현 손실 분포를 모델이 약속한 범위 안에 두는 것"이다. 그래서 핵심 판정은
통제 후 수익률에 대한 VaR 백테스트(risk/var_backtest.py)이고, 수익률·MDD 변화는 '통제의 비용'으로 함께 보고한다.
비용이 0일 수는 없다. 비용이 0으로 나오면 통제가 한 번도 발동하지 않았다는 뜻이다.

[한계]
  - 한도의 출처가 데이터라고 해서 미래에 맞는다는 보장은 없다. 부트스트랩은 과거 분포의 재표본이다.
  - 통제가 비중을 바꾸면 그 자체가 거래비용이다. 여기서는 비용을 반영하지 않는다(리스크 측정 모듈이라 수익률 분포만 다룸).
  - 일봉 기준이라 장중 손실에는 반응하지 못한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from risk.positions import closes_frame, portfolio_returns
from risk.var import EWMA_LAMBDA, compute_var, ewma_vol

DEFAULT_ALPHA = 0.975          # ES 신뢰수준(FRTB 관례)
BLOCK_LEN = 20                 # 블록 부트스트랩 블록 길이(거래일 ≈ 1개월). 자기상관을 보존하기 위한 관행값


@dataclass
class DerivedLimits:
    es_target: float               # 1일 ES 한도(자산 대비 손실률)
    name_budget: float             # 종목별 |w|×σ 예산
    dd_limit: float                # 낙폭 서킷브레이커 발동 수준
    alpha: float
    provenance: dict = field(default_factory=dict)   # 각 한도가 어디서 나왔는지

    def describe(self) -> str:
        p = self.provenance
        return (f"ES 한도 {self.es_target:.2%} ({p.get('es_target', '')}); "
                f"종목 예산 {self.name_budget:.2%} ({p.get('name_budget', '')}); "
                f"낙폭 한도 {self.dd_limit:.2%} ({p.get('dd_limit', '')})")


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """단순수익률 시계열의 최대낙폭(양수, 자산 대비)."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max())


def bootstrap_mdd_quantile(returns: pd.Series, horizon: int = 252, block_len: int = BLOCK_LEN,
                           n_sims: int = 2000, quantile: float = 0.95, seed: int = 0) -> tuple[float, np.ndarray]:
    """
    블록 부트스트랩으로 '길이 horizon의 경로'를 n_sims개 만들어 각 경로의 최대낙폭을 구하고 그 분위를 돌려준다.
    블록을 통째로 이어 붙여 변동성 군집(자기상관)을 어느 정도 보존한다. 반환: (분위값, 전체 MDD 배열).
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < block_len * 3:
        return float("nan"), np.array([])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]).reshape(n_sims, -1)[:, :horizon]
    paths = r[idx]                                     # (n_sims, horizon)
    eq = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    mdd = (1.0 - eq / peak).max(axis=1)
    return float(np.quantile(mdd, quantile)), mdd


def derive_limits(returns: pd.Series, alpha: float = DEFAULT_ALPHA, es_quantile: float = 0.5,
                  name_budget_frac: float = 0.5, dd_quantile: float = 0.95, window: int = 250,
                  method: str = "fhs", seed: int = 0) -> DerivedLimits:
    """
    통제 전 포트폴리오 수익률에서 세 한도를 도출한다.
      es_target  : 롤링 ES[t] 시계열의 es_quantile(기본 중앙값). "평소 수준의 꼬리 위험"을 목표로 두고, 그보다
                   위험이 커지면 노출을 줄인다는 뜻. 중앙값이라 데이터의 절반에서 축소가 발동한다 — 이것이 의도다.
      name_budget: es_target × name_budget_frac. 한 종목이 포트폴리오 꼬리 위험의 절반 이상을 혼자 들지 않게.
      dd_limit   : 블록 부트스트랩 1년 MDD 분포의 dd_quantile.
    """
    v = compute_var(returns, method, alpha, window)
    es = v["es"].dropna()
    if es.empty:
        raise ValueError(f"ES를 계산할 표본이 없습니다(window={window}, n={len(returns.dropna())})")
    es_target = float(es.quantile(es_quantile))
    name_budget = es_target * name_budget_frac
    dd_limit, mdd = bootstrap_mdd_quantile(returns, quantile=dd_quantile, seed=seed)
    prov = dict(
        es_target=f"{method} ES{alpha:.1%}[t] 시계열({len(es)}일)의 {es_quantile:.0%} 분위",
        name_budget=f"ES 한도 × {name_budget_frac:g}",
        dd_limit=(f"블록({BLOCK_LEN}일) 부트스트랩 {len(mdd)}경로 1년 MDD의 {dd_quantile:.0%} 분위"
                  if len(mdd) else "표본 부족으로 미도출"),
    )
    return DerivedLimits(es_target=es_target, name_budget=name_budget, dd_limit=dd_limit, alpha=alpha, provenance=prov)


@dataclass
class ControlResult:
    weights: pd.DataFrame            # 통제 후 비중
    raw_weights: pd.DataFrame        # 통제 전 비중(인덱스 정렬)
    returns: pd.Series               # 통제 후 포트폴리오 수익률
    raw_returns: pd.Series           # 통제 전
    log: pd.DataFrame                # 일별: es, scale_es, n_name_capped, halt_next, dd, dd_active, undefined
    limits: DerivedLimits

    def counts(self) -> dict:
        lg = self.log
        return dict(days=int(len(lg)), undefined=int(lg["undefined"].sum()),
                    es_scaled=int((lg["scale_es"] < 1 - 1e-12).sum()),
                    name_capped_days=int((lg["n_name_capped"] > 0).sum()),
                    daily_halts=int(lg["halt_next"].sum()), dd_active_days=int(lg["dd_active"].sum()))


def apply_controls(weights: pd.DataFrame, closes, limits: DerivedLimits, *, alpha: float | None = None,
                   window: int = 250, method: str = "fhs", lam: float = EWMA_LAMBDA,
                   max_scale: float = 1.0, dd_cut: float = 0.5) -> ControlResult:
    """
    원 비중에 네 한도를 순서대로 적용한다. 설명은 모듈 docstring 참고.
    ES[t]는 '통제 전' 포트폴리오 수익률로 계산한다 — 통제 후 수익률로 계산하면 통제가 자기 입력을 바꾸는
    순환이 생기고, 축소된 포트폴리오의 ES는 축소된 만큼 작아져 한도의 의미가 사라진다.
    """
    alpha = limits.alpha if alpha is None else alpha
    px = closes_frame(closes)
    cols = list(weights.columns)
    w_raw = weights.reindex(px.index).fillna(0.0).astype(float)[cols]
    r_raw = portfolio_returns(w_raw, px)
    es = compute_var(r_raw, method, alpha, window)["es"]            # ES[t]: t-1까지 정보
    sym_vol = pd.DataFrame({c: ewma_vol(px[c].pct_change(fill_method=None), lam) for c in cols})   # σ_i[t]: t-1까지 정보

    n = len(px.index)
    w_out = np.zeros((n, len(cols)))
    log = dict(es=np.full(n, np.nan), scale_es=np.ones(n), n_name_capped=np.zeros(n, dtype=int),
               halt_next=np.zeros(n, dtype=bool), dd=np.zeros(n), dd_active=np.zeros(n, dtype=bool),
               undefined=np.zeros(n, dtype=bool))
    equity, peak = 1.0, 1.0
    dd_active = False
    halt = False                   # 전날 손실이 ES를 넘어 오늘 '증가' 금지
    prev_w = np.zeros(len(cols))
    es_v, wv, rr = es.to_numpy(), w_raw.to_numpy(), r_raw.to_numpy()
    sv = sym_vol.to_numpy()
    has_dd = np.isfinite(limits.dd_limit)

    for t in range(n):
        # 통제 후 자산 경로(t일 수익률은 t-1 비중 × t 수익률 = prev_w · r_i[t]) — 낙폭 판단용
        if t > 0:
            ri = (px.iloc[t] / px.iloc[t - 1] - 1.0).to_numpy()
            if np.isfinite(ri).all() or (prev_w == 0).all():
                equity *= 1.0 + float(np.nansum(prev_w * ri))
        peak = max(peak, equity)
        dd = 1.0 - equity / peak
        log["dd"][t] = dd

        w = wv[t].copy()
        e = es_v[t]
        s = sv[t]
        if not np.isfinite(e) or not np.isfinite(s[w != 0]).all():
            log["undefined"][t] = True
            w[:] = 0.0                                              # 리스크를 못 재면 들지 않는다
        else:
            log["es"][t] = e
            # (2) 종목 예산
            risk_i = np.abs(w) * s
            over = risk_i > limits.name_budget + 1e-15
            if over.any():
                w[over] = w[over] * (limits.name_budget / risk_i[over])
                log["n_name_capped"][t] = int(over.sum())
            # (1) ES 한도
            scale = min(max_scale, limits.es_target / e) if e > 0 else 1.0
            if abs(scale - 1.0) > 1e-12:
                w = w * scale
            log["scale_es"][t] = scale
            # (3) 일일손실: 전날 손실이 ES를 넘었으면 오늘은 |비중| 증가 금지(축소·유지만)
            if halt:
                inc = np.abs(w) > np.abs(prev_w) + 1e-15
                w[inc] = prev_w[inc]
            # (4) 낙폭 서킷브레이커(히스테리시스)
            if has_dd:
                if not dd_active and dd > limits.dd_limit:
                    dd_active = True
                elif dd_active and dd < limits.dd_limit / 2:
                    dd_active = False
                if dd_active:
                    w = w * dd_cut
            log["dd_active"][t] = dd_active
        # 오늘 손실이 ES[t]를 넘었나 → 내일 증가 금지
        halt = bool(np.isfinite(e) and np.isfinite(rr[t]) and (-rr[t] > e))
        log["halt_next"][t] = halt
        w_out[t] = w
        prev_w = w

    w_ctrl = pd.DataFrame(w_out, index=px.index, columns=cols)
    r_ctrl = portfolio_returns(w_ctrl, px)
    logdf = pd.DataFrame(log, index=px.index)
    return ControlResult(weights=w_ctrl, raw_weights=w_raw, returns=r_ctrl, raw_returns=r_raw, log=logdf, limits=limits)


def compare_controlled(res: ControlResult, periods_per_year: int = 252) -> pd.DataFrame:
    """통제 전/후 요약: 연환산 수익률·변동성, MDD, 평균 총노출. 비용(수익률 감소)과 효과(변동성·MDD 감소)를 나란히."""
    rows = []
    for name, r, w in (("통제 전", res.raw_returns, res.raw_weights), ("통제 후", res.returns, res.weights)):
        rr = r.dropna()
        ann_ret = float((1 + rr).prod() ** (periods_per_year / len(rr)) - 1) if len(rr) else np.nan
        rows.append(dict(구분=name, 연환산수익률=ann_ret, 연환산변동성=float(rr.std(ddof=1) * np.sqrt(periods_per_year)),
                         MDD=max_drawdown(rr), 평균총노출=float(w.abs().sum(axis=1).mean()), 일수=int(len(rr))))
    return pd.DataFrame(rows)
