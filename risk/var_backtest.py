"""
VaR 백테스트 — 리스크 모델이 맞았는지를 '손실이 VaR를 넘은 날'의 횟수와 패턴으로 검정한다.

이 저장소에서 신호의 진위를 t-stat·BH·DSR로 판정하듯, 리스크 모델의 진위는 여기서 판정한다.
VaR를 계산만 하고 백테스트하지 않는 것은 신호를 백테스트 없이 쓰는 것과 같다.

[검정 세 가지 — 서로 다른 질문]
  1. Kupiec(1995) POF  : 초과 횟수 x가 기대치 n(1-alpha)와 맞는가. "너무 많이 넘었나(과소추정)",
                         "너무 적게 넘었나(과대추정 — 자본을 놀린다)" 둘 다 잡는 양측 검정.
                         LR_uc = -2 ln[ (1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x) ] ~ χ²(1), p = 1-alpha.
  2. Christoffersen(1998) 독립성 : 초과가 몰려서 오는가. 초과일 다음 날 또 초과할 확률 π11이 평상시
                         초과 확률 π01과 같은가. 변동성 군집을 못 따라가는 모델(롤링 역사적)은 횟수는 맞아도
                         여기서 떨어진다. LR_ind ~ χ²(1).
  3. 조건부 커버리지    : LR_cc = LR_uc + LR_ind ~ χ²(2). 둘을 합친 판정.
  4. 바젤 신호등        : 250일·99%에서 초과 0~4회 녹색, 5~9회 황색(승수 가산), 10회 이상 적색. 규제 관행이며
                         n=250, alpha=0.99 가 아니면 '해당없음'.

[판정 규칙]
  p < 0.05 이면 그 검정은 '기각'. 하나라도 기각이면 모델 판정은 '부적합'. 전부 통과면 '적합'. 표본이
  모자라 검정이 정의되지 않으면(초과 0회 또는 전부 초과, 전이 셀 비어 있음) '판정불가'.
  → '적합'은 "이 표본에서 틀렸다는 증거를 못 찾았다"는 뜻이지 모델이 맞다는 증명이 아니다.

[한계]
  - 검정력이 약하다. 250일에서 99% VaR는 기대 초과가 2.5회라 x=5(두 배)도 Kupiec p≈0.13으로 통과한다.
    표본이 길어야 하고, 97.5%·95%처럼 낮은 신뢰수준은 검정력이 좋으니 함께 보는 것을 권한다.
  - 초과 여부만 쓰고 초과 크기는 안 본다. 크기는 ES 백테스트(미구현)의 영역이다.
  - 같은 표본으로 모델 파라미터(window, λ)를 고르고 백테스트하면 표본 내다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.stats import chi2_sf

PASS, FAIL, UNKNOWN, SKIP = "통과", "기각", "판정불가", "해당없음"
FIT, UNFIT = "적합", "부적합"


@dataclass
class VarBacktestResult:
    n: int                                  # 비교에 쓴 날 수(VaR·손실 둘 다 있는 날)
    n_exceed: int
    expected_exceed: float
    exceed_rate: float
    alpha: float
    kupiec: dict                            # lr, p, status
    independence: dict                      # lr, p, status, n00, n01, n10, n11
    conditional: dict                       # lr, p, status
    basel: dict                             # zone, status
    exceed_dates: list = field(default_factory=list)
    verdict: str = UNKNOWN

    def summary_row(self) -> dict:
        return dict(n=self.n, n_exceed=self.n_exceed, expected=round(self.expected_exceed, 2),
                    rate=self.exceed_rate, kupiec_p=self.kupiec["p"], indep_p=self.independence["p"],
                    cc_p=self.conditional["p"], basel=self.basel["zone"], verdict=self.verdict)


def _xlogy(x: float, y: float) -> float:
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(n: int, x: int, p: float) -> dict:
    """비율 검정. x=0 또는 x=n이면 MLE가 경계에 있어 LR이 정의되지만 해석이 어려워 x=0은 별도 처리."""
    if n <= 0:
        return dict(lr=np.nan, p=np.nan, status=UNKNOWN)
    pi = x / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - pi) + _xlogy(x, pi)
    lr = -2.0 * (ll_null - ll_alt)
    pval = chi2_sf(lr, 1)
    return dict(lr=lr, p=pval, status=(FAIL if pval < 0.05 else PASS))


def christoffersen_independence(hits: np.ndarray) -> dict:
    """1차 마코프 전이 기반 독립성 검정. hits: 0/1 배열(시간순)."""
    h = np.asarray(hits, dtype=int)
    if len(h) < 2:
        return dict(lr=np.nan, p=np.nan, status=UNKNOWN, n00=0, n01=0, n10=0, n11=0)
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    cells = dict(n00=n00, n01=n01, n10=n10, n11=n11)
    if (n01 + n11) == 0 or (n00 + n10) == 0 or (n10 + n11) == 0:
        # 초과가 없거나, 초과 뒤의 전이가 하나도 없으면 π11을 추정할 수 없다
        return dict(lr=np.nan, p=np.nan, status=UNKNOWN, **cells)
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = -2.0 * (ll_null - ll_alt)
    pval = chi2_sf(lr, 1)
    return dict(lr=lr, p=pval, status=(FAIL if pval < 0.05 else PASS), **cells)


def basel_traffic_light(n: int, x: int, alpha: float) -> dict:
    """바젤 신호등. 250일·99% 전용. 다른 설정은 해당없음."""
    if n != 250 or abs(alpha - 0.99) > 1e-9:
        return dict(zone="해당없음", status=SKIP, note="n=250, alpha=0.99 에서만 정의")
    if x <= 4:
        return dict(zone="녹색", status=PASS, note="초과 0~4회")
    if x <= 9:
        return dict(zone="황색", status=PASS, note="초과 5~9회 — 자본 승수 가산 구간")
    return dict(zone="적색", status=FAIL, note="초과 10회 이상 — 모델 거부")


def var_backtest(returns: pd.Series, var: pd.Series, alpha: float = 0.99) -> VarBacktestResult:
    """
    returns[t]의 손실 -returns[t] 가 var[t]를 넘은 날을 세어 세 검정을 돌린다. var[t]는 t-1까지의
    정보로 만든 t일 예측이어야 한다(risk/var.py가 그렇게 만든다). 둘 다 값이 있는 날만 쓴다.
    """
    df = pd.DataFrame({"loss": -returns.astype(float), "var": var.astype(float)}).dropna()
    n = len(df)
    hits = (df["loss"] > df["var"]).to_numpy().astype(int)
    x = int(hits.sum())
    p = 1.0 - alpha
    kup = kupiec_pof(n, x, p)
    ind = christoffersen_independence(hits)
    if np.isfinite(kup["lr"]) and np.isfinite(ind["lr"]):
        lr_cc = kup["lr"] + ind["lr"]
        p_cc = chi2_sf(lr_cc, 2)
        cc = dict(lr=lr_cc, p=p_cc, status=(FAIL if p_cc < 0.05 else PASS))
    else:
        cc = dict(lr=np.nan, p=np.nan, status=UNKNOWN)
    bas = basel_traffic_light(n, x, alpha)
    statuses = [kup["status"], ind["status"], cc["status"], bas["status"]]
    if n == 0:
        verdict = UNKNOWN
    elif FAIL in statuses:
        verdict = UNFIT
    elif UNKNOWN in statuses:
        verdict = UNKNOWN
    else:
        verdict = FIT
    return VarBacktestResult(n=n, n_exceed=x, expected_exceed=n * p, exceed_rate=(x / n if n else np.nan),
                             alpha=alpha, kupiec=kup, independence=ind, conditional=cc, basel=bas,
                             exceed_dates=list(df.index[hits == 1]), verdict=verdict)


def format_var_backtest(res: VarBacktestResult, method: str = "") -> str:
    tag = f" [{method}]" if method else ""
    L = [f"=== VaR 백테스트{tag} (신뢰수준 {res.alpha:.1%}) ==="]
    L.append(f"비교 일수 {res.n}, 초과 {res.n_exceed}회 (기대 {res.expected_exceed:.1f}회, "
             f"초과율 {res.exceed_rate:.2%} vs 목표 {1 - res.alpha:.2%})")
    k, i, c, b = res.kupiec, res.independence, res.conditional, res.basel
    L.append(f"  Kupiec 비율       LR={k['lr']:.3f} p={k['p']:.3f} -> {k['status']}" if np.isfinite(k["lr"])
             else f"  Kupiec 비율       -> {k['status']}")
    L.append(f"  Christoffersen 독립성 LR={i['lr']:.3f} p={i['p']:.3f} -> {i['status']} "
             f"(전이 00={i['n00']} 01={i['n01']} 10={i['n10']} 11={i['n11']})" if np.isfinite(i["lr"])
             else f"  Christoffersen 독립성 -> {i['status']} (전이 00={i['n00']} 01={i['n01']} 10={i['n10']} 11={i['n11']})")
    L.append(f"  조건부 커버리지   LR={c['lr']:.3f} p={c['p']:.3f} -> {c['status']}" if np.isfinite(c["lr"])
             else f"  조건부 커버리지   -> {c['status']}")
    L.append(f"  바젤 신호등       {b['zone']} ({b['note']})")
    L.append(f"판정: {res.verdict} — '적합'은 이 표본에서 틀렸다는 증거를 못 찾았다는 뜻이지 모델이 맞다는 증명이 아님")
    return "\n".join(L)
