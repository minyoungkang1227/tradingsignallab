"""
신호 채택 게이트 — 이 저장소에서 "이 신호를 믿을 것인가"를 판단하는 단일 구현.

[왜 만들었나]
같은 질문에 세 진입점이 서로 다른 답을 내고 있었다.

  decision/rule_based.py  표본수 >= 10, |t| >= 3.0, BH, 알파 |t| >= 3.0, 부호 일치
  alerts/check_signals.py BH, DSR  (t 임계값도 표본수 기준도 없음)
  advisor/evaluator.py    |t| >= t_threshold  (BH도 DSR도 알파도 없음)

세 축(t 임계값·다중검정·알파) 전부에서 기준이 엇갈려서, 어느 쪽이 더 엄격한지
줄 세울 수조차 없었다. 특히 매수/보유/매도를 직접 추천하는 advisor가 가장 느슨한
기준을 쓰면서, 하필 유니버스를 스캔해 종목 여럿을 동시에 보는 구조라 다중검정
문제가 가장 큰 자리에서 보정이 가장 약했다.

[설계]
게이트는 '층(layer)'의 순서 있는 묶음이다. 각 층은 통과/실패/해당없음/판정불가
넷 중 하나를 내고, 전체 판정은 아래 규칙으로 합친다.

  실패가 하나라도 있으면            -> 기각
  실패는 없는데 판정불가가 있으면   -> 보류
  전부 통과이거나 해당없음이면      -> 채택

'해당없음'과 '판정불가'를 구분하는 게 핵심이다. 코인은 KRX 팩터를 적용할 수 없어
알파 층이 구조적으로 '해당없음'이고(이건 흠이 아니다), 국내주식인데 팩터 표본이
모자라 알파를 못 구한 건 '판정불가'다(이건 보류 사유다). 둘을 뭉개면 코인
어드바이저가 영원히 보류로 떨어지거나, 반대로 검증 못 한 알파를 통과로 흘려보낸다.

[공통 바닥]
표본수 + 개별 t-stat + BH + DSR은 세 진입점 모두의 최소 기준이다. 알파는 팩터
데이터가 있을 때만 요구한다. t 임계값은 3.0으로 통일했다 — Harvey, Liu & Zhu(2016)가
금융 신호의 최소 기준으로 제시한 값이고, BH·DSR과는 거르는 축이 달라 대체재가
아니라 이중 방어선이다.

[한계]
이 게이트는 '통계적으로 진짜인가'만 본다. '그래서 손절 걸고 매매하면 벌 수 있나'는
다른 질문이고 backtest/economics.py(R 기대값·손익분기 승률)가 답한다. 게이트를
통과했다고 수익이 나는 게 아니다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# 층 판정값
PASS = "통과"
FAIL = "실패"
SKIP = "해당없음"
UNKNOWN = "판정불가"

# 최종 판정
ADOPT = "채택"
HOLD = "보류"
REJECT = "기각"

DEFAULT_MIN_N_EVENTS = 10
DEFAULT_T_THRESHOLD = 3.0


@dataclass
class Layer:
    name: str
    status: str
    reason: str = ""

    def __repr__(self) -> str:  # 디버깅·리포트용
        return f"{self.name}:{self.status}"


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def evaluate_gate(*, n_events, t_stat,
                   bh_significant=None, dsr_significant=None,
                   alpha=None, alpha_t_stat=None, factor_available=False,
                   mean_return=None,
                   min_n_events: int = DEFAULT_MIN_N_EVENTS,
                   t_threshold: float = DEFAULT_T_THRESHOLD,
                   require_bh: bool = True, require_dsr: bool = True) -> dict:
    """
    n_events: 이 신호의 이벤트(표본) 수.
    t_stat: 비용·체결지연을 반영한 이벤트 수익률의 t-통계량.
    bh_significant / dsr_significant: 각각 BH(FDR) 보정과 DSR을 통과했는지.
        True/False는 판정 결과, None은 '계산되지 않음'이다. require_* 가 True면
        None은 판정불가(보류)가 되고, False면 해당없음으로 건너뛴다.
        None을 '통과'로 취급하는 경로는 두지 않는다 — 검사하지 않은 것을 통과로
        흘려보내는 게 이 게이트를 만든 이유였던 결함이기 때문이다.
    alpha / alpha_t_stat: 팩터 조정 회귀 결과. factor_available=False면 둘 다 무시하고
        알파 층을 해당없음으로 둔다(코인: KRX 팩터 정의가 적용되지 않음).
    mean_return: 주면 알파와 부호가 같은지 본다(다르면 사람의 재확인이 필요해 보류).
    require_bh / require_dsr: 해당 층을 요구할지. rule_based의 기존 호출부처럼
        '넘겨주지 않았으면 건너뛴다'는 동작을 보존해야 하는 경우 False로 준다.

    반환: dict(verdict, layers, reasons, ...)
      verdict : "채택" / "보류" / "기각"
      layers  : [Layer, ...] 평가 순서대로
      reasons : 판정을 좌우한 층의 사유 문자열(기각/보류 사유가 앞에 온다)
    """
    layers: list[Layer] = []

    # --- 1) 표본수 ----------------------------------------------------------
    n = int(n_events) if n_events is not None and not _is_nan(n_events) else 0
    if n < min_n_events:
        layers.append(Layer("표본수", UNKNOWN,
                             f"표본 수 부족 (n_events={n} < {min_n_events}) — 우연일 가능성을 배제할 수 없음"))
    else:
        layers.append(Layer("표본수", PASS, f"표본 {n}건 (최소 {min_n_events}건)"))

    # --- 2) 개별 t-stat -----------------------------------------------------
    if _is_nan(t_stat):
        layers.append(Layer("개별 t-stat", UNKNOWN,
                             "원 신호의 t-stat을 계산할 수 없음 (수익률 데이터 부족)"))
    elif abs(float(t_stat)) < t_threshold:
        layers.append(Layer("개별 t-stat", FAIL,
                             f"원 신호 자체가 통계적으로 유의하지 않음 "
                             f"(|t|={abs(float(t_stat)):.2f} < {t_threshold})"))
    else:
        layers.append(Layer("개별 t-stat", PASS, f"|t|={abs(float(t_stat)):.2f} >= {t_threshold}"))

    # --- 3) BH 다중검정 보정 -------------------------------------------------
    layers.append(_binary_layer(
        "BH 다중검정", bh_significant, require_bh,
        fail_reason=(f"개별 t-stat은 유의했지만, 여러 신호를 동시에 비교한 "
                      f"Benjamini-Hochberg 다중검정 보정을 통과하지 못함 — "
                      f"우연히 유의하게 나왔을 가능성(p-hacking)"),
        unknown_reason="BH 다중검정 보정 결과가 없음 — 묶음 차원의 보정을 확인할 수 없어 판정 보류",
        pass_reason="BH(FDR) 다중검정 보정 통과",
        skip_reason="BH 보정을 요구하지 않는 호출(묶음 정보 없음)"))

    # --- 4) DSR -------------------------------------------------------------
    layers.append(_binary_layer(
        "DSR", dsr_significant, require_dsr,
        fail_reason=("디플레이티드 샤프비율(DSR) 기준 미달 — 수익률의 비정규성과 "
                      "'여러 번 시도해 고른 최고값'이라는 선택편의를 감안하면 유의하다고 볼 수 없음"),
        unknown_reason="DSR 결과가 없음 — 선택편의 보정을 확인할 수 없어 판정 보류",
        pass_reason="DSR 통과",
        skip_reason="DSR을 요구하지 않는 호출"))

    # --- 5) 팩터 조정 알파 ---------------------------------------------------
    if not factor_available:
        layers.append(Layer("팩터 알파", SKIP,
                             "팩터 데이터 없음 — 알파 검증 해당 없음(가상자산은 KRX 사이즈·가치·모멘텀 "
                             "팩터 정의가 적용되지 않아 구조적으로 해당 없음)"))
    elif _is_nan(alpha_t_stat):
        layers.append(Layer("팩터 알파", UNKNOWN,
                             "팩터 회귀 표본 부족으로 알파 유의성을 검증할 수 없음"))
    elif abs(float(alpha_t_stat)) < t_threshold:
        layers.append(Layer("팩터 알파", FAIL,
                             f"원 신호는 유의했지만 알파는 유의하지 않음"
                             f"(|alpha_t|={abs(float(alpha_t_stat)):.2f}) — "
                             f"SMB/HML/WML 팩터 프리미엄의 재포장일 가능성"))
    elif (mean_return is not None and not _is_nan(mean_return) and not _is_nan(alpha)
            and (float(mean_return) > 0) != (float(alpha) > 0)):
        layers.append(Layer("팩터 알파", UNKNOWN,
                             f"원 평균수익률(부호 {'+' if float(mean_return) > 0 else '-'})과 "
                             f"알파(부호 {'+' if float(alpha) > 0 else '-'})의 방향이 달라 "
                             f"해석에 사람의 재확인이 필요함"))
    else:
        layers.append(Layer("팩터 알파", PASS,
                             f"알파도 같은 방향으로 유의 (|alpha_t|={abs(float(alpha_t_stat)):.2f})"))

    # --- 종합 ---------------------------------------------------------------
    # 층은 전부 평가해서 보고용으로 남기되, 판정은 '순서상 처음 걸린 층'이 결정한다.
    # 모아서 보는 방식(실패 우선)이 아니라 순서 우선인 이유는 두 가지다. 첫째,
    # 사람이 읽을 때 "어디서 처음 막혔나"가 조치로 이어진다 — 표본이 모자란 신호에
    # "알파가 유의하지 않음"을 들이대는 건 쓸모가 없다. 둘째, 기존 judge_signal의
    # 분기 순서와 정확히 같아야 그쪽 동작을 바꾸지 않고 이 게이트로 옮길 수 있다.
    blocking = next((l for l in layers if l.status in (FAIL, UNKNOWN)), None)
    if blocking is None:
        verdict = ADOPT
        reasons = [" / ".join(l.reason for l in layers if l.status == PASS)]
    elif blocking.status == FAIL:
        verdict = REJECT
        reasons = [blocking.reason]
    else:
        verdict = HOLD
        reasons = [blocking.reason]

    return dict(verdict=verdict, layers=layers, reasons=reasons,
                 n_events=n, t_stat=t_stat,
                 bh_significant=bh_significant, dsr_significant=dsr_significant,
                 alpha=alpha, alpha_t_stat=alpha_t_stat,
                 factor_available=bool(factor_available))


def _binary_layer(name: str, value, required: bool, *, fail_reason: str,
                   unknown_reason: str, pass_reason: str, skip_reason: str) -> Layer:
    """True/False/None 하나짜리 층의 공통 처리."""
    if value is None:
        return Layer(name, UNKNOWN if required else SKIP,
                      unknown_reason if required else skip_reason)
    return Layer(name, PASS if bool(value) else FAIL,
                  pass_reason if bool(value) else fail_reason)


def format_layers(result: dict) -> str:
    """사람이 읽는 한 줄 요약 — 어느 층에서 걸렸는지 그대로 보여준다."""
    return " | ".join(f"{l.name} {l.status}" for l in result["layers"])
