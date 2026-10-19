"""
파이프라인 7단계 — 판단 레이어 (규칙 기반).

지금까지의 파이프라인(1~6단계)은 전부 "이 신호가 통계적으로 유의미한가"만
다뤘다. 이 모듈은 그 결과를 받아서 "그래서 이 신호를 채택할지 말지"를
결정론적 규칙으로 판정하는 마지막 단계다.

TradingAgents(Xiao et al. 2024)류 LLM 멀티에이전트 판단 구조를 참고했지만,
Xia et al.(2026) 서베이가 지적한 재현성·look-ahead bias 문제를 피하기 위해
지금 단계에서는 LLM 호출 없이 순수 규칙(if/else)으로만 구현했다 — 같은 입력을
넣으면 항상 같은 판정이 나오고, API 비용·지연시간·비결정성이 전혀 없다.
나중에 LLM 기반 판단(예: bull/bear 논거 생성)을 추가하고 싶다면, 이 규칙
기반 판정을 "베이스라인"으로 두고 LLM 판단과 결과를 나란히 비교하는 방식을
권한다(하나로 교체하지 말 것 — 재현 가능한 대조군이 있어야 LLM이 실제로
도움이 되는지 검증 가능하다).

판정 기준 (전부 README에 이미 명시했던 경험칙을 그대로 코드화)
------------------------------------------------------------
1. n_events < min_n_events(기본 10) → "보류" (표본 부족, 우연일 가능성 배제 못 함)
2. |raw_t_stat| < t_threshold(기본 3.0) → "기각" (원 신호 자체가 유의하지 않음)
3. bh_significant가 False로 명시됨 → "기각" (개별 t-stat은 유의해도, 여러 지표·
   방향·보유기간을 동시에 비교한 Benjamini-Hochberg 다중검정 보정을 통과하지
   못함 — p-hacking으로 우연히 유의하게 나왔을 가능성)
4. alpha_t_stat이 NaN → "보류" (팩터 회귀 표본 부족으로 알파 검증 불가)
5. |alpha_t_stat| < t_threshold → "기각" (원 신호는 유의했지만 알파는 유의하지 않음
   — 팩터 프리미엄 재포장 가능성, 지난번 논의한 "누락변수편향" 문제)
6. 원 수익률과 알파의 부호가 다름 → "보류" (방향성 해석이 애매해 사람이 재확인 필요)
7. 위 전부 통과 → "채택" (팩터로 설명 안 되는 예측력으로 판단)

t_threshold 기본값 근거 (2.0 -> 3.0으로 상향)
------------------------------------------------------------
Harvey, Liu & Zhu(2016, "...and the Cross-Section of Expected Returns")는
학계가 지금까지 제시한 수백 개의 팩터 대부분이 다중 비교(multiple testing)
상황에서 우연히 유의하게 나온 것(p-hacking)이라는 걸 보였고, 단일 가설
검정의 관행적 기준(|t|>=2.0, 유의수준 5%)을 다중 비교 상황에 그대로 쓰면
가짜 신호를 걸러내지 못한다며 |t|>=3.0을 최소 기준으로 제안했다. 이 저장소는
5개 지표 x 롱/숏 x 여러 보유기간을 동시에 검증하는 전형적인 다중 비교
상황이라 이 권고를 그대로 받아들여 t_threshold 기본값을 3.0으로 올렸다.
Benjamini-Hochberg(compare_all()의 significant_bh)가 "사후에" 다중검정을
보정하는 것과 달리, 이건 "사전에" 개별 유의성 기준 자체를 보수적으로 잡는
것이라 서로 대체재가 아니라 이중 방어선에 가깝다. 필요하면 t_threshold를
직접 지정해 기존 기준(2.0)으로 되돌릴 수 있다.
"""
from __future__ import annotations
import pandas as pd


def judge_signal(n_events: int, raw_mean_return: float, raw_t_stat: float,
                  alpha: float, alpha_t_stat: float, r_squared: float,
                  min_n_events: int = 10, t_threshold: float = 3.0,
                  bh_significant: bool | None = None,
                  dsr_significant: bool | None = None) -> dict:
    """
    순수 판정 함수 — 이미 계산된 통계치만 받아서 규칙을 적용한다(직접 호출해도 되고,
    아래 judge_indicator()가 이벤트 스터디·팩터 회귀까지 한 번에 해서 이 함수에 넘겨준다).

    bh_significant: backtest.engine.compare_all()이 계산하는 `significant_bh`
        (다중 검정 보정 후에도 유의한지)를 그대로 넘겨받는 파라미터. 이 함수는
        지표 하나만 보고 판정하므로 그 자체로는 "여러 지표를 동시에 비교했을 때"의
        다중검정 문제를 알 수 없다 — 그래서 compare_all()이 전체 요약표 차원에서
        계산한 결과를 호출부가 여기로 전달해줘야 반영된다. 기본 None(모른다는
        뜻)이면 이 검사를 건너뛰고 개별 |raw_t_stat| 기준으로만 판정한다(이전
        동작과 동일). False로 명시되면, 개별 t-stat이 유의해도 "기각"으로 판정한다.

    반환: dict(verdict, reasons, n_events, raw_t_stat, alpha, alpha_t_stat, r_squared,
              bh_significant)
    verdict은 "채택" / "보류" / "기각" 중 하나.
    """
    # 판정 규칙 자체는 decision/gate.py 한 곳에 있다. 이 함수는 그 게이트에
    # "팩터 데이터가 있는 국내주식 경로"의 설정으로 질의하는 얇은 래퍼다.
    # bh_significant=None이면 건너뛰는 기존 동작을 보존하려고 require_bh를
    # None 여부로 정하고, DSR은 이 함수의 호출부가 아직 넘기지 않으므로
    # 기본 None + require_dsr=False로 둔다(넘기면 그때부터 반영된다).
    from decision.gate import evaluate_gate

    res = evaluate_gate(
        n_events=n_events, t_stat=raw_t_stat,
        bh_significant=bh_significant, require_bh=bh_significant is not None,
        dsr_significant=dsr_significant, require_dsr=dsr_significant is not None,
        alpha=alpha, alpha_t_stat=alpha_t_stat, factor_available=True,
        mean_return=raw_mean_return,
        min_n_events=min_n_events, t_threshold=t_threshold,
    )
    verdict = res["verdict"]
    reasons = list(res["reasons"])
    if verdict == "채택":
        # 채택 사유 문구는 이 함수가 원래 쓰던 형태를 유지한다(리포트·테스트 호환).
        reasons = [
            f"원 신호 유의(t={raw_t_stat:.2f}), 알파도 같은 방향으로 유의"
            f"(|alpha_t|={abs(alpha_t_stat):.2f})"
            + (", 다중검정 보정도 통과" if bh_significant else "")
            + " — 팩터로 설명되지 않는 예측력으로 판단"
        ]
    layers = res["layers"]

    return dict(
        verdict=verdict, reasons=reasons, n_events=n_events,
        raw_mean_return=raw_mean_return, raw_t_stat=raw_t_stat,
        alpha=alpha, alpha_t_stat=alpha_t_stat, r_squared=r_squared,
        bh_significant=bh_significant, dsr_significant=dsr_significant,
        layers=layers,
    )


def judge_indicator(df: pd.DataFrame, event_col: str, direction: int, factor_df: pd.DataFrame,
                     horizon: int = 20, close_col: str = "close",
                     factor_cols: tuple[str, ...] = ("SMB", "HML", "WML"),
                     min_n_events: int = 10, t_threshold: float = 3.0,
                     entry_lag: int = 0, entry_col: str | None = None,
                     cost_bps: float = 0.0, slippage_bps: float = 0.0,
                     bh_significant: bool | None = None,
                  dsr_significant: bool | None = None) -> dict:
    """
    5개 지표 중 하나의 이벤트 컬럼에 대해, [3단계 이벤트 스터디] + [5단계 팩터
    조정 알파]를 전부 계산한 뒤 judge_signal()로 최종 판정까지 한 번에 낸다.
    파이프라인 3~7단계를 한 함수 호출로 잇는 통합 진입점.

    df, event_col, direction, horizon, close_col, factor_cols: event_study()/
    event_alpha()와 동일한 의미. factor_df: factors.kr_fama_french.combine_factors()
    반환값(월별 SMB/HML/WML 시계열). 모멘텀 필터를 이미 [2]단계에서 적용한
    신호라면 factor_cols에서 WML을 빼는 것을 권장한다(event_alpha() 참고).
    entry_lag, entry_col, cost_bps, slippage_bps: event_study()·event_alpha()에
        동일하게 전달된다(전부 같은 체결 가정을 써야 [3]단계 원 신호와 [5]단계
        알파가 같은 기준으로 비교됨). 기본값은 이전 버전과 동일(당일 종가 즉시
        진입, 비용·슬리피지 0).
    bh_significant: judge_signal()에 그대로 전달. 이 함수는 지표 하나만 처리하므로
        "여러 지표를 동시에 비교했을 때"의 다중검정 보정(Benjamini-Hochberg)은 이
        함수 내부에서 계산할 수 없다 — backtest.engine.compare_all()이 전체
        요약표에서 계산한 significant_bh 값을 호출부(run_backtest.py 등)가 이
        지표·방향·보유기간에 해당하는 행에서 찾아 넘겨줘야 한다. 기본 None이면
        다중검정 보정 없이 이전 동작과 동일하게 판정한다.
    """
    from backtest.engine import event_study, event_alpha

    res = event_study(df, event_col, direction=direction, horizons=(horizon,),
                       max_curve_horizon=horizon, entry_lag=entry_lag,
                       entry_col=entry_col, cost_bps=cost_bps, slippage_bps=slippage_bps)
    row = res.table.loc[horizon]
    n_events = int(row["n"]) if pd.notna(row["n"]) else 0
    raw_mean = float(row["mean"]) if pd.notna(row["mean"]) else float("nan")
    raw_t = float(row["t_stat"]) if pd.notna(row["t_stat"]) else float("nan")

    factor_result = event_alpha(df, event_col, factor_df, direction=direction,
                                 horizon=horizon, close_col=close_col, factor_cols=factor_cols,
                                 entry_lag=entry_lag, entry_col=entry_col,
                                 cost_bps=cost_bps, slippage_bps=slippage_bps)

    return judge_signal(
        n_events=n_events, raw_mean_return=raw_mean, raw_t_stat=raw_t,
        alpha=factor_result["alpha"], alpha_t_stat=factor_result["alpha_t_stat"],
        r_squared=factor_result["r_squared"],
        min_n_events=min_n_events, t_threshold=t_threshold,
        bh_significant=bh_significant, dsr_significant=dsr_significant,
    )
