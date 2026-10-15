"""
"이 종목 지금 사야 돼, 팔아야 돼, 들고 있어야 돼?"라는 질문에 답하는 레이어.

[변경] combo_filter(candle_volume_entry_filter.pine 포팅)를 _compute_indicator_dfs에
추가했다 — evaluate_symbol()은 SIGNAL_SPECS를 순회하는 구조라 자동으로 같이 판정
대상에 포함된다. combo_* 파라미터는 getattr(cfg, ..., 원본 Pine 기본값) 경로로
읽으므로 cfg에 이 속성들이 없어도(기존 호출부) 그대로 동작한다. 그 외 하이브리드
신뢰도·다수결 로직은 바꾸지 않았다.

지금까지의 파이프라인([1]~[7]단계)은 전부 "과거에 이 신호가 통계적으로
유의미한 예측력이 있었는가"를 검증하는 데 쓰였다. 이 모듈은 그 검증 결과를
"지금 이 순간"에 적용한다: 최신 봉에서 어떤 신호가 떠 있는지 보고, 그 신호가
과거에 검증됐던 신호인지 확인한 뒤, 매수/보유/매도 중 하나로 합친다.

신뢰도 판단은 하이브리드로 한다: 그 종목 자신의 과거 이벤트 수가
min_own_events 이상이면 종목 자체 통계를 쓰고, 부족하면(신규상장주·알트코인처럼
역사가 짧은 경우) 같은 신호 유형을 여러 종목에서 풀링한 횡단면 통계
(pooled_stats, universe_scan.compute_pooled_stats()의 결과)로 대체한다 —
표본이 부족하다고 "판단 불가"로 손 놓는 대신, "이 신호 유형은 다른 종목들
에서는 대체로 어떠했는가"로 보수적으로 보강한다는 뜻이다. 다만 이건 "이
신호의 신뢰도가 종목을 넘어 전이된다"는 가정을 깔고 있다는 점은 명심할 것 —
evidence에 항상 어느 근거(own/pooled)를 썼는지 표시해서 이 가정에 기대고
있다는 걸 숨기지 않는다.

여러 신호가 동시에(때로는 충돌 방향으로) 뜰 수 있으므로, 합산은 "보수적
다수결"로 한다: 검증된(통계적으로 유의하다고 확인된) 강세 신호 개수와 검증된
약세 신호 개수를 비교해서 많은 쪽으로 판정하고, 같거나(0대0 포함) 충돌이
애매하면 무조건 보유로 떨어진다 — 애매한 상황에서 매수/매도 쪽으로 넘어가지
않는 게 이 프로젝트 전체의 설계 철학(judge_signal의 "애매하면 보류")과 같다.

이 레이어는 투자자문이 아니라 "지금까지 검증한 통계가 이 순간 이 종목에
대해 뭐라고 말하는지"를 요약해 보여주는 도구다. 근거(evidence) 없이 결론만
내지 않는다 — 항상 어떤 신호가, 어떤 방향으로, 어떤 근거(own/pooled)로, 어떤
t-stat으로 검증됐는지를 verdict과 함께 반환한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators import (institutional_displacement, delta_trading, big_sales,
                         ultimate_rsi, vp_box, momentum, liquidity, combo_filter)
from backtest.engine import SIGNAL_SPECS, compare_all
from decision.gate import evaluate_gate

MIN_OWN_EVENTS = 10
DEFAULT_T_THRESHOLD = 3.0  # decision/rule_based.py와 동일 근거(Harvey, Liu & Zhu 2016)


@dataclass
class SignalEvidence:
    label: str
    direction: int  # +1 롱(강세), -1 숏(약세)
    active_today: bool
    validated: bool
    basis: str          # "own" | "pooled" | "미검증(표본부족)"
    n_events: int
    mean_return: float
    t_stat: float
    reason: str = ""


@dataclass
class SymbolVerdict:
    symbol: str
    as_of: str
    verdict: str  # "매수" | "보유" | "매도"
    active_signals: list = field(default_factory=list)  # list[SignalEvidence] (오늘 뜬 것만)
    strength: float = 0.0  # 추천 랭킹용 — 검증된 신호들의 |t_stat| 합

    def to_row(self) -> dict:
        validated_active = [s for s in self.active_signals if s.validated]
        return dict(
            symbol=self.symbol, as_of=self.as_of, verdict=self.verdict, strength=self.strength,
            n_active_signals=len(self.active_signals),
            n_validated_signals=len(validated_active),
            signals=" | ".join(
                f"{s.label}({'+' if s.direction == 1 else '-'}, "
                f"{'검증' if s.validated else '미검증'}/{s.basis}, t={s.t_stat:.2f})"
                for s in self.active_signals
            ),
        )


def _compute_indicator_dfs(df: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    dfs = {
        "institutional_displacement": institutional_displacement.compute(
            df, vol_mode=cfg.id_vol_mode, vol_mult=cfg.id_vol_mult, z_threshold=cfg.id_z_threshold),
        "delta_trading": delta_trading.compute(df),
        "big_sales": big_sales.compute(df, length=cfg.big_sales_len),
        "ultimate_rsi": ultimate_rsi.compute(df),
        "vp_box": vp_box.compute(df, bins=cfg.vp_bins, freq=cfg.vp_freq),
        "combo_filter": combo_filter.compute(
            df,
            body_min_pct=getattr(cfg, "combo_body_min_pct", 55.0),
            wick_max_pct=getattr(cfg, "combo_wick_max_pct", 25.0),
            vol_len=getattr(cfg, "combo_vol_len", 20),
            vol_mult=getattr(cfg, "combo_vol_mult", 1.6),
            ma_len=getattr(cfg, "combo_ma_len", 50),
            ma_type=getattr(cfg, "combo_ma_type", "EMA"),
            score_threshold=getattr(cfg, "combo_score_threshold", 0.62),
        ),
    }
    if getattr(cfg, "mom_filter", False):
        mom_df = momentum.compute(df, lookback=cfg.mom_lookback, window=cfg.mom_window)
        dfs = momentum.apply_filter(dfs, mom_df, SIGNAL_SPECS, threshold=cfg.mom_threshold)
    if getattr(cfg, "liq_filter", False):
        liq_df = liquidity.compute(df, window=cfg.liq_window, percentile_window=cfg.liq_percentile_window)
        dfs = liquidity.apply_filter(dfs, liq_df, SIGNAL_SPECS, threshold=cfg.liq_threshold)
    return dfs


def own_symbol_summary(dfs: dict, cfg) -> pd.DataFrame:
    """
    한 종목의 신호 전부를 compare_all()로 '한 묶음'으로 평가한 요약표를 돌려준다.

    [왜 신호별 event_study가 아니라 compare_all인가]
    예전에는 신호마다 event_study()를 따로 불러 t-stat만 뽑고 |t| >= 임계값으로
    검증 여부를 정했다. 그러면 "이 종목에서 12개 신호를 동시에 보고 있다"는
    사실이 통계에 전혀 반영되지 않는다 — 전부 무의미해도 그중 몇 개는 우연히
    유의하게 나온다. compare_all()은 묶음 차원에서 BH(FDR) 보정과 DSR을 계산해
    주므로, 어드바이저도 알림·규칙판정과 같은 기준으로 거를 수 있게 된다.
    """
    summary, _ = compare_all(
        dfs, horizons=(cfg.target_horizon,), max_curve_horizon=cfg.target_horizon,
        entry_lag=cfg.entry_lag, entry_col=cfg.entry_col,
        cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps,
        newey_west=getattr(cfg, "newey_west", False),
        drop_overlapping=getattr(cfg, "drop_overlapping", False),
        fdr=getattr(cfg, "fdr", 0.10),
        deflate=getattr(cfg, "deflate", True),
        dsr_threshold=getattr(cfg, "dsr_threshold", 0.95),
        min_events=getattr(cfg, "dsr_min_events", 20),
        extra_trials=getattr(cfg, "dsr_extra_trials", 0),
    )
    return summary


def _row_for(summary: pd.DataFrame, label: str, horizon: int):
    """요약표에서 (신호, 보유기간) 한 행을 찾는다. 없으면 None."""
    if summary is None or summary.empty:
        return None
    hit = summary[(summary["signal"] == label) & (summary["horizon"] == horizon)]
    return None if hit.empty else hit.iloc[0]


def _stats_from_row(row) -> tuple[int, float, float, object, object]:
    """요약표 한 행에서 (n, mean, t_stat, bh, dsr)을 꺼낸다."""
    if row is None:
        return 0, float("nan"), float("nan"), None, None
    n = int(row["n_events"]) if pd.notna(row.get("n_events")) else 0
    mean = float(row["mean_return_pct"]) if pd.notna(row.get("mean_return_pct")) else float("nan")
    t_stat = float(row["t_stat"]) if pd.notna(row.get("t_stat")) else float("nan")
    bh = bool(row["significant_bh"]) if "significant_bh" in row.index and pd.notna(row["significant_bh"]) else None
    dsr = bool(row["significant_dsr"]) if "significant_dsr" in row.index and pd.notna(row["significant_dsr"]) else None
    return n, mean, t_stat, bh, dsr


def evaluate_symbol(df: pd.DataFrame, symbol: str, cfg, pooled_stats: dict | None = None) -> SymbolVerdict:
    """
    df: 그 종목의 전체 OHLCV 히스토리(가장 최근 봉까지 포함).
    cfg: 지표·필터·판정 파라미터를 담은 설정 객체(run_advisor.py의 argparse
        Namespace, 또는 그와 같은 속성을 가진 임의의 객체). run_backtest.py와
        동일한 파라미터명을 쓴다 — 그래야 "지금 뜬 신호"가 과거에 검증한 것과
        정확히 같은 정의(같은 필터 임계값)를 쓰게 된다.
    pooled_stats: {signal_label: {"n": int, "mean": float, "t_stat": float}} —
        universe_scan.compute_pooled_stats()의 결과. 종목 자체 표본이 부족할
        때(하이브리드 신뢰도 판단) 대체로 쓰인다. None이면 표본 부족 시
        "미검증"으로만 처리한다.

    반환: SymbolVerdict — 오늘(가장 최근 봉) 활성화된 신호 목록과 그 신뢰도,
    그리고 보수적 다수결로 합친 최종 매수/보유/매도.
    """
    dfs = _compute_indicator_dfs(df, cfg)
    as_of = str(df.index[-1].date()) if len(df) else "N/A"

    # 종목 자체 통계는 신호 전부를 한 묶음으로 평가해야 BH·DSR이 성립한다.
    own_summary = own_symbol_summary(dfs, cfg)

    active: list[SignalEvidence] = []
    for indicator_key, event_col, direction, label in SIGNAL_SPECS:
        ind_df = dfs.get(indicator_key)
        if ind_df is None or event_col not in ind_df.columns or len(ind_df) == 0:
            continue
        is_active_today = bool(ind_df[event_col].fillna(False).iloc[-1])
        if not is_active_today:
            continue

        n_own, mean_own, t_own, bh_own, dsr_own = _stats_from_row(
            _row_for(own_summary, label, cfg.target_horizon))

        if n_own >= cfg.min_own_events:
            n, mean, t_stat, basis = n_own, mean_own, t_own, "own"
            bh, dsr = bh_own, dsr_own
        elif pooled_stats and label in pooled_stats and pooled_stats[label].get("n", 0) > 0:
            p = pooled_stats[label]
            n, mean, t_stat, basis = p["n"], p["mean"], p["t_stat"], "pooled"
            # 풀링 통계도 유니버스 전체를 한 묶음으로 보정한 결과를 같이 받는다.
            bh, dsr = p.get("significant_bh"), p.get("significant_dsr")
        else:
            n, mean, t_stat, basis = n_own, mean_own, t_own, "미검증(표본부족)"
            bh, dsr = bh_own, dsr_own

        # [변경] 예전에는 |t| >= 임계값 하나로 validated를 정했다. 그러면 유니버스를
        # 스캔하며 종목·신호를 동시에 여럿 보는 이 모듈이 하필 다중검정 보정을
        # 전혀 하지 않는 상태가 된다. 이제 알림·규칙판정과 같은 공통 게이트를 쓴다.
        # 팩터 알파는 이 경로에서 계산하지 않으므로 해당 없음으로 둔다.
        gate = evaluate_gate(
            n_events=n, t_stat=t_stat, bh_significant=bh, dsr_significant=dsr,
            factor_available=False,
            min_n_events=cfg.min_own_events, t_threshold=cfg.t_threshold,
        )
        validated = (basis in ("own", "pooled")) and gate["verdict"] == "채택"
        if basis == "미검증(표본부족)":
            reason = f"종목 자체 이벤트 {n}개 < 최소 {cfg.min_own_events}개, 풀링 통계도 없음"
        else:
            basis_kr = "종목 자체" if basis == "own" else "동일 신호유형 풀링"
            reason = (f"{basis_kr} n={n} t={t_stat:.2f} -> {gate['verdict']}: "
                      f"{gate['reasons'][0]}")

        active.append(SignalEvidence(label=label, direction=direction, active_today=True,
                                      validated=validated, basis=basis, n_events=n,
                                      mean_return=mean, t_stat=t_stat, reason=reason))

    validated_signals = [s for s in active if s.validated]
    validated_long = sum(1 for s in validated_signals if s.direction == 1)
    validated_short = sum(1 for s in validated_signals if s.direction == -1)

    if validated_long > validated_short:
        verdict = "매수"
    elif validated_short > validated_long:
        verdict = "매도"
    else:
        verdict = "보유"  # 0대0, 동률, 혹은 애매한 충돌 -> 항상 보유(보수적 기본값)

    strength = float(sum(abs(s.t_stat) for s in validated_signals if pd.notna(s.t_stat)))

    return SymbolVerdict(symbol=symbol, as_of=as_of, verdict=verdict,
                          active_signals=active, strength=strength)
