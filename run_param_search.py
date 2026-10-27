#!/usr/bin/env python3
"""
mom_threshold/liq_threshold를 "여러 종목의 IS(In-Sample) 구간 데이터"에서 좌표별
순차탐색(Anchor Search)으로 찾고, Plateau(강인한 구간) 여부를 판정하고, 시도
횟수(K)를 반영한 임계값(Bonferroni / sqrt(2*ln K))을 통과하는지 확인한 뒤,
탐색 과정이 전혀 들여다보지 않은 OOS(Out-of-Sample) 구간에서 그대로 재현되는지
검증하는 파이프라인.

지금까지 run_backtest.py의 --mom-threshold/--liq-threshold는 연구자가 논문
근거로 고정한 상수였다. 이 스크립트는 그 값을 데이터에서 "찾되", 파라미터
탐색이 만드는 과최적화 위험(문제는 세 가지 — research/param_search.py 모듈
docstring 참고)을 각각 완화하는 장치를 전부 거친 뒤에만 채택한다.

사용 예:
  python run_param_search.py --market krx --symbols 005930,000660,035420 \\
    --start 20180101 --end 20260901 --split-date 20240101 \\
    --target-signal "델타트레이딩-골든크로스" --target-horizon 5

주의: 이 스크립트가 결정하는 건 mom_threshold/liq_threshold 두 값뿐이다.
--target-signal이 momentum/liquidity 필터의 영향을 받지 않는 지표(예: VP Box는
momentum/liquidity 필터 대상이지만 필터 자체가 모든 지표에 동일하게 걸린다는
점에 유의)라도 동일한 절차로 돌아간다 — 필터는 지표와 무관하게 이벤트 컬럼에
AND 마스크를 적용하는 구조이기 때문이다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from indicators import (institutional_displacement, delta_trading, big_sales,
                         ultimate_rsi, vp_box, momentum, liquidity, combo_filter)
from backtest.engine import SIGNAL_SPECS, event_returns, cluster_robust_t_stat
from research.param_search import (
    sequential_anchor_search, bonferroni_t_threshold, deflated_t_threshold_sqrt2lnk,
    pool_event_returns, split_is_oos,
)


def _find_signal_spec(target_signal: str):
    for spec in SIGNAL_SPECS:
        if spec[3] == target_signal:
            return spec
    labels = ", ".join(s[3] for s in SIGNAL_SPECS)
    sys.exit(f"--target-signal '{target_signal}'을 찾을 수 없습니다. 가능한 값: {labels}")


def _compute_target_indicator(df: pd.DataFrame, indicator_key: str, args) -> pd.DataFrame:
    """SIGNAL_SPECS의 indicator_key 하나에 대해서만 지표를 계산한다(탐색 루프
    안에서 필요 없는 지표까지 계산하는 낭비를 피하기 위해 target-signal에
    해당하는 것만 계산)."""
    if indicator_key == "institutional_displacement":
        return institutional_displacement.compute(
            df, vol_mode=args.id_vol_mode, vol_mult=args.id_vol_mult,
            z_threshold=args.id_z_threshold)
    if indicator_key == "delta_trading":
        return delta_trading.compute(df)
    if indicator_key == "big_sales":
        return big_sales.compute(df, length=args.big_sales_len)
    if indicator_key == "ultimate_rsi":
        return ultimate_rsi.compute(df)
    if indicator_key == "vp_box":
        return vp_box.compute(df, bins=args.vp_bins, freq=args.vp_freq)
    if indicator_key == "combo_filter":
        return combo_filter.compute(
            df, body_min_pct=args.combo_body_min_pct, wick_max_pct=args.combo_wick_max_pct,
            vol_len=args.combo_vol_len, vol_mult=args.combo_vol_mult,
            ma_len=args.combo_ma_len, ma_type=args.combo_ma_type,
            score_threshold=args.combo_score_threshold)
    raise ValueError(f"알 수 없는 indicator_key: {indicator_key}")


class SymbolContext:
    """종목 하나당 한 번만 계산해두는 것들(지표 원신호, 모멘텀/유동성 퍼센타일).
    탐색 루프가 매 시도(trial)마다 다시 계산하지 않도록 캐싱하는 역할."""

    def __init__(self, symbol: str, df: pd.DataFrame, indicator_key: str,
                 event_col: str, direction: int, args):
        self.symbol = symbol
        self.indicator_key = indicator_key
        self.event_col = event_col
        self.direction = direction
        self.indicator_df = _compute_target_indicator(df, indicator_key, args)
        self.mom_df = momentum.compute(df, lookback=args.mom_lookback, window=args.mom_window)
        self.liq_df = liquidity.compute(df, window=args.liq_window,
                                         percentile_window=args.liq_percentile_window)

    def event_returns_full_history(self, mom_threshold: float, liq_threshold: float,
                                    horizon: int, args) -> pd.Series:
        """trial 파라미터(mom_threshold, liq_threshold)로 필터를 적용한 뒤,
        전체 기간(IS+OOS)에 대한 이벤트별 순방향수익률을 반환한다. IS/OOS로
        자르는 건 이 함수 밖(split_is_oos)에서 한다 — 그래야 지표/필터 계산은
        항상 전체 히스토리 기준으로 한 번만 하고, 날짜 절단만 매번 값싸게
        반복할 수 있다."""
        spec = [(self.indicator_key, self.event_col, self.direction, "target")]
        filtered = {self.indicator_key: self.indicator_df}
        filtered = momentum.apply_filter(filtered, self.mom_df, spec, threshold=mom_threshold)
        filtered = liquidity.apply_filter(filtered, self.liq_df, spec, threshold=liq_threshold)
        return event_returns(filtered[self.indicator_key], self.event_col, horizon=horizon,
                              direction=self.direction, entry_lag=args.entry_lag,
                              entry_col=args.entry_col, cost_bps=args.cost_bps,
                              slippage_bps=args.slippage_bps)


def _t_passes(t_stat: float, threshold: float) -> bool:
    """부호를 본다: 수익률에 direction이 곱해져 있어 t>=+임계만 '신호 방향으로 유의'다.
    abs()를 쓰면 유의하게 손실 나는 조합도 통과한다."""
    return bool(pd.notna(t_stat) and t_stat >= threshold)


def _pooled_t_stat(contexts: list[SymbolContext], mom_threshold: float, liq_threshold: float,
                    horizon: int, split_date, period: str, args) -> tuple[float, float, int]:
    """period='is' 또는 'oos'. 풀링된 (mean, t_stat, n_events)를 반환한다."""
    per_symbol = {}
    for ctx in contexts:
        full = ctx.event_returns_full_history(mom_threshold, liq_threshold, horizon, args)
        is_part, oos_part = split_is_oos(full, split_date)
        per_symbol[ctx.symbol] = is_part if period == "is" else oos_part

    vals, cluster_ids = pool_event_returns(per_symbol)
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    mean, t_stat = cluster_robust_t_stat(vals, cluster_ids)
    return mean, t_stat, len(vals)


def run_search(dfs_by_symbol: dict[str, pd.DataFrame], args) -> dict:
    """
    dfs_by_symbol: {"005930": ohlcv_df, "000660": ohlcv_df, ...} — 각 df는 전체
    기간(IS+OOS 둘 다 포함)의 OHLCV. 실제 CLI(main())는 pykrx/업비트에서 이걸
    받아오고, 테스트는 합성 데이터를 직접 넣어 이 함수를 호출한다(네트워크 없이
    검증하기 위함 — 이 프로젝트의 다른 스모크 테스트들과 동일한 패턴).

    반환: dict(
        target_signal, target_horizon, split_date,
        final_params, per_param(plateau 진단),
        n_evaluations(K, +1은 최종 재평가),
        bonferroni_threshold, deflated_threshold,
        is_mean, is_t_stat, is_n_events, is_pass,
        oos_mean, oos_t_stat, oos_n_events, oos_pass,
    )
    """
    indicator_key, event_col, direction, label = _find_signal_spec(args.target_signal)

    contexts = [
        SymbolContext(symbol, df, indicator_key, event_col, direction, args)
        for symbol, df in dfs_by_symbol.items()
    ]
    if not contexts:
        sys.exit("종목 데이터가 하나도 없습니다.")

    def objective_is(params: dict) -> float:
        _, t_stat, _ = _pooled_t_stat(contexts, params["mom_threshold"], params["liq_threshold"],
                                       args.target_horizon, args.split_date, "is", args)
        return t_stat

    param_grid = {
        "mom_threshold": [float(x) for x in args.mom_threshold_grid.split(",")],
        "liq_threshold": [float(x) for x in args.liq_threshold_grid.split(",")],
    }
    defaults = {"mom_threshold": args.mom_threshold_default, "liq_threshold": args.liq_threshold_default}

    search = sequential_anchor_search(param_grid, defaults, objective_is,
                                       top_frac=args.top_frac)
    final_params = search["final_params"]

    # 최종 확정 조합의 IS 지표를 한 번 더(K에 +1) 평가 — plateau의 중앙값이
    # 원래 격자 후보와 정확히 일치하지 않을 수도 있어(짝수 개 묶음이면 평균),
    # 탐색 중 캐시된 값을 재사용하지 않고 다시 정직하게 계산한다.
    is_mean, is_t_stat, is_n = _pooled_t_stat(contexts, final_params["mom_threshold"],
                                               final_params["liq_threshold"], args.target_horizon,
                                               args.split_date, "is", args)
    k_trials = search["n_evaluations"] + 1

    bonferroni_threshold = bonferroni_t_threshold(k_trials, alpha=args.alpha)
    deflated_threshold = deflated_t_threshold_sqrt2lnk(k_trials)
    required_threshold = max(bonferroni_threshold, deflated_threshold)
    is_pass = _t_passes(is_t_stat, required_threshold)

    oos_mean, oos_t_stat, oos_n = _pooled_t_stat(contexts, final_params["mom_threshold"],
                                                  final_params["liq_threshold"], args.target_horizon,
                                                  args.split_date, "oos", args)
    oos_pass = _t_passes(oos_t_stat, 2.0)

    return dict(
        target_signal=label, target_horizon=args.target_horizon, split_date=str(args.split_date),
        final_params=final_params, per_param=search["per_param"], n_evaluations=k_trials,
        bonferroni_threshold=bonferroni_threshold, deflated_threshold=deflated_threshold,
        is_mean=is_mean, is_t_stat=is_t_stat, is_n_events=is_n, is_pass=is_pass,
        oos_mean=oos_mean, oos_t_stat=oos_t_stat, oos_n_events=oos_n, oos_pass=oos_pass,
    )


def print_report(result: dict):
    print(f"\n=== IS/OOS 파라미터 탐색 결과: {result['target_signal']} "
          f"(보유기간 {result['target_horizon']}봉, 분할일 {result['split_date']}) ===")
    for name, plateau in result["per_param"].items():
        print(f"  [{name}] {plateau.reason}")
    print(f"  최종 파라미터: {result['final_params']}")
    print(f"  시도 횟수(K)={result['n_evaluations']} -> "
          f"Bonferroni 임계 t=+{result['bonferroni_threshold']:.3f}, "
          f"sqrt(2lnK) 임계 t=+{result['deflated_threshold']:.3f}")
    print(f"  IS : n={result['is_n_events']} mean={result['is_mean']:.4%} "
          f"t={result['is_t_stat']:.3f} -> {'통과' if result['is_pass'] else '기각'}"
          f"(임계 {max(result['bonferroni_threshold'], result['deflated_threshold']):.3f} 기준)")
    print(f"  OOS: n={result['oos_n_events']} mean={result['oos_mean']:.4%} "
          f"t={result['oos_t_stat']:.3f} -> {'재현됨(t>=+2)' if result['oos_pass'] else '재현 안 됨'}")
    if result["is_pass"] and result["oos_pass"]:
        print("  => IS에서 통과하고 OOS에서도 재현됨: 이 파라미터 조합을 강인하다고 볼 근거가 있음")
    elif result["is_pass"] and not result["oos_pass"]:
        print("  => IS에서는 통과했지만 OOS에서 재현되지 않음: 여전히 과최적화 가능성을 배제할 수 없음")
    else:
        print("  => IS 단계부터 시도 횟수를 반영한 기준을 통과하지 못함: 이 신호에 대해 "
              "mom_threshold/liq_threshold를 데이터에서 찾으려는 시도 자체를 접는 것을 권장")


def _fetch_one_symbol(symbol: str, args) -> pd.DataFrame:
    """단일 종목 수집 — market/args에 따라 적절한 data.* fetch 함수로 위임만
    한다(수집 계층은 data/collect.py, data/krx.py, data/upbit.py). 실패는
    예외로 전파해 collect_universe()가 기록·집계하게 둔다."""
    from data.upbit import fetch_upbit_ohlcv
    from data.krx import fetch_krx_ohlcv

    if args.market == "upbit":
        return fetch_upbit_ohlcv(market=symbol, unit=args.unit, count=args.count,
                                  minute_unit=args.minute_unit)
    if not args.start or not args.end:
        sys.exit("--market krx 에는 --start / --end (YYYYMMDD)가 필요합니다.")
    return fetch_krx_ohlcv(symbol, args.start, args.end)


def _load_symbols(args) -> dict[str, pd.DataFrame]:
    from data.collect import collect_universe

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    result = collect_universe(symbols, lambda symbol: _fetch_one_symbol(symbol, args),
                               quality_gate=not getattr(args, "skip_quality_gate", False))
    return result.ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", choices=["upbit", "krx"], required=True)
    p.add_argument("--symbols", required=True, help="콤마로 구분한 종목코드 목록(예: 005930,000660)")
    p.add_argument("--unit", default="day", choices=["day", "week", "minute"])
    p.add_argument("--minute-unit", dest="minute_unit", type=int, default=60)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--start", help="krx 시작일 YYYYMMDD")
    p.add_argument("--end", help="krx 종료일 YYYYMMDD")
    p.add_argument("--split-date", dest="split_date", required=True,
                    help="IS(<=)/OOS(>) 분할 기준일. YYYYMMDD 또는 YYYY-MM-DD")
    p.add_argument("--target-signal", dest="target_signal", required=True,
                    help="탐색 대상 신호 라벨(예: '델타트레이딩-골든크로스'). "
                         "backtest.engine.SIGNAL_SPECS의 label과 정확히 일치해야 함")
    p.add_argument("--target-horizon", dest="target_horizon", type=int, default=5)
    p.add_argument("--mom-lookback", dest="mom_lookback", type=int, default=126)
    p.add_argument("--mom-window", dest="mom_window", type=int, default=252)
    p.add_argument("--mom-threshold-grid", dest="mom_threshold_grid",
                    default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--mom-threshold-default", dest="mom_threshold_default", type=float, default=0.7)
    p.add_argument("--liq-window", dest="liq_window", type=int, default=20)
    p.add_argument("--liq-percentile-window", dest="liq_percentile_window", type=int, default=252)
    p.add_argument("--liq-threshold-grid", dest="liq_threshold_grid",
                    default="0.8,0.85,0.9,0.95,0.99")
    p.add_argument("--liq-threshold-default", dest="liq_threshold_default", type=float, default=0.9)
    p.add_argument("--skip-quality-gate", dest="skip_quality_gate", action="store_true",
                   help="데이터 품질 게이트를 건너뛴다(기본은 켜짐: 모순된 데이터의 종목은 제외하고 사유 출력)")
    p.add_argument("--entry-lag", dest="entry_lag", type=int, default=0)
    p.add_argument("--entry-col", dest="entry_col", default=None)
    p.add_argument("--cost-bps", dest="cost_bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=0.0)
    p.add_argument("--top-frac", dest="top_frac", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--id-vol-mode", dest="id_vol_mode", default="multiple", choices=["multiple", "zscore"])
    p.add_argument("--id-vol-mult", dest="id_vol_mult", type=float, default=2.0)
    p.add_argument("--id-z-threshold", dest="id_z_threshold", type=float, default=1.96)
    p.add_argument("--big-sales-len", dest="big_sales_len", type=int, default=7)
    p.add_argument("--vp-bins", dest="vp_bins", type=int, default=20)
    p.add_argument("--vp-freq", dest="vp_freq", default="W")
    # [신규] 콤보필터(candle_volume_entry_filter.pine 포팅) 파라미터 — target-signal로
    # "콤보필터-강세"/"콤보필터-약세"를 지정할 때 _compute_target_indicator가 사용한다.
    p.add_argument("--combo-body-min-pct", dest="combo_body_min_pct", type=float, default=55.0)
    p.add_argument("--combo-wick-max-pct", dest="combo_wick_max_pct", type=float, default=25.0)
    p.add_argument("--combo-vol-len", dest="combo_vol_len", type=int, default=20)
    p.add_argument("--combo-vol-mult", dest="combo_vol_mult", type=float, default=1.6)
    p.add_argument("--combo-ma-len", dest="combo_ma_len", type=int, default=50)
    p.add_argument("--combo-ma-type", dest="combo_ma_type", default="EMA", choices=["SMA", "EMA", "WMA"])
    p.add_argument("--combo-score-threshold", dest="combo_score_threshold", type=float, default=0.62)
    p.add_argument("--output", default="output")
    args = p.parse_args()
    args.split_date = pd.Timestamp(args.split_date)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs_by_symbol = _load_symbols(args)
    if not dfs_by_symbol:
        sys.exit("로딩에 성공한 종목이 없습니다.")
    print(f"[데이터 확인] {len(dfs_by_symbol)}개 종목 로딩 완료: {list(dfs_by_symbol.keys())}")

    result = run_search(dfs_by_symbol, args)
    print_report(result)

    rows = [dict(param=name, **vars(plateau)) for name, plateau in result["per_param"].items()]
    diag_path = out_dir / "param_search_diagnostics.csv"
    pd.DataFrame(rows).to_csv(diag_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {diag_path}")


if __name__ == "__main__":
    main()
