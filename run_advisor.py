#!/usr/bin/env python3
"""
"이 종목/코인 지금 사야 돼, 팔아야 돼, 들고 있어야 돼?"에 답하고, 지금 검증된
신호가 뜬 다른 유망 종목까지 추천하는 CLI.

이 프로젝트가 지금까지 만든 지표([1]) + 필터([2]) + 이벤트 스터디 검증([3]) +
규칙 기반 판정([7])의 "지금 이 순간" 버전이다. 투자자문이 아니라, 지금까지
검증한 통계가 이 순간 이 종목에 대해 뭐라고 말하는지를 근거(t-stat, 표본 수,
검증 기준)와 함께 요약해 보여주는 도구다.

두 모드로 쓴다.

  --mode scan  : 유니버스 전체(예: KRX 300, 업비트 상위 20개)를 훑어서 종목별
      매수/보유/매도와 신호유형별 횡단면 풀링 통계를 계산해 output/ 밑에
      캐시(advisor_scan.csv, advisor_pooled_stats.csv)로 저장한다. 종목 수가
      많으면 느리므로 하루 한 번 정도 배치로 돌리는 걸 권장한다.

  --mode query : 질문받은 종목 하나만 실시간으로 평가하고(최신 데이터 fetch),
      --mode scan이 만들어둔 캐시에서 "다른 유망 종목"을 읽어와 같이 보여준다.
      캐시가 없으면 하이브리드 신뢰도 판단(종목 자체 표본 부족 시 풀링 통계로
      보강)의 풀링 통계를 못 쓰므로, 그 종목 자체 이벤트만으로 판단한다는 점과
      "다른 유망 종목" 추천을 못 준다는 점을 경고로 알려준다.

사용 예:
  # 1) 유니버스 스캔(캐시 생성) — 하루에 한 번 정도
  python run_advisor.py --mode scan --market krx \\
    --symbols 005930,000660,035420,051910,207940,035720 \\
    --start 20200101 --end 20260901

  # 2) 개별 종목 질의 (스캔 캐시가 있으면 다른 유망 종목도 같이 보여줌)
  python run_advisor.py --mode query --market krx --symbol 005930 \\
    --start 20200101 --end 20260901
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from advisor.evaluator import evaluate_symbol, MIN_OWN_EVENTS, DEFAULT_T_THRESHOLD, _compute_indicator_dfs
from advisor.universe_scan import scan_universe
from data.collect import collect_universe

# 데이터 "수집"(data/*.py)과 "분석"(이 파일)의 관심사 분리: 이 파일은
# 어느 심볼을 어떤 순서로 평가할지만 알고, 실제 fetch 방식(업비트 캔들
# 페이징, pykrx/KRX Open API 등)은 data.collect.collect_universe에 위임한다.
# 종목 하나의 수집 실패가 전체 스캔을 멈추지 않는 견고성도 그쪽에서 보장된다.


def _load_symbol(market: str, symbol: str, args) -> pd.DataFrame | None:
    """단일 종목 수집 — market/args에 따라 적절한 data.* fetch 함수로 위임만
    한다. 실패 시 예외를 그대로 전파해 collect_universe()가 기록하게 둔다
    (예외를 여기서 삼키면 실패 사유가 뭉개진다)."""
    from data.upbit import fetch_upbit_ohlcv
    from data.krx import fetch_krx_ohlcv

    if market == "upbit":
        return fetch_upbit_ohlcv(market=symbol, unit=args.unit, count=args.count,
                                  minute_unit=args.minute_unit)
    if not args.start or not args.end:
        sys.exit("--market krx 에는 --start / --end (YYYYMMDD)가 필요합니다.")
    return fetch_krx_ohlcv(symbol, args.start, args.end)


def _load_universe(args) -> dict[str, pd.DataFrame]:
    if args.symbol_file:
        symbols = [line.strip() for line in Path(args.symbol_file).read_text().splitlines() if line.strip()]
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    result = collect_universe(symbols, lambda symbol: _load_symbol(args.market, symbol, args),
                               quality_gate=not getattr(args, "skip_quality_gate", False))
    return result.ok


def _print_verdict(verdict, title: str):
    print(f"\n=== {title}: {verdict.symbol} (기준일 {verdict.as_of}) -> {verdict.verdict} ===")
    if not verdict.active_signals:
        print("  지금 활성화된 신호가 없습니다(관망 근거는 있지만 신규 신호는 없음).")
        return
    for s in verdict.active_signals:
        tag = "✔검증" if s.validated else "미검증"
        print(f"  [{tag}] {s.label} ({'강세' if s.direction == 1 else '약세'}) — {s.reason}")
    validated = [s for s in verdict.active_signals if s.validated]
    if not validated:
        print("  -> 오늘 뜬 신호 중 통계적으로 검증된 건 없어 보유로 판정")
    else:
        long_n = sum(1 for s in validated if s.direction == 1)
        short_n = sum(1 for s in validated if s.direction == -1)
        print(f"  -> 검증된 강세 신호 {long_n}개 vs 약세 신호 {short_n}개 "
              f"(강도 점수 {verdict.strength:.2f}) -> {verdict.verdict}")
    print("  (주의: 이건 과거 통계 기반 신호 요약이지 투자자문이 아닙니다 — 최종 판단은 본인 책임)")


def _print_recommendations(scan_df: pd.DataFrame, exclude_symbol: str | None, top_n: int):
    if scan_df is None or scan_df.empty:
        print("\n[다른 유망 종목] 스캔 캐시가 없어(--mode scan을 먼저 실행) 추천을 줄 수 없습니다.")
        return
    candidates = scan_df[scan_df["verdict"] != "보유"]
    if exclude_symbol:
        candidates = candidates[candidates["symbol"] != exclude_symbol]
    candidates = candidates.head(top_n)

    print(f"\n=== 다른 유망 종목 (스캔 캐시 기준, 상위 {top_n}) ===")
    if candidates.empty:
        print("  지금 검증된 매수/매도 신호가 뜬 다른 종목이 없습니다.")
        return
    with pd.option_context("display.max_colwidth", 80):
        print(candidates[["symbol", "verdict", "strength", "n_validated_signals", "as_of"]]
              .to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["scan", "query"], required=True)
    p.add_argument("--market", choices=["upbit", "krx"], required=True)
    p.add_argument("--symbol", help="query 모드: 질의할 종목 하나")
    p.add_argument("--symbols", help="scan 모드: 콤마로 구분한 종목코드 목록")
    p.add_argument("--symbol-file", dest="symbol_file", default=None,
                    help="scan 모드: 종목코드가 한 줄에 하나씩 있는 파일(예: KRX 300 리스트)")
    p.add_argument("--unit", default="day", choices=["day", "week", "minute"])
    p.add_argument("--minute-unit", dest="minute_unit", type=int, default=60)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--start", help="krx 시작일 YYYYMMDD")
    p.add_argument("--end", help="krx 종료일 YYYYMMDD")
    p.add_argument("--target-horizon", dest="target_horizon", type=int, default=5,
                    help="신뢰도 판단에 쓸 보유기간(봉수). 기본 5")
    p.add_argument("--t-threshold", dest="t_threshold", type=float, default=DEFAULT_T_THRESHOLD)
    p.add_argument("--min-own-events", dest="min_own_events", type=int, default=MIN_OWN_EVENTS,
                    help="이 이상이면 종목 자체 통계, 미만이면 풀링 통계로 대체(하이브리드)")
    p.add_argument("--top-n", dest="top_n", type=int, default=10, help="query 모드: 추천 종목 수")
    p.add_argument("--id-vol-mode", dest="id_vol_mode", default="multiple", choices=["multiple", "zscore"])
    p.add_argument("--id-vol-mult", dest="id_vol_mult", type=float, default=2.0)
    p.add_argument("--id-z-threshold", dest="id_z_threshold", type=float, default=1.96)
    p.add_argument("--big-sales-len", dest="big_sales_len", type=int, default=7)
    p.add_argument("--vp-bins", dest="vp_bins", type=int, default=20)
    p.add_argument("--vp-freq", dest="vp_freq", default="W")
    p.add_argument("--mom-filter", dest="mom_filter", action="store_true")
    p.add_argument("--mom-lookback", dest="mom_lookback", type=int, default=126)
    p.add_argument("--mom-window", dest="mom_window", type=int, default=252)
    p.add_argument("--mom-threshold", dest="mom_threshold", type=float, default=0.7)
    p.add_argument("--liq-filter", dest="liq_filter", action="store_true")
    p.add_argument("--liq-window", dest="liq_window", type=int, default=20)
    p.add_argument("--liq-percentile-window", dest="liq_percentile_window", type=int, default=252)
    p.add_argument("--liq-threshold", dest="liq_threshold", type=float, default=0.9)
    p.add_argument("--skip-quality-gate", dest="skip_quality_gate", action="store_true",
                   help="데이터 품질 게이트를 건너뛴다(기본은 켜짐: 모순된 데이터의 종목은 제외하고 사유 출력)")
    p.add_argument("--entry-lag", dest="entry_lag", type=int, default=0)
    p.add_argument("--entry-col", dest="entry_col", default=None)
    p.add_argument("--cost-bps", dest="cost_bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=0.0)
    # [신규] 콤보필터(candle_volume_entry_filter.pine 포팅) 파라미터
    p.add_argument("--combo-body-min-pct", dest="combo_body_min_pct", type=float, default=55.0)
    p.add_argument("--combo-wick-max-pct", dest="combo_wick_max_pct", type=float, default=25.0)
    p.add_argument("--combo-vol-len", dest="combo_vol_len", type=int, default=20)
    p.add_argument("--combo-vol-mult", dest="combo_vol_mult", type=float, default=1.6)
    p.add_argument("--combo-ma-len", dest="combo_ma_len", type=int, default=50)
    p.add_argument("--combo-ma-type", dest="combo_ma_type", default="EMA", choices=["SMA", "EMA", "WMA"])
    p.add_argument("--combo-score-threshold", dest="combo_score_threshold", type=float, default=0.62)
    # [신규] 메타라벨링 통합 1단계 — 체결 로그 레이어(backtest/execution_log.py) 연결.
    # 기본 꺼짐: 기존 동작(로그 안 남김)을 그대로 보존한다. scan/query 둘 다 지원.
    p.add_argument("--log-executions", dest="log_executions", action="store_true",
                    help="평가 대상 종목(들)의 신호 이벤트를 signal_id 단위로 체결 로그 CSV에 "
                         "기록한다(메타라벨링용 원재료). exit_reason은 전부 'horizon_exit' — "
                         "실제 리스크관리 청산 사유는 RiskManagedTrader 연동 후에만 채워진다. "
                         "기본 꺼짐")
    p.add_argument("--execution-log-path", dest="execution_log_path", default=None,
                    help="--log-executions일 때 기록할 CSV 경로. 기본 None이면 "
                         "<output>/execution_log.csv")
    p.add_argument("--output", default="output")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_cache_path = out_dir / "advisor_scan.csv"
    pooled_cache_path = out_dir / "advisor_pooled_stats.csv"

    if args.mode == "scan":
        if not args.symbols and not args.symbol_file:
            sys.exit("--mode scan 에는 --symbols 또는 --symbol-file 이 필요합니다.")
        dfs_by_symbol = _load_universe(args)
        if not dfs_by_symbol:
            sys.exit("로딩에 성공한 종목이 없습니다.")
        print(f"[데이터 확인] {len(dfs_by_symbol)}개 종목 로딩 완료")

        scan_df, pooled_stats = scan_universe(dfs_by_symbol, args)
        scan_df.to_csv(scan_cache_path, index=False, encoding="utf-8-sig")
        pd.DataFrame([dict(label=k, **v) for k, v in pooled_stats.items()]) \
            .to_csv(pooled_cache_path, index=False, encoding="utf-8-sig")
        print(f"[저장] {scan_cache_path}")
        print(f"[저장] {pooled_cache_path}")

        if args.log_executions:
            from backtest.execution_log import log_all_signals
            log_path = args.execution_log_path or str(out_dir / "execution_log.csv")
            total_new = 0
            for sym, sym_df in dfs_by_symbol.items():
                dfs_by_indicator = _compute_indicator_dfs(sym_df, args)
                total_new += log_all_signals(dfs_by_indicator, symbol=sym,
                                              horizon=args.target_horizon, path=log_path,
                                              entry_lag=args.entry_lag, entry_col=args.entry_col,
                                              cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)
            print(f"[체결 로그] {log_path}에 {total_new}건 신규 기록 ({len(dfs_by_symbol)}개 종목, "
                  f"보유기간={args.target_horizon}봉 기준, exit_reason은 전부 'horizon_exit')")

        print("\n=== 유니버스 스캔 결과 (강도 점수 내림차순) ===")
        with pd.option_context("display.max_colwidth", 80):
            print(scan_df[["symbol", "verdict", "strength", "n_validated_signals", "as_of"]]
                  .to_string(index=False))
        return

    # --mode query
    if not args.symbol:
        sys.exit("--mode query 에는 --symbol 이 필요합니다.")

    # 단일 종목 질의도 collect_universe로 통일해서, 실패 사유(예외 메시지)를
    # _load_symbol 안에서 삼키지 않고 그대로 사용자에게 보여준다.
    single = collect_universe([args.symbol], lambda symbol: _load_symbol(args.market, symbol, args),
                               quality_gate=not getattr(args, "skip_quality_gate", False))
    if not single.ok:
        reason = single.failed.get(args.symbol, "알 수 없는 이유")
        sys.exit(f"{args.symbol} 데이터를 가져오지 못했습니다: {reason}")
    df = single.ok[args.symbol]

    pooled_stats = None
    if pooled_cache_path.exists():
        pooled_df = pd.read_csv(pooled_cache_path)
        pooled_stats = {row["label"]: dict(n=row["n"], mean=row["mean"], t_stat=row["t_stat"])
                        for _, row in pooled_df.iterrows()}
    else:
        print("[안내] 유니버스 스캔 캐시가 없어 하이브리드 신뢰도 판단(풀링 통계)을 쓸 수 없습니다. "
              "--mode scan을 먼저 실행하면 종목 자체 표본이 부족할 때도 신뢰도를 보강할 수 있습니다.")

    verdict = evaluate_symbol(df, args.symbol, args, pooled_stats=pooled_stats)
    _print_verdict(verdict, "종목 평가")

    if args.log_executions:
        from backtest.execution_log import log_all_signals
        log_path = args.execution_log_path or str(out_dir / "execution_log.csv")
        dfs_by_indicator = _compute_indicator_dfs(df, args)
        n_new = log_all_signals(dfs_by_indicator, symbol=args.symbol,
                                 horizon=args.target_horizon, path=log_path,
                                 entry_lag=args.entry_lag, entry_col=args.entry_col,
                                 cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)
        print(f"\n[체결 로그] {log_path}에 {n_new}건 신규 기록 (보유기간={args.target_horizon}봉 기준, "
              f"exit_reason은 전부 'horizon_exit' — 실제 리스크관리 청산 사유 아님)")

    scan_df = pd.read_csv(scan_cache_path) if scan_cache_path.exists() else None
    _print_recommendations(scan_df, exclude_symbol=args.symbol, top_n=args.top_n)


if __name__ == "__main__":
    main()
