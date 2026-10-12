#!/usr/bin/env python3
"""
5개 Pine Script 지표(기관 변동 캔들, Delta Trading, Big Sales, Ultimate RSI, VP Box)와
직접 설계한 콤보필터(candle_volume_entry_filter.pine 포팅)를 파이썬으로 재구현해
각각의 신호가 통계적으로 유의미한지 이벤트 스터디로 검증한다.

[변경] combo_filter(candle_volume_entry_filter.pine 포팅, 7번째 신호)를
dfs_by_indicator에 추가했다 — 다른 호출부(run_judgment 등)는 SIGNAL_SPECS를
순회하는 구조라 자동으로 같이 검증된다. --log-executions(메타라벨링 통합 1단계,
backtest/execution_log.py 연결)를 추가했다 — 기본 꺼짐, 기존 동작과 100% 동일.

사용 예:
  # 업비트 코인
  python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000

  # 국내 주식 (SK하이닉스)
  python run_backtest.py --market krx --symbol 000660 --start 20220101 --end 20260901

결과:
  output/summary.csv          지표별·방향별·보유기간별 평균수익률·승률·t-stat 표
  output/event_curves.png     신호 발생 후 N봉 평균 누적수익률 곡선 (지표별 비교)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _setup_korean_font():
    """OS별 한글 폰트를 찾아 matplotlib에 적용. 못 찾으면 조용히 넘어간다(글자가 네모로 보일 수 있음)."""
    import platform
    candidates = {
        "Windows": ["Malgun Gothic"],
        "Darwin": ["AppleGothic"],
        "Linux": ["NanumGothic", "NanumBarunGothic", "UnDotum"],
    }.get(platform.system(), [])
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_korean_font()

from data.upbit import fetch_upbit_ohlcv
from data.krx import fetch_krx_ohlcv
from indicators import (institutional_displacement, delta_trading, big_sales, ultimate_rsi,
                         vp_box, momentum, liquidity, combo_filter)
from backtest.engine import compare_all, SIGNAL_SPECS
from decision.rule_based import judge_indicator


def load_data(args) -> pd.DataFrame:
    if args.market == "upbit":
        df = fetch_upbit_ohlcv(market=args.symbol, unit=args.unit, count=args.count,
                                minute_unit=args.minute_unit)
    elif args.market == "krx":
        if not args.start or not args.end:
            sys.exit("--market krx 에는 --start / --end (YYYYMMDD)가 필요합니다.")
        df = fetch_krx_ohlcv(args.symbol, args.start, args.end)
    else:
        sys.exit(f"unknown market: {args.market}")

    if len(df) < 60:
        print(f"[경고] 데이터가 {len(df)}봉밖에 없습니다. 지표 계산이 불안정할 수 있습니다.")
    return df


def build_compare_kwargs(args, slippage_arg) -> dict:
    """
    args(argparse 네임스페이스 또는 alerts.config.build_args의 결과)에서 compare_all()에 넘길
    인자를 만든다. run()과 OOS·국면 검증(backtest.oos / backtest.regime)이 같은 설정으로
    같은 통계를 계산하도록 한 곳에서 만든다 — 설정이 갈라지면 학습/검증 결과를 본 분석과
    비교할 수 없다.

    getattr 기본값 경로: 이 필드가 없는 기존 args 네임스페이스(alerts.config의 build_args 등)도
    그대로 동작한다. 새 옵션(deflate, barrier, barrier_sequential, barrier_economics)은 기본
    꺼짐이라 켜지 않으면 출력 컬럼도 이전과 완전히 동일하다.
    """
    horizons = tuple(int(h) for h in args.horizons.split(","))
    return dict(
        horizons=horizons, max_curve_horizon=max(horizons),
        entry_lag=args.entry_lag, entry_col=args.entry_col,
        cost_bps=args.cost_bps, slippage_bps=slippage_arg,
        newey_west=args.newey_west, drop_overlapping=args.drop_overlapping, fdr=args.fdr,
        # 디플레이티드 샤프비율(backtest.deflated)
        deflate=getattr(args, "deflate", False),
        dsr_threshold=getattr(args, "dsr_threshold", 0.95),
        min_events=getattr(args, "dsr_min_events", 20),
        extra_trials=getattr(args, "dsr_extra_trials", 0),
        # 트리플 배리어(backtest.triple_barrier). 기본 False면 tb_* 컬럼이 아예 안 생긴다.
        barrier=getattr(args, "barrier", False),
        pt_mult=getattr(args, "barrier_pt_mult", 2.0),
        sl_mult=getattr(args, "barrier_sl_mult", 1.0),
        vol_span=getattr(args, "barrier_vol_span", 20),
        use_intrabar=getattr(args, "barrier_intrabar", True),
        # 배리어 통계의 겹침 제거(순차 거래)와 경제적 유의성 컬럼. 기본 꺼짐.
        barrier_sequential=getattr(args, "barrier_sequential", False),
        barrier_economics=getattr(args, "barrier_economics", False),
    )


def run(df: pd.DataFrame, args) -> tuple[pd.DataFrame, dict, dict, "float | pd.Series"]:
    dfs_by_indicator = {
        "institutional_displacement": institutional_displacement.compute(
            df, vol_mode=args.id_vol_mode, vol_mult=args.id_vol_mult,
            z_threshold=args.id_z_threshold),
        "delta_trading": delta_trading.compute(df),
        "big_sales": big_sales.compute(df, length=args.big_sales_len),
        "ultimate_rsi": ultimate_rsi.compute(df),
        "vp_box": vp_box.compute(df, bins=args.vp_bins, freq=args.vp_freq),
        "combo_filter": combo_filter.compute(
            df, body_min_pct=args.combo_body_min_pct, wick_max_pct=args.combo_wick_max_pct,
            vol_len=args.combo_vol_len, vol_mult=args.combo_vol_mult,
            ma_len=args.combo_ma_len, ma_type=args.combo_ma_type,
            score_threshold=args.combo_score_threshold),
    }

    if args.mom_filter:
        mom_df = momentum.compute(df, lookback=args.mom_lookback, window=args.mom_window)
        n_before = {(k, c): int(v[c].fillna(False).sum())
                    for k, v in dfs_by_indicator.items()
                    for _, c, _, _ in SIGNAL_SPECS if c in v.columns}
        dfs_by_indicator = momentum.apply_filter(dfs_by_indicator, mom_df, SIGNAL_SPECS,
                                                  threshold=args.mom_threshold)
        n_after = {(k, c): int(v[c].fillna(False).sum())
                   for k, v in dfs_by_indicator.items()
                   for _, c, _, _ in SIGNAL_SPECS if c in v.columns}
        print(f"[모멘텀 필터] lookback={args.mom_lookback}봉 window={args.mom_window}봉 "
              f"threshold={args.mom_threshold} (상/하위 {(1-args.mom_threshold)*100:.0f}%만 통과)")
        for key in n_before:
            print(f"  {key[0]}.{key[1]}: {n_before[key]} -> {n_after[key]}건")

    liq_df = None
    if args.liq_filter or args.dynamic_slippage:
        liq_df = liquidity.compute(df, window=args.liq_window,
                                    percentile_window=args.liq_percentile_window)

    if args.liq_filter:
        n_before = {(k, c): int(v[c].fillna(False).sum())
                    for k, v in dfs_by_indicator.items()
                    for _, c, _, _ in SIGNAL_SPECS if c in v.columns}
        dfs_by_indicator = liquidity.apply_filter(dfs_by_indicator, liq_df, SIGNAL_SPECS,
                                                   threshold=args.liq_threshold)
        n_after = {(k, c): int(v[c].fillna(False).sum())
                   for k, v in dfs_by_indicator.items()
                   for _, c, _, _ in SIGNAL_SPECS if c in v.columns}
        print(f"[유동성 필터 - Amihud(2002)] window={args.liq_window}봉 "
              f"percentile_window={args.liq_percentile_window}봉 threshold={args.liq_threshold} "
              f"(비유동성 상위 {(1-args.liq_threshold)*100:.0f}% 시점의 이벤트 제외)")
        for key in n_before:
            print(f"  {key[0]}.{key[1]}: {n_before[key]} -> {n_after[key]}건")

    if args.dynamic_slippage:
        slippage_arg = liquidity.estimate_slippage_bps(
            liq_df, base_bps=args.slippage_bps, max_extra_bps=args.max_extra_slippage_bps)
        print(f"[동적 슬리피지 - Amihud(2002)] base_bps={args.slippage_bps} "
              f"max_extra_bps={args.max_extra_slippage_bps} "
              f"(비유동성 퍼센타일에 비례해 봉마다 슬리피지를 {args.slippage_bps}~"
              f"{args.slippage_bps + args.max_extra_slippage_bps}bp 사이로 추정)")
    else:
        slippage_arg = args.slippage_bps

    horizons = tuple(int(h) for h in args.horizons.split(","))
    summary, results = compare_all(dfs_by_indicator, **build_compare_kwargs(args, slippage_arg))
    if args.entry_lag > 0 or args.cost_bps > 0 or args.slippage_bps > 0 or args.dynamic_slippage:
        slip_desc = "동적(Amihud 기반, 위 로그 참고)" if args.dynamic_slippage else f"{args.slippage_bps}bp(고정)"
        print(f"[체결 가정] entry_lag={args.entry_lag}봉 "
              f"entry_col={args.entry_col or '(close_col과 동일)'} "
              f"cost_bps={args.cost_bps}bp(수수료) slippage_bps={slip_desc}(슬리피지)")
    if args.newey_west or args.drop_overlapping:
        print(f"[중복 이벤트 보정] newey_west={args.newey_west} drop_overlapping={args.drop_overlapping}")
    return summary, results, dfs_by_indicator, slippage_arg


def run_judgment(dfs_by_indicator: dict, summary: pd.DataFrame, args, judge_horizon: int,
                  slippage_bps: "float | pd.Series" = 0.0) -> pd.DataFrame | None:
    """
    파이프라인 4~7단계(팩터 구성 -> 팩터 조정 알파 -> 규칙 기반 판단)를 실행한다.

    가상자산(--market upbit)은 자동으로 건너뛴다: factors/kr_fama_french.py의
    SMB/HML/WML은 KRX 상장 주식 유니버스를 전제로 구성한 팩터라서, 코인에 그대로
    적용하면 두 시장의 사이즈·가치·모멘텀 정의 자체가 안 맞아 회귀 결과가 의미
    없어진다(팩터로 설명되는지 아닌지조차 판단 불가능한 숫자만 나옴). 4~5단계를
    억지로 돌리는 대신 아예 건너뛰고 그 이유를 출력한다. --factor-csv를 실수로
    같이 넘겨도 market=upbit이면 무시된다.

    factor_cols는 --factor-cols로 명시하지 않으면 자동으로 정한다: --mom-filter가
    켜져 있으면([2]단계에서 이미 Jegadeesh & Titman 모멘텀으로 걸러낸 신호이므로)
    WML을 회귀에서 빼고 SMB/HML만 쓴다 — 안 그러면 모멘텀 효과가 [2]단계 필터와
    [5]단계 WML 회귀 양쪽에서 이중으로 차감돼 알파가 과소평가된다.

    slippage_bps: args.slippage_bps를 그대로 읽지 않고 호출부(main())에서 넘겨받는다
    — --dynamic-slippage가 켜진 경우 run()이 이미 Amihud 기반 pd.Series로 계산해
    둔 값을 그대로 써야 [3]단계(compare_all)와 [5]단계(event_alpha)가 동일한 체결
    가정을 공유한다(하나는 고정값, 하나는 동적값을 쓰면 두 단계의 알파 추정이
    서로 다른 비용 가정 위에서 계산되는 모순이 생긴다).
    """
    if args.market == "upbit":
        if args.factor_csv:
            print("[4~5단계 건너뜀] --market upbit: KRX 기반 SMB/HML/WML 팩터는 "
                  "암호화폐 유니버스에 적용할 수 없어 --factor-csv를 지정해도 무시합니다.")
        return None

    if not args.factor_csv:
        return None  # krx라도 팩터 csv를 안 줬으면 4~7단계는 스킵(선택적 기능)

    factor_df = pd.read_csv(args.factor_csv, index_col=0, parse_dates=True)

    if args.factor_cols:
        factor_cols = tuple(c.strip() for c in args.factor_cols.split(","))
    elif args.mom_filter:
        factor_cols = tuple(c for c in ("SMB", "HML") if c in factor_df.columns)
        print(f"[팩터 열 자동 선택] --mom-filter가 켜져 있어 WML을 회귀에서 제외: {factor_cols} "
              f"(모멘텀 효과의 이중 차감 방지 — 필요하면 --factor-cols로 직접 지정)")
    else:
        factor_cols = tuple(c for c in ("SMB", "HML", "WML") if c in factor_df.columns)

    rows = []
    for indicator_key, event_col, direction, label in SIGNAL_SPECS:
        df_i = dfs_by_indicator.get(indicator_key)
        if df_i is None or event_col not in df_i.columns:
            continue

        sig_row = summary[(summary["signal"] == label) & (summary["horizon"] == judge_horizon)]
        bh_sig = bool(sig_row["significant_bh"].iloc[0]) if not sig_row.empty and "significant_bh" in sig_row.columns else None

        result = judge_indicator(
            df_i, event_col, direction, factor_df, horizon=judge_horizon,
            factor_cols=factor_cols, entry_lag=args.entry_lag, entry_col=args.entry_col,
            cost_bps=args.cost_bps, slippage_bps=slippage_bps, bh_significant=bh_sig,
            t_threshold=args.t_threshold,
        )
        rows.append(dict(signal=label, direction=("LONG" if direction == 1 else "SHORT"), **result))

    return pd.DataFrame(rows)


def plot_curves(results: dict, out_path: Path):
    if not results:
        print("[경고] 플롯할 이벤트가 없습니다 (신호가 한 번도 발생하지 않았을 수 있습니다).")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, res in results.items():
        if res.n_events == 0:
            continue
        ax.plot(res.curve.index, res.curve.values * 100, marker="o", markersize=3,
                label=f"{label} (n={res.n_events})")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("신호 발생 후 경과 봉수")
    ax.set_ylabel("평균 누적 수익률 (%)")
    ax.set_title("신호별 평균 순방향 수익률 경로")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[저장] {out_path}")


def print_ranking(summary: pd.DataFrame, horizon: int, fdr: float):
    if summary.empty:
        print("\n요약할 신호가 없습니다.")
        return
    sub = summary[summary["horizon"] == horizon].copy()
    sub = sub.sort_values("t_stat", ascending=False)
    print(f"\n=== 보유기간 {horizon}봉 기준 t-stat 랭킹 (신호 신뢰도가 통계적으로 높은 순) ===")
    cols = ["signal", "direction", "n_events", "mean_return_pct", "win_rate_pct", "t_stat"]
    if "p_value" in sub.columns:
        cols += ["p_value", "significant_bh"]
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(sub[cols].to_string(index=False))
    print("\n참고: t-stat이 대략 |2| 이상이면 통계적으로 눈여겨볼 만한 신호,"
          " 표본(n_events)이 너무 적으면(예: 10 미만) 신뢰하기 어렵습니다.")
    print(f"significant_bh는 이번 실행에서 검증한 전체 신호 x 방향 x 보유기간 조합을 한꺼번에 놓고"
          f" Benjamini-Hochberg 절차(FDR={fdr:.0%})로 다중 검정 보정을 한 뒤에도 유의하다고 남는"
          f" 신호만 True입니다 — 개별 t_stat만 보고 판단하면 우연히 유의하게 나온 신호를 걸러내지"
          f" 못할 수 있으니, 여러 조합을 비교할 때는 이 컬럼을 우선 참고하세요.")


def run_extended_studies(dfs_by_indicator: dict, args, slippage_arg, out_dir: Path, judge_horizon: int):
    """
    --oos-train-end / --regime 로 요청된 표본 외 검증과 국면별 검증을 실행하고 출력·저장한다.
    둘 다 꺼져 있으면 아무것도 하지 않는다(기존 동작 보존). 계산은 backtest.oos / backtest.regime에
    있고 여기는 인자 연결과 출력만 한다(얇은 껍데기 원칙).
    """
    if not (getattr(args, "oos_train_end", None) or getattr(args, "regime", False)):
        return None
    ck = build_compare_kwargs(args, slippage_arg)
    out = {}

    if args.oos_train_end:
        from backtest.oos import oos_study, format_oos_report
        ck_oos = dict(ck)
        if not ck_oos["deflate"]:
            print("\n[OOS] 신호 선택 규칙이 DSR을 쓰므로 이번 실행에서는 DSR을 자동으로 켭니다.")
            ck_oos["deflate"] = True
        if args.oos_require_barrier and not ck_oos["barrier"]:
            print("[OOS] --oos-require-barrier라서 배리어 통계를 자동으로 켭니다.")
            ck_oos["barrier"] = True
        test_start = args.oos_test_start or str((pd.Timestamp(args.oos_train_end) + pd.Timedelta(days=1)).date())
        res = oos_study(dfs_by_indicator, ck_oos, args.oos_train_end, test_start,
                        judge_horizon=judge_horizon, warmup_bars=args.oos_warmup,
                        min_test_n=args.oos_min_test_n, alpha=args.oos_alpha,
                        require_barrier=args.oos_require_barrier)
        print("\n" + format_oos_report(res))
        res.summary_train.to_csv(out_dir / "oos_train_summary.csv", index=False, encoding="utf-8-sig")
        res.summary_test.to_csv(out_dir / "oos_test_summary.csv", index=False, encoding="utf-8-sig")
        res.selected.to_csv(out_dir / "oos_selected.csv", index=False, encoding="utf-8-sig")
        print(f"[저장] {out_dir / 'oos_train_summary.csv'}, oos_test_summary.csv, oos_selected.csv")
        out["oos"] = res

    if args.regime:
        from backtest.regime import classify_regime, regime_study, format_regime_report
        # 분류에는 지표와 무관한 가격 열이 필요하다 — 어느 지표 df든 같은 OHLCV를 갖고 있다.
        price_df = next(iter(dfs_by_indicator.values()))
        regimes = classify_regime(price_df, ma_len=args.regime_ma_len, ret_len=args.regime_ret_len,
                                  vol_len=args.regime_vol_len, z_thr=args.regime_z_thr)
        rres = regime_study(dfs_by_indicator, regimes, ck)
        print("\n" + format_regime_report(rres, judge_horizon, min_n=args.regime_min_n))
        if not rres.summary.empty:
            rres.summary.to_csv(out_dir / "regime_summary.csv", index=False, encoding="utf-8-sig")
            print(f"[저장] {out_dir / 'regime_summary.csv'}")
        out["regime"] = rres
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", choices=["upbit", "krx"], required=True)
    p.add_argument("--symbol", required=True, help="upbit: 'KRW-BTC' 형식 / krx: '005930' 형식 종목코드")
    p.add_argument("--unit", default="day", choices=["day", "week", "minute"], help="upbit 전용")
    p.add_argument("--minute-unit", dest="minute_unit", type=int, default=60, help="upbit unit=minute일 때 분봉 단위")
    p.add_argument("--count", type=int, default=1000, help="upbit 캔들 개수")
    p.add_argument("--start", help="krx 시작일 YYYYMMDD")
    p.add_argument("--end", help="krx 종료일 YYYYMMDD")
    p.add_argument("--horizons", default="1,3,5,10,20", help="검증할 보유기간(봉수), 콤마구분")
    p.add_argument("--id-vol-mode", dest="id_vol_mode", default="multiple",
                    choices=["multiple", "zscore"],
                    help="기관 변동 캔들의 거래량 이상치 판정 방식: "
                         "multiple=평균의 배수(원본, 기본값), zscore=Kyle(1985) 기반 z-스코어")
    p.add_argument("--id-vol-mult", dest="id_vol_mult", type=float, default=2.0,
                    help="vol-mode=multiple일 때 임계 배수")
    p.add_argument("--id-z-threshold", dest="id_z_threshold", type=float, default=1.96,
                    help="vol-mode=zscore일 때 임계 z값 (기본 1.96 = 95%% 유의수준)")
    p.add_argument("--mom-filter", dest="mom_filter", action="store_true",
                    help="Jegadeesh & Titman(1993) 모멘텀 필터 활성화: 롱 이벤트는 상승 모멘텀 "
                         "상위권, 숏 이벤트는 하락 모멘텀 상위권에서만 인정")
    p.add_argument("--mom-lookback", dest="mom_lookback", type=int, default=126,
                    help="모멘텀 측정 구간(봉수). 일봉 기준 126≈6개월(원 논문 J=6 근사)")
    p.add_argument("--mom-window", dest="mom_window", type=int, default=252,
                    help="모멘텀 퍼센타일을 매길 롤링 윈도우(봉수). 일봉 기준 252≈1년")
    p.add_argument("--mom-threshold", dest="mom_threshold", type=float, default=0.7,
                    help="0.5~1.0. 0.7이면 상/하위 30%% 모멘텀 구간에서만 이벤트 인정")
    p.add_argument("--entry-lag", dest="entry_lag", type=int, default=0,
                    help="신호 확정(t봉 종가) 후 실제 진입까지 걸리는 봉 수. 0=당일 종가 즉시 진입"
                         "(과거 동작과 동일, look-ahead 위험). 1을 주면 t+1봉에서 진입 "
                         "(--entry-col open과 함께 쓰면 '종가에 신호 확정, 익일 시가 진입'이 됨)")
    p.add_argument("--entry-col", dest="entry_col", default=None,
                    help="entry-lag>0일 때 실제 진입가로 쓸 컬럼명 (예: open). 기본 None이면 종가 컬럼 사용")
    p.add_argument("--cost-bps", dest="cost_bps", type=float, default=0.0,
                    help="왕복 거래 수수료(bp, 1bp=0.01%%). 기본 0(미반영). 참고: 업비트 편도 0.05%%x2=10, "
                         "KRX 편도(수수료+거래세) 약 0.18%%x2=36")
    p.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=0.0,
                    help="왕복 슬리피지(bp). cost_bps(확정 수수료)와 별개로 체결 시 가격 미끄러짐을 "
                         "반영한다. 기본 0(미반영). 유동성 충분한 대형주/메이저 코인 기준 5~10bp 정도가 "
                         "흔히 쓰는 근사치 — entry-lag로 t+1 시가 진입을 가정할 때 특히 중요")
    p.add_argument("--newey-west", dest="newey_west", action="store_true",
                    help="보유기간이 길어 이벤트 간 순방향수익률 추적 구간이 겹칠 때(overlapping "
                         "events) 생기는 자기상관을 Newey-West HAC 보정(lag=보유기간-1)으로 반영해 "
                         "t-stat을 계산한다. 기본 꺼짐(단순 std/sqrt(n) t-stat, 이전 동작과 동일)")
    p.add_argument("--drop-overlapping", dest="drop_overlapping", action="store_true",
                    help="보유기간이 겹치는 이벤트를 애초에 표본에서 제거한다(직전 채택 이벤트로부터 "
                         "보유기간 이상 떨어진 이벤트만 남김). newey-west와 함께 쓸 수도, 단독으로 쓸 "
                         "수도 있다. 기본 꺼짐(모든 이벤트 포함, 이전 동작과 동일)")
    p.add_argument("--liq-filter", dest="liq_filter", action="store_true",
                    help="Amihud(2002) 비유동성 필터 활성화: 그 종목 자기 역사 대비 비유동성이 "
                         "높은(--liq-threshold 초과) 구간의 이벤트는 실제 체결 가능성이 낮다고 "
                         "보고 표본에서 제외한다")
    p.add_argument("--liq-window", dest="liq_window", type=int, default=20,
                    help="일별 Amihud 비유동성을 평활화할 롤링 윈도우(봉수). 기본 20봉≈1개월")
    p.add_argument("--liq-percentile-window", dest="liq_percentile_window", type=int, default=252,
                    help="Amihud 값을 그 종목 자기 자신의 과거 분포 대비 퍼센타일로 정규화할 "
                         "롤링 윈도우(봉수). 기본 252봉≈1년")
    p.add_argument("--liq-threshold", dest="liq_threshold", type=float, default=0.9,
                    help="0~1. amihud_percentile이 이 값을 초과하는(가장 비유동적인 상위 "
                         "(1-threshold)) 구간의 이벤트를 제외. 기본 0.9(상위 10%% 제외)")
    p.add_argument("--dynamic-slippage", dest="dynamic_slippage", action="store_true",
                    help="슬리피지를 --slippage-bps 고정값 대신 Amihud(2002) 비유동성 퍼센타일에 "
                         "비례해 봉마다 다르게 추정한다(유동성이 낮을수록 슬리피지가 커짐). "
                         "--slippage-bps 값이 최소치(base_bps)로 쓰이고, --max-extra-slippage-bps만큼 "
                         "더해질 수 있다")
    p.add_argument("--max-extra-slippage-bps", dest="max_extra_slippage_bps", type=float, default=50.0,
                    help="--dynamic-slippage일 때 비유동성이 가장 심한 시점(percentile=1)에 "
                         "--slippage-bps에 추가로 더해질 최대 슬리피지(bp). 기본 50")
    p.add_argument("--deflate", dest="deflate", action="store_true",
                    help="디플레이티드 샤프비율(DSR, Bailey & Lopez de Prado 2014)을 추가로 "
                         "계산한다. BH보정이 쓰는 p-value는 수익률이 정규분포라고 가정하는데, "
                         "실제 이벤트 수익률은 왜도·첨도가 달라 그 가정이 깨진다. DSR은 왜도· "
                         "첨도를 표준오차에 직접 넣고, 동시에 '여러 번 시도해서 고른 최고값'이라는 "
                         "선택편의까지 기준선에 반영한다. 합성 랜덤워크 검증에서 BH만 쓸 때 "
                         "거짓양성 5.2%%가 DSR을 같이 걸면 0.1%% 수준으로 떨어졌다(대신 진짜 신호 "
                         "검출력도 79%%->50%%로 낮아지는 트레이드오프가 있다). "
                         "--drop-overlapping과 같이 쓰기를 권장한다")
    p.add_argument("--dsr-threshold", dest="dsr_threshold", type=float, default=0.95,
                    help="--deflate일 때 significant_dsr 판정 임계값. 기본 0.95(5%% 유의수준)")
    p.add_argument("--dsr-min-events", dest="dsr_min_events", type=int, default=20,
                    help="--deflate일 때 DSR 판정에 요구하는 최소 이벤트 수. 기본 20. "
                         "이벤트가 2~4개인 신호는 샤프비율·왜도·첨도 추정 자체가 무의미하고, "
                         "그런 값이 섞이면 기준선 추정이 오염되어 판정이 거꾸로 뒤집힌다")
    p.add_argument("--dsr-extra-trials", dest="dsr_extra_trials", type=int, default=0,
                    help="--deflate일 때, 이 실행 바깥에서 이미 시도한 검정 횟수. "
                         "파라미터 탐색으로 조합 500개를 훑어본 뒤 제일 좋은 설정으로 돌린 "
                         "거라면 500을 넣어야 선택편의가 제대로 반영된다. 기본 0은 "
                         "'이 실행이 전부'라는 가장 낙관적인 가정")
    p.add_argument("--barrier", dest="barrier", action="store_true",
                    help="같은 신호를 손절/익절을 걸고 매매했을 때의 실현 수익률을 "
                         "tb_* 컬럼으로 나란히 보여준다(트리플 배리어). 기존 "
                         "평균수익률은 '정확히 h봉 뒤 종가' 하나만 보기 때문에, 중간에 "
                         "크게 밀렸다가 회복한 거래도 성공으로 집계된다 — 실제로 손절을 "
                         "걸고 매매했다면 그 거래는 잘렸다. 두 값이 크게 벌어지는 신호일수록 "
                         "실제 운용 난이도가 높다는 뜻")
    p.add_argument("--barrier-pt-mult", dest="barrier_pt_mult", type=float, default=2.0,
                    help="--barrier일 때 익절선을 실현변동성의 몇 배로 둘지. 기본 2.0")
    p.add_argument("--barrier-sl-mult", dest="barrier_sl_mult", type=float, default=1.0,
                    help="--barrier일 때 손절선을 실현변동성의 몇 배로 둘지. 기본 1.0. "
                         "변동성은 종가기준인데 터치 판정은 봉 내부 고저로 하므로, 1.0은 "
                         "생각보다 타이트하다(최대 10봉 설정에서 평균 2~3봉 만에 잘리는 "
                         "수준). 평균 보유봉이 지나치게 짧으면 이 값을 키울 것")
    p.add_argument("--barrier-vol-span", dest="barrier_vol_span", type=int, default=20,
                    help="--barrier일 때 배리어 폭 기준이 되는 실현변동성의 지수가중 구간. 기본 20봉")
    p.add_argument("--barrier-no-intrabar", dest="barrier_intrabar", action="store_false",
                    help="--barrier일 때 봉 내부 고가/저가 대신 종가만으로 배리어 터치를 "
                         "판정한다(일봉 고저를 신뢰하기 어려울 때). 기본은 고저 사용")
    p.add_argument("--barrier-sequential", dest="barrier_sequential", action="store_true",
                    help="--barrier일 때 앞 거래가 청산되기 전에 진입하는 거래를 배리어 통계에서 뺀다"
                         "(순차 거래). 겹치는 이벤트를 그대로 세면 표본이 부풀려 배리어 기반 DSR이 낙관적으로 "
                         "나온다. 기본 꺼짐")
    p.add_argument("--barrier-economics", dest="barrier_economics", action="store_true",
                    help="--barrier일 때 경제적 유의성 컬럼(R 단위 기대값, 손익분기 승률, 비용 대비 배수)을 "
                         "요약표에 추가한다. 승률만으로는 손익비가 비대칭인 전략을 평가할 수 없다. 기본 꺼짐")
    p.add_argument("--oos-train-end", dest="oos_train_end", default=None,
                    help="표본 외(OOS) 검증을 켠다. 학습 구간의 마지막 날(YYYY-MM-DD). 이 날까지의 이벤트로 "
                         "신호를 고르고(BH+DSR 통과) 그 뒤 구간에서 유지되는지 확인한다. 학습 끝의 "
                         "(최대 보유기간+진입지연)봉은 퍼지한다. 선택 규칙이 DSR을 쓰므로 자동으로 DSR을 켠다. "
                         "기본 None(꺼짐)")
    p.add_argument("--oos-test-start", dest="oos_test_start", default=None,
                    help="검증 구간 첫 날(YYYY-MM-DD). 기본은 --oos-train-end 다음 날")
    p.add_argument("--oos-warmup", dest="oos_warmup", type=int, default=60,
                    help="OOS: 지표가 안정되기 전 학습 초기 몇 봉의 이벤트를 뺄지. 기본 60")
    p.add_argument("--oos-min-test-n", dest="oos_min_test_n", type=int, default=10,
                    help="OOS: 검증 구간에서 판정하려면 필요한 최소 이벤트 수. 기본 10(미만이면 판정불가)")
    p.add_argument("--oos-alpha", dest="oos_alpha", type=float, default=0.05,
                    help="OOS: 검증 구간 유의수준. 선택된 신호 수 k로 나눠(Bonferroni) 단측 p와 비교한다. 기본 0.05")
    p.add_argument("--oos-require-barrier", dest="oos_require_barrier", action="store_true",
                    help="OOS: 학습 선택 조건에 배리어 기반 DSR 통과도 요구한다(손절을 걸고도 통과한 신호만). "
                         "자동으로 --barrier를 켠다. 기본 꺼짐")
    p.add_argument("--regime", dest="regime", action="store_true",
                    help="시장 국면(상승/하락/횡보, 종목 자체 가격 기준)별로 나눠 신호 성과를 검증한다. "
                         "국면 전체에 걸친 BH 재보정과 국면 간 부호 일관성을 함께 보여준다. 기본 꺼짐")
    p.add_argument("--regime-ma-len", dest="regime_ma_len", type=int, default=200,
                    help="국면 분류: 추세선(이동평균) 길이. 기본 200봉. 앞의 이 봉수만큼은 분류 불가")
    p.add_argument("--regime-ret-len", dest="regime_ret_len", type=int, default=60,
                    help="국면 분류: 최근 수익률 계산 구간. 기본 60봉")
    p.add_argument("--regime-vol-len", dest="regime_vol_len", type=int, default=60,
                    help="국면 분류: 변동성 계산 구간. 기본 60봉")
    p.add_argument("--regime-z-thr", dest="regime_z_thr", type=float, default=0.5,
                    help="국면 분류: 최근 수익률의 방향성(변동성 대비) 임계값. 기본 0.5 — 관행적 값이며 "
                         "최적화한 값이 아니므로 바꿔가며 결과가 민감한지 확인할 것")
    p.add_argument("--regime-min-n", dest="regime_min_n", type=int, default=5,
                    help="국면 일관성 판단에 쓸 국면별 최소 관측 수. 기본 5")
    p.add_argument("--fdr", dest="fdr", type=float, default=0.10,
                    help="다중 검정 보정(Benjamini-Hochberg) 목표 false discovery rate. "
                         "이번 실행에서 검증하는 전체 신호x방향x보유기간 조합을 한꺼번에 놓고 보정한다. "
                         "기본 0.10(10%%)")
    p.add_argument("--factor-csv", dest="factor_csv", default=None,
                    help="build_kr_factors.py로 만든 SMB/HML/WML csv 경로. 지정하면 4~5단계(팩터 "
                         "구성은 이미 끝난 csv를 읽는 것으로 대체, 팩터 조정 알파 검증)와 7단계(규칙 "
                         "기반 판단)까지 이어서 실행하고 output/judgment.csv를 저장한다. "
                         "--market upbit이면 KRX 전용 팩터라 자동으로 무시(4~5단계 스킵)된다.")
    p.add_argument("--factor-cols", dest="factor_cols", default=None,
                    help="팩터 회귀에 쓸 열 이름(콤마 구분, 예: 'SMB,HML'). 기본 None이면 자동 선택: "
                         "--mom-filter가 켜져 있으면 WML을 제외한 SMB,HML만 사용(모멘텀 효과 이중차감 "
                         "방지), 아니면 factor-csv에 있는 SMB/HML/WML을 전부 사용")
    p.add_argument("--t-threshold", dest="t_threshold", type=float, default=3.0,
                    help="7단계 규칙 기반 판단에서 '유의하다'고 볼 |t-stat| 최소값. 기본 3.0 — "
                         "Harvey, Liu & Zhu(2016)가 다중 비교(여러 지표를 동시에 검증하는) 상황에서는 "
                         "단일 검정 관행(2.0)보다 보수적인 기준을 쓰라고 권고한 데 따름. 기존 기준으로 "
                         "되돌리려면 2.0을 지정")
    p.add_argument("--big-sales-len", dest="big_sales_len", type=int, default=7)
    p.add_argument("--vp-bins", dest="vp_bins", type=int, default=20)
    p.add_argument("--vp-freq", dest="vp_freq", default="W", help="VP Box 구간 (W=주, M=월)")
    # [신규] 콤보필터(candle_volume_entry_filter.pine 포팅) 파라미터 — 전부 원본 Pine
    # 스크립트의 기본값/채택값과 동일. 기존 지표 어느 것도 건드리지 않는 순수 추가.
    p.add_argument("--combo-body-min-pct", dest="combo_body_min_pct", type=float, default=55.0,
                    help="콤보필터: 몸통이 전체 고가-저가 범위에서 차지하는 최소 비율(%%). 원본 기본값 55")
    p.add_argument("--combo-wick-max-pct", dest="combo_wick_max_pct", type=float, default=25.0,
                    help="콤보필터: 진행 방향 반대쪽 꼬리의 최대 허용 비율(%%). 원본 기본값 25")
    p.add_argument("--combo-vol-len", dest="combo_vol_len", type=int, default=20,
                    help="콤보필터: 거래량 SMA 길이. 원본 기본값 20")
    p.add_argument("--combo-vol-mult", dest="combo_vol_mult", type=float, default=1.6,
                    help="콤보필터: 거래량 스파이크 판정 배수. 원본 기본값 1.6")
    p.add_argument("--combo-ma-len", dest="combo_ma_len", type=int, default=50,
                    help="콤보필터: 추세 필터 이동평균 길이. 원본 기본값 50")
    p.add_argument("--combo-ma-type", dest="combo_ma_type", default="EMA", choices=["SMA", "EMA", "WMA"],
                    help="콤보필터: 추세 필터 이동평균 종류. 원본 기본값 EMA")
    p.add_argument("--combo-score-threshold", dest="combo_score_threshold", type=float, default=0.62,
                    help="콤보필터: composite_score 신호 인정 임계값(0~1). 원본이 TradingView 백테스트로 "
                         "0.50~0.75 구간을 스캔해 채택한 값(0.62) — 이 파이프라인 event_study로 재검증 전까지는 "
                         "참고용 채택값으로 취급할 것")
    # [신규] 메타라벨링 통합 1단계 — 체결 로그 레이어(backtest/execution_log.py) 연결.
    # 기본 꺼짐: 기존 동작(로그 안 남김)을 그대로 보존한다.
    p.add_argument("--log-executions", dest="log_executions", action="store_true",
                    help="이번 실행에서 뜬 신호 이벤트들을 signal_id 단위로 체결 로그 CSV에 "
                         "기록한다(메타라벨링용 원재료). 지금은 event_study()와 동일한 정의의 "
                         "'N봉 보유 후 청산' 수익만 기록하고 exit_reason은 전부 'horizon_exit'이다 "
                         "— 실제 손절/시간제한청산/circuit breaker 같은 리스크관리 청산 사유는 "
                         "RiskManagedTrader 연동 후에만 채워진다. 기본 꺼짐")
    p.add_argument("--execution-log-path", dest="execution_log_path", default=None,
                    help="--log-executions일 때 기록할 CSV 경로. 기본 None이면 "
                         "<output>/execution_log.csv (여러 심볼을 같은 경로로 반복 실행하면 "
                         "signal_id 기준 중복 없이 누적된다)")
    p.add_argument("--output", default="output")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[데이터 수집] market={args.market} symbol={args.symbol}")
    df = load_data(args)
    print(f"[데이터 확인] {len(df)}봉, {df.index.min()} ~ {df.index.max()}")

    summary, results, dfs_by_indicator, slippage_arg = run(df, args)

    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[저장] {summary_path}")

    main_horizon = int(args.horizons.split(",")[len(args.horizons.split(",")) // 2])
    print_ranking(summary, main_horizon, args.fdr)

    plot_curves(results, out_dir / "event_curves.png")

    if "tb_expectancy_r" in summary.columns:
        # 경제적 유의성(--barrier-economics): 통계적으로 유의한 것과 실제로 벌 만한 것은 다르다.
        sub = summary[summary["horizon"] == main_horizon].dropna(subset=["tb_expectancy_r"])
        if not sub.empty:
            print(f"\n=== 보유기간 {main_horizon}봉 경제적 유의성 (손절 반영 거래 기준, 비용 차감 후) ===")
            ecols = ["signal", "direction", "tb_n", "tb_win_rate_pct", "tb_breakeven_wr_pct",
                     "tb_wr_edge_pp", "tb_payoff_ratio", "tb_expectancy_r", "tb_cost_multiple"]
            if "tb_significant_dsr" in sub.columns:
                ecols.append("tb_significant_dsr")
            with pd.option_context("display.float_format", "{:.2f}".format):
                print(sub.sort_values("tb_expectancy_r", ascending=False)[ecols].to_string(index=False))
            print("해석: 기대값 +0.2R은 거래당 손절폭의 20%를 번다는 뜻입니다. 실제 승률이 손익분기 승률보다 "
                  "높아야(wr_edge > 0) 기대값이 양수이고, 비용 대비 배수가 1 근처면 슬리피지 가정이 조금만 "
                  "틀려도 사라지는 수준입니다.")

    run_extended_studies(dfs_by_indicator, args, slippage_arg, out_dir, main_horizon)

    judgment = run_judgment(dfs_by_indicator, summary, args, judge_horizon=main_horizon,
                             slippage_bps=slippage_arg)
    if judgment is not None and not judgment.empty:
        judgment_path = out_dir / "judgment.csv"
        judgment.to_csv(judgment_path, index=False, encoding="utf-8-sig")
        print(f"\n[저장] {judgment_path}")
        print(f"=== 보유기간 {main_horizon}봉 기준 규칙 기반 판단 (4~7단계) ===")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(judgment[["signal", "direction", "verdict", "n_events", "raw_t_stat",
                             "alpha", "alpha_t_stat", "bh_significant"]].to_string(index=False))

    if args.log_executions:
        from backtest.execution_log import log_all_signals
        log_path = args.execution_log_path or str(out_dir / "execution_log.csv")
        n_new = log_all_signals(dfs_by_indicator, symbol=args.symbol, horizon=main_horizon,
                                 path=log_path, entry_lag=args.entry_lag, entry_col=args.entry_col,
                                 cost_bps=args.cost_bps, slippage_bps=slippage_arg,
                                 drop_overlapping=args.drop_overlapping)
        print(f"\n[체결 로그] {log_path}에 {n_new}건 신규 기록 (보유기간={main_horizon}봉 기준, "
              f"exit_reason은 전부 'horizon_exit' — 실제 리스크관리 청산 사유 아님)")


if __name__ == "__main__":
    main()
