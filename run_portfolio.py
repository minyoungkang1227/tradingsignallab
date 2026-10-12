"""
감시리스트 여러 종목을 한 계좌로 동시 운용했을 때의 시뮬레이션 CLI (얇은 껍데기).

계산은 backtest/portfolio.py, 신호·통계·비용 설정은 run_backtest(run, build_compare_kwargs)와
backtest/profiles.py를 그대로 쓴다 — 여기서 지표나 백테스트 로직을 다시 구현하지 않는다.

예)
  python run_portfolio.py --market krx --profile realistic --sim-start 2024-01-01
  python run_portfolio.py --market krx --signals "델타트레이딩-골든크로스" --risk-pct 0.005 --max-positions 5

주의: --signals 를 시뮬레이션 구간과 같은 데이터로 골랐다면 결과는 표본 내(in-sample)다. 학습 구간에서 고른
신호를 --sim-start 이후 구간에서만 돌려야 표본 외 성과가 된다(run_backtest --oos-train-end 참고).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest as rb
from alerts.config import args_for_watch_item, load_watchlist
from backtest.portfolio import (buy_hold_benchmark, build_candidates, format_portfolio_report,
                                simulate_portfolio)
from backtest.profiles import apply_profile


def build_parser():
    p = argparse.ArgumentParser(description="감시리스트 포트폴리오 시뮬레이션")
    p.add_argument("--watchlist", default="alerts/watchlist.json", help="종목 목록 JSON. 기본 alerts/watchlist.json")
    p.add_argument("--market", default="krx", choices=["krx", "upbit"],
                   help="한 번에 한 시장만 돌린다(달력·연환산 기준이 달라 섞으면 해석이 흐려진다). 기본 krx")
    p.add_argument("--start", default=None, help="krx 데이터 시작(YYYYMMDD). 기본은 감시리스트 설정(최근 2년)")
    p.add_argument("--end", default=None, help="krx 데이터 끝(YYYYMMDD). 기본 오늘")
    p.add_argument("--profile", default="realistic", choices=["realistic", "none"],
                   help="실행 조건 프리셋. 기본 realistic(다음 봉 시가 진입, 비용, 슬리피지). none이면 비용 0·당일 종가 진입")
    p.add_argument("--cost-bps", dest="cost_bps", type=float, default=None, help="프리셋의 왕복비용을 덮어씀(bp)")
    p.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=None, help="프리셋의 슬리피지를 덮어씀(bp)")
    p.add_argument("--signals", default=None, help="쓸 신호 라벨을 쉼표로. 기본 전부. 라벨은 run_backtest 요약표의 signal 열")
    p.add_argument("--hold-bars", dest="hold_bars", type=int, default=None,
                   help="배리어 최대 보유 봉수. 기본 horizons 중간값(5)")
    p.add_argument("--sim-start", dest="sim_start", default=None, help="이 날짜 이후 진입만 시뮬레이션(YYYY-MM-DD)")
    p.add_argument("--sim-end", dest="sim_end", default=None, help="이 날짜 이전 진입만 시뮬레이션(YYYY-MM-DD)")
    p.add_argument("--risk-pct", dest="risk_pct", type=float, default=0.005,
                   help="1거래 위험(1R)을 자본의 몇 %%로 할지(소수). 기본 0.005 = 0.5%%")
    p.add_argument("--max-positions", dest="max_positions", type=int, default=5, help="최대 동시 보유 종목 수. 기본 5")
    p.add_argument("--max-gross", dest="max_gross", type=float, default=1.0, help="총 노출 상한(자본 대비). 기본 1.0=레버리지 없음")
    p.add_argument("--max-weight", dest="max_weight", type=float, default=0.25, help="종목당 비중 상한. 기본 0.25")
    p.add_argument("--min-fill-frac", dest="min_fill_frac", type=float, default=0.5,
                   help="남은 여력이 의도 비중의 이 비율 미만이면 건너뜀(이상이면 줄여서 매수). 기본 0.5")
    p.add_argument("--priority", default="low_vol", choices=["low_vol", "random"],
                   help="같은 날 후보가 많을 때 순서. low_vol=변동성 낮은 종목 먼저(결정적), random=시드 고정 무작위")
    p.add_argument("--seed", type=int, default=0, help="--priority random의 시드. 기본 0")
    p.add_argument("--seeds", type=int, default=1,
                   help="--priority random일 때 시드 0..N-1로 반복해 결과 분포를 보여준다(우선순위 가정에 대한 민감도). 기본 1")
    p.add_argument("--data-label", dest="data_label", default=None, choices=["real", "synthetic"],
                   help="데이터가 실제 시장 데이터인지 합성인지 리포트 머리글에 표시. 기본 None이면 '미표시'")
    p.add_argument("--output", default="output")
    return p


def prepare_symbol(item, ns, loader):
    """종목 하나의 신호 df와 compare 인자를 만든다. 실패는 호출자가 잡아서 다음 종목으로 넘어간다."""
    args = args_for_watch_item(item)
    if item["market"] == "krx":
        if ns.start: args.start = ns.start
        if ns.end: args.end = ns.end
    args.profile = None if ns.profile == "none" else ns.profile
    if args.profile:
        apply_profile(args, args.profile, explicit=set())
    if ns.cost_bps is not None: args.cost_bps = ns.cost_bps
    if ns.slippage_bps is not None: args.slippage_bps = ns.slippage_bps
    df = loader(args)
    _summary, _results, dfs, slip = rb.run(df, args)
    return dfs, rb.build_compare_kwargs(args, slip)


def run_portfolio(ns, loader=None, out_dir: Path | None = None):
    loader = loader or rb.load_data
    items = [i for i in load_watchlist(ns.watchlist) if i["market"] == ns.market]
    if not items:
        raise SystemExit(f"{ns.watchlist}에 market={ns.market} 종목이 없습니다.")
    dfs_by_symbol, ck_by_symbol, load_fail = {}, {}, {}
    for item in items:
        label = item.get("label", item["symbol"])
        try:
            dfs, ck = prepare_symbol(item, ns, loader)
        except SystemExit as e:
            load_fail[item["symbol"]] = f"데이터 로딩 실패 - {e}"; print(f"[건너뜀] {label}: {load_fail[item['symbol']]}"); continue
        except Exception as e:
            load_fail[item["symbol"]] = f"{type(e).__name__}: {e}"; print(f"[건너뜀] {label}: {load_fail[item['symbol']]}"); continue
        dfs_by_symbol[item["symbol"]] = dfs; ck_by_symbol[item["symbol"]] = ck
    if not dfs_by_symbol:
        raise SystemExit("데이터를 불러온 종목이 하나도 없습니다.")

    any_ck = next(iter(ck_by_symbol.values()))
    horizons = any_ck["horizons"]
    hold = ns.hold_bars or horizons[len(horizons) // 2]
    signals = [s.strip() for s in ns.signals.split(",")] if ns.signals else None
    cand, closes, fails = build_candidates(dfs_by_symbol, ck_by_symbol, hold, signals=signals,
                                           lo=ns.sim_start, hi=ns.sim_end)
    fails = {**load_fail, **fails}
    ppy = 365 if ns.market == "upbit" else 252
    sim_kw = dict(sl_mult=any_ck.get("sl_mult", 1.0), risk_pct=ns.risk_pct, max_positions=ns.max_positions,
                  max_gross=ns.max_gross, max_weight=ns.max_weight, min_fill_frac=ns.min_fill_frac,
                  periods_per_year=ppy)
    seeds = range(ns.seeds) if (ns.priority == "random" and ns.seeds > 1) else [ns.seed]
    results = [simulate_portfolio(cand, closes, priority=ns.priority, seed=sd, **sim_kw) for sd in seeds]
    res = results[0]
    if res.stats.get("n_trades", 0):
        res.benchmark = buy_hold_benchmark(closes, res.equity.index[0], res.equity.index[-1], ppy)
    print(f"[설정] 시장 {ns.market}, 프로필 {ns.profile}, 보유 최대 {hold}봉, 후보 거래 {len(cand)}건, "
          f"종목 {len(closes)}개, 신호 {'전부' if signals is None else signals}")
    from backtest.report_common import portfolio_header
    print(portfolio_header(res, any_ck, len(cand), ns.data_label) + "\n")
    print(format_portfolio_report(res, fails))
    if len(results) > 1:
        tr = np.array([r.stats.get("total_return", np.nan) for r in results])
        dd = np.array([r.stats.get("max_drawdown", np.nan) for r in results])
        print(f"[우선순위 민감도] 무작위 시드 {len(results)}개: 누적수익 최소 {np.nanmin(tr):+.1%} / 중앙 {np.nanmedian(tr):+.1%} / "
              f"최대 {np.nanmax(tr):+.1%}, MDD 최악 {np.nanmin(dd):.1%} / 중앙 {np.nanmedian(dd):.1%}. "
              f"범위가 넓으면 결과가 '어떤 종목을 먼저 사느냐'에 좌우된다는 뜻입니다.")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cand.to_csv(out_dir / "portfolio_candidates.csv", index=False, encoding="utf-8-sig")
        res.trades.to_csv(out_dir / "portfolio_trades.csv", index=False, encoding="utf-8-sig")
        res.equity.to_csv(out_dir / "portfolio_equity.csv", header=True, encoding="utf-8-sig")
        print(f"[저장] {out_dir / 'portfolio_trades.csv'}, portfolio_equity.csv, portfolio_candidates.csv")
    return res, results, cand, fails


def main(argv=None):
    ns = build_parser().parse_args(argv)
    run_portfolio(ns, out_dir=Path(ns.output))


if __name__ == "__main__":
    main()
