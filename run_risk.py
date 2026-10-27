"""
run_risk.py — 지정 포트폴리오의 시장리스크 측정과 리스크 모델 백테스트 CLI.

  python run_risk.py --market krx --symbols 005930,000660 --weights 0.5,0.5 --start 20150101 --end 20260930 --data-label real
  python run_risk.py --market upbit --symbols KRW-BTC --count 2000 --alpha 0.975 --methods fhs,historical

[무엇을 하나]
  1. 종목을 수집하고(데이터 품질 게이트 통과분만) '날짜 × 비중' 테이블(risk/positions.py)로 만든다.
     비중은 매일 같은 값(일별 리밸런싱 가정)이다.
  2. 포트폴리오 일별 수익률을 만든다(t-1 비중 × t 수익률).
  3. 방법별(historical / parametric / fhs)로 1일 VaR·ES를 롤링으로 계산한다.
  4. 각 VaR를 실현 손실과 비교해 Kupiec·Christoffersen·조건부 커버리지·바젤 신호등으로 백테스트한다.
  5. 표준 요약 블록 + 방법별 판정표를 출력하고 CSV로 저장한다.
  6. --controls: 한도를 ES에서 도출(risk/limits.py)해 비중에 적용하고, 통제 전/후를 비교한다. 핵심 판정은
     '통제 후 수익률이 VaR 백테스트를 통과하는가'이고 수익률·MDD 변화는 통제의 비용으로 함께 보고한다.

[왜 지정 포트폴리오인가]
리스크 모델의 적합성은 '포지션이 돈이 되는가'와 무관하게 가격 데이터만으로 검증할 수 있다. 신호 전략의
손익은 표본 내일 가능성이 커서 먼저 쓰지 않는다. 신호가 낸 포지션은 같은 비중 테이블 모양으로 바꿔 넣으면
이 스크립트가 그대로 동작한다(다음 단계).

[정직한 한계]
  - 모든 방법은 과거 수익률이 미래 분포를 대표한다는 가정 위에 있다.
  - '적합'은 틀렸다는 증거를 못 찾았다는 뜻이다. 250일·99%에서 검정력은 약하다(초과 2배도 통과).
  - 같은 표본에서 window/λ를 고르고 백테스트하면 표본 내다. 기본값은 관행값(250일, λ=0.94)이며 최적화하지 않았다.
  - 1일 VaR다. 청산 기간·유동성은 반영하지 않았다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.report_common import format_block
from data.collect import collect_universe
from risk.positions import closes_frame, constant_weights, gross_exposure, portfolio_returns
from risk.var import METHODS, compute_var
from risk.var_backtest import format_var_backtest, var_backtest
from risk.limits import apply_controls, compare_controlled, derive_limits


def _fetch_one(symbol: str, args) -> pd.DataFrame:
    from data.upbit import fetch_upbit_ohlcv
    from data.krx import fetch_krx_ohlcv
    if args.market == "upbit":
        return fetch_upbit_ohlcv(market=symbol, unit="day", count=args.count)
    if not args.start or not args.end:
        sys.exit("--market krx 에는 --start / --end (YYYYMMDD)가 필요합니다.")
    return fetch_krx_ohlcv(symbol, args.start, args.end)


def parse_weights(symbols: list[str], text: str | None) -> dict:
    if not text:
        return {s: 1.0 / len(symbols) for s in symbols}
    vals = [float(v) for v in text.split(",")]
    if len(vals) != len(symbols):
        sys.exit(f"--weights 개수({len(vals)})가 --symbols 개수({len(symbols)})와 다릅니다.")
    return dict(zip(symbols, vals))


def run(closes: dict, weights: dict, methods, alpha: float, window: int):
    """계산 본체 — 네트워크 없이 테스트할 수 있게 수집과 분리."""
    px = closes_frame(closes)
    w = constant_weights(weights, px.index)
    rp = portfolio_returns(w, px)
    results = {}
    rows = []
    for m in methods:
        v = compute_var(rp, m, alpha, window)
        bt = var_backtest(rp, v["var"], alpha)
        results[m] = (v, bt)
        rows.append(dict(method=m, **bt.summary_row()))
    table = pd.DataFrame(rows)
    return rp, w, results, table


def run_controls(closes: dict, weights: dict, alpha: float, window: int, method: str, es_quantile: float,
                 dd_quantile: float, seed: int = 0):
    """한도 도출 → 적용 → 전/후 비교 → 통제 후 VaR 백테스트. 네트워크 없이 테스트 가능하게 분리."""
    px = closes_frame(closes)
    w = constant_weights(weights, px.index)
    rp = portfolio_returns(w, px)
    limits = derive_limits(rp, alpha=alpha, es_quantile=es_quantile, dd_quantile=dd_quantile, window=window,
                           method=method, seed=seed)
    res = apply_controls(w, px, limits, window=window, method=method)
    cmp = compare_controlled(res)
    # 통제 후 수익률의 VaR 백테스트: VaR는 통제 후 수익률로 다시 추정한다(통제된 포트폴리오가 실제로 운용되는 것이므로)
    v_ctrl = compute_var(res.returns, method, alpha, window)
    bt_ctrl = var_backtest(res.returns, v_ctrl["var"], alpha)
    v_raw = compute_var(res.raw_returns, method, alpha, window)
    bt_raw = var_backtest(res.raw_returns, v_raw["var"], alpha)
    return limits, res, cmp, bt_raw, bt_ctrl


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    weights = parse_weights(symbols, args.weights)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        sys.exit(f"알 수 없는 --methods: {bad} (가능: {METHODS})")

    coll = collect_universe(symbols, lambda s: _fetch_one(s, args), quality_gate=not args.skip_quality_gate)
    missing = [s for s in symbols if s not in coll.ok]
    if missing:
        sys.exit(f"다음 종목을 쓸 수 없어 중단합니다(비중이 지정된 종목은 하나라도 빠지면 포트폴리오가 달라짐): "
                 f"{ {s: coll.failed.get(s) for s in missing} }")
    closes = {s: coll.ok[s]["close"] for s in symbols}

    rp, w, results, table = run(closes, weights, methods, args.alpha, args.window)
    valid = rp.dropna()
    period = f"{valid.index[0].date()}~{valid.index[-1].date()} ({len(valid)}거래일)" if len(valid) else "없음"
    gross = float(gross_exposure(w).iloc[0])
    print(format_block(
        "시장리스크 — 1일 VaR/ES 측정과 리스크 모델 백테스트", args.data_label, period,
        f"일별 리밸런싱 가정, 비용·슬리피지 미반영(리스크 측정에는 손익 비용이 아니라 수익률 분포만 쓴다)",
        f"종목 {len(symbols)}개, 총노출 {gross:.2f}, 비중 {weights}",
        f"신뢰수준 {args.alpha:.1%}, 창 {args.window}일, 검정: Kupiec·Christoffersen·조건부(p<0.05 기각)·바젤(250일·99%만)",
        ", ".join(f"{m}={results[m][1].verdict}" for m in methods)))
    for m in methods:
        v, bt = results[m]
        last = v.dropna().iloc[-1] if len(v.dropna()) else None
        print()
        print(format_var_backtest(bt, m))
        if last is not None:
            print(f"  최근 1일 예측: VaR {last['var']:.2%} / ES {last['es']:.2%} (자산 대비 손실률)")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.controls:
        method = methods[0] if len(methods) == 1 else ("fhs" if "fhs" in methods else methods[0])
        limits, cres, cmp, bt_raw, bt_ctrl = run_controls(closes, weights, args.alpha, args.window, method,
                                                          args.es_quantile, args.dd_quantile)
        print("\n=== 통제 계층 (한도 도출 → 적용 → 비용·효과) ===")
        print(f"기준 척도 {method} ES{args.alpha:.1%}. 한도 출처: {limits.describe()}")
        c = cres.counts()
        print(f"발동: 미정의(비중 0) {c['undefined']}일, ES 축소 {c['es_scaled']}일, 종목 예산 {c['name_capped_days']}일, "
              f"일일손실 차단 {c['daily_halts']}회, 낙폭 서킷 {c['dd_active_days']}일 / 전체 {c['days']}일")
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
            print(cmp.to_string(index=False))
        print(f"통제 전 VaR 백테스트: {bt_raw.verdict} (초과 {bt_raw.n_exceed}/{bt_raw.n}, Kupiec p={bt_raw.kupiec['p']:.3f}, "
              f"독립성 p={bt_raw.independence['p'] if np.isfinite(bt_raw.independence['p']) else float('nan'):.3f})")
        print(f"통제 후 VaR 백테스트: {bt_ctrl.verdict} (초과 {bt_ctrl.n_exceed}/{bt_ctrl.n}, Kupiec p={bt_ctrl.kupiec['p']:.3f}, "
              f"독립성 p={bt_ctrl.independence['p'] if np.isfinite(bt_ctrl.independence['p']) else float('nan'):.3f})")
        print("해석: 통제의 목적은 수익률이 아니라 손실 분포를 모델 범위 안에 두는 것. 수익률 감소는 비용이고, "
              "비용이 0이면 통제가 한 번도 발동하지 않았다는 뜻. 한도는 같은 표본에서 도출됐으므로 표본 내 결과.")
        cres.weights.to_csv(out / "risk_controlled_weights.csv", encoding="utf-8-sig")
        cres.log.to_csv(out / "risk_control_log.csv", encoding="utf-8-sig")
        cmp.to_csv(out / "risk_control_compare.csv", index=False, encoding="utf-8-sig")
        print(f"[저장] {out / 'risk_controlled_weights.csv'}, risk_control_log.csv, risk_control_compare.csv")
    table.to_csv(out / "risk_var_backtest.csv", index=False, encoding="utf-8-sig")
    series = pd.DataFrame({"portfolio_return": rp, **{f"var_{m}": results[m][0]["var"] for m in methods},
                           **{f"es_{m}": results[m][0]["es"] for m in methods}})
    series.to_csv(out / "risk_var_series.csv", encoding="utf-8-sig")
    print(f"\n[저장] {out / 'risk_var_backtest.csv'}, risk_var_series.csv")
    if args.data_label != "real":
        print("(데이터 라벨이 real이 아니면 위 수치는 실제 시장 성과·실제 리스크가 아님)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--market", choices=["upbit", "krx"], required=True)
    p.add_argument("--symbols", required=True, help="콤마로 구분한 종목 목록(예: 005930,000660 / KRW-BTC)")
    p.add_argument("--weights", default=None, help="콤마로 구분한 비중(자산 대비, 숏은 음수). 기본 동일비중. 합이 1을 넘으면 레버리지")
    p.add_argument("--start", default=None, help="krx 시작일 YYYYMMDD")
    p.add_argument("--end", default=None, help="krx 종료일 YYYYMMDD")
    p.add_argument("--count", type=int, default=2000, help="upbit 일봉 개수. 기본 2000")
    p.add_argument("--alpha", type=float, default=0.99, help="VaR 신뢰수준. 기본 0.99. 검정력이 필요하면 0.975/0.95도 함께 볼 것")
    p.add_argument("--window", type=int, default=250, help="역사적/FHS 표본 창, 롤링 σ 창. 기본 250일(관행값, 최적화 안 함)")
    p.add_argument("--methods", default="historical,parametric,fhs",
                   help="콤마로 구분한 VaR 방법. historical(경험분포) / parametric(정규·EWMA σ) / fhs(필터드 역사적). 기본 셋 다")
    p.add_argument("--data-label", dest="data_label", default=None, choices=["real", "synthetic"],
                   help="데이터가 실제 시장 데이터인지 합성인지 리포트 머리글에 표시. 코드는 스스로 판별하지 못한다")
    p.add_argument("--skip-quality-gate", dest="skip_quality_gate", action="store_true",
                   help="데이터 품질 게이트를 건너뛴다(기본은 켜짐: 모순 데이터 종목은 제외되고, 하나라도 빠지면 중단)")
    p.add_argument("--controls", action="store_true",
                   help="통제 계층을 켠다: ES에서 한도(ES 목표·종목 예산·낙폭)를 도출해 비중에 적용하고 전/후를 비교. 기본 꺼짐")
    p.add_argument("--es-quantile", dest="es_quantile", type=float, default=0.5,
                   help="--controls: ES 목표 = 롤링 ES 시계열의 이 분위. 기본 0.5(중앙값 = 평소 수준의 꼬리 위험을 상한으로)")
    p.add_argument("--dd-quantile", dest="dd_quantile", type=float, default=0.95,
                   help="--controls: 낙폭 한도 = 블록 부트스트랩 1년 MDD 분포의 이 분위. 기본 0.95")
    p.add_argument("--output", default="output", help="CSV 저장 폴더. 기본 output")
    return p


if __name__ == "__main__":
    main()
