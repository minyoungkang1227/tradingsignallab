#!/usr/bin/env python3
"""
감시 포트폴리오의 일일 리스크 요약과 한도 위반을 텔레그램으로 보낸다 (2026-10 리스크 중심 재구성).

예전 알림(check_signals.py)은 "오늘 새로 뜬 매매 신호"를 보냈다. 지금 저장소의 본론은 리스크 관리이므로
알림도 "지금 들고 있는 포트폴리오에 얼마의 위험이 실려 있고, 한도를 넘었는가"로 바꿨다. 신호 알림은 기본
꺼짐이고 watchlist 항목에 "signal_alerts": true를 적어야만 동작한다.

[무엇을 보내나]  시장(krx / upbit)마다 한 통.
  1. 일일 리스크 요약 — 내일 1일 VaR·ES(FHS 97.5%), 오늘 포트폴리오 손익과 ES 초과 여부, 리스크 모델
     백테스트 판정(적합/부적합/판정불가, 초과 횟수), 네 한도의 오늘 상태(ES 축소 배율, 종목 예산 초과 수,
     낙폭 서킷, 현재 낙폭). 조용한 날에도 보낸다 — "시스템이 살아 있다"는 하트비트다.
  2. 한도 위반 — 오늘 손실이 ES를 넘었거나, 낙폭 서킷이 켜져 있거나, 종목 예산을 넘은 종목이 있으면
     메시지 머리에 ⚠ 표시와 위반 목록을 붙인다.
  3. 데이터 품질 중단 — 종목 수집이 품질 게이트에서 중단되면 그 사실을 알리고 그 시장의 리스크 계산은
     건너뛴다(모순된 데이터로 계산한 VaR를 보내는 것보다 안 보내는 것이 낫다).

[비중]  watchlist 항목의 "weight"(자산 대비, 숏은 음수). 한 시장 안에서 weight가 하나라도 빠져 있으면 그 시장은
동일비중으로 본다(일부만 적힌 비중은 해석이 모호해서 쓰지 않는다). 매일 같은 비중을 든다고 가정한다.

[재실행 안전]  같은 날짜의 요약을 이미 보냈으면 다시 보내지 않는다(state.json의 risk:<market> 키).

[설계 원칙]  데이터 수집은 run_backtest.load_data()(품질 게이트 포함), 리스크 계산은 risk/ 모듈을 그대로
쓴다. 여기서는 숫자를 새로 계산하지 않고 모아서 문장으로 만들 뿐이다 — 알림이 보는 VaR와 run_risk.py가
보는 VaR가 같아야 하기 때문이다.

[한계]  일봉 기준 하루 한 번이라 장중 손실에는 반응하지 못한다. 한도는 같은 데이터에서 도출되므로 표본 내다.
비용·슬리피지는 반영하지 않는다(리스크 측정은 수익률 분포만 쓴다).

사용법(저장소 루트에서): python -m alerts.check_risk [--watchlist alerts/watchlist.json] [--state alerts/state.json]
                          [--dry-run] [--alpha 0.975] [--window 250] [--method fhs]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd

import run_backtest as rb
from alerts.config import args_for_watch_item, load_watchlist
from alerts.state import load_state, save_state
from alerts.telegram import send_telegram
from risk.limits import apply_controls, derive_limits
from risk.positions import closes_frame, constant_weights, portfolio_returns
from risk.var import compute_var
from risk.var_backtest import var_backtest

MARKET_KR = {"krx": "국내주식", "upbit": "코인"}


def resolve_weights(items: list[dict]) -> tuple[dict, str]:
    """시장 하나의 항목들에서 비중 dict를 만든다. 반환: (비중, 출처 설명)."""
    if all("weight" in it for it in items):
        return {it["symbol"]: float(it["weight"]) for it in items}, "watchlist weight"
    n = len(items)
    return {it["symbol"]: 1.0 / n for it in items}, f"동일비중 1/{n} (weight 미지정 항목 있음)"


def load_market_closes(items: list[dict]) -> tuple[dict, list[str]]:
    """항목별로 run_backtest.load_data()를 불러 종가를 모은다. 반환: (종가 dict, 문제 목록)."""
    closes, problems = {}, []
    for it in items:
        label = it.get("label", it["symbol"])
        try:
            df = rb.load_data(args_for_watch_item(it))
        except SystemExit as e:           # 품질 게이트 중단 또는 인자 오류
            problems.append(f"{label}({it['symbol']}): 데이터 중단 — {str(e).splitlines()[0]}")
            continue
        except Exception as e:            # noqa: BLE001 — 한 종목 실패가 시장 전체를 죽이면 안 됨
            problems.append(f"{label}({it['symbol']}): 로딩 오류 — {type(e).__name__}: {e}")
            continue
        if df is None or "close" not in df.columns or len(df) < 60:
            problems.append(f"{label}({it['symbol']}): 데이터 부족({0 if df is None else len(df)}봉)")
            continue
        closes[it["symbol"]] = df["close"].astype(float)
    return closes, problems


def assess_market(closes: dict, weights: dict, *, alpha: float = 0.975, window: int = 250,
                  method: str = "fhs") -> dict:
    """리스크 계산 본체. 네트워크 없이 테스트할 수 있게 수집과 분리. 반환은 보고용 dict."""
    px = closes_frame(closes)
    w = constant_weights(weights, px.index)
    rp = portfolio_returns(w, px)
    v = compute_var(rp, method, alpha, window)
    bt = var_backtest(rp, v["var"], alpha)
    limits = derive_limits(rp, alpha=alpha, window=window, method=method)
    ctrl = apply_controls(w, px, limits, window=window, method=method)
    last = px.index[-1]
    lg = ctrl.log.iloc[-1]
    today_ret = float(rp.iloc[-1]) if np.isfinite(rp.iloc[-1]) else float("nan")
    today_es = float(v["es"].iloc[-1]) if np.isfinite(v["es"].iloc[-1]) else float("nan")
    # '내일' 예측 = 마지막 봉까지의 정보로 한 칸 더 계산해야 하지만, compute_var는 t일 값을 t-1 정보로
    # 만들므로 내일 값은 오늘 수익률을 반영한 다음 행이다. 한 행을 덧붙여 구한다.
    rp_ext = pd.concat([rp, pd.Series([0.0], index=[last + pd.Timedelta(days=1)])])
    v_ext = compute_var(rp_ext, method, alpha, window)
    next_var, next_es = float(v_ext["var"].iloc[-1]), float(v_ext["es"].iloc[-1])
    breaches = []
    if np.isfinite(today_ret) and np.isfinite(today_es) and -today_ret > today_es:
        breaches.append(f"오늘 손실 {-today_ret:.2%} > ES {today_es:.2%} (내일 신규 증가 차단)")
    if bool(lg["dd_active"]):
        breaches.append(f"낙폭 서킷 작동 중 (낙폭 {lg['dd']:.1%} > 한도 {limits.dd_limit:.1%}, 노출 절반)")
    if int(lg["n_name_capped"]) > 0:
        breaches.append(f"종목 리스크 예산 초과 {int(lg['n_name_capped'])}종목 (예산 {limits.name_budget:.2%})")
    return dict(date=last, n_symbols=len(closes), today_ret=today_ret, today_es=today_es,
                next_var=next_var, next_es=next_es, backtest=bt, limits=limits,
                scale_es=float(lg["scale_es"]), undefined=bool(lg["undefined"]), dd=float(lg["dd"]),
                breaches=breaches, method=method, alpha=alpha)


def format_message(market: str, weights: dict, weight_note: str, a: dict, problems: list[str]) -> str:
    bt = a["backtest"]
    head = ("⚠ 한도 위반 — " if a["breaches"] else "") + f"리스크 일일 요약 [{MARKET_KR.get(market, market)}] {a['date'].date()}"
    L = [head]
    if a["undefined"]:
        L.append("리스크 측정 불가(히스토리 부족) — 통제 비중 0. 데이터가 더 쌓여야 합니다.")
    L.append(f"내일 1일 VaR {a['next_var']:.2%} / ES {a['next_es']:.2%} ({a['method']}, {a['alpha']:.1%})")
    if np.isfinite(a["today_ret"]):
        L.append(f"오늘 손익 {a['today_ret']:+.2%} (ES 한도 {a['today_es']:.2%})")
    kp = bt.kupiec["p"]
    L.append(f"모델 백테스트: {bt.verdict} (초과 {bt.n_exceed}/{bt.n}, 기대 {bt.expected_exceed:.1f}"
             + (f", Kupiec p={kp:.2f}" if np.isfinite(kp) else "") + ")")
    L.append(f"한도: ES 축소 배율 {a['scale_es']:.2f} (목표 {a['limits'].es_target:.2%}), 낙폭 {a['dd']:.1%}/{a['limits'].dd_limit:.1%}")
    if a["breaches"]:
        L.append("위반: " + " / ".join(a["breaches"]))
    L.append(f"비중: {', '.join(f'{k} {v:.0%}' for k, v in weights.items())} — {weight_note}")
    if problems:
        L.append("제외: " + " / ".join(problems))
    L.append("(수치는 과거 분포 기반 추정이며 미래 손실의 상한이 아님. 투자자문 아님)")
    return "\n".join(L)


def run_market(market: str, items: list[dict], state: dict, *, alpha: float, window: int, method: str) -> str | None:
    """시장 하나를 처리해 보낼 메시지를 돌려준다(없으면 None). state는 자리에서 갱신."""
    weights, note = resolve_weights(items)
    closes, problems = load_market_closes(items)
    missing = [s for s in weights if s not in closes]
    if missing:
        # 비중이 지정된 종목이 빠지면 포트폴리오가 달라지므로 계산하지 않고 알린다
        return (f"⚠ 리스크 계산 중단 [{MARKET_KR.get(market, market)}] — 종목 {len(missing)}개를 쓸 수 없음\n"
                + "\n".join(problems) + "\n(모순되거나 부족한 데이터로 VaR를 내지 않습니다)")
    a = assess_market(closes, weights, alpha=alpha, window=window, method=method)
    key = f"risk:{market}"
    stamp = str(a["date"].date())
    if state.get(key, {}).get("last_date") == stamp:
        print(f"[{market}] {stamp} 요약은 이미 보냈음 — 건너뜀")
        return None
    state[key] = dict(last_date=stamp, verdict=a["backtest"].verdict, breaches=len(a["breaches"]))
    return format_message(market, weights, note, a, problems)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watchlist", default="alerts/watchlist.json")
    p.add_argument("--state", default="alerts/state.json")
    p.add_argument("--dry-run", action="store_true", help="텔레그램으로 보내지 않고 콘솔에만 출력")
    p.add_argument("--alpha", type=float, default=0.975)
    p.add_argument("--window", type=int, default=250)
    p.add_argument("--method", default="fhs", choices=["historical", "parametric", "fhs"])
    args = p.parse_args(argv)

    watchlist = load_watchlist(args.watchlist)
    state = load_state(args.state)
    by_market: dict[str, list[dict]] = {}
    for it in watchlist:
        if it.get("risk", True):              # "risk": false 면 리스크 포트폴리오에서 제외
            by_market.setdefault(it["market"], []).append(it)

    messages = []
    for market, items in by_market.items():
        msg = run_market(market, items, state, alpha=args.alpha, window=args.window, method=args.method)
        if msg:
            messages.append(msg)

    if not messages:
        print("[결과] 보낼 리스크 요약 없음.")
    for msg in messages:
        print("----")
        print(msg)
        if args.dry_run:
            continue
        try:
            send_telegram(msg)
        except Exception as e:  # noqa: BLE001
            print(f"[경고] 텔레그램 전송 실패: {type(e).__name__}: {e}", file=sys.stderr)
    save_state(args.state, state)
    print(f"[저장] {args.state}")


if __name__ == "__main__":
    main()
