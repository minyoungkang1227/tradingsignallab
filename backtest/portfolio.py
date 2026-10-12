"""
포트폴리오 단위 거래 시뮬레이터 — 여러 종목의 신호를 한 계좌에서 동시에 운용했을 때의 결과.

[왜 필요한가]
이벤트 스터디·배리어 통계는 "거래 하나하나가 독립이고 자본이 무한하다"고 가정한다. 실제 계좌는
(1) 같은 날 신호가 여러 종목에 겹치고, (2) 동시에 들 수 있는 포지션과 자본이 유한하며,
(3) 한 종목에서 롱·숏 신호가 충돌하고, (4) 보유 중 평가손실(MDD)이 있다. 종목별 통계가 모두 좋아도
이 제약 때문에 못 먹는 거래가 생기고, 같은 날 몰린 손실이 계좌 MDD를 키울 수 있다. 이 모듈은 그 차이를
측정한다.

[입력] 후보 거래 = 각 신호가 낸 "이 이벤트를 트리플 배리어로 매매했다면" 결과(진입일·청산일·순수익·변동성).
  시뮬레이터는 후보를 시간순으로 보며 규칙에 따라 실제로 잡을 것만 고른다. 잡지 못한 후보는 일어나지
  않은 거래다(미래를 보고 고르지 않는다).

[규칙 — 모두 매개변수이며 기본값은 임의적 가정이다. 결과가 이 가정에 민감한지 바꿔서 확인할 것]
  포지션 크기   1거래 위험 고정: 1R = 자본의 risk_pct. 손절폭이 sl_mult×target_vol이므로
                비중 w = risk_pct / (sl_mult × target_vol). 변동성이 큰 종목은 자동으로 작게 산다.
                w는 max_weight로 자른다(손절 갭이 나면 1R보다 더 잃을 수 있고, 한 종목 쏠림을 막기 위함).
                자본은 그 거래 진입 전날 종가 기준 평가자산(복리).
  동시 보유     최대 max_positions개. 가득 차면 새 후보는 건너뜀(no_slot).
  총 노출       Σ비중 <= max_gross(기본 1.0 = 레버리지 없음). 남은 여력이 의도 비중의 min_fill_frac 미만이면
                건너뜀(no_capital), 이상이면 남은 만큼만 사서 비중을 줄임(scaled).
  종목 충돌     같은 날 같은 종목에 롱·숏 후보가 함께 있으면 둘 다 건너뜀(conflict) — 방향 확신이 없다고 본다.
                같은 방향 후보가 여러 신호에서 겹치면 한 거래로 합친다(merged_dup, 물타기 없음).
                이미 보유 중인 종목의 새 후보는 방향과 무관하게 건너뜀(holding) — 청산 전 추가·반전 없음.
                청산한 날 같은 종목에 재진입하지 않는다.
  우선순위      같은 날 후보가 슬롯·자본보다 많을 때: 'low_vol'(변동성 낮은 종목 먼저, 기본, 결정적) 또는
                'random'(시드 고정 난수). 이것은 임의 규칙이므로 priority='random'을 여러 시드로 돌려 민감도를
                확인할 수 있다.
  평가         일별 평가자산 = 현금성 자산 + Σ 미청산 포지션 평가손익(종가 기준). 청산일에는 순수익(비용·
                슬리피지 차감)으로 확정한다. MDD는 이 평가자산 기준(실현손익만 보면 보유 중 손실을 놓친다).

[한계]
  - 종목별 달력이 다르면(예: KRX와 코인) 합집합 달력에서 휴장일은 직전 종가를 이어받는다. 연환산 통계는
    시장 하나씩 돌릴 때 의미가 가장 분명하다.
  - 일봉 기준이라 같은 날 진입·청산 순서를 알 수 없다. 청산을 먼저 처리해 자본을 풀어준 뒤 진입한다.
  - 신호 선택(어떤 신호를 쓸지)을 시뮬레이션 구간과 같은 데이터로 고르면 표본 내 결과다. 학습 구간에서 고른
    신호를 검증 구간(lo~hi)에서만 시뮬레이션해야 표본 외 성과가 된다.
  - 합성 데이터 결과는 실제 시장 성과가 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SKIP_KEYS = ("no_slot", "no_capital", "conflict", "holding", "merged_dup", "invalid_vol", "scaled", "capped")


@dataclass
class PortfolioResult:
    equity: pd.Series              # 일별 평가자산(시작 1.0 기준 비율, 초기자본 곱함)
    trades: pd.DataFrame           # 실제로 잡은 거래
    counts: dict                   # 건너뜀·조정 사유별 횟수
    stats: dict                    # 성과 요약
    params: dict = field(default_factory=dict)
    benchmark: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)   # 후보 생성에 실패한 종목 {symbol: 이유}


def build_candidates(dfs_by_symbol: dict, compare_kwargs_by_symbol: dict, hold_bars: int,
                     signals=None, lo=None, hi=None, signal_specs=None):
    """
    종목별 {지표: 신호 df}에서 후보 거래표를 만든다. 반환 (candidates, closes, failures).
    compare_kwargs_by_symbol: 종목별 compare_all 인자(시장마다 비용이 다르므로 종목별로 받는다).
    signals: 쓸 신호 라벨 목록(None이면 전부). lo/hi: 진입일 범위로 제한(표본 외 구간 시뮬레이션용).
    한 종목이 실패해도 나머지는 계속한다(실패는 failures에 이유와 함께 기록).
    """
    from backtest.engine import SIGNAL_SPECS
    from backtest.triple_barrier import realized_volatility, triple_barrier_returns

    specs = signal_specs if signal_specs is not None else SIGNAL_SPECS
    parts, closes, failures = [], {}, {}
    for sym, dfs in dfs_by_symbol.items():
        try:
            ck = compare_kwargs_by_symbol[sym]
            any_df = next(iter(dfs.values()))
            closes[sym] = any_df["close"].astype(float)
            for key, col, direction, label in specs:
                if signals is not None and label not in signals:
                    continue
                df = dfs.get(key)
                if df is None or col not in df.columns:
                    continue
                ev = df.index[df[col].fillna(False)]
                if len(ev) == 0:
                    continue
                vol = realized_volatility(df, span=ck.get("vol_span", 20))
                tb = triple_barrier_returns(
                    df, ev, direction=direction, pt_mult=ck.get("pt_mult", 2.0), sl_mult=ck.get("sl_mult", 1.0),
                    max_horizon=hold_bars, vol=vol, entry_lag=ck.get("entry_lag", 0), entry_col=ck.get("entry_col"),
                    cost_bps=ck.get("cost_bps", 0.0), slippage_bps=ck.get("slippage_bps", 0.0),
                    use_intrabar=ck.get("use_intrabar", True), include_prices=True)
                if tb.empty:
                    continue
                tb = tb.reset_index(drop=True)
                tb.insert(0, "symbol", sym); tb.insert(1, "signal", label); tb.insert(2, "direction", direction)
                parts.append(tb)
        except Exception as e:                      # 한 종목 실패가 전체를 멈추지 않게
            failures[sym] = f"{type(e).__name__}: {e}"
            closes.pop(sym, None)
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["symbol", "signal", "direction", "entry_date", "exit_date", "bars_held", "ret",
                 "exit_reason", "target_vol", "entry_price", "gross_ret"])
    if len(cand):
        if lo is not None:
            cand = cand[cand["entry_date"] >= pd.Timestamp(lo)]
        if hi is not None:
            cand = cand[cand["entry_date"] <= pd.Timestamp(hi)]
        cand = cand.reset_index(drop=True)
    return cand, closes, failures


def simulate_portfolio(candidates: pd.DataFrame, closes: dict, sl_mult: float = 1.0, risk_pct: float = 0.005,
                       max_positions: int = 5, max_gross: float = 1.0, max_weight: float = 0.25,
                       min_fill_frac: float = 0.5, priority: str = "low_vol", seed: int = 0,
                       initial_capital: float = 1.0, periods_per_year: int = 252) -> PortfolioResult:
    """후보 거래를 모듈 docstring의 규칙대로 시간순 시뮬레이션한다."""
    if priority not in ("low_vol", "random"):
        raise ValueError("priority는 'low_vol' 또는 'random'")
    if not (0 < risk_pct < 1) or max_positions < 1 or max_gross <= 0 or max_weight <= 0 or sl_mult <= 0:
        raise ValueError("risk_pct(0~1), max_positions>=1, max_gross>0, max_weight>0, sl_mult>0 이어야 합니다.")
    params = dict(sl_mult=sl_mult, risk_pct=risk_pct, max_positions=max_positions, max_gross=max_gross,
                  max_weight=max_weight, min_fill_frac=min_fill_frac, priority=priority, seed=seed,
                  periods_per_year=periods_per_year)
    counts = {k: 0 for k in SKIP_KEYS}
    if candidates is None or len(candidates) == 0 or not closes:
        return PortfolioResult(pd.Series(dtype=float, name="equity"), pd.DataFrame(), counts,
                               {"n_trades": 0}, params)

    cand = candidates.sort_values(["entry_date", "symbol", "signal"], kind="mergesort").reset_index(drop=True)
    start, end = cand["entry_date"].min(), cand["exit_date"].max()
    cal = sorted(set().union(*[set(c.index) for c in closes.values()]))
    dates = [d for d in cal if start <= d <= end]
    px = {s: c.reindex(pd.DatetimeIndex(cal)).ffill() for s, c in closes.items()}   # 휴장일은 직전 종가
    by_day = {d: g for d, g in cand.groupby("entry_date")}
    rng = np.random.default_rng(seed)

    cash = float(initial_capital)          # 확정 손익을 반영한 자본(미청산 평가손익 제외)
    open_pos, taken, eq_dates, eq_vals, n_open_hist, gross_hist = [], [], [], [], [], []
    prev_equity = float(initial_capital)

    for d in dates:
        # 1) 오늘 청산되는 포지션 확정(비용·슬리피지 차감 순수익)
        still = []
        exited_today = set()
        for p in open_pos:
            if p["exit_date"] == d:
                pnl = p["notional"] * p["ret"]
                cash += pnl
                p["pnl"], p["pnl_r"] = pnl, pnl / p["risk_amt"]
                taken.append(p); exited_today.add(p["symbol"])
            else:
                still.append(p)
        open_pos = still

        # 2) 오늘 진입 후보 정리 — 충돌·중복·보유 중 규칙
        g = by_day.get(d)
        if g is not None:
            held = {p["symbol"] for p in open_pos}
            picks = []
            for sym, gs in g.groupby("symbol", sort=True):
                if sym in held or sym in exited_today:
                    counts["holding"] += len(gs); continue
                if gs["direction"].nunique() > 1:
                    counts["conflict"] += len(gs); continue
                counts["merged_dup"] += len(gs) - 1
                picks.append(gs.sort_values("signal", kind="mergesort").iloc[0])
            if priority == "low_vol":
                picks.sort(key=lambda r: (r["target_vol"], r["symbol"]))
            else:
                order = rng.permutation(len(picks)); picks = [picks[i] for i in order]
            for r in picks:
                tv = r["target_vol"]
                if not np.isfinite(tv) or tv <= 0:
                    counts["invalid_vol"] += 1; continue
                if len(open_pos) >= max_positions:
                    counts["no_slot"] += 1; continue
                w_int = risk_pct / (sl_mult * tv)
                w = min(w_int, max_weight)
                was_capped = w < w_int
                gross_now = sum(p["notional"] for p in open_pos) / prev_equity
                remaining = max_gross - gross_now
                if remaining < min_fill_frac * w:
                    counts["no_capital"] += 1; continue
                was_scaled = remaining < w
                if was_scaled:
                    w = remaining
                counts["capped"] += int(was_capped)        # 실제로 거래가 잡힌 경우만 센다
                counts["scaled"] += int(was_scaled)
                notional = w * prev_equity
                open_pos.append(dict(symbol=r["symbol"], signal=r["signal"], direction=int(r["direction"]),
                                     entry_date=d, exit_date=r["exit_date"], entry_price=float(r["entry_price"]),
                                     ret=float(r["ret"]), gross_ret=float(r["gross_ret"]),
                                     exit_reason=r["exit_reason"], target_vol=float(tv), weight=w,
                                     weight_intended=w_int, notional=notional, equity_at_entry=prev_equity,
                                     risk_amt=risk_pct * prev_equity))

        # 3) 일별 평가자산: 확정 자본 + 미청산 평가손익(종가 기준)
        unreal = 0.0
        for p in open_pos:
            unreal += p["notional"] * p["direction"] * (px[p["symbol"]].loc[d] / p["entry_price"] - 1.0)
        prev_equity = cash + unreal
        eq_dates.append(d); eq_vals.append(prev_equity)
        n_open_hist.append(len(open_pos))
        gross_hist.append(sum(p["notional"] for p in open_pos) / prev_equity if open_pos else 0.0)

    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name="equity")
    trades = pd.DataFrame(taken)
    stats = portfolio_stats(equity, trades, n_open_hist, gross_hist, periods_per_year, initial_capital)
    return PortfolioResult(equity, trades, counts, stats, params)


def portfolio_stats(equity: pd.Series, trades: pd.DataFrame, n_open, gross, periods_per_year: int,
                    initial_capital: float) -> dict:
    out = {"n_trades": int(len(trades))}
    if equity.empty:
        return out
    ret = equity.pct_change().dropna()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    total = equity.iloc[-1] / initial_capital - 1.0
    out.update(
        total_return=float(total),
        cagr=float((equity.iloc[-1] / initial_capital) ** (365.25 / days) - 1.0) if equity.iloc[-1] > 0 else float("nan"),
        ann_vol=float(ret.std(ddof=1) * np.sqrt(periods_per_year)) if len(ret) > 1 else float("nan"),
        sharpe=float(ret.mean() / ret.std(ddof=1) * np.sqrt(periods_per_year))
        if len(ret) > 1 and ret.std(ddof=1) > 0 else float("nan"),
        max_drawdown=float(dd.min()),
        mdd_date=dd.idxmin(),
        max_concurrent=int(max(n_open)) if len(n_open) else 0,
        avg_gross=float(np.mean(gross)) if len(gross) else float("nan"),
        time_in_market=float(np.mean([n > 0 for n in n_open])) if len(n_open) else float("nan"),
        n_days=int(len(equity)),
    )
    if len(trades):
        win = trades["pnl"] > 0
        gp, gl = float(trades.loc[win, "pnl"].sum()), float(-trades.loc[~win, "pnl"].sum())
        out.update(win_rate=float(win.mean()), avg_r=float(trades["pnl_r"].mean()),
                   profit_factor=(gp / gl) if gl > 0 else float("inf") if gp > 0 else float("nan"),
                   avg_weight=float(trades["weight"].mean()))
    return out


def buy_hold_benchmark(closes: dict, start, end, periods_per_year: int = 252) -> dict:
    """같은 기간·같은 종목을 동일비중으로 매일 재조정하며 들고 있었을 때의 총수익·MDD(참고용 기준선)."""
    cal = sorted(set().union(*[set(c.index) for c in closes.values()]))
    idx = pd.DatetimeIndex([d for d in cal if pd.Timestamp(start) <= d <= pd.Timestamp(end)])
    if len(idx) < 2:
        return {}
    rets = pd.concat([c.reindex(pd.DatetimeIndex(cal)).ffill().loc[idx].pct_change() for c in closes.values()],
                     axis=1).mean(axis=1).dropna()
    eq = (1 + rets).cumprod()
    return dict(total_return=float(eq.iloc[-1] - 1.0), max_drawdown=float((eq / eq.cummax() - 1).min()),
                n_symbols=len(closes))


def format_portfolio_report(res: PortfolioResult, failures: dict | None = None) -> str:
    s, c, p = res.stats, res.counts, res.params
    L = ["=== 포트폴리오 시뮬레이션 (여러 종목 동시 운용) ===",
         f"규칙: 1거래 위험 {p['risk_pct']:.2%}(1R), 최대 동시 {p['max_positions']}개, 총노출 <= {p['max_gross']:.0%}, "
         f"종목당 비중 <= {p['max_weight']:.0%}, 우선순위 {p['priority']}"]
    if failures:
        L.append("후보 생성 실패 종목: " + "; ".join(f"{k}({v})" for k, v in failures.items()))
    if s.get("n_trades", 0) == 0:
        L.append("잡힌 거래가 없습니다(후보가 없거나 모두 제약에 걸림). 성과를 계산할 수 없습니다.")
        L.append("건너뜀 내역: " + ", ".join(f"{k} {v}" for k, v in c.items() if v))
        return "\n".join(L)
    L.append(f"기간 {res.equity.index[0].date()} ~ {res.equity.index[-1].date()} ({s['n_days']}거래일), 거래 {s['n_trades']}건")
    L.append(f"누적수익 {s['total_return']:+.1%}, 연환산(CAGR) {s['cagr']:+.1%}, 변동성 {s['ann_vol']:.1%}, "
             f"샤프 {s['sharpe']:.2f}(무위험 0 가정), 최대낙폭(MDD, 평가자산 기준) {s['max_drawdown']:.1%} ({s['mdd_date'].date()})")
    L.append(f"거래: 승률 {s['win_rate']:.0%}, 평균 {s['avg_r']:+.2f}R, 손익비(PF) {s['profit_factor']:.2f}, 평균 비중 {s['avg_weight']:.1%}")
    L.append(f"포지션: 최대 동시 {s['max_concurrent']}개, 평균 총노출 {s['avg_gross']:.0%}, 시장 참여 {s['time_in_market']:.0%}")
    skipped = {k: v for k, v in c.items() if v}
    L.append("잡지 못했거나 조정된 후보: " + (", ".join(f"{k} {v}" for k, v in skipped.items()) if skipped else "없음")
             + "  (no_slot=슬롯 가득, no_capital=자본 부족, conflict=롱숏 충돌, holding=보유 중, "
               "merged_dup=동일 방향 중복 합침, scaled=여력만큼만 매수, capped=종목당 한도로 비중 축소)")
    if res.benchmark:
        b = res.benchmark
        L.append(f"참고 기준선(같은 {b['n_symbols']}종목 동일비중 보유): 누적 {b['total_return']:+.1%}, MDD {b['max_drawdown']:.1%}")
    L.append("주의: 가정(우선순위·비중·한도)에 따라 달라지는 시뮬레이션이며, 신호를 같은 구간 데이터로 골랐다면 표본 내 결과입니다.")
    return "\n".join(L)
