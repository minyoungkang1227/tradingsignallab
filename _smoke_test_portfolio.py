"""backtest.portfolio 스모크테스트 — 기대값은 손계산. 합성 데이터 결과는 실제 시장 성과가 아님."""
import math
import numpy as np, pandas as pd
from backtest.portfolio import (simulate_portfolio, build_candidates, portfolio_stats, buy_hold_benchmark,
                                format_portfolio_report)

def close(a, b, tol=1e-9): return abs(a - b) <= tol
ix = pd.bdate_range("2024-01-01", periods=14)

def cand(sym, sig, direc, e, x, ret, gross, tv, ep, reason="pt"):
    return dict(symbol=sym, signal=sig, direction=direc, entry_date=ix[e], exit_date=ix[x], bars_held=x - e,
                ret=ret, exit_reason=reason, target_vol=tv, entry_price=ep, gross_ret=gross)
def cs(*rows): return pd.DataFrame(rows)
def flat(v=100.0): return pd.Series(v, index=ix, dtype=float)
def series(d, base=100.0):
    s = flat(base)
    for k, v in d.items(): s.iloc[k] = v
    return s
KW = dict(sl_mult=1.0, risk_pct=0.005, max_positions=5, max_gross=1.0, max_weight=0.25)

# 1) 롱 한 건 일별 평가자산 손계산: w=0.005/0.02=0.25, 평가 = 1 + 0.25*(종가/100-1), 청산일 순수익 0.036 확정
A = series({2: 102, 3: 101, 4: 103, 5: 105})
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 5, 0.036, 0.04, 0.02, 100.0)), {"A": A}, **KW)
exp = [1 + .25*.02, 1 + .25*.01, 1 + .25*.03, 1 + .25*.036]
assert len(r.equity) == 4 and all(close(a, b) for a, b in zip(r.equity.values, exp)), r.equity.values
assert close(r.stats["max_drawdown"], 1.0025/1.005 - 1) and close(r.stats["total_return"], 0.009)
assert close(r.trades.iloc[0]["pnl_r"], 0.25*0.036/0.005) and r.stats["n_trades"] == 1      # 1.8R
print(f"[PASS] 1) 롱 평가자산 {['%.4f' % v for v in exp]}, MDD {1.0025/1.005-1:.5f}, 실현 1.8R 손계산 일치")

# 2) 숏은 가격 하락이 이익
B = series({2: 49, 3: 48}, base=50.0)
r = simulate_portfolio(cs(cand("B", "s", -1, 2, 4, 0.03, 0.034, 0.02, 50.0, "time")), {"B": B}, **KW)
assert close(r.equity.iloc[0], 1 + .25*(-1)*(49/50 - 1)) and close(r.equity.iloc[1], 1 + .25*(-1)*(48/50 - 1))
print("[PASS] 2) 숏 평가: 가격 50->49이면 +0.005, 48이면 +0.01")

# 3) 복리: 두 번째 거래는 첫 거래 확정 후 자본(1.01)으로 사이징. w=0.005/0.04=0.125 -> 명목 0.12625
A3, B3 = series({4: 100}), flat()
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 4, 0.04, 0.044, 0.02, 100.0),
                          cand("B", "s", 1, 6, 8, 0.01, 0.014, 0.04, 100.0)), {"A": A3, "B": B3}, **KW)
t = r.trades.sort_values("entry_date").reset_index(drop=True)
assert close(t.loc[0, "pnl"], 0.25*0.04)                                   # 0.01
assert close(t.loc[1, "equity_at_entry"], 1.01) and close(t.loc[1, "notional"], 0.125*1.01)
assert close(r.equity.iloc[-1], 1.01 + 0.125*1.01*0.01)
print("[PASS] 3) 복리: 두 번째 거래 명목 0.12625 = 0.125 x 1.01, 최종 자본 1.0112625")

# 4) 슬롯 제한 + 우선순위(낮은 변동성 먼저)
two = lambda tvA, tvB: cs(cand("A", "s", 1, 2, 6, 0.01, 0.014, tvA, 100.0), cand("B", "s", 1, 2, 6, 0.01, 0.014, tvB, 100.0))
cl = {"A": flat(), "B": flat()}
r = simulate_portfolio(two(0.02, 0.03), cl, **{**KW, "max_positions": 1})
assert list(r.trades["symbol"]) == ["A"] and r.counts["no_slot"] == 1
r = simulate_portfolio(two(0.03, 0.02), cl, **{**KW, "max_positions": 1})
assert list(r.trades["symbol"]) == ["B"] and r.counts["no_slot"] == 1
print("[PASS] 4) 슬롯 1개: 변동성 낮은 쪽만 진입(A/B 뒤집으면 B), 나머지 no_slot=1")

# 5) 충돌 / 동일 방향 중복 합침 / 보유 중 / 청산일 재진입 금지
r = simulate_portfolio(cs(cand("A", "s1", 1, 2, 5, .01, .014, .02, 100.0), cand("A", "s2", -1, 2, 5, .01, .014, .02, 100.0)), {"A": flat()}, **KW)
assert r.stats["n_trades"] == 0 and r.counts["conflict"] == 2
r = simulate_portfolio(cs(cand("A", "s1", 1, 2, 5, .01, .014, .02, 100.0), cand("A", "s2", 1, 2, 5, .01, .014, .02, 100.0)), {"A": flat()}, **KW)
assert r.stats["n_trades"] == 1 and r.counts["merged_dup"] == 1 and r.trades.iloc[0]["signal"] == "s1"
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 6, .01, .014, .02, 100.0), cand("A", "s", -1, 4, 8, .01, .014, .02, 100.0)), {"A": flat()}, **KW)
assert r.stats["n_trades"] == 1 and r.counts["holding"] == 1
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 4, .01, .014, .02, 100.0), cand("A", "s", 1, 4, 7, .01, .014, .02, 100.0),
                          cand("A", "s", 1, 5, 8, .01, .014, .02, 100.0)), {"A": flat()}, **KW)
assert r.stats["n_trades"] == 2 and r.counts["holding"] == 1             # 청산일(4) 진입은 막고, 5일 진입은 허용
print("[PASS] 5) 롱숏 충돌 둘 다 건너뜀, 동일방향 중복 합침, 보유 중 신규 후보 무시, 청산 당일 재진입 금지")

# 6) 자본 제약: 총노출 0.65, 종목당 한도 1.0. A 비중 0.5, B 의도 0.2 -> 여력 0.15(>= 0.1)라 0.15로 축소 매수
kw = {**KW, "risk_pct": 0.01, "max_weight": 1.0, "max_gross": 0.65}
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 6, .01, .014, 0.02, 100.0), cand("B", "s", 1, 2, 6, .01, .014, 0.05, 100.0)), cl, **kw)
tb = r.trades.set_index("symbol")
assert close(tb.loc["A", "weight"], 0.5) and close(tb.loc["B", "weight"], 0.15) and r.counts["scaled"] == 1
#    B 의도 0.5(tv 0.02)면 여력 0.15 < 0.5x0.5 -> 건너뜀
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 6, .01, .014, 0.02, 100.0), cand("B", "s", 1, 2, 6, .01, .014, 0.02, 100.0)), cl, **kw)
assert list(r.trades["symbol"]) == ["A"] and r.counts["no_capital"] == 1
print("[PASS] 6) 총노출 한도: 여력 0.15는 의도 0.2의 75%라 축소 매수(scaled), 의도 0.5의 30%면 no_capital")

# 7) 종목당 비중 한도(capped), 변동성 불량 후보
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 6, .01, .014, 0.01, 100.0)), {"A": flat()}, **KW)       # 0.005/0.01 = 0.5 -> 0.25
assert close(r.trades.iloc[0]["weight"], 0.25) and close(r.trades.iloc[0]["weight_intended"], 0.5) and r.counts["capped"] == 1
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 6, .01, .014, np.nan, 100.0)), {"A": flat()}, **KW)
assert r.counts["invalid_vol"] == 1 and r.stats["n_trades"] == 0
print("[PASS] 7) 의도 비중 0.5 -> 한도 0.25로 축소(capped), 변동성 NaN 후보는 invalid_vol")

# 8) 동시 손실이 몰리면 MDD가 커진다: A,B 동시 보유, 하루에 둘 다 -4% -> 평가손실 합산
Af, Bf = series({3: 96}), series({3: 96})
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 6, -.04, -.04, 0.02, 100.0), cand("B", "s", 1, 2, 6, -.04, -.04, 0.02, 100.0)),
                       {"A": Af, "B": Bf}, **KW)
assert close(r.equity.iloc[1], 1 + 2*.25*(-.04)) and close(r.stats["max_drawdown"], (1 - .02) / 1 - 1)
assert r.stats["max_concurrent"] == 2
print("[PASS] 8) 동시 보유 2종목이 같은 날 -4%: 평가자산 0.98, MDD -2.0%, 최대 동시 2개")

# 9) 우선순위 'random': 같은 시드면 동일 결과, 'low_vol'은 항상 결정적
many = cs(*[cand(s, "s", 1, 2, 6, .01, .014, 0.02 + i*0.001, 100.0) for i, s in enumerate("ABCDEFGH")])
clm = {s: flat() for s in "ABCDEFGH"}
k1 = {**KW, "max_positions": 3}
a = simulate_portfolio(many, clm, **k1, priority="random", seed=1); b = simulate_portfolio(many, clm, **k1, priority="random", seed=1)
c = simulate_portfolio(many, clm, **k1, priority="random", seed=2)
assert list(a.trades["symbol"]) == list(b.trades["symbol"]) and len(a.trades) == 3
assert any(list(a.trades["symbol"]) != list(simulate_portfolio(many, clm, **k1, priority="random", seed=s).trades["symbol"]) for s in range(2, 12))
assert list(simulate_portfolio(many, clm, **k1).trades["symbol"]) == ["A", "B", "C"]
print("[PASS] 9) random은 시드 고정 시 재현, 시드 바꾸면 선택이 달라짐, low_vol은 변동성 낮은 A,B,C")

# 10) 성과 지표 손계산: 평가자산 [1, 1.1, 0.99, 1.089] -> 일수익 +10%, -10%, +10%
eq = pd.Series([1.0, 1.1, 0.99, 1.089], index=ix[:4])
st = portfolio_stats(eq, pd.DataFrame(), [1, 1, 1, 0], [.5, .5, .5, 0], 252, 1.0)
assert close(st["total_return"], 0.089, 1e-12) and close(st["max_drawdown"], -0.1, 1e-12)
mean = 0.1/3
sd = math.sqrt(((0.1 - mean)**2 * 2 + (-0.1 - mean)**2) / 2)          # 표본표준편차(ddof=1)
assert close(st["sharpe"], mean/sd*math.sqrt(252), 1e-9) and close(st["time_in_market"], 0.75) and close(st["avg_gross"], 0.375)
print(f"[PASS] 10) 총수익 8.9%, MDD -10%, 샤프 {st['sharpe']:.3f}, 시장참여 75%, 평균노출 37.5% 손계산 일치")

# 11) 동일비중 보유 기준선 손계산: A +10%/+10%, B -10%/0% -> 일평균 0%, +5% -> 누적 +5%, MDD 0
ca = pd.Series([100, 110, 121.0], index=ix[:3]); cb = pd.Series([100, 90, 90.0], index=ix[:3])
bm = buy_hold_benchmark({"A": ca, "B": cb}, ix[0], ix[2])
assert close(bm["total_return"], 0.05, 1e-12) and close(bm["max_drawdown"], 0.0, 1e-12) and bm["n_symbols"] == 2
print("[PASS] 11) 기준선: 일평균 0%, +5% -> 누적 +5.0%")

# 12) build_candidates: 직접 호출과 일치, ret = 총수익 - 비용, 종목 실패 격리, 구간 필터
from backtest.triple_barrier import triple_barrier_returns, realized_volatility
rng = np.random.default_rng(4); n = 300
cx = 100*np.exp(np.cumsum(rng.normal(0, 0.015, n))); ixx = pd.bdate_range("2022-01-01", periods=n)
d = pd.DataFrame({"open": cx, "high": cx*1.005, "low": cx*0.995, "close": cx, "volume": 1e5}, index=ixx)
d["crossover_confirmed"] = False; d["crossunder_confirmed"] = False
d.iloc[np.arange(40, 240, 20), d.columns.get_loc("crossover_confirmed")] = True
ck = dict(pt_mult=2.0, sl_mult=1.0, vol_span=20, use_intrabar=True, entry_lag=1, entry_col="open", cost_bps=30.0, slippage_bps=10.0)
cand_, closes, fails = build_candidates({"X": {"delta_trading": d}, "BAD": {"delta_trading": "oops"}}, {"X": ck, "BAD": ck}, hold_bars=5)
assert "BAD" in fails and "BAD" not in closes and set(cand_["symbol"]) == {"X"} and len(cand_) > 0
ev = d.index[d["crossover_confirmed"]]
tb = triple_barrier_returns(d, ev, 1, 2.0, 1.0, 5, realized_volatility(d, span=20), entry_lag=1, entry_col="open",
                            cost_bps=30.0, slippage_bps=10.0, include_prices=True)
assert np.allclose(cand_["ret"].values, tb["ret"].values) and np.allclose(cand_["entry_price"].values, tb["entry_price"].values)
assert np.allclose(cand_["gross_ret"] - cand_["ret"], 0.0040)                      # (30+10)bp
lo, hi = cand_["entry_date"].iloc[2], cand_["entry_date"].iloc[4]
c2, _, _ = build_candidates({"X": {"delta_trading": d}}, {"X": ck}, 5, lo=lo, hi=hi)
assert list(c2["entry_date"]) == list(cand_["entry_date"].iloc[2:5])
c3, _, _ = build_candidates({"X": {"delta_trading": d}}, {"X": ck}, 5, signals=["없는신호"])
assert len(c3) == 0
tb0 = triple_barrier_returns(d, ev, 1, 2.0, 1.0, 5, realized_volatility(d, span=20), entry_lag=1, entry_col="open")
assert "entry_price" not in tb0.columns                                        # include_prices 기본 False면 열 불변
print("[PASS] 12) 후보 생성: 직접 호출과 일치, gross-ret=40bp, 한 종목 실패 격리, 진입일 구간 필터, 기본 열 불변")

# 13) 입력 검증과 보고서
for bad in [dict(risk_pct=0), dict(max_positions=0), dict(priority="x"), dict(sl_mult=0)]:
    try: simulate_portfolio(cs(cand("A", "s", 1, 2, 5, .01, .014, .02, 100.0)), {"A": flat()}, **{**KW, **bad}); raise SystemExit("예외 필요")
    except ValueError: pass
r = simulate_portfolio(cs(cand("A", "s", 1, 2, 5, .036, .04, .02, 100.0)), {"A": A}, **KW)
rep = format_portfolio_report(r, {"Z": "ValueError: x"})
for tok in ["포트폴리오 시뮬레이션", "MDD", "1R", "no_slot", "후보 생성 실패 종목", "표본 내 결과"]:
    assert tok in rep, tok
assert "잡힌 거래가 없습니다" in format_portfolio_report(simulate_portfolio(cs(), {}, **KW))
print("[PASS] 13) 잘못된 규칙은 예외, 보고서에 규칙·MDD·건너뜀 사유·실패 종목·표본 내 경고 출력, 거래 0건은 정직하게 표시")
print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
