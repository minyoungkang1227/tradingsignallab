"""
경제적 유의성(backtest.economics) + 순차 거래 필터 + barrier_study/compare_all 연결 스모크테스트.
기대값은 전부 손계산한 값이다. 합성 데이터 결과는 실제 시장 성과가 아니다.
"""
import numpy as np
import pandas as pd

from backtest.economics import economic_summary, r_multiples
from backtest.triple_barrier import (
    barrier_study, drop_overlapping_trades, triple_barrier_returns, realized_volatility)

def close(a, b, tol=1e-9): return abs(a - b) <= tol

def mk_tb(rets, vols):
    n = len(rets)
    d = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"entry_date": d, "exit_date": d + pd.Timedelta(days=1), "bars_held": 1,
                         "ret": rets, "exit_reason": "time", "target_vol": vols}, index=d)

# --- 1) 손계산 A: 승 2건(+2R), 패 3건(-1R), 손익비 2:1, 비용 0.002 ---------------------
tb = mk_tb([0.04, 0.04, -0.02, -0.02, -0.02], [0.02] * 5)
e = economic_summary(tb, pt_mult=2.0, sl_mult=1.0, round_trip_cost=0.002)
assert e["n"] == 5
assert close(e["expectancy_r"], 0.2), e                      # 0.4*2 - 0.6*1
assert close(e["win_rate"], 0.4) and close(e["avg_win_r"], 2.0) and close(e["avg_loss_r"], 1.0)
assert close(e["payoff_ratio"], 2.0)
assert close(e["breakeven_wr"], 1 / 3)                        # 1/(1+2)
assert close(e["wr_edge"], 0.4 - 1 / 3)
assert close(e["nominal_breakeven_wr"], 1 / 3)               # sl/(sl+pt)
assert close(e["net_to_cost"], 2.0)                          # 평균 0.004 / 비용 0.002
print("[PASS] 1) 손계산 A: 기대값 +0.2R, 손익분기 승률 33.3%, 승률 40%(+6.7%p), 비용대비 2.0배")

# --- 2) 손계산 B: 변동성이 이벤트마다 다름 -> R이 위험 단위로 정규화되는가 -------------
tb = mk_tb([0.03, -0.02, 0.01], [0.01, 0.04, 0.02])
e = economic_summary(tb, 2.0, 1.0, 0.0)
assert np.allclose(r_multiples(tb, 1.0), [3.0, -0.5, 0.5])
assert close(e["expectancy_r"], 1.0)
p, W, L = e["win_rate"], e["avg_win_r"], e["avg_loss_r"]
assert close(W, 1.75) and close(L, 0.5) and close(p, 2 / 3)
assert close(p * W - (1 - p) * L, e["expectancy_r"])         # 항등식: 기대값 = pW - (1-p)L
assert close(e["breakeven_wr"], 0.5 / 2.25)
assert np.isnan(e["net_to_cost"])                             # 비용 0 -> 비용 대비 배수 NaN
print("[PASS] 2) 손계산 B: R 정규화 [3,-0.5,0.5], 항등식 E = pW-(1-p)L 성립")

# --- 3) sl_mult가 다르면 R 단위가 달라진다 ------------------------------------------
e2 = economic_summary(mk_tb([0.04, -0.02], [0.02, 0.02]), 2.0, 2.0, 0.0)   # 1R = 0.04
assert close(e2["expectancy_r"], (1.0 + -0.5) / 2)
print("[PASS] 3) 손절폭 2σ면 1R=0.04: R=[1,-0.5] -> 기대값 +0.25R")

# --- 4) 경계 사례 ---------------------------------------------------------------
e = economic_summary(mk_tb([-0.01, -0.02], [0.01, 0.01]), 2.0, 1.0, 0.001)
assert close(e["expectancy_r"], -1.5) and close(e["avg_win_r"], 0.0) and close(e["breakeven_wr"], 1.0)
e = economic_summary(mk_tb([0.02, 0.01], [0.01, 0.01]), 2.0, 1.0, 0.001)
assert np.isinf(e["payoff_ratio"]) and close(e["breakeven_wr"], 0.0)
e = economic_summary(mk_tb([0.02, -0.01], [0.01, 0.01]), 2.0, 0.0, 0.001)      # 손절 끔 -> R 정의 불가
assert np.isnan(e["expectancy_r"]) and np.isnan(e["breakeven_wr"]) and np.isnan(e["nominal_breakeven_wr"])
e = economic_summary(pd.DataFrame(columns=["ret", "target_vol"]), 2.0, 1.0, 0.001)
assert e["n"] == 0 and np.isnan(e["expectancy_r"])
print("[PASS] 4) 경계 사례: 승 없음(손익분기 100%), 패 없음(payoff 무한), 손절 끔(NaN), 빈 입력")

# --- 5) 순차 거래 필터 손계산 ---------------------------------------------------------
d = lambda k: pd.Timestamp("2024-01-01") + pd.Timedelta(days=k)
tbs = pd.DataFrame({"entry_date": [d(1), d(2), d(3), d(4), d(6)],
                    "exit_date":  [d(3), d(4), d(5), d(6), d(7)],
                    "ret": [0.0] * 5, "bars_held": 1, "exit_reason": "time", "target_vol": 0.01})
kept = drop_overlapping_trades(tbs)
# d1~d3 채택 / d2 진입은 d3 청산 전이라 제외 / d3 진입은 청산 봉과 같아 제외(엄격히 뒤여야 함)
# d4 진입 채택(d4~d6) / d6 진입은 d6 청산과 같은 봉이라 제외
assert list(kept["entry_date"]) == [d(1), d(4)], list(kept["entry_date"])
unsorted = tbs.iloc[::-1].reset_index(drop=True)
assert list(drop_overlapping_trades(unsorted)["entry_date"]) == [d(1), d(4)]   # 입력 순서와 무관
print("[PASS] 5) 순차 필터: 5건 -> 2건(1일·4일 진입), 청산 봉 동시 진입도 겹침, 입력 순서 무관")

# --- 6) barrier_study 연결: 기본값은 이전 컬럼 그대로, 플래그 켜면 손계산과 일치 ----------
rng = np.random.default_rng(11)
n = 700
idx = pd.bdate_range("2021-01-01", periods=n)
c = 10000 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n)))
o = c * np.exp(rng.normal(0, 0.004, n))
h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
df = pd.DataFrame(dict(open=o, high=h, low=l, close=c, volume=1e5), index=idx)
ev = pd.Series(False, index=idx)
ev.iloc[60::7] = True                      # 일부러 촘촘하게(보유기간과 겹치게) 이벤트를 만든다
df["evt"] = ev
specs = [("ind", "evt", 1, "테스트-롱")]
kw = dict(horizons=(5, 10), pt_mult=2.0, sl_mult=1.0, entry_lag=1, entry_col="open",
          cost_bps=36.0, slippage_bps=20.0)
base = barrier_study({"ind": df}, specs, **kw)
OLD = ["signal", "direction", "horizon", "tb_n", "tb_mean_ret_pct", "tb_win_rate_pct", "tb_sharpe",
       "tb_t_stat", "tb_skew", "tb_kurt", "tb_pt_rate", "tb_sl_rate", "tb_time_rate", "tb_avg_bars"]
assert list(base.columns) == OLD, "기본 호출의 컬럼이 이전과 달라졌다"
seq = barrier_study({"ind": df}, specs, sequential=True, **kw)
eco = barrier_study({"ind": df}, specs, economics=True, **kw)
assert list(eco.columns) == OLD + ["tb_expectancy_r", "tb_payoff_ratio", "tb_breakeven_wr_pct",
                                    "tb_wr_edge_pp", "tb_cost_multiple"]
assert (eco[OLD].fillna(-999).values == base.fillna(-999).values).all()   # economics가 기존 값을 건드리지 않음
# 손계산: 이벤트 간격 7봉, 진입은 이벤트 다음 봉(t+1). h=5면 거래가 아무리 길어도 t+6에 끝나고 다음 진입은 t+8이라
# 겹칠 수 없다 -> 순차 필터 전후 같아야 한다. h=10이면 t+11까지 갈 수 있어 다음 진입(t+8)과 겹치는 거래가 생긴다 -> 줄어야 한다.
bn, sn = base.set_index("horizon")["tb_n"], seq.set_index("horizon")["tb_n"]
assert sn[5] == bn[5], (sn[5], bn[5])
assert sn[10] < bn[10], (sn[10], bn[10])

# 손계산(독립 경로): 같은 이벤트를 triple_barrier_returns -> drop_overlapping_trades -> economic_summary로
vol = realized_volatility(df, span=20)
ev_idx = df.index[df["evt"]]
for hz in (5, 10):
    t = triple_barrier_returns(df, ev_idx, direction=1, pt_mult=2.0, sl_mult=1.0, max_horizon=hz, vol=vol,
                                entry_lag=1, entry_col="open", cost_bps=36.0, slippage_bps=20.0)
    exp_n = len(drop_overlapping_trades(t))
    got = seq[seq["horizon"] == hz]["tb_n"].iloc[0]
    assert got == exp_n, (hz, got, exp_n)
    ref = economic_summary(t, 2.0, 1.0, (36.0 + 20.0) / 1e4)
    row = eco[eco["horizon"] == hz].iloc[0]
    assert close(row["tb_expectancy_r"], ref["expectancy_r"])
    assert close(row["tb_cost_multiple"], ref["net_to_cost"])
    assert close(row["tb_breakeven_wr_pct"], ref["breakeven_wr"] * 100)
print(f"[PASS] 6) barrier_study: 기본 컬럼 불변, 순차 필터 tb_n {base['tb_n'].tolist()} -> {seq['tb_n'].tolist()}, "
      f"economics 값이 독립 계산과 일치")

# --- 7) compare_all까지 연결 -------------------------------------------------------
from alerts.config import build_args
import run_backtest as rb
import contextlib, io
a = build_args(market="krx", symbol="TEST", horizons="3,5,10", entry_lag=1, entry_col="open",
               cost_bps=36.0, slippage_bps=20.0, deflate=True, barrier=True)
with contextlib.redirect_stdout(io.StringIO()):
    summ0, _, dfs, _ = rb.run(df, a)
from backtest.engine import compare_all
kw2 = dict(horizons=(3, 5, 10), max_curve_horizon=10, entry_lag=1, entry_col="open", cost_bps=36.0,
           slippage_bps=20.0, drop_overlapping=True, deflate=True, barrier=True)
s_def = compare_all(dfs, **kw2)[0]
assert "tb_expectancy_r" not in s_def.columns                      # 기본값: 새 컬럼 없음(이전과 동일)
s_new = compare_all(dfs, barrier_sequential=True, barrier_economics=True, **kw2)[0]
assert "tb_expectancy_r" in s_new.columns and s_new.attrs["barrier_sequential"] is True
m = s_def.merge(s_new, on=["signal", "direction", "horizon"], suffixes=("_a", "_b"))
assert (m["tb_n_b"].fillna(0) <= m["tb_n_a"].fillna(0)).all()      # 순차 필터는 표본을 늘리지 않는다
print("[PASS] 7) compare_all: 기본 호출은 새 컬럼 없음, 켜면 tb_* 경제 지표 추가 + 순차 필터로 tb_n 감소/동일")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
