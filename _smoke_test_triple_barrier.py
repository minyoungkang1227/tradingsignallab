"""
backtest/triple_barrier.py 스모크테스트.

변동성을 상수(0.01)로 고정해서 배리어 위치를 손으로 확정할 수 있게 만든 뒤,
각 청산 경로(익절/손절/시간만료/갭/동시터치)가 기대한 값과 정확히 일치하는지
확인한다. pt_mult=2.0, sl_mult=1.0이므로

    익절선 = +0.02,  손절선 = -0.01
"""
import numpy as np
import pandas as pd

from backtest.engine import forward_return
from backtest.triple_barrier import (
    barrier_summary, realized_volatility, triple_barrier_returns,
)

TOL = 1e-12
VOL = 0.01
PT, SL = 2.0, 1.0        # -> 익절 +0.02, 손절 -0.01
PT_LEVEL, SL_LEVEL = 0.02, -0.01


def make_df(bars):
    """bars: [(open, high, low, close), ...] -> 날짜 인덱스 OHLCV DataFrame"""
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="D")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1000.0
    df.index.name = "date"
    return df


def const_vol(df):
    return pd.Series(VOL, index=df.index)


def run(df, event_pos=0, direction=1, max_horizon=5, **kw):
    return triple_barrier_returns(
        df, event_idx=df.index[[event_pos]], direction=direction,
        pt_mult=PT, sl_mult=SL, max_horizon=max_horizon, vol=const_vol(df), **kw)


# --- 1) 익절 터치 -----------------------------------------------------------
# 진입 100(봉0 종가). 봉1: high=103 -> r=+0.03 >= +0.02 -> 익절
# 시가 100.5(r=+0.005)는 익절선 아래이므로 익절선 0.02에 체결
df = make_df([(100, 100, 100, 100),
              (100.5, 103, 99.5, 102),
              (102, 102, 102, 102)])
res = run(df)
assert len(res) == 1
assert res["exit_reason"].iloc[0] == "pt"
assert abs(res["ret"].iloc[0] - PT_LEVEL) < TOL, res["ret"].iloc[0]
assert res["bars_held"].iloc[0] == 1
print(f"[PASS] 1) 익절 터치: ret={res['ret'].iloc[0]:+.4f} (손계산 +0.0200), 1봉 보유")


# --- 2) 손절 터치 -----------------------------------------------------------
# 봉1: low=98.5 -> r=-0.015 <= -0.01 -> 손절. 시가 100.1은 손절선 위라 손절선 체결
df = make_df([(100, 100, 100, 100),
              (100.1, 100.2, 98.5, 99),
              (99, 99, 99, 99)])
res = run(df)
assert res["exit_reason"].iloc[0] == "sl"
assert abs(res["ret"].iloc[0] - SL_LEVEL) < TOL, res["ret"].iloc[0]
print(f"[PASS] 2) 손절 터치: ret={res['ret'].iloc[0]:+.4f} (손계산 -0.0100)")


# --- 3) 갭 하락: 손절선보다 나쁘게 체결되어야 한다 --------------------------
# 봉1 시가 97 -> r=-0.03. 손절선(-0.01)을 이미 뚫고 열렸으므로 -0.03에 체결
df = make_df([(100, 100, 100, 100),
              (97, 97.5, 96, 96.5),
              (96, 96, 96, 96)])
res = run(df)
assert res["exit_reason"].iloc[0] == "sl"
assert abs(res["ret"].iloc[0] - (-0.03)) < TOL, res["ret"].iloc[0]
print(f"[PASS] 3) 갭하락 손절: ret={res['ret'].iloc[0]:+.4f} (손절선 -0.0100보다 "
      f"나쁜 -0.0300에 체결 — 갭 반영)")


# --- 4) 같은 봉에서 익절·손절 동시 터치 -> 손절 우선(보수적) ----------------
# 봉1: high=103(r=+0.03, 익절선 초과), low=98.5(r=-0.015, 손절선 초과)
df = make_df([(100, 100, 100, 100),
              (100, 103, 98.5, 101),
              (101, 101, 101, 101)])
res = run(df)
assert res["exit_reason"].iloc[0] == "sl", res["exit_reason"].iloc[0]
assert abs(res["ret"].iloc[0] - SL_LEVEL) < TOL
print(f"[PASS] 4) 동시터치 -> 손절 우선: ret={res['ret'].iloc[0]:+.4f} (유리한 쪽을 "
      f"가정하지 않음)")


# --- 5) 시간 만료 -----------------------------------------------------------
# 어느 배리어도 안 닿게 ±0.5% 안에서만 움직이다 max_horizon(3)째 종가로 청산
df = make_df([(100, 100, 100, 100),
              (100, 100.4, 99.7, 100.2),
              (100.2, 100.5, 99.8, 100.3),
              (100.3, 100.5, 99.6, 100.5),
              (100.5, 101, 100, 100.8)])
res = run(df, max_horizon=3)
assert res["exit_reason"].iloc[0] == "time"
assert res["bars_held"].iloc[0] == 3
assert abs(res["ret"].iloc[0] - (100.5 / 100 - 1)) < TOL, res["ret"].iloc[0]
print(f"[PASS] 5) 시간만료: 3봉 뒤 종가 100.5 -> ret={res['ret'].iloc[0]:+.4f}")


# --- 6) 핵심 케이스: 중간에 급락했다 회복 -----------------------------------
# 고정 보유기간은 "+2% 성공"으로 집계하지만, 실제로는 손절에 잘렸어야 한다.
# 봉1에서 low=95(r=-5%)로 손절선을 한참 뚫고, 봉5 종가는 102로 회복.
df = make_df([(100, 100, 100, 100),
              (99.8, 100, 95, 96),      # 여기서 -5% -> 손절
              (96, 98, 95.5, 97),
              (97, 100, 97, 99),
              (99, 101, 99, 101),
              (101, 102.5, 101, 102)])  # 5봉째 종가 102 = +2%
res = run(df, max_horizon=5)
tb_ret = res["ret"].iloc[0]

fwd = forward_return(df, horizon=5, direction=1)
fixed_ret = fwd.iloc[0]

assert res["exit_reason"].iloc[0] == "sl"
assert abs(tb_ret - SL_LEVEL) < TOL
assert abs(fixed_ret - 0.02) < 1e-12, fixed_ret
assert tb_ret < 0 < fixed_ret
print(f"[PASS] 6) 급락 후 회복 케이스 — 고정보유기간: {fixed_ret:+.4f}(성공으로 집계) "
      f"vs 트리플배리어: {tb_ret:+.4f}(손절). 측정 대상 자체가 다름")


# --- 7) 숏 방향 -------------------------------------------------------------
# direction=-1. 가격이 97로 내려가면 숏은 +3% -> 익절선 +0.02 터치
df = make_df([(100, 100, 100, 100),
              (99, 100.5, 97, 97.5),
              (97.5, 97.5, 97.5, 97.5)])
res = run(df, direction=-1)
assert res["exit_reason"].iloc[0] == "pt"
assert abs(res["ret"].iloc[0] - PT_LEVEL) < TOL, res["ret"].iloc[0]
print(f"[PASS] 7) 숏 익절: 가격 100->97에서 ret={res['ret'].iloc[0]:+.4f} "
      f"(방향 조정 정상)")

# 숏인데 가격이 오르면 손절
df = make_df([(100, 100, 100, 100),
              (100.1, 101.5, 100, 101.2),
              (101, 101, 101, 101)])
res = run(df, direction=-1)
assert res["exit_reason"].iloc[0] == "sl"
assert abs(res["ret"].iloc[0] - SL_LEVEL) < TOL
print(f"[PASS] 8) 숏 손절: 가격 100->101.5에서 ret={res['ret'].iloc[0]:+.4f}")


# --- 9) 비용·슬리피지 차감 --------------------------------------------------
df = make_df([(100, 100, 100, 100),
              (100.5, 103, 99.5, 102),
              (102, 102, 102, 102)])
res = run(df, cost_bps=36.0, slippage_bps=10.0)   # KRX 왕복 36bp + 슬리피지 10bp
expected = PT_LEVEL - 36 / 10000 - 10 / 10000     # 0.02 - 0.0036 - 0.0010 = 0.0154
assert abs(res["ret"].iloc[0] - expected) < TOL, res["ret"].iloc[0]
print(f"[PASS] 9) 비용 차감: 익절 +0.0200 - 36bp - 10bp = {res['ret'].iloc[0]:+.4f}")


# --- 10) 배리어 비활성화 ----------------------------------------------------
# sl_mult=0 -> 손절 없음. 위 6번 케이스에서 -5%를 맞아도 안 잘리고 끝까지 간다.
df = make_df([(100, 100, 100, 100),
              (99.8, 100, 95, 96),
              (96, 98, 95.5, 97),
              (97, 100, 97, 99),
              (99, 101, 99, 101),
              (101, 102.5, 101, 102)])
res = triple_barrier_returns(df, df.index[[0]], direction=1, pt_mult=0, sl_mult=0,
                              max_horizon=5, vol=const_vol(df))
assert res["exit_reason"].iloc[0] == "time"
assert abs(res["ret"].iloc[0] - 0.02) < TOL
print(f"[PASS] 10) 배리어 전부 끄면 고정보유기간과 동일: ret={res['ret'].iloc[0]:+.4f} "
      f"(= forward_return과 일치)")


# --- 11) entry_lag: 다음 봉 시가 진입 ---------------------------------------
# entry_lag=1, entry_col='open' -> 봉1 시가 101에 진입(신호는 봉0 종가에 확정)
df = make_df([(100, 100, 100, 100),
              (101, 101.5, 100.8, 101.2),
              (101.2, 104, 101, 103.5),
              (103.5, 103.5, 103.5, 103.5)])
res = triple_barrier_returns(df, df.index[[0]], direction=1, pt_mult=PT, sl_mult=SL,
                              max_horizon=5, vol=const_vol(df),
                              entry_lag=1, entry_col="open")
# 진입가 101. 봉2 high=104 -> r = 104/101-1 = +0.0297 >= 0.02 -> 익절
assert res["entry_date"].iloc[0] == df.index[1], res["entry_date"].iloc[0]
assert res["exit_reason"].iloc[0] == "pt"
assert abs(res["ret"].iloc[0] - PT_LEVEL) < TOL
print(f"[PASS] 11) entry_lag=1+시가진입: 진입 {res['entry_date'].iloc[0].date()} "
      f"(신호봉 다음날), ret={res['ret'].iloc[0]:+.4f}")


# --- 12) 데이터 부족 이벤트는 조용히 제외 -----------------------------------
df = make_df([(100, 100, 100, 100), (100, 101, 99, 100)])
res = triple_barrier_returns(df, df.index[[1]], direction=1, pt_mult=PT, sl_mult=SL,
                              max_horizon=5, vol=const_vol(df))
assert len(res) == 0, res
print("[PASS] 12) 시계열 끝이라 추종 불가능한 이벤트 -> 제외(빈 결과)")


# --- 13) realized_volatility: 미래 정보를 안 쓰는지 -------------------------
rng = np.random.default_rng(7)
prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
df = make_df([(p, p, p, p) for p in prices])
v = realized_volatility(df, span=20)
# shift(1) 되어 있으므로 첫 값은 NaN이고, 각 시점 값은 "직전 봉까지"로 계산됨
assert np.isnan(v.iloc[0])
ret = df["close"].pct_change()
manual = ret.ewm(span=20, min_periods=10).std()
assert abs(v.iloc[50] - manual.iloc[49]) < 1e-15, (v.iloc[50], manual.iloc[49])
print(f"[PASS] 13) realized_volatility: shift(1) 적용 확인 — t시점 값이 t-1까지의 "
      f"EWM std와 일치({v.iloc[50]:.6f})")


# --- 14) barrier_summary ----------------------------------------------------
tb = pd.DataFrame({
    "ret": [0.02, -0.01, 0.02, -0.01, 0.005],
    "exit_reason": ["pt", "sl", "pt", "sl", "time"],
    "bars_held": [1, 2, 3, 4, 5],
})
s = barrier_summary(tb)
assert s["n"] == 5
assert abs(s["mean_ret"] - 0.005) < 1e-12, s["mean_ret"]   # (0.02-0.01+0.02-0.01+0.005)/5
assert abs(s["win_rate"] - 0.6) < 1e-12
assert abs(s["pt_rate"] - 0.4) < 1e-12
assert abs(s["sl_rate"] - 0.4) < 1e-12
assert abs(s["time_rate"] - 0.2) < 1e-12
assert abs(s["avg_bars"] - 3.0) < 1e-12
print(f"[PASS] 14) barrier_summary: 평균 {s['mean_ret']:+.4f}, 승률 {s['win_rate']:.0%}, "
      f"익절 {s['pt_rate']:.0%}/손절 {s['sl_rate']:.0%}/시간 {s['time_rate']:.0%}")

assert barrier_summary(pd.DataFrame())["n"] == 0
print("[PASS] 15) barrier_summary: 빈 입력 안전 처리")

print("\n모든 스모크테스트 통과.")
