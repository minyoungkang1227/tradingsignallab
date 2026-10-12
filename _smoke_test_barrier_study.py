"""
barrier_study() + compare_all(barrier=True) 스모크테스트.

핵심 검증 세 가지:
  1) 기본값(barrier=False)에서 컬럼이 하나도 안 늘어나는가 (하위호환)
  2) 배리어를 전부 끄면(pt_mult=sl_mult=0) 고정 보유기간 집계와 정확히 일치하는가
     -> 배선이 올바른지 확인하는 가장 강한 테스트
  3) tb_* 값이 triple_barrier_returns()를 직접 돌린 결과와 일치하는가
"""
import numpy as np
import pandas as pd

from backtest.engine import SIGNAL_SPECS, compare_all
from backtest.triple_barrier import (
    barrier_study, barrier_summary, realized_volatility, triple_barrier_returns,
)
from indicators import delta_trading, ultimate_rsi, vp_box

TOL = 1e-9


def synth(n=1200, seed=11):
    rng = np.random.default_rng(seed)
    c = 50000 * np.exp(np.cumsum(rng.normal(0, 0.018, n)))
    sp = np.abs(rng.normal(0, 0.008, n)) * c
    o = c * (1 + rng.normal(0, 0.004, n))
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + sp,
                        "low": np.minimum(o, c) - sp, "close": c,
                        "volume": rng.lognormal(13, 0.6, n)},
                       index=pd.date_range("2019-01-01", periods=n, freq="B"))
    df.index.name = "date"
    return df


df = synth()
dfs = {"delta_trading": delta_trading.compute(df.copy()),
       "ultimate_rsi": ultimate_rsi.compute(df.copy()),
       "vp_box": vp_box.compute(df.copy())}
HZ = (1, 3, 5, 10, 20)


# --- 1) 하위호환: barrier=False면 컬럼이 안 늘어난다 ------------------------
base, _ = compare_all(dfs, horizons=HZ)
base_cols = set(base.columns)
assert not any(c.startswith("tb_") for c in base_cols), base_cols
print(f"[PASS] 1) barrier=False(기본값): 컬럼 {len(base_cols)}개, tb_* 없음 -> 하위호환")


# --- 2) 배리어를 전부 끄면 고정 보유기간과 정확히 일치 ----------------------
# pt_mult=0, sl_mult=0 -> 어느 배리어도 안 닿으므로 전부 시간만료 청산 =
# "정확히 h봉 뒤 종가" = forward_return과 같은 정의.
#
# (a) 이벤트 단위 정확 비교 — 집계값만 보는 것보다 훨씬 강한 검증이다.
#     같은 이벤트에 대해 두 경로가 글자 그대로 같은 숫자를 내야 한다.
from backtest.engine import forward_return  # noqa: E402

d_ = dfs["delta_trading"]
ev_ = d_.index[d_["crossover_confirmed"].fillna(False)]
for h in HZ:
    tb_direct = triple_barrier_returns(d_, ev_, direction=1, pt_mult=0, sl_mult=0,
                                        max_horizon=h,
                                        vol=realized_volatility(d_, span=20))
    fw = forward_return(d_, horizon=h, direction=1).loc[ev_].dropna()
    # 배리어를 껐으므로 표본 자체가 완전히 같아야 한다
    assert set(tb_direct.index) == set(fw.index), (
        h, len(tb_direct), len(fw),
        fw.index.difference(tb_direct.index), tb_direct.index.difference(fw.index))
    assert (fw.loc[tb_direct.index] - tb_direct["ret"]).abs().max() < TOL, h
print(f"[PASS] 2a) 배리어 끈 상태에서 보유기간 {HZ} 전부 — 이벤트 집합과 "
      f"개별 수익률이 forward_return과 글자 그대로 일치 (n={len(ev_)})")

# (b) 집계 레벨에서도 일치
off, _ = compare_all(dfs, horizons=HZ, barrier=True, pt_mult=0, sl_mult=0)
merged = off.dropna(subset=["tb_mean_ret_pct", "mean_return_pct"])
assert len(merged) > 0
worst = (merged["mean_return_pct"] - merged["tb_mean_ret_pct"]).abs().max()
assert worst < 1e-9, worst
print(f"[PASS] 2b) 집계 레벨에서도 {len(merged)}개 행 전부 일치 (최대 오차 {worst:.2e})")

# 전부 시간만료로 나갔는지도 확인
assert (off["tb_time_rate"].dropna() == 1.0).all(), off["tb_time_rate"].dropna().unique()
assert (off["tb_pt_rate"].dropna() == 0.0).all()
assert (off["tb_sl_rate"].dropna() == 0.0).all()
print("[PASS] 3) 배리어 끈 상태에서 청산 사유가 전부 '시간만료'(익절·손절 0%)")


# --- 3) tb_* 값이 직접 돌린 결과와 일치하는가 -------------------------------
s, _ = compare_all(dfs, horizons=HZ, barrier=True, pt_mult=2.0, sl_mult=1.0)

# 델타트레이딩-골든크로스, horizon=5를 손으로 다시 계산
key_signal, key_h = "델타트레이딩-골든크로스", 5
row = s[(s["signal"] == key_signal) & (s["horizon"] == key_h)].iloc[0]

d = dfs["delta_trading"]
ev = d.index[d["crossover_confirmed"].fillna(False)]
vol = realized_volatility(d, span=20)
tb_direct = triple_barrier_returns(d, ev, direction=1, pt_mult=2.0, sl_mult=1.0,
                                    max_horizon=key_h, vol=vol)
man = barrier_summary(tb_direct)

assert row["tb_n"] == man["n"], (row["tb_n"], man["n"])
assert abs(row["tb_mean_ret_pct"] - man["mean_ret"] * 100) < TOL
assert abs(row["tb_win_rate_pct"] - man["win_rate"] * 100) < TOL
assert abs(row["tb_sharpe"] - man["sharpe"]) < TOL
assert abs(row["tb_pt_rate"] - man["pt_rate"]) < TOL
assert abs(row["tb_sl_rate"] - man["sl_rate"]) < TOL
print(f"[PASS] 4) tb_* 값이 직접 계산과 일치: {key_signal} h={key_h}, "
      f"n={man['n']}, 평균 {man['mean_ret']*100:+.3f}%, "
      f"익절 {man['pt_rate']:.0%}/손절 {man['sl_rate']:.0%}/시간 {man['time_rate']:.0%}")

# 청산 사유 비율의 합은 1이어야 한다
tot = s[["tb_pt_rate", "tb_sl_rate", "tb_time_rate"]].dropna().sum(axis=1)
assert (tot - 1.0).abs().max() < 1e-12, tot.describe()
print("[PASS] 5) 청산 사유 비율(익절+손절+시간) 합이 모든 행에서 정확히 1")


# --- 4) 손절을 조이면 손절 비율이 올라가야 한다(단조성) ---------------------
rates = []
for sl in (3.0, 1.5, 0.75):
    x, _ = compare_all(dfs, horizons=(10,), barrier=True, pt_mult=2.0, sl_mult=sl)
    rates.append(float(x["tb_sl_rate"].mean()))
assert rates[0] < rates[1] < rates[2], rates
print(f"[PASS] 6) 손절폭을 조일수록 손절 비율 증가: "
      f"{' -> '.join(f'{r:.0%}' for r in rates)} (sl_mult 3.0/1.5/0.75)")


# --- 5) deflate와 같이 켜면 배리어 기준 DSR도 나온다 ------------------------
both, _ = compare_all(dfs, horizons=HZ, barrier=True, deflate=True,
                       drop_overlapping=True)
for c in ("deflated_sr", "significant_dsr", "tb_deflated_sr", "tb_significant_dsr"):
    assert c in both.columns, c
assert "tb_dsr_benchmark" in both.attrs
print(f"[PASS] 7) barrier+deflate 동시: 고정 기준선 SR "
      f"{both.attrs['dsr_benchmark']:.4f} / 배리어 기준선 SR "
      f"{both.attrs['tb_dsr_benchmark']:.4f} 각각 산출")

# 임시 컬럼이 남아있지 않은지
assert not any(c.startswith("_") for c in both.columns), \
    [c for c in both.columns if c.startswith("_")]
print("[PASS] 8) 내부 임시 컬럼(_sharpe 등)이 결과에 남지 않음")


# --- 6) 고정 vs 배리어 비교 출력 --------------------------------------------
print("\n" + "=" * 76)
print("고정 보유기간 vs 손절 반영 — 같은 신호, 같은 시간예산 (합성 데이터)")
print("=" * 76)
view = both[both["horizon"] == 10][
    ["signal", "n_events", "mean_return_pct", "tb_mean_ret_pct",
     "tb_sl_rate", "tb_pt_rate", "tb_avg_bars"]].dropna()
view = view.rename(columns={"signal": "신호", "n_events": "표본",
                             "mean_return_pct": "고정10봉(%)",
                             "tb_mean_ret_pct": "손절반영(%)",
                             "tb_sl_rate": "손절률", "tb_pt_rate": "익절률",
                             "tb_avg_bars": "평균보유"})
for c in ("고정10봉(%)", "손절반영(%)"):
    view[c] = view[c].map(lambda v: f"{v:+.2f}")
for c in ("손절률", "익절률"):
    view[c] = view[c].map(lambda v: f"{v:.0%}")
view["평균보유"] = view["평균보유"].map(lambda v: f"{v:.1f}봉")
print(view.to_string(index=False))
print("\n※ 합성(가상) 데이터 기준이며 실제 시장 성과가 아님.")

print("\n모든 스모크테스트 통과.")
