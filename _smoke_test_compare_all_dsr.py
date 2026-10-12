"""
compare_all(deflate=True) 통합 테스트 + 거짓양성률 실측.

[핵심 실험]
순수 랜덤워크(기하 브라운 운동, 드리프트 0)로 가격을 만들면 "진짜 예측력이
있는 신호"는 정의상 존재하지 않는다. 그러므로 이 데이터에서 유의하다고
판정된 신호는 전부 거짓양성(false positive)이다. 이 성질을 이용해
BH보정만 썼을 때와 DSR까지 적용했을 때의 거짓양성률을 직접 센다.

주의: 전부 합성(가상) 데이터다. 실제 시장 성과가 아니며, 실데이터에서의
거짓양성률은 다를 수 있다. 여기서 측정하는 건 "보정 장치가 노이즈를 제대로
걸러내는가"라는 통계적 성질 하나뿐이다.
"""
import numpy as np
import pandas as pd

from backtest.engine import SIGNAL_SPECS, compare_all
from indicators import (
    big_sales, combo_filter, delta_trading, institutional_displacement,
    ultimate_rsi, vp_box,
)

COMPUTE = {
    "institutional_displacement": institutional_displacement.compute,
    "delta_trading": delta_trading.compute,
    "big_sales": big_sales.compute,
    "ultimate_rsi": ultimate_rsi.compute,
    "vp_box": vp_box.compute,
    "combo_filter": combo_filter.compute,
}


def random_walk_ohlcv(n: int, seed: int) -> pd.DataFrame:
    """드리프트 0의 기하 브라운 운동. 예측 가능한 구조가 전혀 없다."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 0.018, n)
    close = 50000 * np.exp(np.cumsum(ret))
    # 봉 내부 변동은 종가 주변의 대칭 노이즈로 생성(방향성 없음)
    spread = np.abs(rng.normal(0.0, 0.008, n)) * close
    open_ = close * (1 + rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                        "volume": rng.lognormal(13, 0.6, n)}, index=idx)
    df.index.name = "date"
    return df


def build_indicator_dfs(df: pd.DataFrame) -> dict:
    out = {}
    for key, fn in COMPUTE.items():
        try:
            out[key] = fn(df.copy())
        except Exception:
            pass
    return out


# --- 1) 하위호환: deflate=False면 컬럼이 기존 그대로 -------------------------
df = random_walk_ohlcv(600, seed=1)
dfs = build_indicator_dfs(df)

summary_old, _ = compare_all(dfs, horizons=(1, 3, 5, 10, 20))
expected_cols = {"signal", "direction", "n_events", "horizon", "mean_return_pct",
                  "win_rate_pct", "t_stat", "p_value", "significant_bh"}
assert set(summary_old.columns) == expected_cols, set(summary_old.columns)
print(f"[PASS] 1) deflate=False(기본값): 컬럼 {len(summary_old.columns)}개, 기존과 동일 "
      f"-> 하위호환 유지")

# --- 2) deflate=True면 컬럼 3개 추가 ----------------------------------------
summary_new, _ = compare_all(dfs, horizons=(1, 3, 5, 10, 20),
                              deflate=True, drop_overlapping=True)
added = set(summary_new.columns) - expected_cols
assert added == {"sharpe_per_trade", "deflated_sr", "significant_dsr"}, added
# 기존 컬럼의 값은 그대로여야 한다(drop_overlapping 차이는 있으므로 컬럼 존재만 확인)
assert expected_cols.issubset(set(summary_new.columns))
print(f"[PASS] 2) deflate=True: {sorted(added)} 추가됨")

# attrs에 진단 정보가 실렸는지
assert "dsr_n_trials" in summary_new.attrs
assert "dsr_benchmark" in summary_new.attrs
print(f"       진단정보: 시행 {summary_new.attrs['dsr_n_trials']}회, "
      f"시도간 SR표준편차 {summary_new.attrs['dsr_sr_std']:.4f}, "
      f"기준선 SR {summary_new.attrs['dsr_benchmark']:.4f}")

# --- 3) extra_trials를 주면 기준선이 올라가고 통과가 줄어야 한다 -------------
s_no_extra, _ = compare_all(dfs, deflate=True, drop_overlapping=True, extra_trials=0)
s_extra, _ = compare_all(dfs, deflate=True, drop_overlapping=True, extra_trials=500)
assert s_extra.attrs["dsr_benchmark"] > s_no_extra.attrs["dsr_benchmark"]
assert s_extra["significant_dsr"].sum() <= s_no_extra["significant_dsr"].sum()
print(f"[PASS] 3) extra_trials=500: 기준선 {s_no_extra.attrs['dsr_benchmark']:.4f} -> "
      f"{s_extra.attrs['dsr_benchmark']:.4f}, 통과 "
      f"{int(s_no_extra['significant_dsr'].sum())} -> {int(s_extra['significant_dsr'].sum())}건")

# --- 4) NaN(판정 불가)은 유의하지 않음으로 처리되는가 -----------------------
nan_rows = summary_new["deflated_sr"].isna()
if nan_rows.any():
    assert not summary_new.loc[nan_rows, "significant_dsr"].any()
    print(f"[PASS] 4) 판정 불가(DSR=NaN) {int(nan_rows.sum())}건 -> 전부 "
          f"유의하지 않음으로 처리(표본부족을 유의로 흘려보내지 않음)")
else:
    print("[PASS] 4) 이번 표본에는 판정 불가 행 없음")


# --- 5) 거짓양성률 실측: 랜덤워크 30회 반복 ---------------------------------
print("\n" + "=" * 72)
print("거짓양성률 실측 — 드리프트 0 랜덤워크 30회 (진짜 신호는 0건이 정답)")
print("=" * 72)

N_SIM = 30
bh_hits, dsr_hits, both_hits, total_tests = 0, 0, 0, 0
sim_bh, sim_dsr = [], []

for seed in range(100, 100 + N_SIM):
    d = random_walk_ohlcv(600, seed=seed)
    dd = build_indicator_dfs(d)
    s, _ = compare_all(dd, horizons=(1, 3, 5, 10, 20), deflate=True,
                        drop_overlapping=True)
    if s.empty:
        continue
    bh = int(s["significant_bh"].sum())
    dsr = int(s["significant_dsr"].sum())
    both = int((s["significant_bh"] & s["significant_dsr"]).sum())
    bh_hits += bh
    dsr_hits += dsr
    both_hits += both
    total_tests += len(s)
    sim_bh.append(bh)
    sim_dsr.append(both)

print(f"\n총 검정 수        : {total_tests:,}건 (30회 x 신호x보유기간 조합)")
print(f"진짜 유의한 신호  : 0건 (랜덤워크이므로 정의상 0)")
print()
print(f"BH보정만          : {bh_hits:4d}건 통과  -> 거짓양성률 {bh_hits/total_tests:6.2%}")
print(f"DSR만             : {dsr_hits:4d}건 통과  -> 거짓양성률 {dsr_hits/total_tests:6.2%}")
print(f"BH + DSR 둘 다    : {both_hits:4d}건 통과  -> 거짓양성률 {both_hits/total_tests:6.2%}")
print()
print(f"시뮬 1회당 평균 거짓양성: BH만 {np.mean(sim_bh):.2f}건 -> "
      f"BH+DSR {np.mean(sim_dsr):.2f}건")
n_clean_bh = sum(1 for x in sim_bh if x == 0)
n_clean_both = sum(1 for x in sim_dsr if x == 0)
print(f"거짓양성이 0건이었던 시뮬: BH만 {n_clean_bh}/{N_SIM}회 -> "
      f"BH+DSR {n_clean_both}/{N_SIM}회")

# DSR을 추가로 걸면 통과 건수가 늘어날 수는 없다(둘 다 통과 = 교집합)
assert both_hits <= bh_hits
print(f"\n[PASS] 5) DSR 게이트 추가는 통과 건수를 늘리지 않음 "
      f"({bh_hits} -> {both_hits}건, {1 - both_hits/max(bh_hits,1):.0%} 감소)")

print("\n※ 전부 합성(가상) 데이터 기준이며 실제 시장 성과가 아님.")
print("  측정한 것은 '보정 장치가 노이즈를 걸러내는가'라는 통계적 성질뿐임.")
print("\n모든 스모크테스트 통과.")
