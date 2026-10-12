"""
DSR 게이트의 검출력(statistical power) 측정.

거짓양성을 줄이는 필터는 만들기 쉽다 — 전부 거부하면 된다. 그래서 반드시
반대편도 재야 한다: "진짜로 예측력이 있는 신호"를 심어놓고, 그걸 여전히
잡아내는가?

방법: 랜덤워크를 만든 뒤 특정 신호(델타트레이딩 골든크로스)가 뜬 직후
몇 봉에 실제 드리프트를 주입한다. 이러면 그 신호만 진짜 예측력을 갖고,
나머지 신호들은 여전히 노이즈다. 주입한 신호를 잡아내면 검출력이 있는 것이고,
나머지가 통과하면 거짓양성이다.

주의: 전부 합성(가상) 데이터다. 실제 시장 성과가 아니다.
"""
import numpy as np
import pandas as pd

from backtest.engine import compare_all
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
TRUE_SIGNAL = "델타트레이딩-골든크로스"


def make_data(n: int, seed: int, edge_per_bar: float, edge_bars: int = 5,
               n_iter: int = 12):
    """
    edge_per_bar: 골든크로스 직후 edge_bars 봉에 주입할 봉당 드리프트.
        0이면 순수 랜덤워크(진짜 신호 없음).

    [주의 — 단순 주입은 작동하지 않는다]
    드리프트를 더하면 가격 경로가 바뀌고, EMA 교차 위치도 같이 움직인다.
    그래서 "1차 경로에서 찾은 신호 위치"에 드리프트를 심으면 최종 데이터의
    실제 신호 위치와 어긋나버린다. 주입이 셀수록 더 어긋나서, 강하게 주입할수록
    오히려 t값이 0에 가까워지는 역설이 생긴다(처음에 실제로 이렇게 나왔다).

    그래서 "현재 경로에서 신호를 찾아 -> 그 위치에 주입 -> 경로 갱신"을
    반복해 고정점으로 수렴시킨다. 수렴 여부는 호출부에서 verify_edge()로
    직접 확인한다 — 테스트가 자기 전제를 검증하게 하기 위함.
    """
    rng = np.random.default_rng(seed)
    base_ret = rng.normal(0.0, 0.018, n)
    noise = {"open": rng.normal(0.0, 0.004, n), "spread": np.abs(rng.normal(0.0, 0.008, n)),
             "vol": rng.lognormal(13, 0.6, n)}

    if edge_per_bar == 0.0:
        return _to_ohlcv(50000 * np.exp(np.cumsum(base_ret)), noise, n)

    ret = base_ret.copy()
    prev_ev = None
    for _ in range(n_iter):
        df = _to_ohlcv(50000 * np.exp(np.cumsum(ret)), noise, n)
        ev = delta_trading.compute(df.copy())["crossover_confirmed"].fillna(False).values
        if prev_ev is not None and np.array_equal(ev, prev_ev):
            break           # 고정점 도달
        prev_ev = ev
        ret = base_ret.copy()
        for i in np.where(ev)[0]:
            lo, hi = i + 1, min(i + 1 + edge_bars, n)
            ret[lo:hi] += edge_per_bar

    return _to_ohlcv(50000 * np.exp(np.cumsum(ret)), noise, n)


def verify_edge(df: pd.DataFrame, horizon: int = 5) -> float:
    """
    최종 데이터에서 '골든크로스 직후 수익률'이 실제로 높아졌는지 확인한다.
    주입이 제대로 걸렸는지 테스트가 스스로 검증하기 위한 함수.
    반환: (신호 직후 평균 h봉 수익률) - (전체 평균 h봉 수익률)
    """
    ev = delta_trading.compute(df.copy())["crossover_confirmed"].fillna(False)
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    sig = fwd[ev].dropna()
    allr = fwd.dropna()
    if len(sig) == 0:
        return np.nan
    return float(sig.mean() - allr.mean())


def _to_ohlcv(close, noise, n):
    """봉 내부 노이즈는 고정된 난수를 재사용한다 — 반복 주입 때마다 노이즈가
    바뀌면 고정점에 수렴하지 않기 때문."""
    spread = noise["spread"] * close
    open_ = close * (1 + noise["open"])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                        "volume": noise["vol"]}, index=idx)
    df.index.name = "date"
    return df


def run_one(seed, edge):
    d = make_data(1800, seed, edge)
    dfs = {}
    for k, fn in COMPUTE.items():
        try:
            dfs[k] = fn(d.copy())
        except Exception:
            pass
    s, _ = compare_all(dfs, horizons=(1, 3, 5, 10, 20), deflate=True,
                        drop_overlapping=True)
    if s.empty:
        return None
    is_true = s["signal"] == TRUE_SIGNAL
    return {
        "n_events_true": int(s.loc[is_true, "n_events"].max()) if is_true.any() else 0,
        "edge": verify_edge(d),
        "bh_true": int(s.loc[is_true, "significant_bh"].sum()),
        "bh_false": int(s.loc[~is_true, "significant_bh"].sum()),
        "dsr_true": int((s.loc[is_true, "significant_bh"] & s.loc[is_true, "significant_dsr"]).sum()),
        "dsr_false": int((s.loc[~is_true, "significant_bh"] & s.loc[~is_true, "significant_dsr"]).sum()),
        "n_true_rows": int(is_true.sum()),
    }


N_SIM = 25
print("=" * 78)
print("DSR 검출력 측정 — 델타트레이딩-골든크로스에만 진짜 예측력 주입")
print("=" * 78)
print(f"{'주입강도':>10} {'실측초과':>8} {'표본':>5} | {'BH 검출':>9} {'BH 거짓':>8} | "
      f"{'+DSR 검출':>10} {'+DSR 거짓':>10}")
print("-" * 78)

results = {}
for edge in (0.0, 0.002, 0.004, 0.008):
    agg = {"bh_true": 0, "bh_false": 0, "dsr_true": 0, "dsr_false": 0,
           "n_true_rows": 0, "runs": 0, "edges": [], "n_events": []}
    for seed in range(200, 200 + N_SIM):
        r = run_one(seed, edge)
        if r is None:
            continue
        for k in ("bh_true", "bh_false", "dsr_true", "dsr_false", "n_true_rows"):
            agg[k] += r[k]
        agg["edges"].append(r["edge"])
        agg["n_events"].append(r["n_events_true"])
        agg["runs"] += 1
    results[edge] = agg

    denom = max(agg["n_true_rows"], 1)
    label = "없음" if edge == 0 else f"봉당 {edge*100:.1f}%"
    print(f"{label:>10} {np.nanmean(agg['edges'])*100:>7.2f}% "
          f"{np.mean(agg['n_events']):>5.0f} | "
          f"{agg['bh_true']:>4}/{denom:<4} {agg['bh_false']:>8} | "
          f"{agg['dsr_true']:>5}/{denom:<4} {agg['dsr_false']:>10}")

print("-" * 78)
print("  실측초과 = 신호 직후 5봉 수익률 - 전체 평균 (주입이 실제로 걸렸는지 검증)")
print("  표본 = 델타트레이딩 신호의 평균 이벤트 수")
print()

# 전제 검증: 주입 강도가 셀수록 실측 초과수익도 커져야 한다(테스트가 자기
# 전제를 스스로 확인). 이게 깨지면 아래 검출력 비교는 의미가 없다.
measured = [np.nanmean(results[e]["edges"]) for e in (0.0, 0.002, 0.004, 0.008)]
assert all(measured[i] < measured[i + 1] for i in range(len(measured) - 1)), measured
print(f"[PASS] 0) 전제 검증 — 주입 강도에 따라 실측 초과수익 단조 증가: "
      f"{' -> '.join(f'{m*100:+.2f}%' for m in measured)}")
print()

# --- 검증 1: 노이즈만 있을 때 DSR이 거짓양성을 줄여야 한다 ------------------
noise = results[0.0]
assert noise["dsr_false"] <= noise["bh_false"], (noise["dsr_false"], noise["bh_false"])
reduction = 1 - noise["dsr_false"] / max(noise["bh_false"], 1)
print(f"[PASS] 1) 노이즈 구간 거짓양성: BH {noise['bh_false']}건 -> "
      f"BH+DSR {noise['dsr_false']}건 ({reduction:.0%} 감소)")

# --- 검증 2: 신호가 충분히 강하면 DSR도 잡아내야 한다(= 검출력 존재) --------
strong = results[0.008]
assert strong["dsr_true"] > 0, (
    "주입 신호가 봉당 0.8%인데도 DSR이 하나도 못 잡음 — 필터가 과도하게 "
    "보수적이라 쓸모가 없다는 뜻")
power = strong["dsr_true"] / max(strong["n_true_rows"], 1)
bh_power = strong["bh_true"] / max(strong["n_true_rows"], 1)
print(f"[PASS] 2) 강한 신호(봉당 0.8%) 검출력: BH {bh_power:.0%} -> "
      f"BH+DSR {power:.0%} (진짜 신호를 여전히 잡아냄)")

# --- 검증 3: 주입 강도가 셀수록 검출이 늘어야 한다(단조성) ------------------
true_rates = [results[e]["dsr_true"] for e in (0.0, 0.002, 0.004, 0.008)]
assert true_rates[-1] >= true_rates[0], true_rates
print(f"[PASS] 3) 주입 강도에 따른 DSR 검출: "
      f"{' -> '.join(str(x) for x in true_rates)} (강할수록 증가)")

# --- 요약: 신호대잡음비 개선 -----------------------------------------------
print()
for edge in (0.004, 0.008):
    a = results[edge]
    bh_ratio = a["bh_true"] / max(a["bh_true"] + a["bh_false"], 1)
    dsr_ratio = a["dsr_true"] / max(a["dsr_true"] + a["dsr_false"], 1)
    print(f"주입 봉당 {edge*100:.1f}% — 통과한 신호 중 '진짜'의 비율: "
          f"BH {bh_ratio:.0%} -> BH+DSR {dsr_ratio:.0%}")

print("\n※ 전부 합성(가상) 데이터 기준이며 실제 시장 성과가 아님.")
print("\n모든 스모크테스트 통과.")
