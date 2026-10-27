"""
data/collect.py 배관 검증: 수집 계약 3원칙(표준화된 출력 / fetched_at 등
메타데이터 attrs / 종목 하나 실패해도 나머지는 계속 수집)이 실제로
지켜지는지 확인한다. (run_advisor/run_param_search 위임 검사는 2026-10 리스크 중심
재구성에서 해당 스크립트와 함께 제거. 지금 이 계층을 쓰는 진입점은 run_risk.py.)
"""
import numpy as np
import pandas as pd

from data.collect import attach_fetch_meta, collect_universe


def make_df(n=100):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    return pd.DataFrame({"open": close, "high": close, "low": close,
                          "close": close, "volume": np.random.uniform(100, 200, n)}, index=dates)


# ---------------------------------------------------------------------------
# 1) attach_fetch_meta: fetched_at/source/symbol이 attrs에 정확히 남는지
# ---------------------------------------------------------------------------
df = make_df()
attach_fetch_meta(df, symbol="005930", source="pykrx")
assert df.attrs["symbol"] == "005930"
assert df.attrs["source"] == "pykrx"
assert "fetched_at" in df.attrs and "T" in df.attrs["fetched_at"]  # ISO 8601 형식 확인
print("[OK] attach_fetch_meta(): symbol/source/fetched_at attrs 정상 부착")

# Series에도 동일하게 동작해야 함(ecos.py가 Series를 반환하므로)
s = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
attach_fetch_meta(s, symbol="731Y001:0000001", source="ecos")
assert s.attrs["source"] == "ecos"
print("[OK] attach_fetch_meta(): pd.Series에도 동일하게 동작")

# ---------------------------------------------------------------------------
# 2) collect_universe: 종목 하나 실패(예외)해도 나머지는 계속 수집
# ---------------------------------------------------------------------------
def flaky_fetch(symbol):
    if symbol == "BAD":
        raise RuntimeError("네트워크 오류(시뮬레이션)")
    return make_df(n=100)


result = collect_universe(["A", "BAD", "B"], flaky_fetch, verbose=False)
assert set(result.ok.keys()) == {"A", "B"}
assert "BAD" in result.failed and "RuntimeError" in result.failed["BAD"]
assert len(result) == 2
print(f"[OK] collect_universe(): 예외 발생 종목(BAD)을 건너뛰고 나머지 계속 수집 — {result.summary()}")

# ---------------------------------------------------------------------------
# 3) collect_universe: 데이터가 min_len 미만이면 실패로 분류(예외 없이도)
# ---------------------------------------------------------------------------
def short_fetch(symbol):
    return make_df(n=10) if symbol == "SHORT" else make_df(n=100)


result2 = collect_universe(["OK1", "SHORT"], short_fetch, min_len=60, verbose=False)
assert "OK1" in result2.ok and "SHORT" not in result2.ok
assert "데이터 부족" in result2.failed["SHORT"]
print("[OK] collect_universe(): 표본 부족(min_len 미만) 종목도 실패로 분류")

print("\nSMOKE TEST (data/collect.py 수집 계약) OK")
