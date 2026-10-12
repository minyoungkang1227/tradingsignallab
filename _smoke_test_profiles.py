"""backtest.profiles 스모크테스트 — 기대값은 손계산. 합성 데이터 결과는 실제 시장 성과가 아님."""
import argparse, io, sys, tempfile, contextlib
from pathlib import Path
import numpy as np, pandas as pd
from backtest.profiles import PROFILES, apply_profile, explicit_dests

def ns(**kw):
    base = dict(market="krx", entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0, newey_west=False,
                drop_overlapping=False, deflate=False, barrier=False, barrier_sequential=False,
                barrier_economics=False)
    base.update(kw); return argparse.Namespace(**base)

# 1) 프로필 없음 = 아무것도 안 바뀜
a = ns(); before = vars(a).copy()
assert apply_profile(a, None) == [] and vars(a) == before
print("[PASS] 1) 프로필 없음: 변경 0건, args 불변(기존 동작 보존)")

# 2) realistic(krx): 손계산한 값
a = ns(); ch = apply_profile(a, "realistic")
assert (a.entry_lag, a.entry_col, a.cost_bps, a.slippage_bps) == (1, "open", 36.0, 10.0)
assert a.deflate and a.barrier and a.barrier_sequential and a.barrier_economics and a.newey_west and a.drop_overlapping
assert {c[0] for c in ch} == set(PROFILES["realistic"]["fixed"]) | {"cost_bps"}
a = ns(market="upbit"); apply_profile(a, "realistic"); assert a.cost_bps == 10.0
print("[PASS] 2) realistic: 다음봉 시가 진입, krx 36bp/upbit 10bp, 슬리피지 10bp, 보정 옵션 전부 켜짐")

# 3) 사용자가 직접 준 옵션이 우선(0으로 직접 준 경우 포함)
a = ns(cost_bps=0.0, entry_lag=0); apply_profile(a, "realistic", explicit={"cost_bps", "entry_lag"})
assert a.cost_bps == 0.0 and a.entry_lag == 0 and a.slippage_bps == 10.0
print("[PASS] 3) 직접 지정한 옵션(값이 기본과 같아도)은 프리셋이 덮어쓰지 않음")

# 4) 알 수 없는 프로필/시장은 조용히 넘어가지 않고 예외
for bad in [dict(name="nope"), dict(name="realistic", market="nyse")]:
    try:
        apply_profile(ns(market=bad.get("market", "krx")), bad["name"]); raise SystemExit("예외가 나야 함")
    except ValueError: pass
print("[PASS] 4) 알 수 없는 프로필·시장은 ValueError")

# 5) explicit_dests: --x val, --x=val 둘 다 감지, 미지정은 제외
p = argparse.ArgumentParser(); p.add_argument("--cost-bps", dest="cost_bps", type=float, default=0.0)
p.add_argument("--entry-lag", dest="entry_lag", type=int, default=0)
assert explicit_dests(p, ["--cost-bps", "0"]) == {"cost_bps"}
assert explicit_dests(p, ["--entry-lag=0"]) == {"entry_lag"} and explicit_dests(p, []) == set()
print("[PASS] 5) 명시 옵션 감지: '--x v'와 '--x=v' 모두, 미지정은 제외")

# 6) CLI 통합: 프로필이 실제로 결과를 바꾸고(비용·지연 반영), 직접 지정이 우선
import run_backtest as rb
rng = np.random.default_rng(3); n = 700
ix = pd.bdate_range("2021-01-01", periods=n); c = 100*np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
o = np.r_[c[0], c[:-1]]*(1+rng.normal(0, 0.003, n))
DF = pd.DataFrame({"open": o, "high": np.maximum(o, c)*1.004, "low": np.minimum(o, c)*0.996, "close": c,
                   "volume": rng.integers(1000, 5000, n).astype(float)}, index=ix); DF.index.name = "date"
rb.load_data = lambda args: DF.copy()
def cli(extra):
    out = tempfile.mkdtemp(); old = sys.argv
    sys.argv = ["run_backtest.py", "--market", "upbit", "--symbol", "SYN", "--output", out, "--horizons", "1,3,5"] + extra
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): rb.main()
    finally: sys.argv = old
    return pd.read_csv(Path(out)/"summary.csv"), buf.getvalue()
s0, t0 = cli([]); s1, t1 = cli(["--profile", "realistic"]); s2, t2 = cli(["--profile", "realistic", "--cost-bps", "0"])
assert "[프로필" not in t0 and "[프로필 realistic]" in t1 and "entry-lag: 0 -> 1" in t1
assert "cost-bps: 0.0 -> 10.0" in t1 and "cost-bps" not in t2.split("프로필 realistic")[1].split("직접")[0]
assert "tb_expectancy_r" in s1.columns and "tb_expectancy_r" not in s0.columns
m = s0.merge(s1, on=["signal", "direction", "horizon"], suffixes=("_0", "_1"))
assert (m["mean_return_pct_0"] != m["mean_return_pct_1"]).any()      # 비용·지연 반영으로 평균수익이 달라짐
print("[PASS] 6) CLI: 프로필 켜면 조건 변경 목록 출력·경제성 컬럼 생성·평균수익 변화, 끄면 기존과 동일, 직접 지정 우선")
print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
