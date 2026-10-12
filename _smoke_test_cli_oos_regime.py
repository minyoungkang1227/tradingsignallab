"""
run_backtest.py CLI 통합 스모크테스트 — 새 옵션(--barrier-economics, --oos-*, --regime)이
(1) 꺼져 있으면 기존 동작·출력과 동일하고 (2) 켜면 해당 파일/컬럼을 만드는지 확인한다.
합성 데이터 결과이며 실제 시장 성과가 아니다.
"""
import io, sys, tempfile, contextlib
from pathlib import Path
import numpy as np
import pandas as pd

import run_backtest as rb
from backtest.engine import compare_all

rng = np.random.default_rng(7)
n = 900
ix = pd.bdate_range("2021-01-01", periods=n)
ret = rng.normal(0.0004, 0.02, n)
close = 100 * np.exp(np.cumsum(ret))
open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
vol = rng.integers(1_000, 5_000, n).astype(float) * (1 + 3 * (rng.random(n) < 0.05))
DF = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=ix)
DF.index.name = "date"
rb.load_data = lambda args: DF.copy()          # 네트워크 없이 합성 데이터 주입

def run_cli(extra):
    out = tempfile.mkdtemp()
    argv = ["run_backtest.py", "--market", "upbit", "--symbol", "SYN", "--output", out,
            "--horizons", "1,3,5", "--cost-bps", "5", "--drop-overlapping"] + extra
    old = sys.argv; sys.argv = argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rb.main()
    finally:
        sys.argv = old
    return Path(out), buf.getvalue()

# --- 1) 기본 실행: 새 파일/컬럼이 전혀 생기지 않는다 --------------------------------
out0, txt0 = run_cli([])
s0 = pd.read_csv(out0 / "summary.csv")
assert not any(c.startswith("tb_") for c in s0.columns), "기본 실행에 tb_ 컬럼이 생김"
assert not (out0 / "oos_selected.csv").exists() and not (out0 / "regime_summary.csv").exists()
assert "표본 외(OOS)" not in txt0 and "국면별" not in txt0
print("[PASS] 1) 새 옵션을 안 켜면 summary에 tb_* 없고 OOS/국면 파일도 생기지 않음(기존 동작 보존)")

# --- 2) build_compare_kwargs 가 직접 호출과 동일한 결과를 낸다 ------------------------
class A: pass
a = A()
for k, v in dict(horizons="1,3,5", entry_lag=0, entry_col=None, cost_bps=5.0, newey_west=False,
                 drop_overlapping=True, fdr=0.10).items():
    setattr(a, k, v)
ck = rb.build_compare_kwargs(a, 0.0)
assert ck["deflate"] is False and ck["barrier"] is False
assert ck["barrier_sequential"] is False and ck["barrier_economics"] is False
assert ck["horizons"] == (1, 3, 5) and ck["max_curve_horizon"] == 5
print("[PASS] 2) build_compare_kwargs: 필드가 없는 args에서도 새 옵션 모두 꺼짐이 기본값")

# --- 3) --barrier --barrier-economics: 경제성 컬럼과 출력 ------------------------------
out1, txt1 = run_cli(["--barrier", "--barrier-sequential", "--barrier-economics"])
s1 = pd.read_csv(out1 / "summary.csv")
for c in ["tb_expectancy_r", "tb_payoff_ratio", "tb_breakeven_wr_pct", "tb_wr_edge_pp", "tb_cost_multiple"]:
    assert c in s1.columns, c
assert "경제적 유의성" in txt1
# 겹침 제거는 거래 수를 늘리지 않는다
s_ov = pd.read_csv(run_cli(["--barrier"])[0] / "summary.csv")
m = s1.merge(s_ov, on=["signal", "direction", "horizon"], suffixes=("_seq", "_ov"))
assert (m["tb_n_seq"].fillna(0) <= m["tb_n_ov"].fillna(0)).all()
# 기본 14개 열(신호 통계)은 --barrier-economics 여부와 무관하게 동일해야 한다
shared = [c for c in s_ov.columns if c in s1.columns and not c.startswith("tb_")]
pd.testing.assert_frame_equal(s1[shared], s_ov[shared], check_exact=False, rtol=1e-12)
print("[PASS] 3) 경제성 컬럼 5개 생성·출력, 순차 거래는 거래수를 늘리지 않고 신호 통계 열은 불변")

# --- 4) --oos-train-end: DSR 자동 켜짐, 결과 파일 3개 ------------------------------------
tr_end = str(ix[629].date())
out2, txt2 = run_cli(["--oos-train-end", tr_end, "--oos-warmup", "60"])
assert "DSR을 자동으로 켭니다" in txt2
for f in ["oos_train_summary.csv", "oos_test_summary.csv", "oos_selected.csv"]:
    assert (out2 / f).exists(), f
assert "표본 외(OOS) 검증" in txt2 and "퍼지" in txt2
# OOS를 켜도 본 분석(summary.csv)은 바뀌지 않는다(DSR 자동 켜기는 OOS 내부에서만)
s2 = pd.read_csv(out2 / "summary.csv")
assert "significant_dsr" not in s2.columns
pd.testing.assert_frame_equal(s2[s0.columns], s0, check_exact=False, rtol=1e-12)
print("[PASS] 4) OOS: 파일 3개 생성, 보고서 출력, 본 분석 summary는 OOS 옵션과 무관하게 동일")

# --- 5) --regime ----------------------------------------------------------------------
out3, txt3 = run_cli(["--regime"])
assert (out3 / "regime_summary.csv").exists() and "시장 국면별 검증" in txt3
rs = pd.read_csv(out3 / "regime_summary.csv")
assert set(rs["regime"]).issubset({"bull", "bear", "sideways"}) and "significant_bh_global" in rs.columns
print("[PASS] 5) 국면: regime_summary.csv 생성, 국면 라벨·전역 BH 컬럼 존재")

# --- 6) OOS 시작일이 학습 종료 이전이면 명확한 오류 ----------------------------------------
try:
    run_cli(["--oos-train-end", tr_end, "--oos-test-start", str(ix[100].date())])
    raise SystemExit("예외가 나야 하는데 통과")
except ValueError:
    pass
print("[PASS] 6) test-start가 train-end보다 앞서면 ValueError(조용히 진행하지 않음)")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
