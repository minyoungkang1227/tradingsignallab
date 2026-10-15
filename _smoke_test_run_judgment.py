"""
run_backtest.py::run_judgment()의 두 가지 정책을 검증한다.
1) --market upbit이면 --factor-csv를 줘도 4~5단계(팩터 조정 알파)를 자동으로
   건너뛴다 (KRX 전용 SMB/HML/WML 팩터를 코인에 적용하면 무의미하므로).
2) --mom-filter가 켜져 있으면 factor_cols를 명시하지 않아도 자동으로 WML을
   빼고 SMB/HML만 쓴다 (모멘텀 필터와 WML 회귀의 이중차감 방지).
"""
import types
import numpy as np
import pandas as pd
import run_backtest as rb
from indicators import institutional_displacement, delta_trading, big_sales, ultimate_rsi, vp_box

np.random.seed(7)
n = 300
dates = pd.date_range("2023-01-01", periods=n, freq="D")
close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
high = close * (1 + np.random.uniform(0, 0.01, n))
low = close * (1 - np.random.uniform(0, 0.01, n))
open_ = close * (1 + np.random.uniform(-0.005, 0.005, n))
volume = np.random.uniform(1000, 5000, n)
df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

dfs_by_indicator = {
    "institutional_displacement": institutional_displacement.compute(df),
    "delta_trading": delta_trading.compute(df),
    "big_sales": big_sales.compute(df),
    "ultimate_rsi": ultimate_rsi.compute(df),
    "vp_box": vp_box.compute(df),
}

factor_dates = pd.date_range("2023-01-31", periods=12, freq="ME")
factor_df = pd.DataFrame({
    "SMB": np.random.normal(0, 0.01, 12),
    "HML": np.random.normal(0, 0.01, 12),
    "WML": np.random.normal(0, 0.01, 12),
}, index=factor_dates)
factor_df.index.name = "date"
factor_csv_path = "/tmp/_smoke_test_factors_for_judgment.csv"
factor_df.to_csv(factor_csv_path)

from backtest.engine import compare_all
summary, _ = compare_all(dfs_by_indicator, horizons=(5,), max_curve_horizon=5)


def make_args(**overrides):
    a = types.SimpleNamespace(
        market="krx", mom_filter=False, factor_csv=factor_csv_path, factor_cols=None,
        entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0, t_threshold=3.0,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a

# 1) market=upbit -> factor_csv를 줘도 None(스킵)
args_upbit = make_args(market="upbit")
result_upbit = rb.run_judgment(dfs_by_indicator, summary, args_upbit, judge_horizon=5)
assert result_upbit is None, "market=upbit이면 factor_csv를 줘도 4~5단계를 건너뛰어야 함"
print("[OK] --market upbit -> run_judgment()가 None을 반환 (4~5단계 자동 스킵)")

# 2) market=krx, factor_csv 없음 -> None(선택 기능이므로 스킵)
args_no_csv = make_args(market="krx", factor_csv=None)
result_no_csv = rb.run_judgment(dfs_by_indicator, summary, args_no_csv, judge_horizon=5)
assert result_no_csv is None
print("[OK] --market krx인데 --factor-csv 미지정 -> None (선택 기능이므로 스킵)")

# 3) market=krx, factor_csv 있음, mom_filter=False -> factor_cols 자동선택 시 WML 포함(SMB,HML,WML)
args_krx = make_args(market="krx", mom_filter=False)
result_krx = rb.run_judgment(dfs_by_indicator, summary, args_krx, judge_horizon=5)
assert result_krx is not None and not result_krx.empty
assert "verdict" in result_krx.columns and "bh_significant" in result_krx.columns
print(f"[OK] --market krx + factor_csv -> judgment 실행됨 (행 수={len(result_krx)})")

# 4) market=krx, mom_filter=True, factor_cols 미지정 -> WML 제외하고 SMB/HML만 써야 함
#    (event_alpha 내부 팩터 열 선택을 직접 검증하기 위해 monkeypatch로 factor_cols를 가로챈다)
captured_factor_cols = []
from decision import rule_based as rb_module
orig_judge_indicator = rb_module.judge_indicator


def spy_judge_indicator(*args, **kwargs):
    captured_factor_cols.append(kwargs.get("factor_cols"))
    return orig_judge_indicator(*args, **kwargs)


rb.judge_indicator = spy_judge_indicator
args_mom = make_args(market="krx", mom_filter=True)
# 팩터 수를 줄이면 소표본에서 설계행렬이 특이해질 수 있다. 예전에는 그때
# np.linalg.LinAlgError가 올라와 파이프라인 전체가 멈췄고, 이 테스트도 예외를
# 잡아 넘겨야 했다. 지금은 factor_adjust()가 NaN + 사유로 돌려주므로 예외가
# 올라오면 안 된다 — 회귀가 성립하든 안 하든 run_judgment()는 끝까지 돌아야 한다.
try:
    rb.run_judgment(dfs_by_indicator, summary, args_mom, judge_horizon=5)
except np.linalg.LinAlgError as exc:
    raise AssertionError(f"특이행렬이 예외로 올라옴(NaN 반환으로 처리돼야 함): {exc}")
rb.judge_indicator = orig_judge_indicator  # 원상복구

assert len(captured_factor_cols) > 0
assert all(cols == ("SMB", "HML") for cols in captured_factor_cols), \
    f"mom_filter=True인데 WML이 factor_cols에 남아있음: {captured_factor_cols}"
print(f"[OK] --mom-filter 켜짐 -> factor_cols 자동으로 WML 제외: {captured_factor_cols[0]}")

# 5) factor_cols를 명시하면 자동 선택을 무시하고 그대로 써야 함
captured_factor_cols.clear()
rb.judge_indicator = spy_judge_indicator
args_explicit = make_args(market="krx", mom_filter=True, factor_cols="SMB,HML,WML")
try:
    rb.run_judgment(dfs_by_indicator, summary, args_explicit, judge_horizon=5)
except np.linalg.LinAlgError as exc:
    raise AssertionError(f"특이행렬이 예외로 올라옴(NaN 반환으로 처리돼야 함): {exc}")
rb.judge_indicator = orig_judge_indicator

assert all(cols == ("SMB", "HML", "WML") for cols in captured_factor_cols)
print(f"[OK] --factor-cols 명시 시 자동 선택을 무시하고 그대로 사용: {captured_factor_cols[0]}")

print("\nSMOKE TEST (run_judgment: upbit 자동 스킵 + factor_cols 자동 선택) OK")
