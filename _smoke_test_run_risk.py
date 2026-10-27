"""
run_risk.py 배관 스모크테스트 — 네트워크 없이 합성 종가를 주입해 CLI가 끝까지 돌고 파일·판정표를 만드는지.
합성 데이터 결과는 실제 시장 성과·실제 리스크가 아님.
"""
import contextlib, io, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_risk as rr

rng = np.random.default_rng(9)
n = 1200
ix = pd.bdate_range("2020-01-01", periods=n)
def mk(sig):
    c = 100 * np.cumprod(1 + rng.normal(0, sig, n))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=ix)
FRAMES = {"AAA": mk(0.012), "BBB": mk(0.025)}

# 1) run() 직접: 비중·수익률·판정표
rp, w, results, table = rr.run({k: v["close"] for k, v in FRAMES.items()}, {"AAA": 0.6, "BBB": 0.4},
                               ["historical", "parametric", "fhs"], 0.99, 250)
assert set(table["method"]) == {"historical", "parametric", "fhs"} and set(table.columns) >= {"n_exceed", "verdict", "kupiec_p"}
assert (w.sum(axis=1) == 1.0).all() and np.isnan(rp.iloc[0]) and rp.iloc[1:].notna().all()
# 손계산: 두 번째 날 수익률 = .6*rA + .4*rB
rA = FRAMES["AAA"]["close"].iloc[1] / FRAMES["AAA"]["close"].iloc[0] - 1
rB = FRAMES["BBB"]["close"].iloc[1] / FRAMES["BBB"]["close"].iloc[0] - 1
assert abs(rp.iloc[1] - (0.6 * rA + 0.4 * rB)) < 1e-12
print("[PASS] 1) run(): 세 방법 판정표, 비중 합 1, 둘째 날 수익률 손계산 일치")

# 2) CLI 전체(수집은 몽키패치), 품질 게이트가 끼어 있는지(모순 데이터면 중단)
rr._fetch_one = lambda s, args: FRAMES[s].copy()
def cli(extra):
    out = tempfile.mkdtemp()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rr.main(["--market", "krx", "--symbols", "AAA,BBB", "--weights", "0.6,0.4", "--start", "20200101",
                 "--end", "20241231", "--data-label", "synthetic", "--output", out] + extra)
    return Path(out), buf.getvalue()
out, txt = cli([])
assert (out / "risk_var_backtest.csv").exists() and (out / "risk_var_series.csv").exists()
assert "[실행 조건·검증 요약]" in txt and "VaR 백테스트 [fhs]" in txt and "실제 시장 성과·실제 리스크가 아님" in txt
tb = pd.read_csv(out / "risk_var_backtest.csv"); assert len(tb) == 3
se = pd.read_csv(out / "risk_var_series.csv", index_col=0); assert {"var_fhs", "es_historical", "portfolio_return"} <= set(se.columns)
out2, txt2 = cli(["--methods", "fhs", "--alpha", "0.975"])
assert pd.read_csv(out2 / "risk_var_backtest.csv")["method"].tolist() == ["fhs"] and "97.5%" in txt2
print("[PASS] 2) CLI: 표준 블록·방법별 보고·CSV 2개, --methods/--alpha 반영")

bad = {k: v.copy() for k, v in FRAMES.items()}; bad["BBB"].iloc[10, bad["BBB"].columns.get_loc("high")] = 1.0   # high < close
rr._fetch_one = lambda s, args: bad[s].copy()
try:
    cli([]); raise AssertionError("모순 데이터인데 중단되지 않음")
except SystemExit as e:
    assert "쓸 수 없어 중단" in str(e) and "BBB" in str(e)
print("[PASS] 3) 품질 게이트: 한 종목이 모순이면 포트폴리오 전체 중단(사유에 종목 표시)")

# 4) 비중 개수 불일치 / 알 수 없는 방법
rr._fetch_one = lambda s, args: FRAMES[s].copy()
for extra, key in ((["--weights", "1.0"], "개수"), (["--methods", "montecarlo"], "알 수 없는")):
    try:
        cli(extra); raise AssertionError(extra)
    except SystemExit as e:
        assert key in str(e), str(e)
print("[PASS] 4) 입력 검증: 비중 개수 불일치·미지원 방법은 즉시 종료")

# 5) --controls: 한도 도출·적용·비교·통제 후 백테스트가 출력되고 파일 3개가 생긴다. 끄면 생기지 않는다(기본 보존).
rr._fetch_one = lambda s, args: FRAMES[s].copy()
out5, txt5 = cli(["--controls", "--methods", "fhs", "--alpha", "0.975"])
for tok in ("통제 계층", "한도 출처", "ES 축소", "통제 전 VaR 백테스트", "통제 후 VaR 백테스트", "표본 내"):
    assert tok in txt5, tok
for f in ("risk_controlled_weights.csv", "risk_control_log.csv", "risk_control_compare.csv"):
    assert (out5 / f).exists(), f
assert not (out / "risk_control_log.csv").exists()
cw = pd.read_csv(out5 / "risk_controlled_weights.csv", index_col=0)
assert (cw.abs().sum(axis=1) <= 1.0 + 1e-12).all()                 # max_scale=1: 레버리지 금지
limits, cres, cmp, bt_raw, bt_ctrl = rr.run_controls({k: v["close"] for k, v in FRAMES.items()}, {"AAA": 0.6, "BBB": 0.4},
                                                    0.975, 250, "fhs", 0.5, 0.95)
assert cmp.shape[0] == 2 and cres.counts()["es_scaled"] > 0 and bt_ctrl.n > 0
print("[PASS] 5) --controls: 한도·발동·비교표·전/후 백테스트 출력, CSV 3개, 총노출 ≤ 1, 기본 꺼짐")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
