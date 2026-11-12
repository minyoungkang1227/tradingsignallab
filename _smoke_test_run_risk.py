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
assert pd.read_csv(out2 / "risk_var_backtest.csv")["method"].tolist() == ["fhs"]
# 숫자가 '어딘가' 있는지로는 부족하다. 97.5%가 신뢰수준 자리에 쓰였는지, 그리고 기본값
# 99.0%가 어느 경로에도 남지 않았는지를 본다. 예전 검증("97.5%" in txt2)은 그 숫자가
# 전혀 다른 맥락(분위·임계값)에 찍혀도 통과했다.
assert "신뢰수준 97.5%" in txt2, "신뢰수준 자리에 --alpha가 반영되지 않음"
assert "VaR 백테스트 [fhs] (신뢰수준 97.5%)" in txt2, "방법별 보고 머리글에 반영되지 않음"
assert "99.0%" not in txt2, "기본 신뢰수준이 남아 있음 → --alpha가 일부 경로에만 반영됨"
assert "97.5%" not in txt, "기본 실행에도 97.5%가 있으면 위 검증이 --alpha를 가리지 못한다"
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

# 6) 낙폭 서킷 비활성이 CLI 출력에 드러나는 경계 (회귀)
# 품질 게이트는 60봉 이상을 요구하고, 낙폭 부트스트랩은 수익률 60일(= 61봉) 이상을 요구한다.
# 그 사이의 딱 한 칸(60봉 = 수익률 59일)에서 네 통제 중 낙폭 서킷만 빠진 채 돌아간다.
# 예전에는 '낙폭 서킷 0일'로 찍혀서 '발동 안 함'과 구별되지 않았다.
def short_cli(bars):
    ixs = pd.bdate_range("2023-01-02", periods=bars)
    def mkb(sig):
        c = 100 * np.cumprod(1 + np.random.default_rng(3).normal(0, sig, bars))
        return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=ixs)
    sf = {"AAA": mkb(0.012), "BBB": mkb(0.02)}
    rr._fetch_one = lambda s, args: sf[s].copy()
    outd = tempfile.mkdtemp(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rr.main(["--market", "krx", "--symbols", "AAA,BBB", "--weights", "0.5,0.5", "--start", "20230102",
                 "--end", "20231231", "--data-label", "synthetic", "--output", outd, "--controls",
                 "--methods", "fhs", "--window", "25"])
    return buf.getvalue()

t60 = short_cli(60)                                      # 수익률 59일 -> 한도 미도출
assert "낙폭 한도 없음 → 서킷 비활성" in t60, "한도 출처가 비활성을 알리지 않음"
assert "낙폭 서킷 비활성" in t60 and "낙폭 서킷 0일" not in t60, "0일과 비활성이 구별되지 않음"
assert "경고:" in t60 and "낙폭 서킷브레이커는 적용되지 않았으며" in t60, "경고가 없음"
assert "수익률 59일 < 필요 60일" in t60, "사유에 실제/필요 표본 일수가 없음"
t61 = short_cli(61)                                      # 수익률 60일 -> 한도 도출
assert "낙폭 서킷 0일" in t61 and "서킷 비활성" not in t61, "한도가 나오는데 비활성으로 보고됨"
assert "경고:" not in t61
print("[PASS] 6) 60봉: 낙폭 서킷 비활성·경고가 출력에 드러남 / 61봉: 정상 활성")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
