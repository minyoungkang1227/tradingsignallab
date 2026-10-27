"""
data/quality.py 스모크 테스트 — 데이터 품질 게이트.

(1) 깨끗한 데이터는 전부 통과하는가.
(2) 오염을 하나씩 심으면 '그 검사만' 정확히 걸리는가 (검사끼리 독립).
(3) 심각도: 실패=중단, 경고=계속.
(4) collect_universe / run_backtest.load_data 연결, 기본값이 이전 동작을 보존하는가.
합성 데이터 — 실제 시장 성과가 아니다.
"""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.quality import (FAIL, PASS, SKIP, WARN, V_OK, V_STOP, V_WARN, run_quality_checks)

NAME_OHLC, NAME_IDX, NAME_MISS, NAME_GAP, NAME_RET = (
    "OHLC 정합성", "인덱스(중복·정렬)", "결측값", "날짜 공백", "수익률 이상치")


def clean(n=100):
    """영업일 n개, 시가=종가=100, 고가 101, 저가 99, 거래량 1000. 수익률 0, 최대 간격 3일(주말)."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                         "volume": 1000.0}, index=idx)


def statuses(rep):
    return {c.name: c.status for c in rep.checks}


def only_fail(rep, name, want=FAIL):
    """name 검사만 want 상태이고 나머지 네 검사는 전부 통과인지."""
    st = statuses(rep)
    others = {k: v for k, v in st.items() if k != name}
    assert st[name] == want, st
    assert all(v == PASS for v in others.values()), st


# 1) 깨끗한 데이터 -----------------------------------------------------------
rep = run_quality_checks(clean(), "CLEAN")
assert rep.verdict == V_OK and all(c.status == PASS for c in rep.checks), rep.format()
assert len(rep.checks) == 5
print("[OK] 1) 깨끗한 영업일 100봉 -> 5개 검사 전부 통과, 판정 통과")

# 2) OHLC: high < close ------------------------------------------------------
# 손계산: 행10 close=102 > high=101 이므로 위반 1행. 종가 100->102 는 +2%라 이상치 아님.
d = clean(); d.iloc[10, d.columns.get_loc("close")] = 102.0
rep = run_quality_checks(d); only_fail(rep, NAME_OHLC)
assert "high<max(open,close) 1행" in rep.by_name()[NAME_OHLC].detail, rep.by_name()[NAME_OHLC].detail
assert rep.verdict == V_STOP
print("[OK] 2) high<close 1행 -> OHLC 정합성만 실패, 나머지 독립적으로 통과, 중단")

# 3) OHLC: low > open --------------------------------------------------------
d = clean(); d.iloc[20, d.columns.get_loc("low")] = 100.5      # open=100 < low=100.5
rep = run_quality_checks(d); only_fail(rep, NAME_OHLC)
assert "low>min(open,close) 1행" in rep.by_name()[NAME_OHLC].detail
print("[OK] 3) low>open 1행 -> OHLC 정합성만 실패")

# 4) 거래정지일처럼 가격이 0 -------------------------------------------------
# 가격 0 행은 OHLC 검사가 잡고, 수익률 이상치 검사는 close<=0 을 제외하므로 중복 신고하지 않는다.
d = clean(); d.iloc[5, d.columns.get_loc("open")] = 0.0
d.iloc[5, d.columns.get_loc("high")] = 0.0
d.iloc[5, d.columns.get_loc("low")] = 0.0
d.iloc[5, d.columns.get_loc("close")] = 0.0
rep = run_quality_checks(d); only_fail(rep, NAME_OHLC)
assert "가격<=0 1행" in rep.by_name()[NAME_OHLC].detail
print("[OK] 4) 가격 0 -> OHLC만 실패(수익률 이상치로 이중 신고하지 않음)")

# 5) 날짜 중복 / 역순 --------------------------------------------------------
d = clean(); d = pd.concat([d, d.iloc[[-1]]])
rep = run_quality_checks(d); only_fail(rep, NAME_IDX)
assert "중복 날짜 1개" in rep.by_name()[NAME_IDX].detail
d = clean(); d = d.iloc[[1, 0] + list(range(2, 100))]
rep = run_quality_checks(d); only_fail(rep, NAME_IDX)
assert "오름차순 아님" in rep.by_name()[NAME_IDX].detail
d = clean().reset_index(drop=True)
rep = run_quality_checks(d)
assert statuses(rep)[NAME_IDX] == FAIL and statuses(rep)[NAME_GAP] == SKIP, statuses(rep)
print("[OK] 5) 날짜 중복·역순·비날짜 인덱스 -> 인덱스 검사 실패, 다른 검사는 독립")

# 6) 결측: 가격 NaN -> 실패, 거래량 NaN -> 경고 --------------------------------
d = clean(); d.iloc[7, d.columns.get_loc("close")] = np.nan
rep = run_quality_checks(d); only_fail(rep, NAME_MISS)       # NaN 행은 OHLC 검사에서 제외되므로 OHLC는 통과
d = clean(); d.iloc[8, d.columns.get_loc("volume")] = np.nan
rep = run_quality_checks(d); only_fail(rep, NAME_MISS, want=WARN)
assert rep.verdict == V_WARN
print("[OK] 6) 가격 NaN=실패(중단), 거래량 NaN=경고(계속)")

# 7) 날짜 공백 ---------------------------------------------------------------
# 2024-01-01..01-31 일간 31개 뒤 2024-02-20. 1/31->2/20 = 20일. 한도 7일 초과 1곳, 최대 20일.
idx = list(pd.date_range("2024-01-01", "2024-01-31", freq="D")) + list(pd.date_range("2024-02-20", periods=40, freq="D"))
d = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=pd.DatetimeIndex(idx))
rep = run_quality_checks(d); only_fail(rep, NAME_GAP, want=WARN)
assert "최대 20일" in rep.by_name()[NAME_GAP].detail and "1곳" in rep.by_name()[NAME_GAP].detail
assert run_quality_checks(d, max_gap_days=30).verdict == V_OK        # 한도를 올리면 통과
print("[OK] 7) 20일 공백 1곳 -> 경고. max_gap_days=30이면 통과")

# 8) 수익률 이상치 -----------------------------------------------------------
# 행50만 140(+40%), 다음 날 100으로 복귀(-28.57%, 한도 30% 미만). 이상치는 +40% 하나뿐.
d = clean()
for col, v in (("open", 140.0), ("high", 141.0), ("low", 139.0), ("close", 140.0)):
    d.iloc[50, d.columns.get_loc(col)] = v
rep = run_quality_checks(d); only_fail(rep, NAME_RET, want=WARN)
assert "1개" in rep.by_name()[NAME_RET].detail and "+40.0%" in rep.by_name()[NAME_RET].detail, rep.by_name()[NAME_RET].detail
assert run_quality_checks(d, max_abs_return=0.5).verdict == V_OK
assert run_quality_checks(d, max_abs_return=0.25).by_name()[NAME_RET].detail.count("2개") == 1   # -28.6%도 걸림
print("[OK] 8) +40% 하루 -> 이상치 1개(경고). 한도 50%면 통과, 25%면 -28.6%까지 2개")

# 9) 일봉보다 짧은 간격이면 공백 기준 미적용 ---------------------------------
idx = pd.date_range("2024-01-01", periods=200, freq="h")
d = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
assert statuses(run_quality_checks(d))[NAME_GAP] == SKIP
print("[OK] 9) 시간봉 -> 날짜 공백 검사 해당없음")

# 10) 필수 컬럼 없음 / None -> 크래시 없이 실패 --------------------------------
rep = run_quality_checks(clean().drop(columns=["high"]))
assert statuses(rep)[NAME_OHLC] == FAIL and rep.verdict == V_STOP
assert run_quality_checks(None).verdict == V_STOP
print("[OK] 10) high 컬럼 없음/None -> 예외 없이 중단 판정")

# 11) 여러 오염 동시 -> 각 검사가 독립적으로 모두 보고 ------------------------
d = clean(); d.iloc[10, d.columns.get_loc("close")] = 102.0; d.iloc[8, d.columns.get_loc("volume")] = np.nan
rep = run_quality_checks(d)
assert statuses(rep)[NAME_OHLC] == FAIL and statuses(rep)[NAME_MISS] == WARN
print("[OK] 11) 오염 둘 -> 실패와 경고가 한 보고서에 함께 나옴")

# 12) collect_universe 연결 ---------------------------------------------------
from data.collect import collect_universe

bad = clean(); bad.iloc[10, bad.columns.get_loc("close")] = 102.0
warn = clean(); warn.iloc[8, warn.columns.get_loc("volume")] = np.nan
frames = {"GOOD": clean(), "BAD": bad, "WARN": warn}


def fetch(sym):
    if sym == "BOOM":
        raise RuntimeError("network")
    return frames[sym].copy()


# 기본값(quality_gate=False)은 이전 동작: 모순 데이터도 그대로 통과한다.
r0 = collect_universe(["GOOD", "BAD", "WARN"], fetch, verbose=False)
assert set(r0.ok) == {"GOOD", "BAD", "WARN"} and r0.quality == {}
r1 = collect_universe(["GOOD", "BAD", "WARN", "BOOM"], fetch, verbose=False, quality_gate=True)
assert set(r1.ok) == {"GOOD", "WARN"}, r1.ok.keys()
assert "데이터 품질 실패" in r1.failed["BAD"] and "OHLC" in r1.failed["BAD"], r1.failed["BAD"]
assert "RuntimeError" in r1.failed["BOOM"]                       # 한 종목의 예외가 나머지를 막지 않음
assert r1.quality["WARN"].verdict == V_WARN and r1.quality["GOOD"].verdict == V_OK
print("[OK] 12) collect_universe: 기본 꺼짐=이전과 동일, 켜면 BAD만 제외(사유 기록)·WARN은 포함")

# 13) run_backtest.load_data 연결 ----------------------------------------------
import run_backtest as rb

orig = rb.fetch_krx_ohlcv
args = SimpleNamespace(market="krx", symbol="X", start="20240101", end="20240601", skip_quality_gate=False)
try:
    rb.fetch_krx_ohlcv = lambda *a, **k: bad.copy()
    try:
        rb.load_data(args)
        raise AssertionError("모순 데이터인데 멈추지 않음")
    except SystemExit as e:
        assert "데이터 품질 오류" in str(e) and "--skip-quality-gate" in str(e)
    args.skip_quality_gate = True
    assert len(rb.load_data(args)) == 100                         # 무시 옵션이면 통과
    del args.skip_quality_gate                                    # 옵션 자체가 없는 네임스페이스(alerts 등)는 게이트 켜짐
    try:
        rb.load_data(args)
        raise AssertionError("옵션 없는 네임스페이스에서 게이트가 꺼져 있음")
    except SystemExit:
        pass
    rb.fetch_krx_ohlcv = lambda *a, **k: warn.copy()
    assert len(rb.load_data(args)) == 100                         # 경고는 계속 진행
finally:
    rb.fetch_krx_ohlcv = orig
print("[OK] 13) load_data: 모순=SystemExit, 경고=계속, --skip-quality-gate=통과, 옵션 없는 args=게이트 켜짐")

print("\nSMOKE TEST (quality) OK  (합성 데이터 — 실제 시장 성과가 아님)")
