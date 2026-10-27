"""
alerts/check_risk.py 스모크테스트 — 일일 리스크 요약·한도 위반 알림의 배관.
수집(rb.load_data)은 합성 종가로 몽키패치. 합성 데이터 결과는 실제 시장 성과·리스크가 아님.
"""
import contextlib, io, json, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_backtest as rb
import alerts.check_risk as cr

rng = np.random.default_rng(21)
n = 700
ix = pd.bdate_range("2023-06-01", periods=n)


def mk(sig, seed):
    r = np.random.default_rng(seed).normal(0, sig, n)
    c = 100 * np.cumprod(1 + r)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=ix)


FRAMES = {"AAA": mk(0.012, 1), "BBB": mk(0.02, 2), "KRW-BTC": mk(0.035, 3)}
rb.load_data = lambda args: FRAMES[args.symbol].copy()

# --- 1) 비중 해석 ------------------------------------------------------------------------
w, note = cr.resolve_weights([{"symbol": "AAA", "weight": 0.7}, {"symbol": "BBB", "weight": 0.3}])
assert w == {"AAA": 0.7, "BBB": 0.3} and "watchlist" in note
w2, note2 = cr.resolve_weights([{"symbol": "AAA", "weight": 0.7}, {"symbol": "BBB"}])
assert w2 == {"AAA": 0.5, "BBB": 0.5} and "동일비중" in note2                # 일부만 적히면 동일비중
print("[PASS] 1) weight 전부 있으면 그대로, 하나라도 없으면 동일비중")

# --- 2) assess_market: 숫자가 risk/ 모듈과 같은가 ------------------------------------------------
from risk.positions import closes_frame, constant_weights, portfolio_returns
from risk.var import compute_var
closes = {k: FRAMES[k]["close"] for k in ("AAA", "BBB")}
a = cr.assess_market(closes, {"AAA": 0.6, "BBB": 0.4}, alpha=0.975, window=250)
px = closes_frame(closes); rp = portfolio_returns(constant_weights({"AAA": 0.6, "BBB": 0.4}, px.index), px)
v = compute_var(rp, "fhs", 0.975, 250)
assert a["date"] == ix[-1] and abs(a["today_ret"] - rp.iloc[-1]) < 1e-15 and abs(a["today_es"] - v["es"].iloc[-1]) < 1e-15
assert a["next_es"] > 0 and a["next_var"] > 0 and a["next_es"] >= a["next_var"]
assert a["backtest"].n > 0 and a["limits"].es_target > 0
print(f"[PASS] 2) assess_market: 오늘 손익·ES가 risk 모듈과 일치, 내일 VaR {a['next_var']:.2%}/ES {a['next_es']:.2%}, 판정 {a['backtest'].verdict}")

# --- 3) 위반 감지: 마지막 날 -15% -> 오늘 손실 > ES 위반 -------------------------------------
crash = {k: v.copy() for k, v in closes.items()}
crash["AAA"].iloc[-1] = crash["AAA"].iloc[-2] * 0.85
crash["BBB"].iloc[-1] = crash["BBB"].iloc[-2] * 0.85
a3 = cr.assess_market(crash, {"AAA": 0.6, "BBB": 0.4})
assert any("오늘 손실" in b for b in a3["breaches"]), a3["breaches"]
assert a3["next_es"] > a["next_es"]                                             # 폭락 후 내일 ES는 커진다(EWMA 반응)
msg3 = cr.format_message("krx", {"AAA": 0.6, "BBB": 0.4}, "watchlist weight", a3, [])
assert msg3.startswith("⚠ 한도 위반") and "위반:" in msg3 and "투자자문 아님" in msg3
msg2 = cr.format_message("krx", {"AAA": 0.6, "BBB": 0.4}, "watchlist weight", a, [])
assert not msg2.startswith("⚠") and "리스크 일일 요약 [국내주식]" in msg2 and "모델 백테스트" in msg2
print("[PASS] 3) 폭락일은 ⚠ 위반 머리말 + 위반 목록, 평상시는 요약만")

# --- 4) main 배관: 시장별 한 통, dry-run, state 중복 방지, risk:false 제외 ------------------------
def run(watch, state_path):
    wl = tempfile.mktemp(suffix=".json")
    Path(wl).write_text(json.dumps(watch, ensure_ascii=False), encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cr.main(["--watchlist", wl, "--state", state_path, "--dry-run", "--window", "250"])
    return buf.getvalue()

st = tempfile.mktemp(suffix=".json")
watch = [{"market": "krx", "symbol": "AAA", "label": "A", "weight": 0.6},
         {"market": "krx", "symbol": "BBB", "label": "B", "weight": 0.4},
         {"market": "upbit", "symbol": "KRW-BTC", "label": "비트코인", "weight": 1.0}]
out = run(watch, st)
assert out.count("리스크 일일 요약") == 2 and "[국내주식]" in out and "[코인]" in out, out
state = json.loads(Path(st).read_text(encoding="utf-8"))
assert set(state) == {"risk:krx", "risk:upbit"} and state["risk:krx"]["last_date"] == str(ix[-1].date())
out2 = run(watch, st)
assert "이미 보냈음" in out2 and "리스크 일일 요약" not in out2                 # 같은 날 재실행은 중복 발송 안 함
st2 = tempfile.mktemp(suffix=".json")
out3 = run(watch[:2] + [dict(watch[2], risk=False)], st2)
assert out3.count("리스크 일일 요약") == 1 and "[코인]" not in out3            # risk:false 제외
print("[PASS] 4) main: 시장별 한 통, state로 같은 날 중복 방지, risk:false 제외")

# --- 5) 데이터 품질 중단: 종목 하나가 SystemExit -> 그 시장은 계산 대신 중단 알림 -------------------
def load_with_halt(args):
    if args.symbol == "BBB":
        raise SystemExit("[데이터 품질 오류] 분석을 중단합니다.\n  [X] OHLC 정합성 실패")
    return FRAMES[args.symbol].copy()
rb.load_data = load_with_halt
out5 = run(watch[:2], tempfile.mktemp(suffix=".json"))
assert "리스크 계산 중단" in out5 and "데이터 중단" in out5 and "리스크 일일 요약" not in out5, out5
rb.load_data = lambda args: FRAMES[args.symbol].copy()
print("[PASS] 5) 품질 게이트 중단 종목이 있으면 VaR를 내지 않고 중단 알림")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과·리스크가 아님)")
