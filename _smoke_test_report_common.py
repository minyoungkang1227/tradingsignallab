"""backtest.report_common 스모크테스트 — 문자열·개수는 손계산. 합성 데이터 결과는 실제 시장 성과가 아님."""
import numpy as np, pandas as pd
from backtest.report_common import (describe_execution, format_block, data_label_text, FIELDS,
                                    oos_header, walk_forward_header, regime_header, portfolio_header)
from backtest.oos import oos_study
from backtest.walkforward import walk_forward
from backtest.regime import classify_regime, regime_study
from backtest.portfolio import simulate_portfolio

# 1) 체결 가정 문장 — 손으로 쓴 기대 문자열과 일치
ck0 = dict(entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0)
assert describe_execution(ck0) == "신호 당일 종가 진입(look-ahead 가능), 수수료·세금 0bp(왕복), 슬리피지 0bp(왕복, 고정), ※비용 미반영"
ck1 = dict(entry_lag=1, entry_col="open", cost_bps=36.0, slippage_bps=10.0, drop_overlapping=True, newey_west=True,
           deflate=True, dsr_threshold=0.95, barrier=True, pt_mult=2.0, sl_mult=1.0, barrier_sequential=True)
assert describe_execution(ck1) == ("신호 1봉 뒤 시가 진입, 수수료·세금 36bp(왕복), 슬리피지 10bp(왕복, 고정), "
                                   "보정 겹침 제거·Newey-West·DSR≥0.95·배리어 익절2σ/손절1σ·순차")
ix = pd.bdate_range("2024-01-01", periods=4)
ck2 = dict(entry_lag=1, entry_col="open", cost_bps=15.0, slippage_bps=pd.Series([5.0, 15.0, 25.0, 35.0], index=ix))
assert "슬리피지 동적(평균 20.0bp, 최대 35.0bp)" in describe_execution(ck2) and "※비용 미반영" not in describe_execution(ck2)
print("[PASS] 1) 체결 가정 문장: 비용 0이면 '※비용 미반영' 경고, 비용·보정·동적 슬리피지 요약 손계산 일치")

# 2) 데이터 구분 라벨 — 합성은 '실제 시장 성과가 아님', 미지정은 '미표시', 잘못된 값은 예외
assert "실제 시장 성과가 아님" in data_label_text("synthetic") and data_label_text("real") == "실제 시장 데이터"
assert "미표시" in data_label_text(None)
try: data_label_text("fake"); raise SystemExit("예외 필요")
except ValueError: pass
print("[PASS] 2) 데이터 라벨: synthetic은 '실제 시장 성과가 아님', 미지정은 '미표시', 잘못된 값은 예외")

# 3) 네 가지 헤더가 같은 항목을 같은 순서로 낸다
N = 1200; EVT = np.arange(12, N - 5, 8)
CK = dict(horizons=(1, 3, 5), max_curve_horizon=5, entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
          newey_west=False, drop_overlapping=False, fdr=0.10, deflate=True, dsr_threshold=0.95, min_events=20,
          extra_trials=0, barrier=False)
def make(edge, seed=11):
    rng = np.random.default_rng(seed); r = rng.normal(0, 0.005, N)
    for e in EVT: r[e + 1] += edge
    c = 10000 * np.cumprod(1 + r); ixx = pd.bdate_range("2019-01-01", periods=N)
    d = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e5}, index=ixx)
    d["crossover_confirmed"] = False; d.iloc[EVT, d.columns.get_loc("crossover_confirmed")] = True
    d["crossunder_confirmed"] = False
    return ixx, {"delta_trading": d}
ixx, dfs = make(0.015)
oos = oos_study(dfs, CK, ixx[599], ixx[600], warmup_bars=60)
wf = walk_forward(dfs, CK, ixx[599], test_bars=150, warmup_bars=60)
rg = regime_study(dfs, classify_regime(dfs["delta_trading"]), CK)
cand = pd.DataFrame([dict(symbol="A", signal="s", direction=1, entry_date=ixx[300], exit_date=ixx[304], bars_held=4, ret=0.02,
                          exit_reason="pt", target_vol=0.02, entry_price=100.0, gross_ret=0.024)])
pf = simulate_portfolio(cand, {"A": pd.Series(100.0, index=ixx)})
heads = {"oos": oos_header(oos, CK, "synthetic"), "wf": walk_forward_header(wf, CK, "synthetic"),
         "regime": regime_header(rg, CK, 3, "synthetic"), "pf": portfolio_header(pf, CK, 1, "synthetic")}
for name, h in heads.items():
    lines = [l for l in h.splitlines() if l.startswith("  ")]
    assert h.startswith("[실행 조건·검증 요약]") and [l.split("  ")[1] for l in lines] == list(FIELDS), (name, h)
    assert "합성 데이터 — 실제 시장 성과가 아님" in h and "비용 미반영" in h, name
print("[PASS] 3) OOS·Walk-forward·국면·포트폴리오 헤더 모두 같은 7개 항목(분석/데이터/기간/체결/표본/기준/결과)을 같은 순서로 출력")

# 4) 내용: OOS 선택·판정 개수, WF 판정 개수, 포트폴리오는 '해당 없음'
sel_n = oos.k; kept = int((oos.selected["verdict"] == "유지됨").sum())
assert f"선택 {sel_n}개 중 검증 통과(유지됨) {kept}개" in heads["oos"] and "IS(학습)" in heads["oos"] and "OOS(검증)" in heads["oos"]
assert "퍼지 5봉" in heads["oos"] and "단측 p < α/k" in heads["oos"]
vc = wf.by_signal["verdict"].value_counts()
assert f"안정 {int(vc.get('안정', 0))}개" in heads["wf"] and f"fold {wf.n_folds_planned}개 중 실행 {wf.n_folds_ok}개" in heads["wf"]
assert "해당 없음" in heads["pf"] and "후보 거래 1건 중 체결 1건" in heads["pf"]
assert "전역 BH 유의" in heads["regime"] and "보유 3봉 기준 국면별 관측 수" in heads["regime"]
print(f"[PASS] 4) 내용: OOS 선택 {sel_n}개·유지됨 {kept}개, 퍼지 5봉, WF fold·판정 개수, 포트폴리오는 판정 '해당 없음'")

# 5) 통과 신호가 없는 경우를 '통과'로 오해하지 않게 표시
ixx0, dfs0 = make(0.0, seed=5)
oos0 = oos_study(dfs0, CK, ixx0[599], ixx0[600], warmup_bars=60)
h0 = oos_header(oos0, CK, None)
assert "통과 신호 없음" in h0 and "미표시" in h0 and oos0.k == 0
wf0 = walk_forward(dfs0, CK, ixx0[599], test_bars=150, warmup_bars=60)
assert "통과 신호 없음" in walk_forward_header(wf0, CK, None)
print("[PASS] 5) 선택 신호 0개면 '통과 신호 없음 — 검증할 대상이 없음'으로 표시, 데이터 라벨 미지정은 '미표시'")

# 6) format_block은 항목 이름이 어긋나면 만들지 않는다(형식 고정)
assert format_block("x", "real", "p", "e", "s", "c", "r").count("\n") == 7
print("[PASS] 6) 표준 블록은 항상 제목 1줄 + 항목 7줄")
print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
