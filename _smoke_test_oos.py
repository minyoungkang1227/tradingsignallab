"""
backtest.oos 스모크테스트 — 기대값은 손계산. 합성 데이터 결과는 실제 시장 성과가 아님.
"""
import numpy as np
import pandas as pd

from backtest.oos import (OOSWindows, purged_windows, mask_events, one_sided_p, judge_oos,
                          select_train_signals, rank_stability, oos_study, format_oos_report)

def close(a, b, tol=1e-9): return abs(a - b) <= tol

# --- 1) 퍼지 구간 손계산 -------------------------------------------------------
idx = pd.bdate_range("2024-01-01", periods=100)
w = purged_windows(idx, train_end=idx[59], test_start=idx[60], max_horizon=5, entry_lag=1, warmup_bars=10)
assert w.train_lo == idx[10] and w.train_hi == idx[59 - 6]          # 퍼지 = 5 + 1 = 6봉
assert w.test_lo == idx[60] and w.test_hi == idx[99]
assert w.n_train_bars == 53 - 10 + 1 == 44 and w.n_test_bars == 40 and w.purged_bars == 6
w2 = purged_windows(idx, idx[59] + pd.Timedelta(hours=12), idx[60], 5, 1, 10)   # 학습 종료일이 봉 사이여도
assert w2.train_hi == idx[53]
for bad in [dict(train_end=idx[60], test_start=idx[60]),                          # 같은 날: 겹침
            dict(train_end=idx[60], test_start=idx[50]),                          # 역순
            dict(train_end=idx[59], test_start=idx[60], warmup_bars=55),          # 학습 이벤트 구간 소멸
            dict(train_end=idx[59], test_start=idx[99] + pd.Timedelta(days=5))]:  # 검증 데이터 없음
    kw = dict(max_horizon=5, entry_lag=1, warmup_bars=10); kw.update(bad)
    try:
        purged_windows(idx, **kw); raise SystemExit(f"예외가 나야 하는데 통과: {bad}")
    except ValueError:
        pass
print("[PASS] 1) 퍼지 구간: 학습 이벤트 10~53(44봉), 검증 60~99(40봉), 겹침·역순·소멸·데이터없음은 예외")

# --- 2) 이벤트 마스킹 손계산 --------------------------------------------------
ev = pd.Series(False, index=idx); ev.iloc[[5, 12, 15, 18, 25, 40]] = True
dfm = pd.DataFrame({"close": 100.0, "evt": ev}, index=idx)
specs = [("ind", "evt", 1, "x")]
masked, removed = mask_events({"ind": dfm}, specs, idx[10], idx[20], blocked=[idx[15], idx[30]])
got = list(np.flatnonzero(masked["ind"]["evt"].values))
assert got == [12, 18], got                      # 구간 [10,20] 안에서 15는 차단 -> 12, 18만
assert removed == 1                              # 차단 때문에 빠진 것은 15 하나(30은 이벤트가 아님, 5/25/40은 구간 밖)
assert dfm["evt"].sum() == 6                     # 원본은 건드리지 않음
assert (masked["ind"]["close"] == 100.0).all()   # 가격 열 불변
print("[PASS] 2) 마스킹: 이벤트 6개 중 구간[10,20]·차단 반영 -> 12, 18만 남고 제외 1건, 원본 불변")

# --- 3) 단측 p / 판정 ------------------------------------------------------------
assert close(one_sided_p(1.0, 0.04), 0.02) and close(one_sided_p(-1.0, 0.04), 0.98)
assert np.isnan(one_sided_p(np.nan, 0.04)) and np.isnan(one_sided_p(1.0, np.nan))
v, p1, a = judge_oos(12, 1.0, 0.01, k=2)
assert (v, close(p1, 0.005), close(a, 0.025)) == ("유지됨", True, True)
v, p1, a = judge_oos(12, 1.0, 0.06, k=2)                    # 단측 0.03 > 0.025
assert v == "유지 안 됨" and close(p1, 0.03)
v, _, _ = judge_oos(12, 1.0, 0.06, k=1)                     # 같은 p라도 k=1이면 0.03 < 0.05 -> 통과 (Bonferroni 효과)
assert v == "유지됨"
assert judge_oos(9, 1.0, 0.001, k=1)[0] == "판정불가"        # 표본 미달
assert judge_oos(12, -1.0, 0.01, k=1)[0] == "유지 안 됨"     # 부호 반대
assert judge_oos(12, np.nan, np.nan, k=1)[0] == "판정불가"
print("[PASS] 3) 단측 p와 판정: 같은 p(0.06)가 k=2에선 탈락, k=1에선 통과; 표본<10 판정불가; 부호 반대 탈락")

# --- 4) 선택 규칙 ------------------------------------------------------------------
s = pd.DataFrame({
    "signal": list("AAABBBCCCDDD"), "horizon": [3, 5, 10] * 4,
    "significant_bh":  [1, 1, 1,  1, 1, 1,  0, 0, 0,  1, 1, 1],
    "significant_dsr": [0, 1, 0,  1, 0, 0,  1, 1, 1,  1, 1, 1],
    "tb_significant_dsr": [0, 1, 0,  0, 0, 0,  1, 1, 1,  1, 0, 0]}).astype({"significant_bh": bool, "significant_dsr": bool, "tb_significant_dsr": bool})
sel = select_train_signals(s, 5)
assert sorted(sel["signal"]) == ["A", "D"]                    # h=5: A(BH,DSR), D(BH,DSR); B는 DSR 미통과, C는 BH 미통과
assert sorted(select_train_signals(s, 5, exclude=["A"])["signal"]) == ["D"]
assert sorted(select_train_signals(s, 5, require_barrier=True)["signal"]) == ["A"]   # 배리어 DSR까지: D는 h=5에서 False
try:
    select_train_signals(s.drop(columns=["significant_dsr"]), 5); raise SystemExit("컬럼 없으면 예외여야 함")
except ValueError:
    pass
print("[PASS] 4) 선택 규칙: BH AND DSR(h=5) -> A, D / 제외·배리어 DSR 옵션 / 컬럼 없으면 예외")

# --- 5) 순위 안정성 -----------------------------------------------------------------
jn = pd.DataFrame({"tr_n_obs": 10, "te_n_obs": 10, "tr_mean_return_pct": np.arange(1, 11, dtype=float),
                   "te_mean_return_pct": np.arange(1, 11, dtype=float)})
assert close(rank_stability(jn)[0], 1.0)
jn2 = jn.assign(te_mean_return_pct=jn["te_mean_return_pct"][::-1].values)
assert close(rank_stability(jn2)[0], -1.0)
c, n_ = rank_stability(jn.head(9)); assert np.isnan(c) and n_ == 9        # 행 부족 -> 계산 안 함
c, n_ = rank_stability(jn.assign(te_n_obs=3)); assert np.isnan(c) and n_ == 0   # 표본 미달 행은 제외
print("[PASS] 5) 순위상관: 같은 순서 +1, 역순 -1, 행<10 또는 표본<5면 NaN")

# --- 6) oos_study 통합: 주입한 우위가 학습·검증 모두에 있을 때/검증에서 사라질 때 -----------------
N = 600
EVT = np.arange(12, N - 5, 8)                 # 이벤트 12, 20, ..., 356, 364, ... (간격 8봉)
CK = dict(horizons=(1, 3, 5), max_curve_horizon=5, entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
          newey_west=False, drop_overlapping=False, fdr=0.10, deflate=True, dsr_threshold=0.95, min_events=20,
          extra_trials=0, barrier=False)

def make(edge_train, edge_test, seed=3):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.005, N)
    for e in EVT:
        r[e + 1] += edge_train if e < 360 else edge_test       # 이벤트 다음 봉 수익률에 우위를 심는다
    c = 10000 * np.cumprod(1 + r)
    ix = pd.bdate_range("2021-01-01", periods=N)
    d = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e5}, index=ix)
    d["crossover_confirmed"] = False
    d.iloc[EVT, d.columns.get_loc("crossover_confirmed")] = True
    d["crossunder_confirmed"] = False
    return ix, {"delta_trading": d}

ix, dfs = make(0.015, 0.015)
TRAIN_END, TEST_START = ix[359], ix[360]
res = oos_study(dfs, CK, TRAIN_END, TEST_START, warmup_bars=60, min_test_n=10)
# 손계산: 학습 이벤트 구간 = [워밍업 60, 학습종료 359 - 퍼지(5+0)] = [60, 354].
lo_pos, hi_pos = 60, 359 - 5
exp_tr = int(((EVT >= lo_pos) & (EVT <= hi_pos)).sum())
exp_te = int((EVT >= 360).sum())
row_tr = res.summary_train[(res.summary_train["signal"] == "델타트레이딩-골든크로스") & (res.summary_train["horizon"] == 3)].iloc[0]
row_te = res.summary_test[(res.summary_test["signal"] == "델타트레이딩-골든크로스") & (res.summary_test["horizon"] == 3)].iloc[0]
assert row_tr["n_events"] == exp_tr, (row_tr["n_events"], exp_tr)       # 학습 이벤트 수가 퍼지 반영 손계산과 일치
assert row_te["n_events"] == exp_te, (row_te["n_events"], exp_te)
assert 356 in EVT and 356 > hi_pos     # 이벤트 356: 순방향 창이 361까지라 학습 종료(359)를 넘는다 -> 학습에서 빠져야 함
assert 348 in EVT and 348 <= hi_pos    # 이벤트 348: 창이 353까지라 안전 -> 학습에 포함
assert not (EVT[(EVT >= lo_pos) & (EVT <= hi_pos)] == 356).any()
assert 356 < 360                       # 356은 검증 구간(>=360)에도 들어가지 않는다 = 어느 쪽에도 쓰이지 않는 퍼지 이벤트
assert res.k == 1 and res.selected.iloc[0]["verdict"] == "유지됨", res.selected
assert close(res.alpha_adj, 0.05)
print(f"[PASS] 6a) 우위가 학습·검증 모두 있음: 학습 {exp_tr}건 / 검증 {exp_te}건 손계산 일치, "
      f"선택 {res.k}개 -> '{res.selected.iloc[0]['verdict']}' (검증 t={res.selected.iloc[0]['te_t_stat']:.1f})")

ix, dfs = make(0.015, -0.005)                                         # 검증 구간에선 우위가 사라지고 손실
res2 = oos_study(dfs, CK, ix[359], ix[360], warmup_bars=60)
assert res2.k == 1 and res2.selected.iloc[0]["verdict"] == "유지 안 됨", res2.selected
print("[PASS] 6b) 학습에서만 우위(과적합 사례): 선택은 되지만 검증에서 '유지 안 됨'")

res3 = oos_study(make(0.015, 0.015)[1], CK, ix[359], ix[360], warmup_bars=60, min_test_n=50)
assert res3.selected.iloc[0]["verdict"] == "판정불가"
print("[PASS] 6c) 검증 이벤트가 기준(50건)에 못 미치면 '판정불가'")

ix, dfs = make(0.0, 0.0, seed=5)                                      # 우위 전혀 없음
res4 = oos_study(dfs, CK, ix[359], ix[360], warmup_bars=60)
assert res4.k == 0
rep = format_oos_report(res4)
assert "판단할 수 없다" in rep and "탐색용 후보" in rep
print("[PASS] 6d) 우위 없음: 선택 0개, 보고서가 '판단할 수 없다'와 탐색용 후보(검증 아님)를 표시")

ix, dfs = make(0.015, 0.015)
res5 = oos_study(dfs, CK, ix[359], ix[360], warmup_bars=60, blocked=[ix[68]])   # 68은 이벤트(12+8*7)
assert res5.removed_train == 1 and res5.removed_test == 0
r5 = res5.summary_train[(res5.summary_train["signal"] == "델타트레이딩-골든크로스") & (res5.summary_train["horizon"] == 3)].iloc[0]
assert r5["n_events"] == exp_tr - 1
print("[PASS] 6e) 차단 날짜: 학습 이벤트 1건 제외 및 건수 보고")

rep = format_oos_report(res)
for token in ["표본 외(OOS) 검증", "퍼지", "유지됨", "단측 p <", "[안정성]"]:
    assert token in rep, token
print("[PASS] 6f) 보고서에 구간·퍼지·기준·판정·안정성이 모두 근거와 함께 출력")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
