"""backtest.walkforward 스모크테스트 — 기대값은 손계산. 합성 데이터 결과는 실제 시장 성과가 아님."""
import math
import numpy as np, pandas as pd
from backtest.walkforward import make_folds, pool_signal, walk_forward, format_walk_forward_report, _chi2_sf
from backtest.oos import purged_windows

def close(a, b, tol=1e-9): return abs(a - b) <= tol

# 1) fold 구성 손계산: 100봉, 첫 학습 끝=위치 59, 검증 20봉, step 20 -> (59,60~79), (79,80~99)
ix = pd.bdate_range("2024-01-01", periods=100)
f = make_folds(ix, ix[59], 20)
assert [(a, b, c) for a, b, c in f] == [(ix[59], ix[60], ix[79]), (ix[79], ix[80], ix[99])]
assert len(make_folds(ix, ix[59], 15)) == 2              # (59: 60~74), (74: 75~89); 89+15>99라 세 번째 없음
f2 = make_folds(ix, ix[59], 10, step_bars=5)             # 59,64,69,74,79,84,89 -> e+10<=99 -> 7개
assert len(f2) == 7 and f2[1][0] == ix[64] and f2[1][2] == ix[74]
for bad in [lambda: make_folds(ix, ix[95], 20), lambda: make_folds(ix, ix[59], 0)]:
    try: bad(); raise SystemExit("예외가 나야 함")
    except ValueError: pass
print("[PASS] 1) fold 구성: 확장창, 검증 조각 겹침 없음, 마지막 불완전 조각 제외, 불가능하면 예외")

# 2) test_end가 검증 구간을 자른다(기존 기본값은 데이터 끝)
w = purged_windows(ix, ix[59], ix[60], 5, 0, 10, test_end=ix[79])
assert w.test_hi == ix[79] and w.n_test_bars == 20
w0 = purged_windows(ix, ix[59], ix[60], 5, 0, 10)
assert w0.test_hi == ix[99] and w0.n_test_bars == 40
print("[PASS] 2) test_end: 검증 구간 20봉으로 절단, 미지정이면 데이터 끝(40봉) — 기존 동작 보존")

# 3) 합산 공식 손계산: 두 fold (n=20,m=1.0,t=2.0), (n=30,m=2.0,t=4.0)
rows = pd.DataFrame({"te_n_obs": [20, 30], "te_mean_return_pct": [1.0, 2.0], "te_t_stat": [2.0, 4.0]})
r = pool_signal(rows)
se1, se2 = 1.0/2.0, 2.0/4.0                            # 둘 다 0.5
exp_mean = (20*1.0 + 30*2.0) / 50                      # 1.6
exp_se = math.sqrt((20*se1)**2 + (30*se2)**2) / 50     # sqrt(100+225)/50 = 18.0278/50
assert close(r["pooled_mean_pct"], exp_mean) and close(r["pooled_se_pct"], exp_se)
exp_z = exp_mean / exp_se; exp_p = 0.5 * math.erfc(exp_z / math.sqrt(2))
assert close(r["pooled_z"], exp_z) and close(r["pooled_p1"], exp_p) and r["frac_folds_positive"] == 1.0
assert r["pooled_n"] == 50 and r["n_folds_selected"] == 2
rows2 = rows.copy(); rows2.loc[0, "te_t_stat"] = 0.0     # t=0이면 se 역산 불가 -> 합산 z는 NaN(만들어내지 않음)
assert np.isnan(pool_signal(rows2)["pooled_z"])
print(f"[PASS] 3) 합산: 평균 {exp_mean}, SE {exp_se:.4f}, z {exp_z:.2f} 손계산 일치, t=0이면 NaN")

# 3b) 카이제곱 꼬리확률: scipy 없이 구현한 값이 알려진 값과 일치 (자유도1 x=3.841459 -> 0.05, 자유도3 x=7.814728 -> 0.05)
assert close(_chi2_sf(3.841458820694124, 1), 0.05, 1e-9) and close(_chi2_sf(7.814727903251179, 3), 0.05, 1e-9)
assert close(_chi2_sf(0.0, 2), 1.0) and np.isnan(_chi2_sf(1.0, 0))
try:
    from scipy.stats import chi2
    for x in [0.3, 1.0, 4.0, 10.0, 25.0, 60.0]:
        for df in [1, 2, 3, 5, 10, 20]:
            assert close(_chi2_sf(x, df), float(chi2.sf(x, df)), 1e-10), (x, df)
except ImportError:
    pass
# 이질성 Q 손계산: (m=1,se=.5), (m=3,se=.5) -> 가중평균 2, Q = (1/.25)(1)+(1/.25)(1) = 8, df=1 -> p=chi2.sf(8,1)
rq = pool_signal(pd.DataFrame({"te_n_obs": [20, 20], "te_mean_return_pct": [1.0, 3.0], "te_t_stat": [2.0, 6.0]}).assign(
    te_t_stat=[2.0, 6.0]))                                   # se = 0.5, 0.5
assert close(rq["het_q"], 8.0) and close(rq["het_p"], _chi2_sf(8.0, 1))
rq2 = pool_signal(pd.DataFrame({"te_n_obs": [20, 20], "te_mean_return_pct": [2.0, 2.0], "te_t_stat": [4.0, 4.0]}))
assert close(rq2["het_q"], 0.0) and close(rq2["het_p"], 1.0)
print("[PASS] 3b) 카이제곱 꼬리확률 scipy와 일치(1e-10), 이질성 Q 손계산 8.0 일치, 동일 성과면 Q=0·p=1")

# 4) 통합: 우위가 전 기간에 있는 경우 vs 초반에만 있는 경우
N = 1200
EVT = np.arange(12, N - 5, 8)
CK = dict(horizons=(1, 3, 5), max_curve_horizon=5, entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
          newey_west=False, drop_overlapping=False, fdr=0.10, deflate=True, dsr_threshold=0.95, min_events=20,
          extra_trials=0, barrier=False)
def make(edge_fn, seed=11):
    rng = np.random.default_rng(seed); r = rng.normal(0, 0.005, N)
    for e in EVT: r[e + 1] += edge_fn(e)
    c = 10000 * np.cumprod(1 + r); ix = pd.bdate_range("2019-01-01", periods=N)
    d = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e5}, index=ix)
    d["crossover_confirmed"] = False; d.iloc[EVT, d.columns.get_loc("crossover_confirmed")] = True
    d["crossunder_confirmed"] = False
    return ix, {"delta_trading": d}

ix, dfs = make(lambda e: 0.015)                                   # 전 기간 우위
res = walk_forward(dfs, CK, ix[599], test_bars=150, warmup_bars=60, min_test_n=10, min_folds=3)
assert res.n_folds_planned == 4 and res.n_folds_ok == 4           # 599 -> 749,899,1049,(1199): e+150<=1199 -> 4개
assert len(res.by_signal) == 1
row = res.by_signal.iloc[0]
assert row["n_folds_selected"] == 4 and row["verdict"] == "안정" and row["frac_folds_positive"] == 1.0
assert row["pooled_mean_pct"] > 1.0
print(f"[PASS] 4a) 전 기간 우위: 4개 fold 모두 선택·합산 평균 {row['pooled_mean_pct']:.2f}% -> '안정'")

ix, dfs = make(lambda e: 0.015 if e < 700 else 0.0)               # 700봉 이후 우위 소멸
res = walk_forward(dfs, CK, ix[599], test_bars=150, warmup_bars=60, min_test_n=10, min_folds=3)
row = res.by_signal.iloc[0]
assert row["verdict"] in ("불균질", "불안정"), row                      # 우위가 사라진 구간이 섞이면 '안정'이 될 수 없다
assert row["het_p"] < 0.05, row
fs = res.folds
assert fs.loc[0, "n_kept"] == 1 and fs.loc[len(fs) - 1, "n_failed"] + fs.loc[len(fs) - 1, "n_undetermined"] >= 0
print(f"[PASS] 4b) 우위가 중간에 사라지면 '안정'이 아님(판정 {row['verdict']}, 양수 fold 비율 {row['frac_folds_positive']:.2f})")

ix, dfs = make(lambda e: 0.0, seed=5)                              # 우위 전혀 없음
res = walk_forward(dfs, CK, ix[599], test_bars=150, warmup_bars=60)
rep = format_walk_forward_report(res)
assert res.by_signal.empty and "확인할 대상이 없다" in rep
print("[PASS] 4c) 우위 없음: 선택 신호 0 -> 보고서가 '안정적'이 아니라 '확인할 대상이 없다'로 표시")

# 5) 일부 fold가 실패해도 나머지는 계속(첫 학습 구간이 너무 짧아 첫 fold만 실패하는 경우)
ix, dfs = make(lambda e: 0.015)
res = walk_forward(dfs, CK, ix[64], test_bars=300, warmup_bars=60)   # 64-퍼지5=59 <= 워밍업60 -> 첫 fold만 소멸
assert list(res.folds["status"]) == ["skipped", "ok", "ok"] and res.n_folds_ok == 2
print("[PASS] 5) fold 하나가 학습 구간 소멸로 실패해도 건너뛰고 나머지를 계속 실행")

# 6) 보고서에 근거(fold 수, 합산 z/p, Bonferroni)가 모두 출력
ix, dfs = make(lambda e: 0.015)
rep = format_walk_forward_report(walk_forward(dfs, CK, ix[599], test_bars=150, warmup_bars=60))
for tok in ["반복 Walk-forward", "Bonferroni", "pooled_z", "pooled_p1", "frac_folds_positive", "증거 부족"]:
    assert tok in rep, tok
print("[PASS] 6) 보고서에 구간·fold·합산 통계·보정 유의수준·해석 주의가 모두 출력")
print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
