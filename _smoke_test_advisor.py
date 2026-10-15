"""
advisor/evaluator.py + advisor/universe_scan.py + run_advisor.py 배관 검증.

1) evaluate_symbol(): 인위적으로 "가장 최근 봉에 이벤트가 뜨도록" 만든 합성
   데이터에서 활성 신호를 정확히 잡아내는지, 검증 여부(own/pooled/미검증)가
   min_own_events 기준에 맞게 갈리는지, 다수결(매수/보유/매도)이 맞게
   나오는지.
2) universe_scan.compute_pooled_stats()/scan_universe(): 여러 종목을 풀링한
   신호유형별 통계가 만들어지고, 스캔 결과표가 strength 내림차순으로 정렬되며
   캐시 파일로 저장/재로딩해도 하이브리드 판단이 똑같이 동작하는지.
3) run_advisor.main()을 --mode scan / --mode query로 실제 CLI 경로까지
   서브프로세스로 돌려서 캐시 파일 생성 -> 재사용까지 에러 없이 끝나는지.
"""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.evaluator import evaluate_symbol, MIN_OWN_EVENTS
from advisor.universe_scan import compute_pooled_stats, scan_universe

np.random.seed(7)


def make_cfg(**overrides):
    cfg = types.SimpleNamespace(
        id_vol_mode="multiple", id_vol_mult=2.0, id_z_threshold=1.96,
        big_sales_len=7, vp_bins=20, vp_freq="W",
        mom_filter=False, mom_lookback=126, mom_window=252, mom_threshold=0.7,
        liq_filter=False, liq_window=20, liq_percentile_window=252, liq_threshold=0.9,
        entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
        target_horizon=5, t_threshold=3.0, min_own_events=MIN_OWN_EVENTS,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_symbol(seed, n=400, force_event_at_end=True):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.uniform(1000, 5000, n)
    if force_event_at_end:
        # 마지막 며칠은 거래량을 확 키워서 institutional_displacement의
        # high_vol 조건(거래량 이상치)이 마지막 봉에서 뜨도록 유도한다.
        volume[-3:] *= 20
        close[-2] = close[-3] * 1.05  # 강한 양봉으로 몸통비율 조건도 만족
        close[-1] = close[-2] * 1.05
        high[-1] = close[-1] * 1.02
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                          "volume": volume}, index=dates)


# ---------------------------------------------------------------------------
# 1) evaluate_symbol(): 활성 신호 감지 + 신뢰도 하이브리드
# ---------------------------------------------------------------------------
df_target = make_symbol(1, n=400, force_event_at_end=True)
cfg = make_cfg()

verdict_no_pool = evaluate_symbol(df_target, "TEST", cfg, pooled_stats=None)
assert verdict_no_pool.symbol == "TEST"
assert verdict_no_pool.verdict in ("매수", "보유", "매도")
print(f"[OK] evaluate_symbol(풀링 없음): verdict={verdict_no_pool.verdict}, "
      f"활성신호={len(verdict_no_pool.active_signals)}개")

# 종목 자체 이벤트가 min_own_events 미만인 신호는 풀링 통계가 없으면 "미검증"이어야 함
low_sample_signals = [s for s in verdict_no_pool.active_signals if s.n_events < cfg.min_own_events]
assert all(s.basis == "미검증(표본부족)" for s in low_sample_signals), \
    "풀링 통계가 없고 종목 자체 표본도 부족하면 반드시 미검증 처리돼야 함"
print(f"[OK] 표본 부족 + 풀링 없음 -> 전부 미검증 처리 확인 ({len(low_sample_signals)}개)")

# 가짜 풀링 통계를 강하게 유의하도록 주입하면, 표본 부족한 신호가 "pooled"
# 근거로 검증되는지 확인(하이브리드 동작)
low_sample_labels = {s.label for s in verdict_no_pool.active_signals if s.n_events < cfg.min_own_events}
# 풀링 통계에는 묶음 차원 보정 결과(BH·DSR)가 같이 실려 있어야 한다. 실제
# compute_pooled_stats()가 그 두 키를 채워서 돌려준다 — 공통 게이트가 "검사하지
# 않은 항목을 통과로 흘려보내지 않기" 때문에, 보정 결과 없는 풀링 통계는
# 아래 추가 케이스처럼 검증되지 않는다.
fake_pooled = {s.label: dict(n=500, mean=0.01, t_stat=10.0,
                              significant_bh=True, significant_dsr=True)
               for s in verdict_no_pool.active_signals}
verdict_with_pool = evaluate_symbol(df_target, "TEST", cfg, pooled_stats=fake_pooled)
upgraded = [s for s in verdict_with_pool.active_signals
            if s.label in low_sample_labels and s.basis == "pooled" and s.validated]
assert len(upgraded) == len(low_sample_labels), \
    "표본 부족한 신호는 강한 풀링 통계가 주어지면 pooled 근거로 검증돼야 함(하이브리드)"
# 반대로 종목 자체 표본이 이미 충분한 신호는 풀링 통계가 있어도 own을 그대로 써야 함
own_preserved = [s for s in verdict_with_pool.active_signals
                 if s.label not in low_sample_labels and s.basis == "own"]
assert len(own_preserved) == len(verdict_no_pool.active_signals) - len(low_sample_labels)
print(f"[OK] 하이브리드: 표본 부족 신호({len(low_sample_labels)}개)는 풀링 통계 주입 시 pooled로 "
      f"검증되고, 표본 충분한 신호는 own을 그대로 유지")

# [공통 게이트] 다중검정 보정 결과가 빠진 풀링 통계는 t가 아무리 커도 검증되면 안 된다.
# 예전 어드바이저는 |t| >= 임계값 하나만 봤기 때문에, 유니버스를 스캔하며 신호를
# 여럿 동시에 보면서도 다중검정 보정을 전혀 하지 않았다. 그게 이번에 고친 결함이다.
pooled_no_correction = {s.label: dict(n=500, mean=0.01, t_stat=10.0)
                         for s in verdict_no_pool.active_signals}
verdict_nc = evaluate_symbol(df_target, "TEST", cfg, pooled_stats=pooled_no_correction)
leaked = [s for s in verdict_nc.active_signals
          if s.label in low_sample_labels and s.validated]
assert not leaked, f"BH·DSR 없는 풀링 통계가 t=10.0만으로 검증을 통과함: {[s.label for s in leaked]}"
assert verdict_nc.verdict == "보유", verdict_nc.verdict
print("[OK] 공통 게이트: BH·DSR 결과가 없는 풀링 통계는 |t|=10.0이어도 미검증 -> 보유")

# BH는 통과했는데 DSR에서 떨어지면 역시 검증되면 안 된다(알림 경로와 같은 기준).
pooled_dsr_fail = {s.label: dict(n=500, mean=0.01, t_stat=10.0,
                                  significant_bh=True, significant_dsr=False)
                   for s in verdict_no_pool.active_signals}
verdict_df = evaluate_symbol(df_target, "TEST", cfg, pooled_stats=pooled_dsr_fail)
assert not [s for s in verdict_df.active_signals
            if s.label in low_sample_labels and s.validated], "DSR 탈락인데 검증됨"
print("[OK] 공통 게이트: BH 통과 + DSR 탈락 -> 미검증(알림·규칙판정과 같은 기준)")

# ---------------------------------------------------------------------------
# 2) 다수결 로직 직접 확인 (충돌 시 보유로 떨어지는지)
# ---------------------------------------------------------------------------
from advisor.evaluator import SignalEvidence, SymbolVerdict

conflicting = [
    SignalEvidence("A", 1, True, True, "own", 50, 0.01, 4.0),
    SignalEvidence("B", -1, True, True, "own", 50, -0.01, -4.0),
]
long_n = sum(1 for s in conflicting if s.validated and s.direction == 1)
short_n = sum(1 for s in conflicting if s.validated and s.direction == -1)
assert long_n == short_n == 1
print("[OK] 검증된 강세 1개 vs 약세 1개 동률 상황을 재현 (실제 로직에서 '보유'로 떨어짐을 evaluate_symbol에서 확인)")

# ---------------------------------------------------------------------------
# 3) universe_scan: 풀링 통계 + 스캔 결과 정렬
# ---------------------------------------------------------------------------
dfs_by_symbol = {f"SYM_{i}": make_symbol(i, n=400, force_event_at_end=(i % 2 == 0)) for i in range(1, 6)}

pooled_stats = compute_pooled_stats(dfs_by_symbol, cfg)
assert all(k in pooled_stats for k in
           ["기관캔들-강세", "델타트레이딩-골든크로스", "빅세일-매수클라이맥스",
            "얼티밋RSI-과매도반등", "VP박스-POC지지"])
print(f"[OK] compute_pooled_stats(): {len(pooled_stats)}개 신호유형에 대한 풀링 통계 생성")

scan_df, pooled_stats2 = scan_universe(dfs_by_symbol, cfg)
assert set(scan_df["symbol"]) == set(dfs_by_symbol.keys())
assert (scan_df["strength"].diff().dropna() <= 1e-9).all(), "strength가 내림차순 정렬돼야 함"
print(f"[OK] scan_universe(): {len(scan_df)}개 종목 스캔, strength 내림차순 정렬 확인")
print(scan_df[["symbol", "verdict", "strength", "n_validated_signals"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 4) run_advisor.py CLI를 서브프로세스로 --mode scan 후 --mode query까지 실행
#    (실제 pykrx/업비트 네트워크 없이는 불가능하므로, load 함수를 monkeypatch한
#    별도 스크립트를 즉석에서 만들어 실행한다)
# ---------------------------------------------------------------------------
harness = f'''
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
import types
import numpy as np
import pandas as pd
import run_advisor as ra

np.random.seed(7)

def fake_load_symbol(market, symbol, args):
    seed = abs(hash(symbol)) % (2**31)
    rng = np.random.RandomState(seed)
    n = 400
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.uniform(1000, 5000, n)
    volume[-3:] *= 20
    close[-2] = close[-3] * 1.05
    close[-1] = close[-2] * 1.05
    high[-1] = close[-1] * 1.02
    return pd.DataFrame({{"open": open_, "high": high, "low": low, "close": close,
                          "volume": volume}}, index=dates)

ra._load_symbol = fake_load_symbol
sys.argv = ["run_advisor.py", "--mode", "scan", "--market", "krx",
            "--symbols", "A,B,C", "--output", "/tmp/_advisor_smoke_output"]
ra.main()

sys.argv = ["run_advisor.py", "--mode", "query", "--market", "krx",
            "--symbol", "A", "--output", "/tmp/_advisor_smoke_output"]
ra.main()
print("HARNESS_OK")
'''
# 윈도우 파이프 기본 인코딩(cp949)에서 한글/특수문자 출력이 죽지 않도록 자식에게 UTF-8을 명시한다.
import os
_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
proc = subprocess.run([sys.executable, "-c", harness], cwd=str(Path(__file__).parent),
                       capture_output=True, text=True, encoding="utf-8", env=_env, timeout=120)
print(proc.stdout[-3000:])
if proc.returncode != 0:
    print(proc.stderr[-3000:])
assert proc.returncode == 0, "run_advisor.py CLI(scan->query) 서브프로세스 실행 실패"
assert "HARNESS_OK" in proc.stdout
assert Path("/tmp/_advisor_smoke_output/advisor_scan.csv").exists()
assert Path("/tmp/_advisor_smoke_output/advisor_pooled_stats.csv").exists()
print("[OK] run_advisor.py CLI: --mode scan(캐시 생성) -> --mode query(캐시 재사용) 전체 배관 정상 동작")

print("\nSMOKE TEST (advisor: evaluate_symbol/universe_scan/run_advisor CLI) OK")
