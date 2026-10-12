"""alerts/check_signals.py 스모크테스트: 상태추적(중복방지) + 시장별 BH보정 조건.
실제 네트워크 호출(rb.load_data) / 지표 수학(rb.run)은 몽키패치로 대체하고,
check_one()이 state와 require_significance(시장별 기본값 + override)를
올바르게 쓰는지 손으로 계산한 기대값과 비교한다.
"""
import pandas as pd
import run_backtest as rb
import alerts.check_signals as cs

_state_box = {"last_date": pd.Timestamp("2024-03-01")}


def _make_df(last_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(end=last_date, periods=61, freq="D")
    df = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
    }, index=dates)
    df.index.name = "date"
    return df


def fake_load_data(args):
    return _make_df(_state_box["last_date"])


def make_fake_run(event_true_on_last: bool, bh_significant: bool, judge_horizon: int = 5,
                   dsr_significant: bool | None = None, with_dsr_col: bool = True,
                   tb: dict | None = None):
    """
    dsr_significant: DSR 게이트 통과 여부. None이면 bh_significant를 따라간다
        (DSR 도입 전 테스트들이 그대로 통과하도록).
    with_dsr_col: False면 summary에 significant_dsr 컬럼 자체를 넣지 않는다
        (구버전 engine과의 하위호환 경로 검증용).
    """
    if dsr_significant is None:
        dsr_significant = bh_significant

    def fake_run(df_, args):
        dfi = pd.DataFrame({"crossover_confirmed": False}, index=df_.index)
        dfi.loc[dfi.index[-1], "crossover_confirmed"] = event_true_on_last
        row = {"signal": "델타트레이딩-골든크로스", "horizon": judge_horizon,
               "significant_bh": bh_significant}
        if with_dsr_col:
            row["significant_dsr"] = dsr_significant
        if tb is not None:
            row.update(tb)
        summary = pd.DataFrame([row])
        return summary, None, {"delta_trading": dfi}, 0.0
    return fake_run


rb.load_data = fake_load_data

# --- krx 종목: 기본 require_significance=True (market != "upbit") ---
krx_item = {"market": "krx", "symbol": "005930", "label": "삼성전자"}
krx_key = "krx:005930:delta_trading:crossover_confirmed"

rb.run = make_fake_run(event_true_on_last=True, bh_significant=False)
state = {}
msgs = cs.check_one(krx_item, state)
assert len(msgs) == 0, f"krx: BH 미통과인데도 알림 생성됨: {len(msgs)}건"
print("[PASS] 1) krx 기본값: BH보정 미통과 -> 알림 없음")

rb.run = make_fake_run(event_true_on_last=True, bh_significant=True)
msgs = cs.check_one(krx_item, state)
assert len(msgs) == 1 and "BH보정" in msgs[0] and "통과 신호" in msgs[0]
assert krx_key in state
print("[PASS] 2) krx 기본값: BH보정 통과 -> 알림 1건('BH보정 통과' 문구 포함)")

# --- upbit 종목: 기본 require_significance=False (원시 이벤트 즉시 알림) ---
upbit_item = {"market": "upbit", "symbol": "KRW-BTC", "label": "비트코인"}
upbit_key = "upbit:KRW-BTC:delta_trading:crossover_confirmed"

_state_box["last_date"] = pd.Timestamp("2024-03-02")
rb.run = make_fake_run(event_true_on_last=True, bh_significant=False)  # BH는 거짓이어도
state2 = {}
msgs = cs.check_one(upbit_item, state2)
assert len(msgs) == 1, f"upbit 기본값: BH와 무관하게 알림이 와야 하는데 {len(msgs)}건"
assert "검증(BH보정) 없이" in msgs[0]
assert upbit_key in state2
print("[PASS] 3) upbit 기본값: BH와 무관하게 즉시 알림 1건('검증 없이' 문구 포함)")

# --- upbit인데 watchlist.json에서 require_significance=True로 override ---
upbit_strict_item = {"market": "upbit", "symbol": "KRW-ETH", "label": "이더리움",
                      "require_significance": True}
_state_box["last_date"] = pd.Timestamp("2024-03-03")
rb.run = make_fake_run(event_true_on_last=True, bh_significant=False)
state3 = {}
msgs = cs.check_one(upbit_strict_item, state3)
assert len(msgs) == 0, f"override 무시됨: BH 미통과인데 {len(msgs)}건 생성"
print("[PASS] 4) upbit + require_significance=True override -> BH 미통과시 알림 억제 확인")

# --- krx인데 require_significance=False로 override (즉시 알림) ---
krx_loose_item = {"market": "krx", "symbol": "000660", "label": "SK하이닉스",
                   "require_significance": False}
_state_box["last_date"] = pd.Timestamp("2024-03-04")
rb.run = make_fake_run(event_true_on_last=True, bh_significant=False)
state4 = {}
msgs = cs.check_one(krx_loose_item, state4)
assert len(msgs) == 1, f"override 무시됨: 즉시 알림이어야 하는데 {len(msgs)}건"
print("[PASS] 5) krx + require_significance=False override -> BH와 무관하게 즉시 알림 확인")

# --- 중복 방지는 두 모드 모두에서 동일하게 동작해야 함 ---
_state_box["last_date"] = pd.Timestamp("2024-03-02")  # upbit_item을 처음 체크했던 날짜로 복원
rb.run = make_fake_run(event_true_on_last=True, bh_significant=False)
msgs_dup = cs.check_one(upbit_item, state2)  # 같은 날짜로 재체크
assert len(msgs_dup) == 0
print("[PASS] 6) upbit 즉시알림 모드에서도 같은 날짜 재체크 -> 중복 억제 확인")

# --- DSR 2차 게이트 ---------------------------------------------------------
# BH는 통과했지만 DSR에서 떨어지면 알림이 나가면 안 된다.
_state_box["last_date"] = pd.Timestamp("2024-04-01")
rb.run = make_fake_run(event_true_on_last=True, bh_significant=True, dsr_significant=False)
state5 = {}
msgs = cs.check_one(krx_item, state5)
assert len(msgs) == 0, f"BH통과·DSR탈락인데 알림 {len(msgs)}건 생성됨"
assert krx_key not in state5, "알림을 안 보냈는데 state가 갱신됨"
print("[PASS] 7) krx: BH 통과 + DSR 탈락 -> 알림 억제(2차 게이트 작동)")

# 둘 다 통과하면 알림이 나가고 메시지에 DSR이 명시돼야 한다
_state_box["last_date"] = pd.Timestamp("2024-04-02")
rb.run = make_fake_run(event_true_on_last=True, bh_significant=True, dsr_significant=True)
state6 = {}
msgs = cs.check_one(krx_item, state6)
assert len(msgs) == 1, f"BH·DSR 둘 다 통과인데 알림 {len(msgs)}건"
assert "DSR" in msgs[0], msgs[0]
assert krx_key in state6
print("[PASS] 8) krx: BH + DSR 둘 다 통과 -> 알림 1건('DSR' 문구 포함)")

# upbit(즉시 알림)은 DSR과 무관하게 나가야 한다
_state_box["last_date"] = pd.Timestamp("2024-04-03")
rb.run = make_fake_run(event_true_on_last=True, bh_significant=False, dsr_significant=False)
state7 = {}
msgs = cs.check_one(upbit_item, state7)
assert len(msgs) == 1, f"upbit 즉시알림인데 {len(msgs)}건"
assert "검증(BH보정) 없이" in msgs[0]
print("[PASS] 9) upbit: DSR 도입 후에도 즉시 알림 경로는 영향 없음")

# 구버전 engine처럼 significant_dsr 컬럼이 아예 없으면 BH만으로 판정(하위호환)
_state_box["last_date"] = pd.Timestamp("2024-04-04")
rb.run = make_fake_run(event_true_on_last=True, bh_significant=True, with_dsr_col=False)
state8 = {}
msgs = cs.check_one(krx_item, state8)
assert len(msgs) == 1, f"DSR 컬럼 없을 때 BH만으로 통과해야 하는데 {len(msgs)}건"
assert "DSR" not in msgs[0], msgs[0]
assert "BH보정(다중검정)" in msgs[0], msgs[0]
print("[PASS] 10) significant_dsr 컬럼 자체가 없으면 BH만으로 판정(하위호환)")

# --- 설정 기본값 확인 -------------------------------------------------------
from alerts.config import DEFAULTS, args_for_watch_item

assert DEFAULTS["deflate"] is True
assert DEFAULTS["drop_overlapping"] is True
a = args_for_watch_item({"market": "krx", "symbol": "005930"})
assert a.deflate is True and a.drop_overlapping is True
assert a.dsr_min_events == 20 and a.dsr_threshold == 0.95
print("[PASS] 11) 알림 경로 기본값: deflate=True, drop_overlapping=True, "
      f"min_events={a.dsr_min_events}, threshold={a.dsr_threshold}")

assert DEFAULTS["barrier"] is True and DEFAULTS["require_barrier_profit"] is False
assert a.barrier is True and a.barrier_sl_mult == 1.0 and a.barrier_pt_mult == 2.0
print("[PASS] 12) 배리어 기본값: barrier=True(정보로 표시), "
      "require_barrier_profit=False(게이트로는 안 씀)")

# --- 배리어 실적이 메시지에 실리는가 ----------------------------------------
TB_GOOD = {"tb_n": 23, "tb_mean_ret_pct": 1.85, "tb_pt_rate": 0.52,
           "tb_sl_rate": 0.39, "tb_time_rate": 0.09, "tb_avg_bars": 4.3}
TB_BAD = {**TB_GOOD, "tb_mean_ret_pct": -0.74, "tb_pt_rate": 0.31, "tb_sl_rate": 0.64}

_state_box["last_date"] = pd.Timestamp("2024-05-01")
rb.run = make_fake_run(True, bh_significant=True, dsr_significant=True, tb=TB_GOOD)
state9 = {}
msgs = cs.check_one(krx_item, state9)
assert len(msgs) == 1
m = msgs[0]
assert "손절 반영 과거실적" in m, m
assert "23건" in m and "+1.85%" in m, m
assert "익절 52%" in m and "손절 39%" in m, m
assert "4.3봉" in m, m
print("[PASS] 13) 알림 메시지에 손절 반영 실적 포함: "
      "'23건 중 익절 52%·손절 39%, 평균 +1.85%, 평균보유 4.3봉'")

# 손절 반영하면 손실인 신호도 기본값에서는 알림이 나가되, 숫자가 그대로 보여야 한다
_state_box["last_date"] = pd.Timestamp("2024-05-02")
rb.run = make_fake_run(True, bh_significant=True, dsr_significant=True, tb=TB_BAD)
state10 = {}
msgs = cs.check_one(krx_item, state10)
assert len(msgs) == 1, "기본값에서는 배리어 손실이어도 알림은 나가야 함(게이트 아님)"
assert "-0.74%" in msgs[0] and "손절 64%" in msgs[0], msgs[0]
print("[PASS] 14) 배리어 실적이 손실이어도 기본값에선 알림 발송 + 숫자 그대로 노출 "
      "(판단은 사람이)")

# require_barrier_profit=True면 게이트로 작동해야 한다
import types as _types

_state_box["last_date"] = pd.Timestamp("2024-05-03")
rb.run = make_fake_run(True, bh_significant=True, dsr_significant=True, tb=TB_BAD)
_orig = cs.args_for_watch_item
cs.args_for_watch_item = lambda it: _types.SimpleNamespace(
    **{**vars(_orig(it)), "require_barrier_profit": True})
state11 = {}
msgs = cs.check_one(krx_item, state11)
assert len(msgs) == 0, f"require_barrier_profit=True인데 손실 신호가 통과함: {msgs}"
assert krx_key not in state11
print("[PASS] 15) require_barrier_profit=True -> 배리어 손실 신호 억제(게이트로 전환)")

# 같은 설정에서 수익 신호는 통과해야 한다
_state_box["last_date"] = pd.Timestamp("2024-05-04")
rb.run = make_fake_run(True, bh_significant=True, dsr_significant=True, tb=TB_GOOD)
state12 = {}
msgs = cs.check_one(krx_item, state12)
assert len(msgs) == 1
cs.args_for_watch_item = _orig
print("[PASS] 16) require_barrier_profit=True에서도 배리어 수익 신호는 통과")

# tb_ 컬럼이 없으면 그 줄만 빠지고 나머지는 정상 (하위호환)
_state_box["last_date"] = pd.Timestamp("2024-05-05")
rb.run = make_fake_run(True, bh_significant=True, dsr_significant=True, tb=None)
state13 = {}
msgs = cs.check_one(krx_item, state13)
assert len(msgs) == 1
assert "손절 반영 과거실적" not in msgs[0], msgs[0]
print("[PASS] 17) tb_* 컬럼이 없으면 해당 줄만 생략, 알림 자체는 정상(하위호환)")

print("\n모든 스모크테스트 통과.")
