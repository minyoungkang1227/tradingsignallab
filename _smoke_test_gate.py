"""
decision/gate.py 스모크 테스트 — 신호 채택 게이트의 단일 구현.

(1) 층별 판정: 각 층이 통과/실패/해당없음/판정불가를 손계산대로 내는가.
(2) 종합 규칙: '순서상 처음 걸린 층'이 판정을 결정하는가.
(3) 해당없음 vs 판정불가 구분: 코인(팩터 구조적 부재)과 표본부족(검증 못 함)을
    섞지 않는가. 이걸 뭉개면 코인이 영원히 보류로 떨어지거나, 검증 못 한 항목을
    통과로 흘려보낸다.
(4) 세 진입점이 같은 입력에 같은 답을 내는가 — 이 게이트를 만든 이유 자체.
"""
import math

from decision.gate import (ADOPT, FAIL, HOLD, PASS, REJECT, SKIP, UNKNOWN,
                            evaluate_gate, format_layers)

NAN = float("nan")


def statuses(res):
    return {l.name: l.status for l in res["layers"]}


# --- 1) 전부 통과하는 기준 케이스 -------------------------------------------
# 손계산: 표본 30 >= 10(통과), |t|=4.0 >= 3.0(통과), BH True(통과), DSR True(통과),
#         팩터 없음(해당없음) -> 걸리는 층이 하나도 없으므로 채택.
base = dict(n_events=30, t_stat=4.0, bh_significant=True, dsr_significant=True)
r = evaluate_gate(**base)
assert r["verdict"] == ADOPT, r
assert statuses(r) == {"표본수": PASS, "개별 t-stat": PASS, "BH 다중검정": PASS,
                        "DSR": PASS, "팩터 알파": SKIP}, statuses(r)
print("[OK] 1) 표본 30 / |t|=4.0 / BH·DSR 통과 / 팩터 없음 -> 채택")

# --- 2) 층별로 하나씩 떨어뜨리기 --------------------------------------------
cases = [
    ("표본 9건(<10)", dict(base, n_events=9), HOLD, "표본수", UNKNOWN),
    ("t=NaN", dict(base, t_stat=NAN), HOLD, "개별 t-stat", UNKNOWN),
    ("|t|=2.99 < 3.0", dict(base, t_stat=2.99), REJECT, "개별 t-stat", FAIL),
    ("BH 탈락", dict(base, bh_significant=False), REJECT, "BH 다중검정", FAIL),
    ("BH 결과 없음", dict(base, bh_significant=None), HOLD, "BH 다중검정", UNKNOWN),
    ("DSR 탈락", dict(base, dsr_significant=False), REJECT, "DSR", FAIL),
    ("DSR 결과 없음", dict(base, dsr_significant=None), HOLD, "DSR", UNKNOWN),
]
for name, kwargs, want_verdict, layer_name, want_status in cases:
    res = evaluate_gate(**kwargs)
    assert res["verdict"] == want_verdict, (name, res["verdict"], res["reasons"])
    assert statuses(res)[layer_name] == want_status, (name, statuses(res))
print(f"[OK] 2) 층별 단일 탈락 {len(cases)}가지 전부 기대대로 (탈락=기각, 미확인=보류)")

# --- 3) 검사하지 않은 항목을 통과로 흘려보내지 않는가 -----------------------
# require_*=True(기본)에서 None은 '판정불가'다. 이게 어드바이저가 BH·DSR을 아예
# 안 보면서도 신호를 검증 처리하던 결함의 핵심이었다.
r_none = evaluate_gate(n_events=100, t_stat=99.0)   # BH·DSR 미지정
assert r_none["verdict"] == HOLD, r_none
assert statuses(r_none)["BH 다중검정"] == UNKNOWN
# 반대로 '이 호출은 묶음 정보를 못 받는다'를 명시하면 해당없음으로 건너뛴다.
r_skip = evaluate_gate(n_events=100, t_stat=99.0, require_bh=False, require_dsr=False)
assert r_skip["verdict"] == ADOPT, r_skip
assert statuses(r_skip)["BH 다중검정"] == SKIP
print("[OK] 3) 미지정(None)은 기본적으로 판정불가 -> 보류. require_*=False일 때만 해당없음")

# --- 4) 순서 우선 규칙 -------------------------------------------------------
# 표본 5건(판정불가)이면서 |t|=1.0(실패)이면, 모아서 보면 실패가 있으니 기각이지만
# 순서상 표본수가 먼저이므로 보류다. 표본이 모자란 신호에 "유의하지 않다"고
# 말하는 건 조치로 이어지지 않기 때문에 순서를 택했다.
r_order = evaluate_gate(n_events=5, t_stat=1.0, bh_significant=False, dsr_significant=False)
assert r_order["verdict"] == HOLD, r_order
assert "표본 수 부족" in r_order["reasons"][0], r_order["reasons"]
print("[OK] 4) 순서 우선: 표본부족 + t실패 + BH실패 -> 가장 앞 층(표본수) 기준 보류")

# --- 5) 팩터 알파: 해당없음 vs 판정불가 vs 실패 -----------------------------
# 코인: 팩터가 구조적으로 없음 -> 해당없음이고, 그것 때문에 보류되지 않는다.
r_coin = evaluate_gate(**base, factor_available=False)
assert r_coin["verdict"] == ADOPT and statuses(r_coin)["팩터 알파"] == SKIP
# 국내주식인데 팩터 표본이 모자라 알파를 못 구함 -> 판정불가(보류).
r_na = evaluate_gate(**base, factor_available=True, alpha=NAN, alpha_t_stat=NAN)
assert r_na["verdict"] == HOLD and statuses(r_na)["팩터 알파"] == UNKNOWN
# 알파가 유의하지 않음 -> 실패(기각).
r_weak = evaluate_gate(**base, factor_available=True, alpha=0.01, alpha_t_stat=1.2)
assert r_weak["verdict"] == REJECT and statuses(r_weak)["팩터 알파"] == FAIL
# 알파는 유의한데 원 수익률과 부호가 반대 -> 사람 재확인 필요(보류).
r_sign = evaluate_gate(**base, factor_available=True, alpha=-0.01, alpha_t_stat=4.0,
                        mean_return=0.02)
assert r_sign["verdict"] == HOLD and statuses(r_sign)["팩터 알파"] == UNKNOWN
# 둘 다 같은 방향으로 유의 -> 채택.
r_ok = evaluate_gate(**base, factor_available=True, alpha=0.01, alpha_t_stat=4.0,
                      mean_return=0.02)
assert r_ok["verdict"] == ADOPT and statuses(r_ok)["팩터 알파"] == PASS
print("[OK] 5) 팩터 알파: 코인=해당없음, 표본부족=판정불가, 미유의=실패, 부호불일치=보류, 정상=통과")

# --- 6) 임계값은 인자로 바뀌고 기본은 3.0 -----------------------------------
assert evaluate_gate(**dict(base, t_stat=2.5))["verdict"] == REJECT
assert evaluate_gate(**dict(base, t_stat=2.5), t_threshold=2.0)["verdict"] == ADOPT
assert evaluate_gate(**dict(base, n_events=9), min_n_events=5)["verdict"] == ADOPT
print("[OK] 6) t_threshold 기본 3.0, min_n_events 기본 10, 둘 다 인자로 조정 가능")

# --- 7) 세 진입점이 같은 입력에 같은 답을 내는가 ----------------------------
# 이게 이 게이트를 만든 이유다. 예전에는 같은 신호를 두고
#   rule_based: |t|>=3.0 요구          -> 기각
#   alerts    : BH·DSR만 봄            -> 통과(알림 발송)
#   advisor   : |t|>=임계값만 봄        -> 기각
# 처럼 답이 갈렸다. 특히 |t|=2.5 + BH·DSR 통과가 전형적인 엇갈림 사례였다.
from decision.rule_based import judge_signal

conflict = dict(n_events=40, t_stat=2.5, bh_significant=True, dsr_significant=True)
gate_verdict = evaluate_gate(**conflict, factor_available=False)["verdict"]
# rule_based 경로(팩터 있음, 알파도 유의)에서도 t 층에서 먼저 걸려 같은 기각이 나와야 한다.
rb = judge_signal(n_events=40, raw_mean_return=0.01, raw_t_stat=2.5,
                   alpha=0.01, alpha_t_stat=5.0, r_squared=0.1,
                   bh_significant=True, dsr_significant=True)
assert gate_verdict == REJECT and rb["verdict"] == REJECT, (gate_verdict, rb["verdict"])
assert "2.5" in rb["reasons"][0] and "3.0" in rb["reasons"][0], rb["reasons"]
print("[OK] 7) |t|=2.5 + BH·DSR 통과: 게이트와 rule_based 모두 기각(예전엔 알림만 통과했음)")

# --- 8) format_layers는 걸린 지점을 그대로 보여준다 -------------------------
line = format_layers(evaluate_gate(**dict(base, dsr_significant=False)))
assert "DSR 실패" in line and "BH 다중검정 통과" in line, line
print("[OK] 8) format_layers:", line)

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
