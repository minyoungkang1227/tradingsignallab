"""
검증 게이트의 '부호' 처리 스모크테스트 (decision/gate, decision/rule_based).
(advisor·run_param_search 대상 검사는 2026-10 리스크 중심 재구성에서 해당 모듈과 함께 제거.)

배경: event_returns()의 수익률에는 direction(롱 +1 / 숏 -1)이 이미 곱해져 있다.
따라서 t>0 = 신호 방향이 맞았다, t<0 = 유의하게 반대로 갔다(손실 신호).
예전에는 abs(t)>=임계로 판정해서 t=-3.2 같은 손실 신호도 '개별 t-stat 통과'로 분류됐다.
이 테스트는 (1) 손실 신호가 더 이상 통과하지 않고, (2) 양(+) 신호의 기존 동작은 그대로임을 확인한다.
합성 데이터/가짜 풀링 통계이므로 출력의 t값은 성과가 아니다.
"""
from decision.gate import evaluate_gate, ADOPT, REJECT
from decision.rule_based import judge_signal

T = 3.0
BASE = dict(n_events=100, bh_significant=True, dsr_significant=True)

# 1) evaluate_gate: 손실 신호(t=-5)는 BH·DSR이 다 통과여도 기각 (예전엔 채택)
r = evaluate_gate(t_stat=-5.0, **BASE)
assert r["verdict"] == REJECT and "반대 방향" in r["reasons"][0], r
# 경계: t == +임계 통과(>=), 바로 아래 기각, 약한 음수는 '유의하지 않음' 문구
assert evaluate_gate(t_stat=3.0, **BASE)["verdict"] == ADOPT
assert evaluate_gate(t_stat=2.99, **BASE)["verdict"] == REJECT
r = evaluate_gate(t_stat=-1.0, **BASE)
assert r["verdict"] == REJECT and "유의하지 않음" in r["reasons"][0], r
assert evaluate_gate(t_stat=-3.0, **BASE)["verdict"] == REJECT      # t == -임계 도 기각
print("[OK] evaluate_gate: t<=-임계 기각(반대 방향 문구), t==+임계 통과, 2.99 기각, 약한 음수는 '유의하지 않음'")

# 2) judge_signal(규칙판정): 손실 신호 + 알파도 같은 방향 유의 -> 예전엔 채택, 이제 기각
r = judge_signal(n_events=50, raw_mean_return=-0.02, raw_t_stat=-3.5,
                 alpha=-0.01, alpha_t_stat=-3.5, r_squared=0.4, t_threshold=T)
assert r["verdict"] == "기각", r
assert judge_signal(50, 0.02, 3.5, 0.01, 3.5, 0.4, t_threshold=T)["verdict"] == "채택"
print("[OK] judge_signal: 손실 신호는 기각, 양(+) 신호는 기존대로 채택")

print("\nSMOKE TEST (sign gate) OK")
