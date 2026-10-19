"""
검증 게이트의 '부호' 처리 스모크테스트 (decision/gate, decision/rule_based, advisor/evaluator, run_param_search).

배경: event_returns()의 수익률에는 direction(롱 +1 / 숏 -1)이 이미 곱해져 있다.
따라서 t>0 = 신호 방향이 맞았다, t<0 = 유의하게 반대로 갔다(손실 신호).
예전에는 abs(t)>=임계로 판정해서 t=-3.2 같은 손실 신호도 '개별 t-stat 통과'로 분류됐다.
이 테스트는 (1) 손실 신호가 더 이상 통과하지 않고, (2) 양(+) 신호의 기존 동작은 그대로임을 확인한다.
합성 데이터/가짜 풀링 통계이므로 출력의 t값은 성과가 아니다.
"""
from advisor.evaluator import evaluate_symbol
from decision.gate import evaluate_gate, ADOPT, REJECT
from decision.rule_based import judge_signal
from run_param_search import _t_passes
from _smoke_helpers import make_cfg, make_symbol

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

# 3) run_param_search._t_passes
assert _t_passes(3.0, 3.0) is True and _t_passes(2.99, 3.0) is False
assert _t_passes(-5.0, 3.0) is False        # abs()였다면 True
assert _t_passes(float("nan"), 3.0) is False
print("[OK] _t_passes: 부호 반영(-5.0 불통과), 경계(>=), NaN 처리")

# 4) evaluate_symbol: 같은 신호에 풀링 t를 +10 / -10 으로 주입 (BH·DSR은 둘 다 통과로 줌)
df = make_symbol(1)
cfg = make_cfg()
base = evaluate_symbol(df, "T", cfg, pooled_stats=None)
low = {s.label for s in base.active_signals if s.n_events < cfg.min_own_events}
assert low, "자체 표본이 부족한 활성 신호가 있어야 풀링 경로 테스트가 의미가 있다"


def pooled(t, mean):
    return {s.label: dict(n=500, mean=mean, t_stat=t, significant_bh=True, significant_dsr=True)
            for s in base.active_signals}


pos = evaluate_symbol(df, "T", cfg, pooled_stats=pooled(+10.0, 0.01))
neg = evaluate_symbol(df, "T", cfg, pooled_stats=pooled(-10.0, -0.01))
assert all(s.validated for s in pos.active_signals if s.label in low), "t=+10 은 검증(기존 동작 보존)"
assert not any(s.validated for s in neg.active_signals if s.label in low), "t=-10(손실 신호)은 검증되면 안 됨"
assert neg.verdict == "보유" and neg.strength == 0.0, (neg.verdict, neg.strength)
assert all("반대 방향" in s.reason for s in neg.active_signals if s.label in low)
print("[OK] evaluate_symbol: 풀링 t=+10 -> 검증, t=-10 -> 미검증·보유·strength=0, 사유에 '반대 방향' 표기")

print("\nSMOKE TEST (sign gate) OK")
