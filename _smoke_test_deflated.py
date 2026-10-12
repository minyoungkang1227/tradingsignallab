"""
backtest/deflated.py 스모크테스트.

전부 손으로 계산 가능한 값과 대조한다. 특히 norm_ppf는 E[최대 SR]의 꼬리
분위수를 책임지는 핵심 함수라, 알려진 표준정규 분위수와 직접 맞춰본다.
"""
import math

import numpy as np

from backtest.deflated import (
    EULER_GAMMA, deflated_sharpe_ratio, expected_max_sharpe, norm_cdf, norm_ppf,
    probabilistic_sharpe_ratio, sharpe_dispersion, sharpe_std_error,
)

TOL = 1e-9


# --- 1) norm_ppf: 알려진 표준정규 분위수와 대조 -----------------------------
# 통계표 표준값(소수점 10자리)
KNOWN = {
    0.5: 0.0,
    0.95: 1.6448536269514722,
    0.975: 1.9599639845400545,
    0.99: 2.3263478740408408,
    0.995: 2.5758293035489004,
    0.999: 3.0902323061678132,
    0.025: -1.9599639845400545,
    0.005: -2.5758293035489004,
}
for p, expected in KNOWN.items():
    got = norm_ppf(p)
    assert abs(got - expected) < 1e-9, f"norm_ppf({p}): {got} != {expected}"
print("[PASS] 1) norm_ppf: 알려진 표준정규 분위수 8개와 1e-9 이내 일치")

# 꼬리 영역(시행횟수 N이 클 때 들어오는 구간) 정확도 확인
for n_trials in (10, 100, 1000, 10000):
    p = 1.0 - 1.0 / n_trials
    # 역함수를 다시 CDF에 통과시키면 원래 p로 돌아와야 함
    assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-12, f"왕복 실패 at N={n_trials}"
print("[PASS] 2) norm_ppf: N=10~10000 꼬리 구간 CDF 왕복 1e-12 이내")


# --- 2) sharpe_std_error: 손계산 대조 ---------------------------------------
# 정규분포 가정(skew=0, kurt=3)이면 Var = (1 + SR^2/2)/(n-1)
# SR=0.1, n=101 -> (1 + 0.005)/100 = 0.01005 -> se = 0.100249687...
se = sharpe_std_error(0.1, 101, skew=0.0, kurt=3.0)
expected_se = math.sqrt(1.005 / 100)
assert abs(se - expected_se) < TOL, f"{se} != {expected_se}"
print(f"[PASS] 3) sharpe_std_error 정규분포 케이스: {se:.9f} (손계산 {expected_se:.9f})")

# 팻테일이면 표준오차가 커져야 한다(= 유의성 판정이 보수적으로 바뀜)
# SR=0.5, n=101, kurt=3  -> (1 + 0.5*0.25)/100 = 0.01125   -> se = 0.106066017
# SR=0.5, n=101, kurt=10 -> (1 + 2.25*0.25)/100 = 0.015625 -> se = 0.125 (정확)
se_normal = sharpe_std_error(0.5, 101, skew=0.0, kurt=3.0)
se_fat = sharpe_std_error(0.5, 101, skew=0.0, kurt=10.0)
assert abs(se_normal - math.sqrt(0.01125)) < TOL
assert abs(se_fat - 0.125) < TOL, f"{se_fat} != 0.125"
assert se_fat > se_normal
print(f"[PASS] 4) 팻테일(kurt 3->10)이면 표준오차 {se_normal:.6f} -> {se_fat:.6f}로 증가")

# 음의 왜도도 표준오차를 키운다
# SR=0.5, n=101, skew=-1, kurt=3 -> (1 + 0.5 + 0.125)/100 = 0.01625 -> se=0.127475488
se_skew = sharpe_std_error(0.5, 101, skew=-1.0, kurt=3.0)
assert abs(se_skew - math.sqrt(0.01625)) < TOL
assert se_skew > se_normal
print(f"[PASS] 5) 음의 왜도(skew=-1)면 표준오차 {se_normal:.6f} -> {se_skew:.6f}로 증가")

# n <= 1이면 판정 불가
assert math.isnan(sharpe_std_error(0.5, 1))
assert math.isnan(sharpe_std_error(0.5, 0))
print("[PASS] 6) 표본 1개 이하 -> NaN(판정 불가)")


# --- 3) PSR: 손계산 대조 -----------------------------------------------------
# SR이 기준선과 정확히 같으면 z=0 -> PSR=0.5
psr_eq = probabilistic_sharpe_ratio(0.3, 0.3, 101, skew=0.0, kurt=3.0)
assert abs(psr_eq - 0.5) < TOL, f"{psr_eq} != 0.5"
print("[PASS] 7) PSR: SR == 기준선이면 정확히 0.5")

# z = 1.6448536... 가 되도록 역산하면 PSR = 0.95 가 나와야 한다
se_t = sharpe_std_error(0.2, 101, skew=0.0, kurt=3.0)
sr_for_95 = 0.2  # 관측 SR
bench_for_95 = sr_for_95 - 1.6448536269514722 * se_t
psr95 = probabilistic_sharpe_ratio(sr_for_95, bench_for_95, 101, skew=0.0, kurt=3.0)
assert abs(psr95 - 0.95) < 1e-9, f"{psr95} != 0.95"
print(f"[PASS] 8) PSR: z=1.6448536이 되도록 역산 -> {psr95:.10f} (기대 0.95)")


# --- 4) expected_max_sharpe: 성질 + 손계산 ----------------------------------
# 시행이 1회 이하면 비교 대상이 없으므로 기준선 0
assert expected_max_sharpe(1, 1.0) == 0.0
assert expected_max_sharpe(0, 1.0) == 0.0
print("[PASS] 9) E[maxSR]: 시행 1회 이하면 기준선 0")

# N=2, sr_std=1 손계산:
#   q1 = Phi^-1(1 - 1/2) = Phi^-1(0.5) = 0
#   q2 = Phi^-1(1 - 1/(2e)) = Phi^-1(0.8160602794...)
#   E[max] = (1-gamma)*0 + gamma*q2 = gamma * q2
q2_hand = norm_ppf(1.0 - 1.0 / (2.0 * math.e))
expected_n2 = EULER_GAMMA * q2_hand
got_n2 = expected_max_sharpe(2, 1.0)
assert abs(got_n2 - expected_n2) < TOL, f"{got_n2} != {expected_n2}"
# q1이 정확히 0이어야 한다는 점도 직접 확인(공식 전개가 맞는지 검증)
assert abs(norm_ppf(0.5)) < 1e-12
print(f"[PASS] 10) E[maxSR] N=2, sr_std=1 -> {got_n2:.9f} (손계산 gamma*Phi^-1(1-1/2e))")

# 시행 횟수가 늘수록 기준선이 올라가야 한다(= 더 엄격해짐)
bench_seq = [expected_max_sharpe(n, 0.5) for n in (2, 10, 60, 1000)]
assert all(bench_seq[i] < bench_seq[i + 1] for i in range(len(bench_seq) - 1)), bench_seq
print(f"[PASS] 11) E[maxSR] 단조증가: N=2,10,60,1000 -> "
      f"{', '.join(f'{b:.4f}' for b in bench_seq)}")

# sr_std에 선형 비례해야 한다
assert abs(expected_max_sharpe(60, 2.0) - 2.0 * expected_max_sharpe(60, 1.0)) < TOL
print("[PASS] 12) E[maxSR]: sr_std에 정확히 선형 비례")


# --- 5) deflated_sharpe_ratio: 전 과정 손계산 대조 --------------------------
# r = [0.01, 0.02, 0.03, 0.04, 0.05]
#   mean = 0.03
#   std(ddof=1) = sqrt(0.001/4) = sqrt(0.00025) = 0.0158113883...
#   SR = 0.03/0.0158113883 = 1.8973665961...  (SR^2 = 3.6 정확)
#   z = [-1.264911, -0.632456, 0, 0.632456, 1.264911]  -> 대칭이므로 skew = 0
#   kurt = mean(z^4) = (2.56 + 0.16 + 0 + 0.16 + 2.56)/5 = 5.44/5 = 1.088
r = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
res = deflated_sharpe_ratio(r, n_trials=1, sr_std=1.0)

assert abs(res["sharpe"] - 0.03 / math.sqrt(0.00025)) < TOL
assert abs(res["sharpe"] ** 2 - 3.6) < 1e-9, res["sharpe"] ** 2
assert abs(res["skew"] - 0.0) < 1e-12, res["skew"]
assert abs(res["kurt"] - 1.088) < 1e-12, res["kurt"]
assert res["n_obs"] == 5
# n_trials=1 이므로 기준선 0
assert res["sr_benchmark"] == 0.0
# se = sqrt((1 + (1.088-1)/4 * 3.6)/4) = sqrt(1.0792/4) = sqrt(0.2698)
se_hand = math.sqrt((1.0 + 0.022 * 3.6) / 4.0)
dsr_hand = norm_cdf(res["sharpe"] / se_hand)
assert abs(res["dsr"] - dsr_hand) < TOL, f"{res['dsr']} != {dsr_hand}"
print(f"[PASS] 13) DSR 전과정 손계산: SR={res['sharpe']:.9f}, skew={res['skew']:.1f}, "
      f"kurt={res['kurt']:.3f}, dsr={res['dsr']:.9f}")

# 표본 부족 / 표준편차 0 -> 전부 NaN (유의하다고 잘못 흘러가지 않아야 함)
assert math.isnan(deflated_sharpe_ratio([0.01], 1, 1.0)["dsr"])
assert math.isnan(deflated_sharpe_ratio([], 1, 1.0)["dsr"])
assert math.isnan(deflated_sharpe_ratio([0.02] * 10, 1, 1.0)["dsr"])  # std=0
print("[PASS] 14) 표본 2개 미만 / 표준편차 0 -> NaN (판정 불가로 안전하게 처리)")


# --- 6) 핵심 성질: 시행 횟수가 늘면 DSR이 떨어진다 --------------------------
rng = np.random.default_rng(42)
r_mod = rng.normal(0.004, 0.02, 200)  # 약한 양의 평균
dsr_1 = deflated_sharpe_ratio(r_mod, n_trials=1, sr_std=0.15)["dsr"]
dsr_60 = deflated_sharpe_ratio(r_mod, n_trials=60, sr_std=0.15)["dsr"]
dsr_1000 = deflated_sharpe_ratio(r_mod, n_trials=1000, sr_std=0.15)["dsr"]
assert dsr_1 > dsr_60 > dsr_1000, (dsr_1, dsr_60, dsr_1000)
print(f"[PASS] 15) 같은 수익률도 시행횟수가 늘면 DSR 하락: "
      f"N=1 {dsr_1:.4f} > N=60 {dsr_60:.4f} > N=1000 {dsr_1000:.4f}")


# --- 7) 핵심 성질: 팻테일이면 DSR이 정규분포 가정보다 낮다 ------------------
# 같은 평균/표준편차인데 꼬리만 두꺼운 분포를 만들어 비교한다.
# [중요] 왜도와 첨도는 서로 반대 방향으로 작용할 수 있고, 분산 공식에서
#   왜도항 = -skew*SR      (SR의 1차)
#   첨도항 = (kurt-1)/4*SR^2 (SR의 2차)
# 이므로 거래당 SR이 작으면 왜도가 첨도를 압도한다. 그래서 "팻테일이면 DSR이
# 낮다"는 성질을 검증하려면 왜도를 반드시 0으로 통제해야 한다. 통제 없이
# t분포 표본을 그냥 쓰면 표본 왜도가 우연히 커져서 결과가 역전될 수 있다
# (실제로 그렇게 나와서 이 통제를 넣었다).
TARGET_SR = 0.12
TARGET_STD = 0.02


def _symmetrize(x):
    """x와 -x를 이어붙여 표본 왜도를 정확히 0으로 만든다(첨도는 보존)."""
    return np.concatenate([x, -x])


def _rescale(x, target_sr, target_std):
    """평균·표준편차를 지정값으로 맞춘다. z=(r-mean)/std는 불변이므로
    왜도·첨도는 이 변환에 영향받지 않는다."""
    x = (x - x.mean()) / x.std(ddof=1)
    return x * target_std + target_sr * target_std


base = _rescale(_symmetrize(rng.standard_normal(300)), TARGET_SR, TARGET_STD)
fat = _rescale(_symmetrize(rng.standard_t(df=3, size=300)), TARGET_SR, TARGET_STD)

# sr_std를 작게 잡아 기준선을 SR 아래로 두면 DSR이 0/1로 포화되지 않고
# 판별 가능한 구간에 들어온다(두 값의 차이를 눈으로 확인하기 위함).
res_base = deflated_sharpe_ratio(base, n_trials=60, sr_std=0.05)
res_fat = deflated_sharpe_ratio(fat, n_trials=60, sr_std=0.05)

# 통제 확인 1: SR이 목표값으로 동일하게 맞춰졌는가
assert abs(res_base["sharpe"] - TARGET_SR) < 1e-9, res_base["sharpe"]
assert abs(res_fat["sharpe"] - TARGET_SR) < 1e-9, res_fat["sharpe"]
# 통제 확인 2: 왜도가 양쪽 모두 0인가(이게 안 되면 아래 비교가 무의미)
assert abs(res_base["skew"]) < 1e-12, res_base["skew"]
assert abs(res_fat["skew"]) < 1e-12, res_fat["skew"]
# 차이는 오직 첨도에서만 온다
assert res_fat["kurt"] > res_base["kurt"] + 2.0, (res_base["kurt"], res_fat["kurt"])
# 핵심: SR·왜도가 완전히 같아도 팻테일 쪽 DSR이 더 낮아야 한다(= 더 보수적)
assert res_fat["dsr"] < res_base["dsr"], (res_fat["dsr"], res_base["dsr"])
print(f"[PASS] 16) SR={TARGET_SR}·왜도=0으로 통제하고 첨도만 다르게: "
      f"정규 kurt={res_base['kurt']:.2f} dsr={res_base['dsr']:.4f} -> "
      f"팻테일 kurt={res_fat['kurt']:.2f} dsr={res_fat['dsr']:.4f} (더 보수적)")

# --- 7-b) 왜도 효과: 음의 왜도(급락 꼬리)는 DSR을 떨어뜨린다 ---------------
# 손절 없는 추세추종처럼 "작게 자주 벌고 가끔 크게 잃는" 수익률 분포가
# 전형적인 음의 왜도다. 이 경우를 공식이 제대로 벌하는지 확인한다.
sym = _symmetrize(rng.standard_normal(300))
neg_skew = _rescale(-np.abs(sym) ** 1.6 * np.sign(sym) - 0.3 * sym ** 2,
                     TARGET_SR, TARGET_STD)
pos_skew = _rescale(-neg_skew, TARGET_SR, TARGET_STD)  # 부호 반전 -> 왜도만 반대

res_neg = deflated_sharpe_ratio(neg_skew, n_trials=60, sr_std=0.05)
res_pos = deflated_sharpe_ratio(pos_skew, n_trials=60, sr_std=0.05)
# 부호 반전이므로 왜도는 정확히 반대, 첨도는 동일 -> 왜도 효과만 분리됨
assert abs(res_neg["skew"] + res_pos["skew"]) < 1e-9, (res_neg["skew"], res_pos["skew"])
assert abs(res_neg["kurt"] - res_pos["kurt"]) < 1e-9
assert res_neg["skew"] < 0, res_neg["skew"]
assert res_neg["dsr"] < res_pos["dsr"], (res_neg["dsr"], res_pos["dsr"])
print(f"[PASS] 16b) 첨도 동일({res_neg['kurt']:.2f}), 왜도만 반대: "
      f"음의왜도 {res_neg['skew']:+.2f} dsr={res_neg['dsr']:.4f} < "
      f"양의왜도 {res_pos['skew']:+.2f} dsr={res_pos['dsr']:.4f}")


# --- 8) sharpe_dispersion ---------------------------------------------------
vals = [0.1, 0.2, 0.3, np.nan, 0.4]
disp = sharpe_dispersion(vals)
assert abs(disp - float(np.std([0.1, 0.2, 0.3, 0.4], ddof=1))) < TOL
assert math.isnan(sharpe_dispersion([0.1]))
assert math.isnan(sharpe_dispersion([np.nan, np.nan]))
print(f"[PASS] 17) sharpe_dispersion: NaN 제외 후 ddof=1 표준편차 {disp:.9f}")

print("\n모든 스모크테스트 통과.")
