"""
backtest/stats.py 스모크 테스트 — 정규분포 꼬리확률 단일 구현.

(1) 손계산 기준값과 일치하는가.
(2) norm_sf(z) + norm_cdf(z) == 1 (대칭성·수치 정합성).
(3) 중복 제거가 실제로 됐는가 — engine/deflated/walkforward가 이 구현을 쓰는지.
    세 모듈이 각자 erfc를 들고 있으면 나중에 근사를 바꿀 때 모듈마다 p값이 갈린다.
"""
import math

from backtest.stats import norm_cdf, norm_sf

# --- 1) 손계산 기준값 -------------------------------------------------------
# 표준정규 분포표의 교과서 값. P(Z > 1.959963985) = 0.025, P(Z > 0) = 0.5.
CASES = [
    (0.0, 0.5),
    (1.0, 0.158655253931),
    (1.959963985, 0.025),
    (2.575829304, 0.005),
    (-1.0, 0.841344746069),
]
for z, expected in CASES:
    got = norm_sf(z)
    assert abs(got - expected) < 1e-10, f"norm_sf({z}) = {got}, 기대 {expected}"
print(f"[OK] 1) 손계산 기준값 {len(CASES)}개 일치 (오차 < 1e-10)")

# --- 2) 항등식 --------------------------------------------------------------
for z in (-6.0, -2.5, -0.3, 0.0, 0.3, 2.5, 6.0):
    assert abs(norm_sf(z) + norm_cdf(z) - 1.0) < 1e-15, z
    assert abs(norm_cdf(-z) - norm_sf(z)) < 1e-15, z
assert norm_sf(40.0) == 0.0 and norm_cdf(40.0) == 1.0   # 극단값에서 언더플로만 날 뿐 예외 없음
print("[OK] 2) norm_sf(z) + norm_cdf(z) = 1, norm_cdf(-z) = norm_sf(z), 극단값 안전")

# --- 3) 중복 제거 확인 ------------------------------------------------------
from backtest.deflated import norm_cdf as deflated_norm_cdf
from backtest.engine import _normal_sf as engine_sf
from backtest.walkforward import _p1_from_z as wf_sf

assert engine_sf is norm_sf, "engine이 stats.norm_sf를 쓰지 않음(구현이 복제됨)"
assert wf_sf is norm_sf, "walkforward가 stats.norm_sf를 쓰지 않음(구현이 복제됨)"
assert deflated_norm_cdf is norm_cdf, "deflated가 stats.norm_cdf를 쓰지 않음(구현이 복제됨)"
print("[OK] 3) engine/walkforward/deflated가 모두 stats.py 구현을 가리킴")

# 소스에 erfc 호출이 stats.py 밖에 남아있지 않은지도 확인한다(주석/문자열 제외).
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
leftovers = []
for name in sorted(os.listdir(os.path.join(ROOT, "backtest"))):
    if not name.endswith(".py") or name == "stats.py":
        continue
    with open(os.path.join(ROOT, "backtest", name), encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            code = line.split("#", 1)[0]
            if re.search(r"\berfc\s*\(", code):
                leftovers.append(f"backtest/{name}:{i}")
assert not leftovers, "stats.py 밖에 erfc 구현이 남아있음: " + ", ".join(leftovers)
print("[OK] 4) stats.py 밖에 남은 erfc 호출 없음")

# p-value 진입점이 실제로 같은 값을 내는지(회귀 방지)
from backtest.engine import p_value_from_t

assert abs(p_value_from_t(1.959963985) - 0.05) < 1e-9
assert math.isnan(p_value_from_t(float("nan")))
print("[OK] 5) p_value_from_t(1.96) = 0.05 (양측), NaN 입력은 NaN")

print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
