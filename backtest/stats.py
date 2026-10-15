"""
정규분포 꼬리확률 — 이 저장소 전체가 공유하는 단일 구현.

[왜 따로 모았나]
engine.py(p_value_from_t), deflated.py(DSR), walkforward.py(합산 z→p)가 각자
`0.5 * math.erfc(...)`를 따로 들고 있었다. 수식이 같아 결과는 같았지만, 나중에
정규분포 근사를 t분포나 부트스트랩 경험분포로 바꿀 때 세 곳을 전부 고쳐야 하고
한 곳을 빠뜨리면 모듈마다 같은 t값에서 다른 p값이 나온다. 검정 기준이 모듈마다
갈리는 건 통계적 결론을 믿을 수 없게 만들므로 진입점을 하나로 둔다.

[왜 scipy를 안 쓰나]
이 저장소는 scipy 의존성 없이 돌아가는 것을 원칙으로 한다(GitHub Actions 알림
러너와 Colab 양쪽에서 설치 부담을 줄이기 위해). math.erfc는 표준 라이브러리이고
배정밀도 한계까지 정확하므로 근사 오차 문제도 없다.

[한계]
표준정규분포이지 t분포가 아니다. 표본이 작으면(대략 30건 미만) 실제 t분포보다
꼬리가 얇아 p-value를 과소평가한다 — 즉 유의하다고 판정하기 쉬워진다. 각 호출부
docstring에 이 점을 명시해 두었다.
"""
from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def norm_sf(z: float) -> float:
    """표준정규분포 생존함수 P(Z > z). 단측 p-value로 바로 쓴다."""
    return 0.5 * math.erfc(float(z) / _SQRT2)


def norm_cdf(z: float) -> float:
    """표준정규분포 누적분포함수 P(Z <= z). norm_sf(-z)와 같다."""
    return 0.5 * math.erfc(-float(z) / _SQRT2)
