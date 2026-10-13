import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
"""사망률 개선율 재추정 (v27).

사용법: KOSIS 「생명표」 완전생명표를 연도별로 내려받아 retire_sim/data/life_table_<연도>.csv
(age,qx_M,qx_F)로 저장한 뒤 실행하면, Lee-Carter 방식으로 연령별 개선율을 추정한다.

Lee-Carter: ln m(x,t) = a_x + b_x k_t + e,  k_t를 시간 추세로 보고 연간 개선율 ≈ -b_x · Δk.
여기서는 연령대별 평균 개선율만 쓰므로, 간단히 ln q(x,t)를 연도에 회귀한 기울기의 음수를 쓴다.
파일이 2개 이상 있어야 추정 가능하며, 없으면 현재 기본값(IMPROVEMENT)을 그대로 보여준다.
"""
import glob
import numpy as np
import pandas as pd
from retire_sim import mortality

files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "retire_sim", "data", "life_table_*.csv")))
years = []
for f in files:
    try:
        years.append((int(os.path.basename(f).split("_")[-1].split(".")[0]), pd.read_csv(f)))
    except ValueError:
        pass

if len(years) < 2:
    print("연도별 생명표가 2개 미만입니다. 현재 기본 개선율을 사용합니다:")
    for lim, v in mortality.IMPROVEMENT:
        print(f"  {lim}세 미만: 연 {v:.1%}")
    print("\nKOSIS에서 여러 해의 완전생명표를 받아 retire_sim/data/life_table_<연도>.csv 로 저장한 뒤 다시 실행하세요.")
    raise SystemExit

ys = np.array([y for y, _ in years], float)
for sex in ("M", "F"):
    Q = np.vstack([d[f"qx_{sex}"].values[:101] for _, d in years])
    lq = np.log(np.clip(Q, 1e-8, 1))
    slope = np.polyfit(ys, lq, 1)[0]                       # 연령별 ln q 의 연간 기울기
    k = -slope
    print(f"\n[{sex}] 연령대별 연간 개선율")
    for lo, hi in ((0, 65), (65, 80), (80, 90), (90, 100)):
        print(f"  {lo}~{hi - 1}세: {k[lo:hi].mean():.2%}")

print("\n위 값을 retire_sim/mortality.py 의 IMPROVEMENT 에 반영하세요.")
