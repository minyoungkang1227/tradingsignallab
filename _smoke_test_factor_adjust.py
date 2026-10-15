"""
backtest/engine.py::factor_adjust()의 OLS 회귀 로직을 네트워크 없이 검증한다.
합성 팩터 시계열과, 그 팩터들의 정확한 선형결합으로 만든 이벤트 수익률을 넣으면
회귀가 alpha·베타를 원래 넣은 값 그대로 복원해야 하고 R^2는 1에 가까워야 한다
(노이즈가 없는 완전선형 관계이므로).
"""
import numpy as np
import pandas as pd
from backtest.engine import factor_adjust, event_alpha

N = 40
DATES = pd.date_range("2020-01-31", periods=N, freq="ME")

SMB = 0.01 * (np.arange(N) % 5 - 2)
HML = 0.01 * (np.arange(N) % 3 - 1)
WML = 0.01 * (np.arange(N) % 4 - 1.5)

TRUE_ALPHA, TRUE_B_SMB, TRUE_B_HML, TRUE_B_WML = 0.015, 0.6, -0.4, 0.25
EVENT_RETURNS = TRUE_ALPHA + TRUE_B_SMB * SMB + TRUE_B_HML * HML + TRUE_B_WML * WML

factor_df = pd.DataFrame({"SMB": SMB, "HML": HML, "WML": WML}, index=DATES)

# 1) 저수준 factor_adjust() 직접 호출
result = factor_adjust(DATES, EVENT_RETURNS, factor_df)
print(result)

assert result["n_obs"] == N
assert abs(result["alpha"] - TRUE_ALPHA) < 1e-9, f"alpha 복원 실패: {result['alpha']}"
assert abs(result["betas"]["SMB"][0] - TRUE_B_SMB) < 1e-9
assert abs(result["betas"]["HML"][0] - TRUE_B_HML) < 1e-9
assert abs(result["betas"]["WML"][0] - TRUE_B_WML) < 1e-9
assert result["r_squared"] > 0.999, f"R^2 낮음: {result['r_squared']}"

# 2) 고수준 event_alpha() — df+event_col 인터페이스로도 같은 결과가 나오는지 확인
# forward_return(close,1)[t] = close[t+1]/close[t]-1 이 EVENT_RETURNS[t]와 정확히
# 일치하도록 close를 역산한다: close[0]=100, close[t+1]=close[t]*(1+EVENT_RETURNS[t]).
closes = 100.0 * np.concatenate([[1.0], np.cumprod(1 + EVENT_RETURNS[:-1])])
df = pd.DataFrame({"close": closes}, index=DATES)
df["dummy_event"] = True
# horizon=1의 순방향수익률이 EVENT_RETURNS와 정확히 일치하도록 close를 역산했으므로
# 마지막 이벤트는 다음 봉이 없어 자동으로 빠진다(n_obs가 1개 준다).
result2 = event_alpha(df, "dummy_event", factor_df, horizon=1)
print(result2)
assert result2["n_obs"] == N - 1
assert abs(result2["alpha"] - TRUE_ALPHA) < 1e-6, f"event_alpha alpha 불일치: {result2['alpha']}"

# 3) 표본 부족 시 NaN 반환하는지 확인 (파라미터 4개인데 관측치 2개)
tiny = factor_adjust(DATES[:2], EVENT_RETURNS[:2], factor_df)
assert tiny["n_obs"] == 2
assert np.isnan(tiny["alpha"])
assert tiny["note"] and "표본 부족" in tiny["note"], tiny["note"]
print("[OK] 3) 표본 부족 -> NaN + 사유")

# 4) 설계행렬 특이(rank 부족) — 예외 대신 NaN + 사유로 돌려주는지.
#    손계산: 이벤트 30건이 전부 정확히 같은 날(2020-03-31, 팩터 리밸런싱일)에 있으면
#    merge_asof가 30행 모두에 같은 팩터 행 1개를 붙인다. 그러면 설계행렬의 30개 행이
#    전부 동일하므로 rank = 1이다. 파라미터는 4개(절편+팩터 3)라 계수가 유일하지 않다.
same_day = pd.to_datetime(["2020-03-31"] * 30)
singular = factor_adjust(same_day, np.linspace(0.01, 0.03, 30), factor_df)
assert singular["n_obs"] == 30
assert np.isnan(singular["alpha"]) and np.isnan(singular["alpha_t_stat"])
assert all(np.isnan(b) and np.isnan(t) for b, t in singular["betas"].values())
assert "rank 1" in singular["note"] and "1종류" in singular["note"], singular["note"]
print("[OK] 4) 이벤트가 한 날에 몰림 -> rank 1 < 4, 예외 없이 NaN + 사유:", singular["note"])

# 4b) 같은 '달'에 몰려도 월말 기준선을 사이에 두면 팩터 행 2종류가 붙는다.
#     merge_asof(direction="nearest")가 3월 초 이벤트는 2020-02-29에, 3월 중순 이후는
#     2020-03-31에 매칭시키기 때문. 그래도 서로 다른 행이 2개뿐이라 rank = 2 < 4.
#     (이 경우가 실데이터에서 파이프라인을 멈추던 원인이다.)
same_month = pd.to_datetime(["2020-03-%02d" % (d + 1) for d in range(30)])
singular2 = factor_adjust(same_month, np.linspace(0.01, 0.03, 30), factor_df)
assert np.isnan(singular2["alpha"])
assert "rank 2" in singular2["note"] and "2종류" in singular2["note"], singular2["note"]
print("[OK] 4b) 한 달에 몰림 -> 팩터 행 2종류, rank 2 < 4:", singular2["note"])

# 5) 완전 공선성(두 팩터가 동일) — rank는 3이고 파라미터는 4이므로 역시 식별 불가.
dup_df = factor_df.copy()
dup_df["WML"] = dup_df["SMB"]          # WML을 SMB와 동일하게 만들어 열 하나를 중복시킴
collinear = factor_adjust(DATES, EVENT_RETURNS, dup_df)
assert np.isnan(collinear["alpha"]), collinear
assert collinear["note"] and "rank 3" in collinear["note"], collinear["note"]
print("[OK] 5) 팩터 열 중복 -> rank 3 < 4, NaN + 사유")

# 6) 정상 케이스는 note=None (위 1번 결과 재확인) — 사유 필드가 항상 존재해야 한다.
assert result["note"] is None and result2["note"] is None
print("[OK] 6) 정상 회귀는 note=None, 사유 필드는 항상 존재")

print("\nSMOKE TEST (factor_adjust) OK")
