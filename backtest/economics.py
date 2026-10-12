"""
경제적 유의성 지표 — "통계적으로 유의하다"와 "실제로 벌 만하다"는 다른 질문이다.

t값·DSR은 평균수익이 0이 아니라는 확신의 정도를 말해줄 뿐, 그 평균이 비용을 낸 뒤
손에 남는 크기인지, 손익비를 감안해도 승률이 충분한지는 말해주지 않는다. 특히
트리플배리어처럼 익절 2σ·손절 1σ로 손익비가 비대칭이면 승률만으로는 전략의 좋고
나쁨을 판단할 수 없다 — 손익비 2:1에서는 승률 34%로도 본전을 넘고, 손익비 1:2였다면
승률 66%라도 본전이다. 그래서 세 가지를 같이 본다.

  1) 기대값(R 단위): 거래 1건의 평균 손익을 "1R = 손절폭"으로 나눈 값.
       R_i = ret_i / (sl_mult × target_vol_i)
     ret_i는 비용·슬리피지를 뺀 순수익이므로 비용이 이미 R에 녹아 있다.
     +0.2R이면 "손절폭의 20%를 거래당 평균으로 번다"는 뜻이다. 종목·시기마다
     변동성이 달라 % 수익률끼리는 비교가 어렵지만 R은 위험 단위로 정규화돼 있다.

  2) 손익분기 승률: 실현된 평균 이익 W(R 단위)와 평균 손실 L(R 단위)로
       손익분기 승률 = L / (W + L)
     실제 승률이 이보다 높아야 기대값이 양수다(정확히는 기대값 = p·W − (1−p)·L).
     명목값(비용 무시, 모든 거래가 배리어에서 끝난다고 가정)인 sl/(sl+pt)도 같이
     준다 — 2:1이면 1/3. 실현값이 명목값보다 높으면 비용과 시간만료 청산이 손익분기를
     밀어올렸다는 뜻이다.

  3) 비용 대비 배수: 평균 순수익 ÷ 왕복비용. 1에 가까우면 비용을 겨우 넘기는 수준이라
     슬리피지 가정이 조금만 틀려도 사라진다. 0 이하면 이미 비용에 먹힌 것이다.

합성 데이터에서 나온 값은 실제 시장 성과가 아니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def r_multiples(tb: pd.DataFrame, sl_mult: float) -> np.ndarray:
    """거래별 R 배수. 손절을 끈(sl_mult<=0) 경우 위험 단위가 없으므로 NaN."""
    if tb is None or len(tb) == 0:
        return np.array([], dtype=float)
    if sl_mult is None or sl_mult <= 0:
        return np.full(len(tb), np.nan)
    ret = tb["ret"].to_numpy(dtype=float)
    unit = sl_mult * tb["target_vol"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(unit > 0, ret / unit, np.nan)
    return r


def economic_summary(tb: pd.DataFrame, pt_mult: float, sl_mult: float,
                     round_trip_cost: float = 0.0) -> dict:
    """
    triple_barrier_returns() 결과(열: ret, target_vol 필요)를 경제적 유의성 지표로 요약.

    round_trip_cost: 왕복 비용+슬리피지를 소수로 환산한 값(예: 56bp -> 0.0056).
        0 이하면 비용 대비 배수는 NaN.

    승·패 구분은 barrier_summary의 win_rate와 같은 기준(ret > 0이면 승, 그 외 패,
    ret == 0은 패에 포함되지만 손실 크기 0이라 평균 L을 키우지는 않는다)이다.
    이 정의에서 기대값 = p·W − (1−p)·L 이 항상 정확히 성립한다.
    승이 하나도 없으면 W=0(손익분기 승률 100%), 패가 하나도 없으면 L=0(손익분기 0%)으로 둔다.
    """
    empty = {"n": 0, "expectancy_r": np.nan, "win_rate": np.nan, "avg_win_r": np.nan,
             "avg_loss_r": np.nan, "payoff_ratio": np.nan, "breakeven_wr": np.nan,
             "wr_edge": np.nan, "nominal_breakeven_wr": np.nan, "net_to_cost": np.nan}
    if tb is None or len(tb) == 0:
        return empty

    R = r_multiples(tb, sl_mult)
    ok = np.isfinite(R)
    ret = tb["ret"].to_numpy(dtype=float)
    mean_ret = float(np.nanmean(ret)) if len(ret) else np.nan
    net_to_cost = (mean_ret / round_trip_cost) if (round_trip_cost and round_trip_cost > 0) else np.nan
    nominal = (sl_mult / (sl_mult + pt_mult)) if (pt_mult > 0 and sl_mult > 0) else np.nan

    out = dict(empty)
    out["n"] = int(len(tb))
    out["nominal_breakeven_wr"] = nominal
    out["net_to_cost"] = net_to_cost
    if not ok.any():
        return out

    Rv = R[ok]
    win = Rv > 0
    p = float(win.mean())
    W = float(Rv[win].mean()) if win.any() else 0.0
    L = float((-Rv[~win]).mean()) if (~win).any() else 0.0
    denom = W + L
    out.update(
        expectancy_r=float(Rv.mean()),
        win_rate=p,
        avg_win_r=W,
        avg_loss_r=L,
        payoff_ratio=(W / L) if L > 0 else (np.inf if W > 0 else np.nan),
        breakeven_wr=(L / denom) if denom > 0 else np.nan,
    )
    out["wr_edge"] = (p - out["breakeven_wr"]) if np.isfinite(out["breakeven_wr"]) else np.nan
    return out
