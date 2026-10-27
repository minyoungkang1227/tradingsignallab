"""
공통 포지션 인터페이스 — 리스크 계산이 받는 유일한 입력 형태.

  weights : DataFrame, index=날짜(오름차순), columns=종목, 값=평가자산 대비 비중(부호 포함, 숏은 음수).
            weights.loc[t] 는 "t일 장 마감 후 들고 있는 비중"이다.
  closes  : dict[종목 -> 종가 Series] 또는 DataFrame(index=날짜, columns=종목).

[룩어헤드 규칙]
t일의 포트폴리오 수익률은 t-1일 마감 비중 × t일 수익률이다. t일 비중을 t일 수익률에 곱하면
그날의 등락을 알고 나서 비중을 정한 셈이 된다. 이 모듈이 그 시프트를 한 곳에서 처리하므로
호출부가 따로 신경 쓸 필요가 없고, 틀릴 자리도 하나뿐이다.

[비중의 단위]
비중은 '자산 대비 비율'이다. 수량·금액으로 들고 있는 계좌는 호출부가 평가자산으로 나눠 비율로
바꿔서 넘긴다. 레버리지는 |비중| 합이 1을 넘는 것으로 표현된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def closes_frame(closes) -> pd.DataFrame:
    """dict[종목->Series] 또는 DataFrame을 종가 DataFrame(index=날짜 합집합, 오름차순)으로."""
    if isinstance(closes, pd.DataFrame):
        df = closes.copy()
    else:
        df = pd.DataFrame({k: v.astype(float) for k, v in closes.items()})
    return df.sort_index()


def constant_weights(weights: dict, index: pd.Index) -> pd.DataFrame:
    """지정 보유 포트폴리오: 매일 같은 비중(일별 리밸런싱 가정). 예) {"005930": 0.5, "000660": 0.5}."""
    return pd.DataFrame({k: float(v) for k, v in weights.items()}, index=pd.Index(index))


def portfolio_returns(weights: pd.DataFrame, closes, *, fill_missing_weight: float = 0.0) -> pd.Series:
    """
    일별 포트폴리오 수익률 r_p[t] = Σ_i w[t-1, i] × r[t, i].

    r[t, i]는 단순수익률(close[t]/close[t-1] - 1). 종목의 종가가 결측인 날은 그 종목 수익률을 0으로
    두지 않고 그 날 전체를 NaN으로 남긴다 — 결측을 0 수익으로 메우면 변동성이 과소추정된다.
    weights에 없는 날(비중 미지정)은 fill_missing_weight(기본 0 = 현금)로 본다.
    반환 Series의 첫 날은 직전 비중이 없어 NaN이다.
    """
    px = closes_frame(closes)
    cols = [c for c in weights.columns if c in px.columns]
    missing = [c for c in weights.columns if c not in px.columns]
    if missing:
        raise KeyError(f"종가가 없는 종목: {missing}")
    rets = px[cols].pct_change()
    w = weights[cols].reindex(rets.index).fillna(fill_missing_weight).astype(float)
    w_prev = w.shift(1)                      # t-1 마감 비중을 t 수익률에 적용
    contrib = w_prev * rets
    out = contrib.sum(axis=1, min_count=1)
    # 비중이 0이 아닌 종목의 수익률이 NaN이면 그 날은 계산 불가(NaN)
    bad = ((w_prev != 0) & rets.isna()).any(axis=1)
    out[bad] = np.nan
    out.name = "portfolio_return"
    return out


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """총 노출 = Σ|비중|. 1을 넘으면 레버리지."""
    return weights.abs().sum(axis=1)
