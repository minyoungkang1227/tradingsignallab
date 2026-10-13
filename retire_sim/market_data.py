"""종목 통계 (v15): 종목코드 → 시장 베타, 기존 자산(KOSPI)과의 상관, 변동성, 금리 민감도.

추정 방식 (월별 로그수익률)
- 베타 β = Cov(r_i, r_m) / Var(r_m),  고유 변동성 σ_ε = std(r_i − α − β r_m) · √12
- 금리 민감도 γ: r_i − β r_m 을 국고채 3년 월간 변화 Δy 에 회귀한 계수 (γ<0이면 금리 상승 시 불리)
- 기대수익은 과거 평균이 아니라 CAPM(무위험금리 + β·위험프리미엄)으로 모델에서 결정
  → 과거 수익률이 높았던 종목을 과대평가하지 않기 위함
데이터: FinanceDataReader(KRX 가격, KOSPI 'KS11'), 금리는 ECOS 키가 있을 때 국고채 3년(721Y001/5020000)
"""
import numpy as np
import pandas as pd


def estimate_from_prices(stock: pd.Series, market: pd.Series, yld: pd.Series | None = None) -> dict:
    """월말 가격 시리즈(인덱스=날짜)로 통계 추정. yld는 월별 금리(소수, 예 0.035)."""
    df = pd.concat({"s": stock, "m": market}, axis=1).dropna()
    r = np.log(df).diff().dropna()
    if len(r) < 24:
        raise ValueError(f"월별 데이터가 {len(r)}개월뿐입니다. 최소 24개월이 필요합니다.")
    rs, rm = r["s"].values, r["m"].values
    beta = np.cov(rs, rm, ddof=1)[0, 1] / np.var(rm, ddof=1)
    alpha = rs.mean() - beta * rm.mean()
    resid = rs - alpha - beta * rm
    out = {"months": len(r), "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
           "beta": float(beta), "corr_market": float(np.corrcoef(rs, rm)[0, 1]),
           "sigma": float(rs.std(ddof=1) * np.sqrt(12)), "idio_sigma": float(resid.std(ddof=1) * np.sqrt(12)),
           "rate_beta": 0.0, "rate_beta_se": None}
    if yld is not None:
        y = yld.reindex(r.index, method="nearest").diff()
        ok = ~np.isnan(y.values)
        if ok.sum() >= 24:
            x = y.values[ok]; z = (rs - beta * rm)[ok]
            X = np.column_stack([np.ones(len(x)), x])
            coef, *_ = np.linalg.lstsq(X, z, rcond=None)
            e = z - X @ coef; se = np.sqrt(e @ e / (len(x) - 2) * np.linalg.inv(X.T @ X)[1, 1])
            out["rate_beta"], out["rate_beta_se"] = float(coef[1]), float(se)
    return out


def fetch_monthly_close(code: str, start: str = "2015-01-01") -> pd.Series:
    import FinanceDataReader as fdr
    d = fdr.DataReader(code, start)
    if d is None or len(d) == 0 or "Close" not in d:
        raise ValueError(f"'{code}' 가격 데이터를 찾지 못했습니다. 6자리 종목코드(예: 005930)를 확인하세요.")
    return d["Close"].resample("ME").last()


def fetch_ecos_yield(api_key: str, start: str = "201501", end: str = "209912") -> pd.Series:
    import requests
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/721Y001/M/{start}/{end}/5020000"
    js = requests.get(url, timeout=30).json()
    rows = js["StatisticSearch"]["row"]
    idx = pd.to_datetime([r["TIME"] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    return pd.Series([float(r["DATA_VALUE"]) / 100 for r in rows], index=idx)


def stock_stats(code: str, years: int = 10, ecos_key: str | None = None) -> dict:
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    s = fetch_monthly_close(code, start); m = fetch_monthly_close("KS11", start)
    y = None
    if ecos_key:
        try:
            y = fetch_ecos_yield(ecos_key, start[:4] + start[5:7])
        except Exception:
            y = None
    out = estimate_from_prices(s, m, y); out["code"] = code
    return out
