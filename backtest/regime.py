"""
시장 국면(Regime)별 검증 — 신호 성과가 상승장·하락장·횡보장 어디에서 나오는가.

[왜 필요한가]
이벤트 스터디 평균은 전 기간을 한 덩어리로 섞은 값이다. 상승장에서만 통하는 신호는
전체 평균이 양수로 나와 통과하지만, 하락장에서는 손실이라 앞으로 국면이 바뀌면 깨진다.
반대로 전체로는 약해 보여도 특정 국면에서만 강한 신호도 있다. 국면별로 나눠서 봐야
"어떤 환경에서 통하는가"와 "국면이 바뀌어도 부호가 유지되는가"를 알 수 있다.

[국면 분류 — 종목 자체 가격 기준]
지수 데이터 없이 같은 가격 시계열만으로 분류한다(pykrx/KRX 접속이 막혀도 동작).
  z = (ret_len봉 수익률) / (일간수익률 표준편차(vol_len봉) × √ret_len)
  상승(bull)  : 종가 > MA(ma_len) 이고 z >  +z_thr
  하락(bear)  : 종가 < MA(ma_len) 이고 z <  -z_thr
  횡보(sideways): 그 외(추세선과 최근 흐름이 엇갈리거나 흐름이 소음 수준)
z는 "최근 움직임이 변동성 대비 얼마나 방향성이 있었나"(신호 대 잡음)라서 변동성이 큰
종목의 출렁임을 추세로 오인하지 않게 해준다. ma_len=200, ret_len=60, vol_len=60,
z_thr=0.5는 관행적으로 흔한 값을 고른 것이며 최적화한 값이 아니다 — 결과가 이 값에
민감한지는 바꿔가며 확인해야 한다.
모든 계산은 해당 봉까지의 과거 데이터만 쓴다(미래 정보 없음). 앞의 ma_len봉은 분류가
불가능해 라벨이 NaN이고 분석에서 빠진다 — 상장 초기 종목은 이 구간이 길어 표본이 줄어든다.

[다중검정]
국면을 3개로 나누면 시험 횟수가 3배가 된다. 국면별 요약은 각 국면 안에서 BH·DSR을 계산하므로
그대로 보면 낙관적이다. 그래서 (1) 모든 국면의 p값을 한 번에 놓고 BH를 다시 돌려
significant_bh_global로 주고, (2) DSR의 시도 횟수에 다른 국면에서 쓴 시험 횟수를 더한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.vol_regime import VOL_REGIME_KR

REGIMES = ("bull", "bear", "sideways")
REGIME_KR = {"bull": "상승", "bear": "하락", "sideways": "횡보", **VOL_REGIME_KR}


def classify_regime(df: pd.DataFrame, ma_len: int = 200, ret_len: int = 60, vol_len: int = 60,
                    z_thr: float = 0.5, close_col: str = "close") -> pd.Series:
    """각 봉을 'bull'/'bear'/'sideways'로 분류한다. 분류 불가 구간은 NaN."""
    c = df[close_col].astype(float)
    ma = c.rolling(ma_len, min_periods=ma_len).mean()
    ret = c / c.shift(ret_len) - 1.0
    vol = c.pct_change().rolling(vol_len, min_periods=vol_len).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ret / (vol * np.sqrt(ret_len))
    valid = ma.notna() & z.notna() & np.isfinite(z)
    out = pd.Series(np.nan, index=df.index, dtype=object, name="regime")
    out[valid] = "sideways"
    out[valid & (c > ma) & (z > z_thr)] = "bull"
    out[valid & (c < ma) & (z < -z_thr)] = "bear"
    return out


def regime_share(regimes: pd.Series, labels=REGIMES) -> pd.Series:
    """라벨별 봉 비율(분류 가능한 봉 기준). labels 기본값은 추세 국면 3종."""
    r = regimes.dropna()
    if r.empty:
        return pd.Series({k: np.nan for k in labels})
    return r.value_counts(normalize=True).reindex(list(labels)).fillna(0.0)


@dataclass
class RegimeResult:
    summary: pd.DataFrame            # 국면 x 신호 x 보유기간 (long). regime, significant_bh_global 포함
    share: pd.Series                 # 국면별 봉 비율
    n_unclassified: int              # 라벨이 NaN인 봉 수(워밍업)
    trials_by_regime: dict = field(default_factory=dict)   # DSR 시도 횟수 기록(검증용)
    params: dict = field(default_factory=dict)
    labels: tuple = REGIMES          # 이 결과가 나눈 국면 라벨(추세 3종 또는 고변동/저변동)


def _mask_to_dates(dfs_by_indicator: dict, signal_specs, allowed: pd.DatetimeIndex):
    out = {k: v.copy() for k, v in dfs_by_indicator.items()}
    for key, col, _d, _label in signal_specs:
        if key in out and col in out[key].columns:
            d = out[key]
            d[col] = d[col].fillna(False).astype(bool) & d.index.isin(allowed)
    return out


def regime_study(dfs_by_indicator: dict, regimes: pd.Series, compare_kwargs: dict,
                 lo=None, hi=None, blocked=None, labels=REGIMES) -> RegimeResult:
    """
    국면별로 이벤트를 나눠 compare_all()을 각각 돌리고 합친다.
    lo/hi: 이벤트를 이 구간으로 제한(예: 검증 구간만). blocked: 제외할 날짜.
    이벤트가 없는 국면·신호도 compare_all()과 같이 n_events=0, 통계 NaN인 행으로 남는다(NaN p값은
    BH 계산에서 제외된다). 보고서(format_regime_report)는 관측이 있는 행만 보여준다.
    labels: 나눌 국면 라벨. 기본은 추세 3종(bull/bear/sideways)이고, 변동성 국면은
    backtest.vol_regime.VOL_REGIMES를 넘긴다. 국면 수가 달라지면 DSR 시도 횟수 가산과
    전역 BH 보정도 그 국면 수에 맞춰 계산된다.
    """
    from backtest.engine import SIGNAL_SPECS, benjamini_hochberg, compare_all
    from backtest.oos import attach_n_obs

    kw = dict(compare_kwargs)
    horizons = tuple(kw["horizons"])
    n_reg = len(labels)
    per_regime_rows = len(SIGNAL_SPECS) * len(horizons)      # 한 국면에서 가능한 시험 수의 상한
    base_extra = int(kw.get("extra_trials", 0))
    if kw.get("deflate"):
        # 다른 국면 (n_reg-1)개가 쓴 시험 횟수를 이 국면의 DSR 시도 횟수에 보태 보수적으로 잡는다.
        kw["extra_trials"] = base_extra + (n_reg - 1) * per_regime_rows

    any_df = next(iter(dfs_by_indicator.values()))
    reg = regimes.reindex(any_df.index)
    block_idx = pd.DatetimeIndex(blocked) if blocked is not None else pd.DatetimeIndex([])
    ok_window = pd.Series(True, index=any_df.index)
    if lo is not None:
        ok_window &= any_df.index >= pd.Timestamp(lo)
    if hi is not None:
        ok_window &= any_df.index <= pd.Timestamp(hi)
    ok_window &= ~any_df.index.isin(block_idx)

    parts, trials = [], {}
    for r in labels:
        allowed = any_df.index[(reg == r).fillna(False).to_numpy() & ok_window.to_numpy()]
        masked = _mask_to_dates(dfs_by_indicator, SIGNAL_SPECS, allowed)
        summ, res = compare_all(masked, **kw)
        if summ.empty:
            continue
        summ = attach_n_obs(summ, res)
        summ.insert(0, "regime", r)
        trials[r] = summ.attrs.get("dsr_n_trials")
        parts.append(summ)

    if parts:
        allsum = pd.concat(parts, ignore_index=True)
        allsum["significant_bh_global"] = benjamini_hochberg(allsum["p_value"].values,
                                                             fdr=kw.get("fdr", 0.10))
    else:
        allsum = pd.DataFrame()
    return RegimeResult(summary=allsum, share=regime_share(reg, labels), n_unclassified=int(reg.isna().sum()),
                        trials_by_regime=trials,
                        params=dict(extra_trials_added=(n_reg - 1) * per_regime_rows if kw.get("deflate") else 0),
                        labels=tuple(labels))


def regime_consistency(summary: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """
    (신호, 보유기간)별로 국면 간 부호가 유지되는지 본다. 관측 min_n 이상인 국면만 대상.
      n_regimes           : 평가 가능한 국면 수
      n_positive          : 평균수익이 양수인 국면 수
      consistent_positive : 평가 가능한 국면이 2개 이상이고 전부 양수
      sign_flip           : 양수인 국면과 음수(0 이하)인 국면이 섞임 — 국면이 바뀌면 깨질 수 있다는 경고
    """
    if summary is None or summary.empty:
        return pd.DataFrame(columns=["signal", "horizon", "n_regimes", "n_positive",
                                     "consistent_positive", "sign_flip"])
    d = summary[(summary["n_obs"] >= min_n) & summary["mean_return_pct"].notna()]
    rows = []
    for (sig, h), g in d.groupby(["signal", "horizon"]):
        n_r, n_pos = len(g), int((g["mean_return_pct"] > 0).sum())
        rows.append(dict(signal=sig, horizon=h, n_regimes=n_r, n_positive=n_pos,
                         consistent_positive=(n_r >= 2 and n_pos == n_r),
                         sign_flip=(n_pos > 0 and n_pos < n_r)))
    return pd.DataFrame(rows, columns=["signal", "horizon", "n_regimes", "n_positive",
                                       "consistent_positive", "sign_flip"])


def format_regime_report(res: RegimeResult, judge_horizon: int, min_n: int = 5) -> str:
    L = ["=== 시장 국면별 검증 (종목 자체 가격 기준) ==="]
    sh = res.share
    L.append("국면 비중(분류 가능한 봉 기준): " + ", ".join(
        f"{REGIME_KR[k]} {sh[k]:.0%}" if pd.notna(sh[k]) else f"{REGIME_KR[k]} -" for k in res.labels)
        + f" | 분류 불가(워밍업) {res.n_unclassified}봉")
    if res.summary.empty:
        L.append("국면별로 이벤트가 하나도 없어 요약할 수 없습니다.")
        return "\n".join(L)
    sub = res.summary[res.summary["horizon"] == judge_horizon].copy()
    cols = ["regime", "signal", "n_obs", "mean_return_pct", "win_rate_pct", "t_stat",
            "significant_bh", "significant_bh_global"]
    for extra in ("significant_dsr", "tb_expectancy_r"):
        if extra in sub.columns:
            cols.append(extra)
    sub = sub[sub["n_obs"] > 0].sort_values(["signal", "regime"])
    L.append(f"[보유 {judge_horizon}봉] significant_bh는 국면 안에서만, significant_bh_global은 "
             f"모든 국면을 한꺼번에 보정한 값입니다(이쪽이 더 엄격하고 정직합니다).")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        L.append(sub[cols].to_string(index=False))
    cons = regime_consistency(res.summary[res.summary["horizon"] == judge_horizon], min_n=min_n)
    flips = cons[cons["sign_flip"]]
    allpos = cons[cons["consistent_positive"]]
    L.append(f"국면 간 일관성(관측 {min_n}건 이상인 국면만): 모든 국면에서 양수 {len(allpos)}개 신호, "
             f"국면에 따라 부호가 뒤집힘 {len(flips)}개 신호")
    if len(flips):
        L.append("  부호 뒤집힘: " + ", ".join(flips["signal"].tolist()))
    return "\n".join(L)
