"""
이벤트 스터디(event study) 기반 신호 검증 엔진.
"이 신호가 뜬 뒤 N봉 동안 평균적으로 수익이 나는가"를 통계적으로 측정한다.
포지션 관리(청산/물타기 등)가 있는 완전한 백테스트가 아니라, 신호 자체의
예측력을 t-검정 수준에서 빠르게 스크리닝하기 위한 도구다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd


def forward_return(df: pd.DataFrame, horizon: int, direction: int = 1,
                    close_col: str = "close", entry_lag: int = 0,
                    entry_col: str | None = None, cost_bps: float = 0.0,
                    slippage_bps: float | pd.Series = 0.0) -> pd.Series:
    """
    이벤트 발생 시점(t) 대비 h봉 뒤까지의 방향조정·거래비용반영 수익률.

    look-ahead bias 방지와 거래비용 반영을 위한 옵션 — 기본값은 이전 버전의
    동작(당일 종가 즉시 진입, 비용 0)과 완전히 동일하다. 실제 백테스트에
    반영하려면 명시적으로 켜야 한다.

      entry_lag: 신호가 t봉 종가에 "확정"된 뒤, 실제 진입까지 걸리는 봉 수.
                 기본 0 = 신호가 뜬 바로 그 봉의 종가로 즉시 진입한다고 가정
                 (과거 동작과 동일 — 그 봉이 마감되기 전엔 신호를 알 수 없으므로
                 엄밀히는 미래 데이터를 쓰는 것과 같은 문제가 있다). 1로 두면
                 "t봉 종가에 신호 확정 → t+1봉에서 진입"이라는 현실적 가정이 된다.
      entry_col: entry_lag>0일 때 실제 진입가로 쓸 컬럼. 기본 None이면 close_col을
                 그대로 쓰고, 'open'을 주면 t+entry_lag봉의 시가로 진입한다고
                 가정한다 — "종가에 신호 확정, 다음날 시가 진입"이 가장 현실적인
                 조합(entry_lag=1, entry_col='open').
      cost_bps: 왕복(진입+청산) 거래 수수료를 basis point(1bp=0.01%)로 지정. 참고
                근사치: 업비트는 편도 0.05%×2=10bp, KRX는 편도(수수료+거래세) 약
                0.18%×2=36bp. 기본 0(비용 미반영, 과거 동작과 동일).
      slippage_bps: 왕복 슬리피지(실제 체결가가 기대한 진입/청산가와 벌어지는
                정도)를 별도 basis point로 지정. cost_bps와 개념이 다르다 —
                cost_bps는 거래소에 내는 확정된 수수료·세금이고, slippage_bps는
                주문 집행 과정에서 생기는 불확실한 가격 미끄러짐이다. 특히
                entry_lag>0으로 장 시작 직후 시가(KRX) 진입을 가정하거나 코인의
                변동성 장세에서는 슬리피지가 수수료 못지않게 커질 수 있어 별도
                파라미터로 분리했다. 기본 0(미반영, 과거 동작과 동일). 참고
                근사치는 시장·유동성마다 크게 다르지만, 유동성이 충분한 대형주/
                메이저 코인 기준 5~10bp 정도를 버퍼로 흔히 쓴다.
      cost_bps, slippage_bps 모두 direction(롱/숏)과 무관하게 항상 수익률에서
                차감한다 — 비용·슬리피지는 포지션 방향과 상관없이 실제로 나가는
                돈이기 때문(숏이라고 비용이 이익으로 바뀌면 안 된다).
                slippage_bps는 스칼라(전 구간 동일값) 대신 df.index와 같은
                인덱스를 가진 pd.Series를 넘길 수도 있다 — 봉(날짜)마다 다른
                슬리피지를 반영하고 싶을 때(예: indicators.liquidity의
                Amihud(2002) 비유동성 기반 동적 슬리피지 추정치) 쓴다. Series를
                주면 df.index에 맞춰 reindex한 뒤 결측은 0으로 채운다.

    반환: (exit_price/entry_price - 1) * direction - (cost_bps+slippage_bps)/10000
    """
    exit_price = df[close_col].shift(-horizon)

    if entry_lag == 0:
        entry_price = df[close_col] if entry_col is None else df[entry_col]
    else:
        col = entry_col or close_col
        entry_price = df[col].shift(-entry_lag)

    raw_ret = exit_price / entry_price - 1

    if isinstance(slippage_bps, pd.Series):
        slip = slippage_bps.reindex(df.index).fillna(0.0)
    else:
        slip = slippage_bps

    return raw_ret * direction - cost_bps / 10000.0 - slip / 10000.0


@dataclass
class EventStudyResult:
    name: str
    direction: int  # +1 long, -1 short
    n_events: int
    table: pd.DataFrame  # index=horizon, columns=[n, mean, std, win_rate, t_stat]
    curve: pd.Series      # index=horizon(1..max_h), 평균 누적수익률 경로


def _dedupe_overlapping(df_index: pd.Index, idx: pd.Index, horizon: int) -> pd.Index:
    """
    보유기간이 horizon봉인 이벤트들 중, 서로의 보유구간이 겹치는(overlapping)
    이벤트를 좌에서 우로 훑으며 그리디하게 제거한다. 시간순으로 정렬한 뒤,
    직전에 채택한 이벤트로부터 horizon봉 이상 떨어진 이벤트만 남긴다 — 이렇게
    하면 남는 이벤트들의 보유구간이 서로 겹치지 않아, 겹친 구간에서 생기는
    잔차 자기상관(t-stat이 부풀려지는 원인) 자체를 없앨 수 있다(Newey-West
    보정과는 다른 접근 — 표본을 줄이는 대신 자기상관을 원천 차단).
    """
    if len(idx) == 0:
        return idx
    positions = df_index.get_indexer(idx)
    positions = positions[positions >= 0]
    positions = np.sort(positions)

    kept = []
    last_kept = -10 ** 9
    for pos in positions:
        if pos - last_kept >= horizon:
            kept.append(pos)
            last_kept = pos
    return df_index[kept]


def _newey_west_t_stat(vals: np.ndarray, lag: int) -> tuple[float, float]:
    """
    Newey-West(1987) HAC(이분산·자기상관에 강건한) 표준오차로 조정한 표본평균의
    t-통계량. 보유기간이 길어질수록 며칠 간격으로 뜬 이벤트들의 순방향수익률
    추적 구간이 겹치게(overlapping) 되는데, 이 경우 관측치 간 자기상관이 생겨
    단순 std/sqrt(n) 방식의 t-stat이 실제보다 부풀려진다. Bartlett 커널로
    자기상관을 보정한 "장기분산(long-run variance)"으로 평균의 표준오차를
    다시 추정한다.

    vals: 이벤트 순방향수익률(반드시 이벤트 발생일 기준 오름차순으로 정렬돼
        있어야 한다 — 자기상관은 시간 순서를 전제로 하는 개념이기 때문).
    lag: 보정에 쓸 최대 시차(보통 보유기간-1를 쓴다 — 겹치는 구간이 최대
        horizon-1봉까지이므로 그 이상의 시차에서는 이론상 자기상관이 없다).

    반환: (mean, t_stat). 유효 관측치가 1개 이하이거나 분산이 0/음수로
        추정되면 t_stat은 NaN.
    """
    n = len(vals)
    mean = float(vals.mean())
    if n <= 1:
        return mean, float("nan")

    e = vals - mean
    gamma0 = float((e * e).sum()) / n
    nw_var = gamma0
    L = max(0, min(lag, n - 1))
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1)
        gamma_k = float((e[k:] * e[:-k]).sum()) / n
        nw_var += 2 * w * gamma_k

    if nw_var <= 0:
        return mean, float("nan")
    se_mean = math.sqrt(nw_var / n)
    t_stat = mean / se_mean if se_mean > 0 else float("nan")
    return mean, t_stat


def event_returns(df: pd.DataFrame, event_col: str, horizon: int, direction: int = 1,
                   close_col: str = "close", entry_lag: int = 0, entry_col: str | None = None,
                   cost_bps: float = 0.0, slippage_bps: float | pd.Series = 0.0,
                   drop_overlapping: bool = False) -> pd.Series:
    """
    event_study()가 내부적으로 계산하는 "이벤트별 순방향수익률"을 집계(평균·
    t-stat) 이전 단계 그대로, 이벤트 발생일을 인덱스로 하는 pd.Series로 반환한다.

    event_study()는 이 값을 바로 평균·표준편차·t-stat으로 뭉개버리는데,
    여러 종목을 풀링하거나(research.param_search.pool_event_returns) IS/OOS
    기간으로 나누는 등 "집계 전" 원시 이벤트 수익률이 필요한 곳(파라미터
    탐색·횡단면 풀링)에서 쓰기 위해 이 단계를 공개 함수로 뺐다. 인자 의미는
    event_study()/forward_return()과 완전히 동일하다.
    """
    idx = df.index[df[event_col].fillna(False)]
    use_idx = _dedupe_overlapping(df.index, idx, horizon) if drop_overlapping else idx
    fwd = forward_return(df, horizon, direction=direction, close_col=close_col,
                          entry_lag=entry_lag, entry_col=entry_col,
                          cost_bps=cost_bps, slippage_bps=slippage_bps)
    return fwd.loc[use_idx].dropna().sort_index()


def cluster_robust_t_stat(vals: np.ndarray, cluster_ids) -> tuple[float, float]:
    """
    여러 종목을 풀링한 이벤트 수익률의 평균에 대해 1-way 클러스터 강건
    (cluster-robust) 표준오차로 계산한 t-통계량.

    Newey-West(_newey_west_t_stat)가 "같은 종목 안에서 보유기간이 겹쳐 생기는
    시간축 자기상관"을 보정한다면, 이건 "여러 종목을 풀링했을 때 같은 날짜에
    여러 종목에서 동시에 이벤트가 몰리는(시장 전체가 흔들리는 날 공통으로 신호가
    뜨는 등) 횡단면 상관"을 보정한다 — 축만 다를 뿐 원리는 같다: 잔차를
    클러스터(여기서는 이벤트 날짜)별로 묶어서, 클러스터 내부의 상관은 그대로
    두되 클러스터 간에는 독립이라고 가정하고 평균의 분산을 다시 추정한다
    (Cameron, Gelbach & Miller류 1-way cluster-robust sandwich를 절편만 있는
    회귀=단순평균에 적용한 특수형).

    풀링으로 표본 수(n)를 몇십 개에서 몇천 개로 늘려도, 그 표본들이 진짜
    독립이 아니라면(같은 날 여러 종목이 같이 움직임) 단순 std/sqrt(n)로 계산한
    t-stat은 다시 부풀려진다 — 이걸 막기 위한 함수다.

    vals: 풀링된 전체 이벤트 수익률.
    cluster_ids: vals와 같은 길이의 클러스터 식별자(보통 이벤트 날짜를 정수로
        바꾼 값). 같은 값을 가진 원소들이 한 클러스터로 묶인다.

    반환: (mean, t_stat). 클러스터가 1개 이하이거나 분산이 0/음수로 추정되면
        t_stat은 NaN.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if n <= 1:
        return mean, float("nan")

    cluster_ids = np.asarray(cluster_ids)
    resid = vals - mean
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    if n_clusters <= 1:
        return mean, float("nan")

    cluster_sum_sq = 0.0
    for c in unique_clusters:
        s = float(resid[cluster_ids == c].sum())
        cluster_sum_sq += s * s

    var_mean = cluster_sum_sq / (n * n)
    if var_mean <= 0:
        return mean, float("nan")
    # 클러스터 수가 적을 때의 소표본 보정(Cameron et al. 권고 — 자유도를
    # n_clusters/(n_clusters-1)배만큼 보수적으로 부풀림)
    dof_correction = n_clusters / (n_clusters - 1)
    se_mean = math.sqrt(var_mean * dof_correction)
    t_stat = mean / se_mean if se_mean > 0 else float("nan")
    return mean, t_stat


def event_study(df: pd.DataFrame, event_col: str, close_col: str = "close",
                 direction: int = 1, horizons=(1, 3, 5, 10, 20),
                 max_curve_horizon: int = 20, name: str | None = None,
                 entry_lag: int = 0, entry_col: str | None = None,
                 cost_bps: float = 0.0, slippage_bps: float | pd.Series = 0.0,
                 newey_west: bool = False, drop_overlapping: bool = False) -> EventStudyResult:
    """
    entry_lag, entry_col, cost_bps, slippage_bps: forward_return()과 동일한 의미.
    기본값은 이전 버전과 동일한 동작(당일 종가 진입, 비용·슬리피지 0)이므로,
    현실적인 백테스트를 하려면 호출부에서 entry_lag=1, entry_col='open',
    cost_bps/slippage_bps=<시장별 왕복비용>을 명시적으로 지정해야 한다.

    newey_west: True면 각 보유기간의 t-stat을 단순 std/sqrt(n) 대신 Newey-West
        HAC 보정(lag=horizon-1, Bartlett 커널)으로 계산한다. 보유기간이 길어
        이벤트 간 순방향수익률 추적 구간이 겹칠 때(overlapping events) 자기상관
        때문에 t-stat이 부풀려지는 문제를 완화한다. 기본 False(이전 동작과 동일).
    drop_overlapping: True면 보유기간이 겹치는 이벤트를 애초에 표본에서 제거한다
        (_dedupe_overlapping — 시간순으로 훑으며 직전 채택 이벤트로부터 horizon봉
        이상 떨어진 것만 남김). newey_west와 동시에 켜도 되고(중복 제거 후 남은
        표본에 추가로 HAC 보정을 적용), 하나만 켤 수도 있다. 기본 False.
    """
    idx = df.index[df[event_col].fillna(False)]
    n_events = len(idx)
    name = name or event_col

    def _fwd(h):
        return forward_return(df, h, direction=direction, close_col=close_col,
                               entry_lag=entry_lag, entry_col=entry_col,
                               cost_bps=cost_bps, slippage_bps=slippage_bps)

    def _event_idx_for_horizon(h):
        return _dedupe_overlapping(df.index, idx, h) if drop_overlapping else idx

    rows = []
    for h in horizons:
        use_idx = _event_idx_for_horizon(h)
        vals = _fwd(h).loc[use_idx].dropna().sort_index()
        if len(vals) == 0:
            rows.append(dict(horizon=h, n=0, mean=np.nan, std=np.nan, win_rate=np.nan,
                              t_stat=np.nan, skew=np.nan, kurt=np.nan))
            continue
        std = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        win_rate = float((vals > 0).mean())
        if newey_west:
            mean, t_stat = _newey_west_t_stat(vals.values, lag=max(0, h - 1))
        else:
            mean = float(vals.mean())
            t_stat = mean / (std / np.sqrt(len(vals))) if std and std > 0 else np.nan
        # 왜도·첨도(원첨도, 정규분포=3). 디플레이티드 샤프비율(backtest.deflated)이
        # 비정규성 보정에 쓴다 — 여기서 vals가 이미 메모리에 있으므로 같이 계산해
        # 나중에 원시 이벤트 수익률을 다시 뽑는 비용을 없앤다. 기존 컬럼은 그대로라
        # table을 읽던 호출부는 영향받지 않는다.
        if len(vals) > 1 and std and std > 0:
            _z = (vals.values - mean) / std
            skew = float(np.mean(_z ** 3))
            kurt = float(np.mean(_z ** 4))
        else:
            skew = kurt = np.nan
        rows.append(dict(horizon=h, n=len(vals), mean=mean, std=std, win_rate=win_rate,
                          t_stat=t_stat, skew=skew, kurt=kurt))
    table = pd.DataFrame(rows).set_index("horizon")

    curve_vals = []
    for h in range(1, max_curve_horizon + 1):
        use_idx = _event_idx_for_horizon(h)
        vals = _fwd(h).loc[use_idx].dropna()
        curve_vals.append(vals.mean() if len(vals) else np.nan)
    curve = pd.Series(curve_vals, index=range(1, max_curve_horizon + 1))

    return EventStudyResult(name=name, direction=direction, n_events=n_events, table=table, curve=curve)


# (indicator_key, event_column, direction(+1 long/-1 short), display_label)
SIGNAL_SPECS = [
    ("institutional_displacement", "bull_shift", 1, "기관캔들-강세"),
    ("institutional_displacement", "bear_shift", -1, "기관캔들-약세"),
    ("delta_trading", "crossover_confirmed", 1, "델타트레이딩-골든크로스"),
    ("delta_trading", "crossunder_confirmed", -1, "델타트레이딩-데드크로스"),
    ("big_sales", "bull_event", 1, "빅세일-매수클라이맥스"),
    ("big_sales", "bear_event", -1, "빅세일-매도클라이맥스"),
    ("ultimate_rsi", "bull_signal", 1, "얼티밋RSI-과매도반등"),
    ("ultimate_rsi", "bear_signal", -1, "얼티밋RSI-과매수반락"),
    ("vp_box", "poc_support_test", 1, "VP박스-POC지지"),
    ("vp_box", "poc_resistance_test", -1, "VP박스-POC저항"),
    ("combo_filter", "bull_signal", 1, "콤보필터-강세"),
    ("combo_filter", "bear_signal", -1, "콤보필터-약세"),
]


def compare_all(dfs_by_indicator: dict[str, pd.DataFrame], horizons=(1, 3, 5, 10, 20),
                 max_curve_horizon: int = 20, entry_lag: int = 0,
                 entry_col: str | None = None, cost_bps: float = 0.0,
                 slippage_bps: float | pd.Series = 0.0, newey_west: bool = False,
                 drop_overlapping: bool = False, fdr: float = 0.10,
                 deflate: bool = False, dsr_threshold: float = 0.95,
                 extra_trials: int = 0, min_events: int = 20,
                 sr_std_method: str = "theoretical",
                 barrier: bool = False, pt_mult: float = 2.0, sl_mult: float = 1.0,
                 vol_span: int = 20, use_intrabar: bool = True,
                 barrier_sequential: bool = False, barrier_economics: bool = False
                 ) -> tuple[pd.DataFrame, dict[str, EventStudyResult]]:
    """
    dfs_by_indicator: {"institutional_displacement": df1, "delta_trading": df2, ...}
    각 df는 해당 indicators.<name>.compute()의 반환값(원본 OHLCV + 신호 컬럼 포함)이어야 한다.

    entry_lag, entry_col, cost_bps, slippage_bps, newey_west, drop_overlapping:
    event_study()/forward_return()과 동일한 의미로 모든 지표에 동일하게 적용된다.
    기본값(0, None, 0.0, 0.0, False, False)은 이전 버전과 동일한 동작(당일 종가
    진입, 비용·슬리피지 미반영, 단순 t-stat, 겹치는 이벤트도 전부 포함)이다.

    fdr: 다중 검정 보정(Benjamini-Hochberg)의 목표 위양성발견율(false discovery
    rate). 5개 지표 x 롱/숏 x 여러 보유기간을 한 번에 비교하다 보면, 개별 t-stat만
    보고 |t|>=2를 "유의하다"고 판단할 경우 우연히 유의하게 나온 신호(p-hacking)를
    걸러내지 못한다. 요약표의 t_stat 전체(모든 지표 x 방향 x 보유기간 조합, 즉
    이 호출 하나가 수행한 전체 검정 묶음)를 대상으로 BH 절차를 적용해
    `p_value`(정규분포 근사 양측검정)와 `significant_bh`(그 절차를 통과했는지)
    컬럼을 추가한다. 이 두 컬럼은 기존 컬럼(signal/direction/n_events/horizon/
    mean_return_pct/win_rate_pct/t_stat)에 추가되는 것이라, 기존 컬럼만 읽는
    호출부는 그대로 동작한다.

    deflate: True면 디플레이티드 샤프비율(backtest.deflated)까지 계산해
    sharpe_per_trade / deflated_sr / significant_dsr 컬럼을 추가한다.
    기본값 False는 기존 동작을 그대로 보존한다(컬럼 자체가 추가되지 않음).

    [왜 BH보정 위에 DSR을 또 얹는가]
    BH가 쓰는 p-value는 p_value_from_t()가 정규분포를 가정해서 만든 값이다.
    이벤트 수익률은 왜도·첨도가 정규분포와 다른 게 보통이라, 그 p-value 자체가
    한쪽으로 치우쳐 있다. DSR은 왜도·첨도를 표준오차에 직접 넣고, 동시에
    "N번 시도해서 고른 최고값"이라는 선택편의까지 기준선에 반영한다.
    두 보정은 대체재가 아니라 서로 다른 구멍을 막는다 — 둘 다 통과한 신호만
    보려면 significant_bh와 significant_dsr을 함께 보면 된다.

    dsr_threshold: significant_dsr 판정 임계값(기본 0.95 = 통상적인 5% 유의수준).
    extra_trials: 이 호출 바깥에서 이미 시도한 검정 횟수. compare_all은 자기가
        수행한 검정 수(요약표 행 수)밖에 모르는데, 실제로는 파라미터 탐색
        (run_param_search)이나 종목을 바꿔가며 돌린 횟수까지 전부 "시도"에
        포함되어야 선택편의가 제대로 반영된다. 예를 들어 파라미터 조합 500개를
        미리 훑어본 뒤 제일 좋은 걸로 이 함수를 불렀다면 extra_trials=500을
        넘겨야 한다. 기본 0은 "이 호출이 전부"라는 가장 낙관적인 가정이므로,
        실제 탐색을 거쳤다면 반드시 채워 넣어야 DSR이 과대평가되지 않는다.

    min_events: DSR 판정에 요구하는 최소 이벤트 수(기본 20). 이보다 적은
        조합은 deflated_sr을 NaN으로 두고 유의 판정에서 제외한다.
        이벤트가 2~4개인 신호는 샤프비율은 물론 왜도·첨도 추정 자체가
        무의미한데(진짜 SR이 0이어도 n=2면 SR 추정치의 표준편차가 발산한다),
        그런 값들이 섞이면 sr_std 추정이 오염되어 기준선이 비현실적으로
        높아지고 "진짜 신호는 떨어뜨리고 표본 적은 노이즈만 통과시키는"
        정반대 결과가 나온다. 실제로 이 저장소의 12개 신호는 빈도 차이가
        극심해서(빅세일 0~8건 vs VP박스 70건대) 이 문제가 그대로 재현됐다.

    sr_std_method: 시도 간 샤프비율 흩어짐(sr_std)을 어떻게 추정할지.
        "theoretical"(기본) — 표본 수로부터 이론적으로 계산(1/sqrt(n-1)).
            표본 수가 제각각인 상황에 강건해서 기본값으로 삼았다.
        "observed" — 실제로 관측된 샤프비율들의 표준편차. 모든 시도의 표본
            수가 비슷할 때만 타당하다.

    barrier: True면 같은 신호를 "손절/익절을 걸고" 매매했을 때의 실현 수익률을
        tb_* 컬럼으로 나란히 붙인다(backtest.triple_barrier.barrier_study).
        기본 False는 기존 동작을 그대로 보존한다(컬럼 자체가 추가되지 않음).

        [왜 나란히 붙이는가]
        기존 mean_return_pct는 "정확히 h봉 뒤 종가" 하나만 본다. 그 사이 경로는
        무시하므로, 중간에 -18%를 찍고 h봉째에 +2%로 끝난 거래가 "+2% 성공"으로
        집계된다. 실제로 손절을 걸고 매매했다면 그 거래는 -18% 근처에서 잘렸다.
        tb_* 컬럼은 같은 신호를 손절 걸고 매매했을 때 실제로 실현됐을 수익률이라,
        둘을 나란히 놓으면 "이 신호의 성과가 경로를 견딜 수 있는 성과인가"를
        바로 볼 수 있다. 두 값이 크게 벌어지는 신호일수록 실제 운용 난이도가 높다.

        시간 배리어는 각 horizon 값을 그대로 쓴다 — 같은 시간 예산으로 맞춰야
        "경로를 보느냐 마느냐"의 차이만 분리된다.

        deflate=True와 같이 켜면 배리어 수익률 기준 DSR(tb_deflated_sr /
        tb_significant_dsr)도 함께 계산한다. 이쪽이 "손절 걸고 매매했을 때의
        성과가 다중검정·비정규성을 감안해도 진짜인가"라는, 실제로 묻고 싶은
        질문에 가장 가까운 판정이다.

    pt_mult / sl_mult: 익절/손절 배리어를 실현변동성의 몇 배로 둘지
        (기본 2.0 / 1.0 = 1시그마 손절, 2시그마 익절, 손익비 2:1).
    vol_span: 배리어 스케일에 쓸 실현변동성의 지수가중 구간(기본 20봉).
    use_intrabar: True(기본)면 봉의 고가/저가로 배리어 터치를 판정한다.
    barrier_sequential: True면 배리어 통계에서 앞 거래가 청산되기 전에 진입하는 거래를 뺀다
        (triple_barrier.drop_overlapping_trades). 기본 False(이전 동작: 겹치는 이벤트도 전부
        센다). drop_overlapping은 고정 보유기간 집계에만 적용되고 배리어 집계에는 적용되지
        않으므로, 배리어 기반 DSR(tb_deflated_sr)이 독립 가정을 지키려면 이것을 켜야 한다.
    barrier_economics: True면 배리어 요약에 경제적 유의성 컬럼(tb_expectancy_r, tb_payoff_ratio,
        tb_breakeven_wr_pct, tb_wr_edge_pp, tb_cost_multiple)을 덧붙인다. 기본 False.

    [DSR 적용 시 주의 — 겹치는 이벤트]
    DSR은 관측치가 서로 독립이라고 가정한다. drop_overlapping=False면 보유기간이
    겹치는 이벤트들이 그대로 들어가 자기상관이 생기므로, 실질 표본 수가 n보다
    적은데도 n으로 계산되어 DSR이 낙관적으로 나온다. deflate=True로 쓸 때는
    drop_overlapping=True를 같이 켜는 것을 권장한다.
    """
    summary_rows = []
    results: dict[str, EventStudyResult] = {}

    for indicator_key, event_col, direction, label in SIGNAL_SPECS:
        df = dfs_by_indicator.get(indicator_key)
        if df is None or event_col not in df.columns:
            continue
        res = event_study(df, event_col, direction=direction, horizons=horizons,
                           max_curve_horizon=max_curve_horizon, name=label,
                           entry_lag=entry_lag, entry_col=entry_col, cost_bps=cost_bps,
                           slippage_bps=slippage_bps, newey_west=newey_west,
                           drop_overlapping=drop_overlapping)
        results[label] = res
        for h, row in res.table.iterrows():
            entry = dict(
                signal=label, direction=("LONG" if direction == 1 else "SHORT"),
                n_events=res.n_events, horizon=h,
                mean_return_pct=row["mean"] * 100 if pd.notna(row["mean"]) else np.nan,
                win_rate_pct=row["win_rate"] * 100 if pd.notna(row["win_rate"]) else np.nan,
                t_stat=row["t_stat"],
            )
            if deflate:
                # 거래당 샤프비율. 연율화하지 않는다 — 관측 1개 = 거래 1건이고
                # DSR 공식의 n도 거래 수이므로, 연율화하면 단위가 어긋난다.
                std = row.get("std", np.nan)
                mean = row.get("mean", np.nan)
                entry["_sharpe"] = (mean / std if pd.notna(mean) and pd.notna(std)
                                     and std > 0 else np.nan)
                entry["_n_obs"] = row.get("n", 0)
                entry["_skew"] = row.get("skew", np.nan)
                entry["_kurt"] = row.get("kurt", np.nan)
            summary_rows.append(entry)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary, results

    summary["p_value"] = summary["t_stat"].apply(p_value_from_t)
    summary["significant_bh"] = benjamini_hochberg(summary["p_value"].values, fdr=fdr)

    if barrier:
        from backtest.triple_barrier import barrier_study

        tb = barrier_study(
            dfs_by_indicator, SIGNAL_SPECS, horizons=horizons,
            pt_mult=pt_mult, sl_mult=sl_mult, vol_span=vol_span,
            entry_lag=entry_lag, entry_col=entry_col, cost_bps=cost_bps,
            slippage_bps=slippage_bps, use_intrabar=use_intrabar,
            sequential=barrier_sequential, economics=barrier_economics)
        # left merge — 요약표의 행 구성은 그대로 두고 배리어 컬럼만 덧붙인다.
        # 배리어 쪽에만 있는 조합(이벤트는 있는데 고정보유기간 집계가 비었던 경우)은
        # 버려지고, 반대로 배리어 쪽에 없는 조합은 tb_* 가 NaN으로 남는다.
        summary = summary.merge(tb, on=["signal", "direction", "horizon"], how="left")
        summary.attrs["barrier_pt_mult"] = pt_mult
        summary.attrs["barrier_sl_mult"] = sl_mult
        summary.attrs["barrier_sequential"] = barrier_sequential
        summary.attrs["barrier_economics"] = barrier_economics

    if deflate:
        from backtest.deflated import (
            expected_max_sharpe, null_sharpe_dispersion, probabilistic_sharpe_ratio,
            sharpe_dispersion,
        )

        def _dsr_block(sharpes, n_obs_arr, skews, kurts):
            """한 벌의 (샤프, 표본수, 왜도, 첨도)에 대해 DSR을 계산한다.
            고정 보유기간 수익률과 배리어 수익률에 똑같이 적용하기 위해 함수로
            뺐다 — 두 경로가 다른 기준선을 쓰면 비교가 성립하지 않으므로,
            각자 자기 분포에서 기준선을 따로 구하되 절차는 동일해야 한다."""
            sharpes = np.asarray(sharpes, dtype=float)
            n_obs_arr = np.asarray(n_obs_arr, dtype=float)

            # 표본이 너무 적은 조합은 애초에 추정 대상에서 뺀다. 포함시키면
            # sr_std가 오염되어 기준선 자체가 망가진다(min_events 설명 참조).
            eligible = np.isfinite(sharpes) & (n_obs_arr >= min_events)

            # 시도 횟수 = 판정 가능한 검정 수 + 바깥에서 이미 써버린 횟수
            n_trials = int(eligible.sum()) + max(0, int(extra_trials))

            if sr_std_method == "observed":
                sr_std = sharpe_dispersion(sharpes[eligible])
            elif sr_std_method == "theoretical":
                sr_std = null_sharpe_dispersion(n_obs_arr[eligible])
            else:
                raise ValueError(
                    f"sr_std_method는 'theoretical' 또는 'observed'여야 합니다: "
                    f"{sr_std_method!r}")

            bench = expected_max_sharpe(n_trials, sr_std)

            vals = []
            for ok, sr, n_obs, skew, kurt in zip(eligible, sharpes, n_obs_arr,
                                                  skews, kurts):
                if not ok or not np.isfinite(skew) or not np.isfinite(kurt):
                    vals.append(np.nan)
                    continue
                vals.append(probabilistic_sharpe_ratio(
                    float(sr), bench, int(n_obs), skew=float(skew), kurt=float(kurt)))
            return vals, n_trials, sr_std, bench, int(eligible.sum())

        # --- 고정 보유기간 수익률 기준 DSR ---
        dsr_vals, n_trials, sr_std, benchmark, n_eligible = _dsr_block(
            summary["_sharpe"].values,
            pd.to_numeric(summary["_n_obs"], errors="coerce").values,
            summary["_skew"].values, summary["_kurt"].values)

        summary["sharpe_per_trade"] = summary["_sharpe"].values
        summary["deflated_sr"] = dsr_vals
        # NaN(판정 불가)은 유의하지 않음으로 처리한다 — 표본 부족을 유의로
        # 흘려보내지 않기 위해.
        summary["significant_dsr"] = summary["deflated_sr"].fillna(0.0) >= dsr_threshold
        summary.attrs["dsr_n_trials"] = n_trials
        summary.attrs["dsr_sr_std"] = sr_std
        summary.attrs["dsr_benchmark"] = benchmark
        summary.attrs["dsr_min_events"] = min_events
        summary.attrs["dsr_n_eligible"] = n_eligible
        summary.attrs["dsr_sr_std_method"] = sr_std_method

        # --- 배리어(손절 반영) 수익률 기준 DSR ---
        # barrier=True일 때만. 이쪽이 "손절 걸고 매매했을 때의 성과가 다중검정·
        # 비정규성을 감안해도 진짜인가"라는, 실제로 묻고 싶은 질문에 가장 가깝다.
        if barrier and "tb_sharpe" in summary.columns:
            tb_vals, tb_trials, tb_sr_std, tb_bench, tb_elig = _dsr_block(
                summary["tb_sharpe"].values, summary["tb_n"].values,
                summary["tb_skew"].values, summary["tb_kurt"].values)
            summary["tb_deflated_sr"] = tb_vals
            summary["tb_significant_dsr"] = (
                summary["tb_deflated_sr"].fillna(0.0) >= dsr_threshold)
            summary.attrs["tb_dsr_n_trials"] = tb_trials
            summary.attrs["tb_dsr_sr_std"] = tb_sr_std
            summary.attrs["tb_dsr_benchmark"] = tb_bench
            summary.attrs["tb_dsr_n_eligible"] = tb_elig

        summary = summary.drop(columns=["_sharpe", "_n_obs", "_skew", "_kurt"])

    return summary, results


# ---------------------------------------------------------------------------
# 다중 검정 보정 (Benjamini-Hochberg) — 여러 개의 t-stat/p-value를 동시에 볼 때
# "이 중 몇 개는 우연히 유의하게 나온 것"이라는 점을 감안해 유의성 판정을 보정한다.
# scipy 없이 정규분포 근사(math.erfc)만으로 p-value를 계산하는, 다른 모듈들과
# 동일한 의존성 최소화 설계.
# ---------------------------------------------------------------------------
def _normal_sf(z: float) -> float:
    """표준정규분포 생존함수 P(Z > z). math.erfc로 직접 계산(scipy 불필요)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def p_value_from_t(t_stat: float) -> float:
    """
    t-stat으로부터 양측검정 p-value를 정규분포 근사로 계산한다(t-분포가 아니라
    표준정규분포로 근사 — 이벤트 스터디 표본이 충분히 크다는 가정, 표본이 작으면
    실제 t-분포보다 p-value를 약간 과소평가할 수 있다는 점에 유의).
    t_stat이 NaN이면 NaN을 반환한다(표본 부족 등으로 t-stat 자체가 없는 경우).
    """
    if t_stat is None or (isinstance(t_stat, float) and math.isnan(t_stat)):
        return float("nan")
    return 2.0 * _normal_sf(abs(float(t_stat)))


def benjamini_hochberg(p_values, fdr: float = 0.10) -> np.ndarray:
    """
    Benjamini-Hochberg 절차. p_values: p-value의 array-like(NaN 허용 — NaN은
    검정 자체가 불가능했던 항목이라 유의성 판정에서 제외된다). fdr: 원하는
    false discovery rate(기본 10%).

    절차: 유효한 p-value를 오름차순 정렬해 순위 i(1부터)를 매기고, p_(i) <=
    (i/m)*fdr 를 만족하는 것 중 "가장 큰 순위"까지를 전부 유의하다고 판정한다
    (그 순위보다 앞선 것들은 전부 자동으로 조건을 만족).

    반환: 입력과 같은 길이의 bool array(True=BH 절차 통과, 유의하다고 판정됨).
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    result = np.zeros(n, dtype=bool)

    valid_idx = np.where(~np.isnan(p))[0]
    m = len(valid_idx)
    if m == 0:
        return result

    order = valid_idx[np.argsort(p[valid_idx])]
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)
    thresholds = ranks / m * fdr

    below = np.where(sorted_p <= thresholds)[0]
    if len(below) > 0:
        max_k = int(below.max())  # 조건을 만족하는 가장 큰 순위(0-indexed)
        result[order[: max_k + 1]] = True

    return result


# ---------------------------------------------------------------------------
# 팩터 조정 알파 (파이프라인 5단계) — Fama & French(1993)/Carhart(1997) 팩터로
# 이벤트 스터디 수익률을 회귀해서, 팩터로 설명 안 되는 순수 알파를 분리한다.
# ---------------------------------------------------------------------------
def factor_adjust(event_dates, event_returns, factor_df: pd.DataFrame,
                   factor_cols: tuple[str, ...] = ("SMB", "HML", "WML")) -> dict:
    """
    event_dates, event_returns: 이벤트 발생일과 그 순방향수익률(이미 direction이
        곱해진 값 — event_study()가 forward_return()*direction으로 만드는 것과 동일).
        길이가 같아야 한다.
    factor_df: index가 월별 리밸런싱 날짜(datetime 또는 그걸로 변환 가능한 값)이고
        factor_cols로 지정한 팩터 열(SMB/HML/WML 등)을 담은 DataFrame — 보통
        factors.kr_fama_french.combine_factors()의 반환값을 그대로 넣는다.

    각 이벤트 날짜를 factor_df에서 가장 가까운(이전/이후 중 더 가까운) 리밸런싱
    시점의 팩터값에 매칭시킨다(merge_asof, direction="nearest"). 이벤트는 일 단위,
    팩터는 월 단위라 정확히 같은 기간이 아니라 "가장 가까운 달의 팩터 수익률로
    근사"한다는 뜻이다 — 이벤트의 보유기간이 한 달보다 훨씬 짧으면 근사오차가
    커질 수 있다는 점에 유의.

    이후 OLS로 아래 회귀를 추정한다.
        event_return = alpha + sum(beta_k * factor_k) + residual
    scipy/statsmodels 없이 numpy만으로 계수·표준오차·t-통계량을 직접 계산한다
    (다른 모듈들과 동일하게 의존성을 최소화하는 설계).

    반환: dict(alpha, alpha_t_stat, betas={factor_col: (beta, t_stat)},
              n_obs, r_squared)
    표본이 파라미터 수보다 적으면(n_obs <= len(factor_cols)+1) 전부 NaN으로 반환.
    """
    ev = pd.DataFrame({"date": pd.to_datetime(pd.Index(event_dates)),
                        "ret": np.asarray(event_returns, dtype=float)})
    ev = ev.dropna().sort_values("date")

    fac = factor_df[list(factor_cols)].copy()
    fac.index = pd.to_datetime(fac.index)
    fac = fac.sort_index().reset_index().rename(columns={fac.index.name or "index": "date"})

    merged = pd.merge_asof(ev, fac, on="date", direction="nearest")
    merged = merged.dropna(subset=list(factor_cols) + ["ret"])

    n = len(merged)
    k = len(factor_cols) + 1  # 절편 포함 파라미터 수
    if n <= k:
        return dict(alpha=np.nan, alpha_t_stat=np.nan,
                     betas={c: (np.nan, np.nan) for c in factor_cols},
                     n_obs=n, r_squared=np.nan)

    X = np.column_stack([np.ones(n)] + [merged[c].values for c in factor_cols])
    y = merged["ret"].values

    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    dof = n - k
    ssr = float((resid ** 2).sum())
    sigma2 = ssr / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t_stats = np.divide(beta_hat, se, out=np.full_like(beta_hat, np.nan), where=se > 0)

    sst = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - ssr / sst if sst > 0 else np.nan

    betas = {factor_cols[i]: (float(beta_hat[i + 1]), float(t_stats[i + 1]))
              for i in range(len(factor_cols))}

    return dict(alpha=float(beta_hat[0]), alpha_t_stat=float(t_stats[0]),
                betas=betas, n_obs=n, r_squared=r_squared)


def event_alpha(df: pd.DataFrame, event_col: str, factor_df: pd.DataFrame,
                 direction: int = 1, horizon: int = 20, close_col: str = "close",
                 factor_cols: tuple[str, ...] = ("SMB", "HML", "WML"),
                 entry_lag: int = 0, entry_col: str | None = None,
                 cost_bps: float = 0.0, slippage_bps: float | pd.Series = 0.0) -> dict:
    """
    factor_adjust()를 event_study()와 같은 인터페이스(df + event_col)로 바로 쓸 수
    있게 감싼 편의 함수. 5개 지표의 이벤트 컬럼을 이 함수에 그대로 넣으면 된다.

    df, event_col, direction, horizon, close_col: event_study()와 동일한 의미
        (df[event_col]이 True인 날짜에서 horizon봉 뒤까지의 순방향수익률을 direction
        방향으로 계산).
    entry_lag, entry_col, cost_bps, slippage_bps: forward_return()과 동일한
        의미(진입 지연·진입가 컬럼·거래비용·슬리피지). 기본값은 이전 버전과
        동일(당일 종가 즉시 진입, 비용·슬리피지 0)이므로 event_study()와 동일한
        실제 체결 가정을 쓰려면 여기도 같은 값을 넘겨야 한다 — 그렇지 않으면
        [3]단계는 비용을 반영하고 [5]단계는 반영하지 않는 불일치가 생긴다.
    factor_cols: factor_adjust()에 그대로 전달할 팩터 열 이름. 모멘텀 필터([2]
        단계)로 이미 걸러낸 신호를 검증할 때 WML(모멘텀 팩터)까지 회귀에 넣으면
        모멘텀 효과가 [2]단계 필터와 [5]단계 회귀 양쪽에서 이중으로 차감돼 알파가
        과소평가될 수 있다 — 이럴 때는 호출부에서 factor_cols=("SMB","HML")처럼
        WML을 뺀 3팩터(사실상 시장 요인은 별도 취급하지 않으므로 Fama-French
        2팩터에 가까움) 조합을 넘기는 것을 권장한다.
    factor_df: factor_adjust()에 그대로 전달.

    반환: factor_adjust()와 동일한 dict.
    """
    idx = df.index[df[event_col].fillna(False)]
    fwd = forward_return(df, horizon, direction=direction, close_col=close_col,
                          entry_lag=entry_lag, entry_col=entry_col,
                          cost_bps=cost_bps, slippage_bps=slippage_bps)
    vals = fwd.loc[idx].dropna()
    return factor_adjust(vals.index, vals.values, factor_df, factor_cols=factor_cols)
