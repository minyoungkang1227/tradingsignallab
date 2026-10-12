"""
트리플 배리어 라벨링(López de Prado, "Advances in Financial Machine Learning" 3장).

[왜 필요한가 — 고정 보유기간 수익률의 구조적 거짓말]

지금 engine.py의 forward_return()은 "진입가 대비 정확히 h봉 뒤 종가"만 본다.
그 사이의 경로(path)는 통째로 무시한다. 문제는 이것이 실제 매매와 다르다는
점이다. 예를 들어 보유기간 20봉으로 측정했을 때

    진입 -> 3봉 만에 -18% -> 이후 회복 -> 20봉째 +2%

이 거래는 현재 집계에서 "+2% 성공"으로 들어간다. 그런데 실제로는 손절선에
걸려 -18%에서 잘려나갔을 것이고, 설령 손절이 없었어도 -18% 구간을 버틸 수
있었을지는 별개 문제다. 즉 고정 보유기간 수익률은 손절을 쓰는 매매의 성과를
체계적으로 과대평가한다. 평균수익률과 t-stat이 아무리 유의해도, 그 숫자가
측정한 대상 자체가 실제로 집행 가능한 전략이 아니다.

트리플 배리어는 매 거래를 세 개의 경계 중 "먼저 닿는 것"으로 청산한다.

    상단(익절)  : +pt_mult * 변동성
    하단(손절)  : -sl_mult * 변동성
    수직(시간)  : max_horizon 봉 경과

이러면 측정 대상이 "h봉 뒤 수익률"에서 "손절을 걸고 매매했을 때 실현 수익률"로
바뀐다. 이건 지표를 하나 더 추가하는 종류의 변경이 아니라, 무엇을 측정하는지
자체를 바꾸는 변경이다.

[변동성으로 경계를 스케일링하는 이유]

손절폭을 "-5% 고정"처럼 절대값으로 두면, 일변동성 1%인 종목과 5%인 종목에
같은 기준을 적용하게 된다. 전자는 거의 안 닿고 후자는 노이즈에 바로 잘린다.
그래서 경계를 그 종목·그 시점의 실현변동성 배수로 잡는다 — 같은 pt_mult=2.0이
어느 종목에서나 "평소 변동의 2배"라는 동일한 의미를 갖게 된다.

[이 구현이 보수적으로 잡은 지점 두 곳]

1) 같은 봉 안에서 익절선과 손절선이 둘 다 닿았으면 손절을 먼저 맞은 것으로
   본다. 일봉 데이터만으로는 봉 내부 순서를 알 수 없고, 유리한 쪽을 가정하면
   성과가 조직적으로 부풀려지기 때문이다.

2) 갭(gap) 처리 — 시가가 이미 경계를 넘어서 열렸으면 경계값이 아니라 그
   시가로 체결됐다고 본다. 국내 주식은 갭이 잦아서 "손절선에 정확히 체결"
   가정은 실제 손실을 체계적으로 과소평가한다. 이 처리를 넣으면 손절 거래의
   실현손실이 설정한 손절폭보다 나빠질 수 있는데, 그게 현실이다.

[파라미터를 고를 때 꼭 알아야 할 상호작용 — sl_mult와 use_intrabar]

변동성은 "종가 대비 종가" 기준으로 재는데, 배리어 터치는 use_intrabar=True면
봉 내부 고가/저가로 판정한다. 두 기준이 다르기 때문에, 손절선을 1시그마로
잡으면 생각보다 훨씬 자주 잘린다 — 하루 치 봉 내부 변동만으로도 종가기준
1시그마를 넘는 일이 흔하기 때문이다.

합성데이터(봉 내부 범위가 종가변동성의 0.88배)로 측정한 예, max_horizon=10:

    use_intrabar=True,  sl=1시그마 -> 손절률 64%, 평균보유 2.5봉
    use_intrabar=False, sl=1시그마 -> 손절률 50%, 평균보유 4.1봉
    use_intrabar=True,  sl=3시그마 -> 손절률 32%, 평균보유 7.4봉

실제 시장은 합성데이터보다 봉 내부 범위가 종가변동성 대비 더 큰 게 보통이라
이 효과가 더 강하게 나타난다. "최대 20봉 보유" 설정으로 돌렸는데 평균 보유가
2~3봉이면 손절이 지나치게 타이트한 것이니 sl_mult를 키우거나
use_intrabar=False를 고려해야 한다. 기본값 sl_mult=1.0은 "타이트한 손절"
쪽이라는 점을 의식하고 쓸 것.

[한계 — 이것도 여전히 진짜 백테스트는 아니다]

동시에 여러 포지션을 들고 있을 때의 자본 배분, 포지션 사이징, 거래 가능
수량(유동성) 제약은 여전히 반영하지 않는다. 여기까지는 "신호 1건을 단독으로
손절 걸고 잡았을 때 얼마가 실현되는가"까지만 측정한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(df: pd.DataFrame, span: int = 20,
                         close_col: str = "close") -> pd.Series:
    """
    배리어 폭의 기준으로 쓸 실현변동성(봉당 수익률의 지수가중 표준편차).

    span: 지수가중 반감 구간. 20봉 기본값은 "최근 한 달쯤의 변동성"에 해당한다.

    단순이동평균이 아니라 지수가중을 쓰는 이유는 변동성 국면이 바뀔 때
    (조용하던 종목이 갑자기 요동치기 시작할 때) 더 빨리 따라가기 위함이다.

    주의: shift(1)을 적용해 "그 봉 시점에 이미 알 수 있었던 변동성"만 쓴다.
    당일 변동성을 그대로 쓰면 미래 정보를 배리어 설정에 끌어쓰는 셈이 된다.
    """
    ret = df[close_col].pct_change()
    return ret.ewm(span=span, min_periods=max(2, span // 2)).std().shift(1)


def triple_barrier_returns(df: pd.DataFrame, event_idx: pd.Index, direction: int = 1,
                            pt_mult: float = 2.0, sl_mult: float = 1.0,
                            max_horizon: int = 20, vol: pd.Series | None = None,
                            vol_span: int = 20, close_col: str = "close",
                            entry_lag: int = 0, entry_col: str | None = None,
                            cost_bps: float = 0.0,
                            slippage_bps: float | pd.Series = 0.0,
                            use_intrabar: bool = True) -> pd.DataFrame:
    """
    이벤트들을 트리플 배리어로 청산했을 때의 실현 수익률을 계산한다.

    df: OHLCV DataFrame(최소 close_col 필요. use_intrabar=True면 high/low도 사용).
    event_idx: 이벤트 발생 시점 인덱스(engine.event_returns()가 쓰는 것과 동일).
    direction: +1 롱, -1 숏.
    pt_mult / sl_mult: 익절/손절 경계를 변동성의 몇 배로 둘지. sl_mult=1.0,
        pt_mult=2.0이면 "1시그마 손절, 2시그마 익절"(손익비 2:1).
        0 이하로 주면 그 경계를 비활성화한다(예: sl_mult=0 -> 손절 없음).
    max_horizon: 수직(시간) 배리어. 이 봉 수가 지나면 종가로 청산.
    vol: 배리어 스케일에 쓸 변동성 Series. None이면 realized_volatility()로 계산.
    entry_lag / entry_col / cost_bps / slippage_bps: engine.forward_return()과
        완전히 동일한 의미·관례(기본값도 동일).
    use_intrabar: True면 high/low로 봉 내부 터치를 판정(현실적), False면 종가만
        본다(보수적으로 덜 잘림 — 일봉 high/low를 못 믿는 경우용).

    반환 DataFrame (index=실제 진입이 가능했던 이벤트 시점):
        entry_date   진입 봉
        exit_date    청산 봉
        bars_held    보유 봉 수
        ret          비용·슬리피지 차감 후 실현 수익률(방향 조정 완료)
        exit_reason  'pt'(익절) / 'sl'(손절) / 'time'(시간만료)
        target_vol   그 이벤트에 적용된 변동성(배리어 폭의 기준값)

    데이터가 모자라 진입/청산을 끝까지 따라갈 수 없는 이벤트는 제외된다
    (시계열 끝부분에서 max_horizon만큼 미래가 없는 경우).
    """
    if vol is None:
        vol = realized_volatility(df, span=vol_span, close_col=close_col)

    close = df[close_col]
    has_hl = use_intrabar and "high" in df.columns and "low" in df.columns
    highs = df["high"] if has_hl else close
    lows = df["low"] if has_hl else close
    opens = df["open"] if "open" in df.columns else close

    if isinstance(slippage_bps, pd.Series):
        slip_series = slippage_bps.reindex(df.index).fillna(0.0)
    else:
        slip_series = None

    pos_of = {ts: i for i, ts in enumerate(df.index)}
    n_bars = len(df)
    rows = []

    for ts in event_idx:
        i = pos_of.get(ts)
        if i is None:
            continue

        # --- 진입 시점/가격 (forward_return과 동일한 관례) ---
        entry_i = i + entry_lag
        if entry_i >= n_bars:
            continue
        if entry_lag == 0:
            entry_price = close.iloc[i] if entry_col is None else df[entry_col].iloc[i]
        else:
            col = entry_col or close_col
            entry_price = df[col].iloc[entry_i]

        if (not np.isfinite(entry_price)) or entry_price <= 0:
            continue

        # 변동성은 배리어 폭을 정하는 데만 쓰인다. 익절·손절을 둘 다 꺼둔
        # 경우(pt_mult=sl_mult=0, 즉 순수 고정 보유기간)에는 필요 없으므로
        # 변동성이 NaN이라는 이유로 이벤트를 버리지 않는다 — 버리면 시계열
        # 초반 이벤트가 까닭 없이 빠져서 forward_return과 표본이 달라진다.
        needs_vol = pt_mult > 0 or sl_mult > 0
        target_vol = vol.iloc[i] if i < len(vol) else np.nan
        if needs_vol and (not np.isfinite(target_vol) or target_vol <= 0):
            continue

        pt_level = pt_mult * target_vol if pt_mult > 0 else np.inf
        sl_level = -sl_mult * target_vol if sl_mult > 0 else -np.inf

        # --- 경로를 따라가며 먼저 닿는 경계를 찾는다 ---
        last_i = min(entry_i + max_horizon, n_bars - 1)
        if last_i <= entry_i:
            continue

        exit_i, exit_ret, reason = None, None, None

        for j in range(entry_i + 1, last_i + 1):
            # 방향 조정된 경로 수익률의 상/하단.
            # direction=+1이면 high가 유리·low가 불리, -1이면 반대가 되는데
            # direction을 곱한 뒤 max/min을 취하면 두 경우가 자동으로 처리된다.
            r_hi = (highs.iloc[j] / entry_price - 1.0) * direction
            r_lo = (lows.iloc[j] / entry_price - 1.0) * direction
            best, worst = max(r_hi, r_lo), min(r_hi, r_lo)
            r_open = (opens.iloc[j] / entry_price - 1.0) * direction

            # 손절을 먼저 검사한다(같은 봉에서 둘 다 닿았으면 손절 우선 — 보수적)
            if worst <= sl_level:
                # 갭: 시가가 이미 손절선을 넘어 열렸으면 그 시가로 체결
                exit_ret = min(r_open, sl_level) if np.isfinite(r_open) else sl_level
                exit_i, reason = j, "sl"
                break
            if best >= pt_level:
                # 갭: 시가가 이미 익절선을 넘어 열렸으면 그 시가로 체결(유리한 갭)
                exit_ret = max(r_open, pt_level) if np.isfinite(r_open) else pt_level
                exit_i, reason = j, "pt"
                break

        if exit_i is None:
            # 어느 경계도 안 닿음 -> 수직(시간) 배리어로 청산.
            # 단, 보유기간을 끝까지 채울 데이터가 없으면(시계열 끝) 결과를
            # 알 수 없으므로 버린다. 있는 데까지만 들고 청산한 것처럼 처리하면
            # 보유기간이 멋대로 짧아져 집계가 편향된다.
            # (배리어에 닿은 경우는 이미 위에서 결과가 확정됐으므로 그대로 쓴다 —
            #  시계열이 더 길었어도 그 거래는 거기서 끝났을 것이기 때문.)
            if entry_i + max_horizon > n_bars - 1:
                continue
            exit_i = last_i
            exit_ret = (close.iloc[last_i] / entry_price - 1.0) * direction
            reason = "time"

        if not np.isfinite(exit_ret):
            continue

        # --- 비용·슬리피지 차감 (forward_return과 동일하게 방향 무관 차감) ---
        slip = float(slip_series.iloc[i]) if slip_series is not None else float(slippage_bps)
        net_ret = exit_ret - cost_bps / 10000.0 - slip / 10000.0

        rows.append({
            "entry_date": df.index[entry_i],
            "exit_date": df.index[exit_i],
            "bars_held": exit_i - entry_i,
            "ret": net_ret,
            "exit_reason": reason,
            "target_vol": float(target_vol),
        })

    if not rows:
        return pd.DataFrame(columns=["entry_date", "exit_date", "bars_held", "ret",
                                      "exit_reason", "target_vol"])

    out = pd.DataFrame(rows)
    out.index = pd.Index([r["entry_date"] for r in rows], name=df.index.name or "date")
    return out


def barrier_summary(tb: pd.DataFrame) -> dict:
    """
    triple_barrier_returns() 결과를 한 줄 요약으로 압축한다.

    고정 보유기간 집계와 달리 "왜 나갔는지"(익절/손절/시간)의 분포가 핵심
    정보다 — 평균수익률이 같아도 익절로 나간 비율이 높은 전략과 시간만료로
    질질 끌다 나온 전략은 실제 운용 난이도가 전혀 다르다.
    """
    if tb is None or len(tb) == 0:
        return {"n": 0, "mean_ret": np.nan, "win_rate": np.nan, "sharpe": np.nan,
                "pt_rate": np.nan, "sl_rate": np.nan, "time_rate": np.nan,
                "avg_bars": np.nan}

    r = tb["ret"].to_numpy(dtype=float)
    std = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    counts = tb["exit_reason"].value_counts()
    n = len(tb)
    mean = float(r.mean())

    # 왜도·첨도(원첨도, 정규분포=3). 디플레이티드 샤프비율이 비정규성 보정에
    # 쓰므로 여기서 같이 계산해둔다(engine.event_study의 처리와 동일한 관례).
    if n > 1 and std and std > 0:
        z = (r - mean) / std
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4))
        t_stat = mean / (std / np.sqrt(n))
    else:
        skew = kurt = t_stat = np.nan

    return {
        "n": n,
        "mean_ret": mean,
        "win_rate": float((r > 0).mean()),
        "sharpe": float(mean / std) if std and std > 0 else np.nan,
        "t_stat": t_stat,
        "skew": skew,
        "kurt": kurt,
        "pt_rate": float(counts.get("pt", 0) / n),
        "sl_rate": float(counts.get("sl", 0) / n),
        "time_rate": float(counts.get("time", 0) / n),
        "avg_bars": float(tb["bars_held"].mean()),
    }


def barrier_study(dfs_by_indicator: dict, signal_specs, horizons=(1, 3, 5, 10, 20),
                   pt_mult: float = 2.0, sl_mult: float = 1.0, vol_span: int = 20,
                   entry_lag: int = 0, entry_col: str | None = None,
                   cost_bps: float = 0.0, slippage_bps: float | pd.Series = 0.0,
                   use_intrabar: bool = True) -> pd.DataFrame:
    """
    SIGNAL_SPECS의 모든 (신호, 방향)에 대해, 각 보유기간을 "시간 배리어"로 삼아
    손절/익절을 걸고 매매했을 때의 실현 수익률을 집계한다.

    engine.compare_all()이 고정 보유기간으로 만드는 요약표와 같은 모양
    (signal x horizon 한 행)으로 돌려주므로, 두 표를 나란히 붙이면 "같은 신호를
    손절 걸고 매매했을 때 어떻게 달라지는가"를 바로 비교할 수 있다.

    [horizon을 시간 배리어로 쓰는 이유]
    고정 보유기간 h와 "최대 h봉까지 들고 있되 손절/익절에 닿으면 먼저 나감"은
    같은 시간 예산을 쓰는 두 가지 운용 방식이다. 같은 h로 맞춰야 "경로를 보느냐
    안 보느냐"의 차이만 분리해서 볼 수 있다. h를 다르게 잡으면 보유기간 효과와
    손절 효과가 섞여버린다.

    signal_specs: engine.SIGNAL_SPECS와 같은 형식의
        (indicator_key, event_col, direction, label) 리스트. 순환 import를
        피하려고 인자로 받는다.

    pt_mult / sl_mult / vol_span / entry_lag / entry_col / cost_bps /
    slippage_bps / use_intrabar: triple_barrier_returns()와 동일한 의미.

    반환 DataFrame 컬럼:
        signal, direction, horizon, tb_n, tb_mean_ret_pct, tb_win_rate_pct,
        tb_sharpe, tb_t_stat, tb_skew, tb_kurt, tb_pt_rate, tb_sl_rate,
        tb_time_rate, tb_avg_bars
    이벤트가 하나도 없는 조합은 행 자체를 만들지 않는다.
    """
    rows = []
    for indicator_key, event_col, direction, label in signal_specs:
        df = dfs_by_indicator.get(indicator_key)
        if df is None or event_col not in df.columns:
            continue

        event_idx = df.index[df[event_col].fillna(False)]
        if len(event_idx) == 0:
            continue

        # 변동성은 보유기간마다 다시 계산할 필요가 없으므로 신호당 한 번만 계산해
        # 모든 horizon에서 재사용한다(같은 변동성 기준을 써야 보유기간 간 비교가
        # 공정하다).
        vol = realized_volatility(df, span=vol_span)

        for h in horizons:
            tb = triple_barrier_returns(
                df, event_idx, direction=direction, pt_mult=pt_mult, sl_mult=sl_mult,
                max_horizon=h, vol=vol, entry_lag=entry_lag, entry_col=entry_col,
                cost_bps=cost_bps, slippage_bps=slippage_bps, use_intrabar=use_intrabar)
            s = barrier_summary(tb)
            if s["n"] == 0:
                continue
            rows.append({
                "signal": label,
                "direction": "LONG" if direction == 1 else "SHORT",
                "horizon": h,
                "tb_n": s["n"],
                "tb_mean_ret_pct": s["mean_ret"] * 100,
                "tb_win_rate_pct": s["win_rate"] * 100,
                "tb_sharpe": s["sharpe"],
                "tb_t_stat": s["t_stat"],
                "tb_skew": s["skew"],
                "tb_kurt": s["kurt"],
                "tb_pt_rate": s["pt_rate"],
                "tb_sl_rate": s["sl_rate"],
                "tb_time_rate": s["time_rate"],
                "tb_avg_bars": s["avg_bars"],
            })

    cols = ["signal", "direction", "horizon", "tb_n", "tb_mean_ret_pct",
            "tb_win_rate_pct", "tb_sharpe", "tb_t_stat", "tb_skew", "tb_kurt",
            "tb_pt_rate", "tb_sl_rate", "tb_time_rate", "tb_avg_bars"]
    return pd.DataFrame(rows, columns=cols)
