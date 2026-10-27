"""
체결 로그 레이어 -- 메타라벨링(meta_labeling.py)이 요구하는 "신호별 실제 거래 결과"를
signal_id 단위로 기록한다. (메타라벨링 통합 1단계)

[중요한 한계 -- 반드시 읽을 것]
이 레이어는 지금 단계에서는 event_study()/event_returns()가 계산하는 "이벤트 발생 후
N봉 고정 보유 수익률"을 거래 결과로 기록한다. 즉 손절(stop_loss)·시간제한청산
(time_limit)·circuit breaker 같은 실제 리스크관리 청산 사유는 여기서 나오지 않는다 --
그건 quant-reversal-system의 RiskManagedTrader가 실제(혹은 페이퍼) 매매를 돌려야만
생기는 정보다. meta_labeling.py의 BAD_EXIT_REASONS={"stop_loss","time_limit",
"circuit_breaker"}를 제대로 채우려면 결국 RiskManagedTrader 연동(3단계 이후 작업)이
필요하다.

그래서 이 레이어는 exit_reason을 두 가지 값으로만 채운다:
  - "horizon_exit": 정상적으로 N봉 보유 후 청산(지금 쓸 수 있는 유일한 사유)
  - 실제 리스크관리 청산 사유(stop_loss 등)는 RiskManagedTrader 연동 후에만 채워진다.

지금 당장은 "과거 신호 이벤트들을 signal_id로 색인해서 재사용 가능한 테이블로
남긴다"는 배관(plumbing)에 집중한다 -- FEATURE_SPEC 연결(2단계)과 실거래 연동(3단계
이후)은 다음 작업이다.

설계 결정(사용자 확인 완료):
  - signal_id: 심볼+날짜+신호라벨 조합을 해시한 결정적(deterministic) 값.
    같은 (symbol, date, label) 입력이면 재실행·재현해도 항상 같은 id가 나오므로
    카운터나 DB auto-increment 같은 별도 상태를 유지할 필요가 없다.
  - 저장 위치: CSV 파일(append, signal_id 기준 중복 방지).
  - 기존 backtest/engine.py와의 관계: event_study()/compare_all()의 통계 로직은
    전혀 건드리지 않고, event_returns()가 이미 계산한 개별 이벤트 수익률 위에
    signal_id 발급 + entry/exit 가격 + exit_reason만 덧씌우는 래퍼 레이어로 둔다.
"""
from __future__ import annotations

import hashlib
import os

import pandas as pd

from backtest.engine import event_returns, SIGNAL_SPECS

EXECUTION_LOG_COLUMNS = [
    "signal_id", "symbol", "date", "label", "direction", "horizon",
    "entry_price", "exit_price", "pnl", "exit_reason", "basis",
]


def generate_signal_id(symbol: str, date, label: str) -> str:
    """심볼+날짜+신호라벨 조합을 해시해 결정적 signal_id를 만든다."""
    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    raw = f"{symbol}|{date_str}|{label}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def log_event_study_trades(df: pd.DataFrame, event_col: str, direction: int, label: str,
                            symbol: str, horizon: int, close_col: str = "close",
                            entry_lag: int = 0, entry_col: str | None = None,
                            cost_bps: float = 0.0, slippage_bps=0.0,
                            drop_overlapping: bool = False) -> pd.DataFrame:
    """
    event_study()와 완전히 같은 정의(같은 이벤트 필터링·entry_lag·cost·slippage)로
    "이 신호가 이 날짜에 떴고 N봉 뒤 이런 수익이 났다"는 개별 이벤트들을 signal_id
    단위 행으로 변환한다. 통계 계산(event_returns)은 그대로 재사용하고, 그 위에
    signal_id 발급 + entry/exit 가격 + exit_reason만 덧붙인다.
    """
    rets = event_returns(df, event_col, horizon, direction=direction, close_col=close_col,
                          entry_lag=entry_lag, entry_col=entry_col, cost_bps=cost_bps,
                          slippage_bps=slippage_bps, drop_overlapping=drop_overlapping)
    if len(rets) == 0:
        return pd.DataFrame(columns=EXECUTION_LOG_COLUMNS)

    close = df[close_col]
    entry_col_name = entry_col or close_col
    n = len(df)
    rows = []
    for event_date, pnl in rets.items():
        pos = df.index.get_loc(event_date)
        entry_pos = pos if entry_lag == 0 else pos + entry_lag
        exit_pos = pos + horizon
        if entry_pos < 0 or entry_pos >= n or exit_pos >= n:
            continue
        entry_price = df[entry_col_name].iloc[entry_pos]
        exit_price = close.iloc[exit_pos]
        rows.append(dict(
            signal_id=generate_signal_id(symbol, event_date, label),
            symbol=symbol, date=pd.Timestamp(event_date).strftime("%Y-%m-%d"),
            label=label, direction=direction, horizon=horizon,
            entry_price=float(entry_price), exit_price=float(exit_price),
            pnl=float(pnl), exit_reason="horizon_exit", basis="backtest_event_study",
        ))
    return pd.DataFrame(rows, columns=EXECUTION_LOG_COLUMNS)


def append_to_log(rows: pd.DataFrame, path: str) -> int:
    """CSV에 append하되 이미 기록된 signal_id는 중복 기록하지 않는다(재실행 안전).
    반환값은 새로 추가된 행 수."""
    if rows.empty:
        return 0
    file_exists = os.path.exists(path)
    if file_exists:
        existing_ids = set(pd.read_csv(path, usecols=["signal_id"])["signal_id"])
        new_rows = rows[~rows["signal_id"].isin(existing_ids)]
    else:
        new_rows = rows
    if new_rows.empty:
        return 0
    new_rows.to_csv(path, mode="a", header=not file_exists, index=False)
    return len(new_rows)


def log_all_signals(dfs_by_indicator: dict[str, pd.DataFrame], symbol: str, horizon: int,
                     path: str, entry_lag: int = 0, entry_col: str | None = None,
                     cost_bps: float = 0.0, slippage_bps=0.0,
                     drop_overlapping: bool = False) -> int:
    """SIGNAL_SPECS 전체를 순회하며 한 종목의 체결 로그를 한 번에 쌓는다.
    run_backtest.py에서 호출할 상위 진입점. 새로 추가된 행 수를 반환한다."""
    total_new = 0
    for indicator_key, event_col, direction, label in SIGNAL_SPECS:
        ind_df = dfs_by_indicator.get(indicator_key)
        if ind_df is None or event_col not in ind_df.columns:
            continue
        rows = log_event_study_trades(ind_df, event_col, direction, label, symbol, horizon,
                                       entry_lag=entry_lag, entry_col=entry_col,
                                       cost_bps=cost_bps, slippage_bps=slippage_bps,
                                       drop_overlapping=drop_overlapping)
        total_new += append_to_log(rows, path)
    return total_new
