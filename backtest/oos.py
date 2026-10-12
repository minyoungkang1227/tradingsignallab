"""
표본 외(Out-of-Sample) 검증 — 학습 구간에서 고른 신호가 한 번도 보지 못한 구간에서도 유지되는가.

[왜 필요한가]
compare_all()의 BH·DSR은 "이번에 시험한 신호들 중 고른 것"의 선택 편의를 통계적으로
깎아주지만, 같은 데이터로 고르고 같은 데이터로 평가하는 구조 자체는 그대로다. 신호를
고르는 데 쓴 기간과 성과를 확인하는 기간을 완전히 갈라야 "과거에만 통한 패턴"과
"앞으로도 통할 패턴"을 구별할 수 있다. 이 모듈이 그 구조를 저장소 본체에 넣는다.

[절차]
  1) 퍼지(purge)된 학습/검증 구간을 정한다(purged_windows).
  2) 학습 구간의 이벤트만으로 compare_all()을 돌려 통과 신호를 고른다(select_train_signals).
     기본 선택 규칙은 알림과 같다: 판정 보유기간에서 BH 보정 통과 AND DSR 통과.
  3) 검증 구간의 이벤트만으로 다시 계산하고, 고른 신호가 유지되는지 판정한다(judge_oos).
  4) 선택 못 한 신호까지 포함해 학습/검증 성적의 순위가 같은 방향인지 본다(rank_stability).

[퍼지가 필요한 이유]
학습 구간 막바지 이벤트의 순방향 수익률은 (보유기간 + 진입지연)봉 뒤까지 가격을
쓴다. 그 가격이 검증 구간에 걸치면 학습 통계 안에 검증 구간의 정보가 들어간다.
그래서 학습 종료일에서 (최대 보유기간 + 진입지연)봉 앞까지의 이벤트만 쓴다.

[검증 구간 판정 기준]
  - 검증 이벤트가 min_n(기본 10)건 이상이어야 판정한다(미달이면 '판정불가').
  - 평균수익이 양수이고 단측 p < alpha/k 여야 '유지됨'. k는 학습에서 선택된 신호 수
    (Bonferroni). 선택된 신호가 여럿이면 그중 하나쯤은 운으로 통과하기 쉬우므로 기준을
    그만큼 엄격하게 잡는다. 수익률은 방향 조정이 끝난 값이라 기대 부호는 항상 양수.

[한계]
종목 하나·짧은 기간에서는 학습 구간 신호별 이벤트가 DSR 최소치(20건)에 못 미쳐
선택되는 신호가 0개일 수 있다. 그때는 '신호가 무의미하다'가 아니라 '이 데이터로는
판단할 수 없다'가 결론이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OOSWindows:
    train_lo: pd.Timestamp      # 학습 이벤트 첫 날(워밍업 이후)
    train_hi: pd.Timestamp      # 학습 이벤트 마지막 날(퍼지 반영)
    test_lo: pd.Timestamp       # 검증 이벤트 첫 날
    test_hi: pd.Timestamp       # 검증 이벤트 마지막 날(데이터 끝)
    n_train_bars: int
    n_test_bars: int
    purged_bars: int            # 학습 끝에서 버린 봉 수 = 최대 보유기간 + 진입지연


@dataclass
class OOSResult:
    windows: OOSWindows
    judge_horizon: int
    summary_train: pd.DataFrame
    summary_test: pd.DataFrame
    selected: pd.DataFrame      # 학습에서 선택된 신호 + 검증 판정 열
    joined: pd.DataFrame        # 학습/검증 나란히 놓은 전체 표
    k: int
    alpha_adj: float            # alpha / k (k=0이면 NaN)
    rank_corr: float
    n_rank_rows: int
    removed_train: int
    removed_test: int
    params: dict = field(default_factory=dict)


def purged_windows(index, train_end, test_start, max_horizon: int, entry_lag: int = 0,
                   warmup_bars: int = 0) -> OOSWindows:
    """
    index: 가격 시계열의 DatetimeIndex(오름차순).
    train_end: 학습 구간의 마지막 날(이 날 이하까지가 학습). test_start: 검증 구간 첫 날.
    max_horizon / entry_lag: 퍼지 폭 = max_horizon + entry_lag 봉.
    warmup_bars: 지표가 안정되기 전 초기 봉은 학습 이벤트에서 뺀다.
    """
    idx = pd.DatetimeIndex(index)
    t_end, t_start = pd.Timestamp(train_end), pd.Timestamp(test_start)
    if t_start <= t_end:
        raise ValueError(f"검증 시작일({t_start.date()})은 학습 종료일({t_end.date()})보다 뒤여야 합니다 "
                         f"— 구간이 겹치면 표본 외 검증이 아닙니다.")
    end_pos = int(idx.searchsorted(t_end, side="right")) - 1
    if end_pos < 0:
        raise ValueError("학습 종료일이 데이터 시작보다 앞입니다.")
    purge = int(max_horizon) + int(entry_lag)
    hi_pos = end_pos - purge
    lo_pos = int(warmup_bars)
    if hi_pos <= lo_pos:
        raise ValueError(f"학습 구간이 너무 짧습니다: 워밍업 {warmup_bars}봉과 퍼지 {purge}봉을 빼면 "
                         f"학습 이벤트 구간이 남지 않습니다(학습 봉 {end_pos + 1}개).")
    test_pos = int(idx.searchsorted(t_start, side="left"))
    if test_pos >= len(idx):
        raise ValueError("검증 시작일 이후 데이터가 없습니다.")
    return OOSWindows(train_lo=idx[lo_pos], train_hi=idx[hi_pos], test_lo=idx[test_pos], test_hi=idx[-1],
                      n_train_bars=hi_pos - lo_pos + 1, n_test_bars=len(idx) - test_pos, purged_bars=purge)


def mask_events(dfs_by_indicator: dict, signal_specs, lo, hi, blocked=None):
    """
    [lo, hi] 밖의 이벤트(와 blocked 날짜의 이벤트)를 False로 가린 사본을 만든다.
    가격 열은 건드리지 않는다 — 순방향 수익률은 전체 가격을 그대로 써야 하고, 이벤트만
    구간으로 제한하는 것이 학습/검증 분리의 정의이기 때문이다.
    반환: (가린 dfs 사본, 구간 안인데 blocked 때문에 빠진 이벤트 수)
    """
    out = {k: v.copy() for k, v in dfs_by_indicator.items()}
    blocked_idx = pd.DatetimeIndex(blocked) if blocked is not None else pd.DatetimeIndex([])
    removed = 0
    for key, col, _direction, _label in signal_specs:
        if key not in out or col not in out[key].columns:
            continue
        d = out[key]
        ev = d[col].fillna(False).astype(bool)
        in_win = (d.index >= pd.Timestamp(lo)) & (d.index <= pd.Timestamp(hi))
        is_blocked = d.index.isin(blocked_idx)
        removed += int((ev & in_win & is_blocked).sum())
        d[col] = ev & in_win & ~is_blocked
    return out, removed


def attach_n_obs(summary: pd.DataFrame, results: dict) -> pd.DataFrame:
    """보유기간별 실제 관측 수(n_obs)를 붙인다. n_events는 신호 전체 이벤트 수라 보유기간마다
    순방향 수익률이 계산되는 개수와 다를 수 있다(시계열 끝 이벤트는 미래가 없어 빠진다)."""
    s = summary.copy()
    s["n_obs"] = [int(results[sig].table.loc[h, "n"]) if (sig in results and h in results[sig].table.index) else 0
                  for sig, h in zip(s["signal"], s["horizon"])]
    return s


def one_sided_p(mean, p_two):
    """양측 p를 '평균이 양수'라는 방향의 단측 p로 바꾼다."""
    if mean is None or p_two is None or pd.isna(mean) or pd.isna(p_two):
        return np.nan
    return p_two / 2.0 if mean > 0 else 1.0 - p_two / 2.0


def judge_oos(n, mean, p_two, k: int, min_n: int = 10, alpha: float = 0.05):
    """
    검증 구간 판정. 반환 (verdict, 단측 p, 보정 유의수준 alpha/k).
    verdict: '유지됨' / '유지 안 됨' / '판정불가'.
    """
    alpha_adj = alpha / k if k and k > 0 else np.nan
    p1 = one_sided_p(mean, p_two)
    if n is None or pd.isna(n) or n < min_n or pd.isna(p1):
        return "판정불가", p1, alpha_adj
    if mean > 0 and p1 < alpha_adj:
        return "유지됨", p1, alpha_adj
    return "유지 안 됨", p1, alpha_adj


def select_train_signals(summary_train: pd.DataFrame, judge_horizon: int, exclude=(),
                         require_barrier: bool = False) -> pd.DataFrame:
    """판정 보유기간에서 BH 통과 AND DSR 통과한 행(알림 게이트와 같은 규칙).
    require_barrier=True면 배리어 기반 DSR(tb_significant_dsr)도 요구한다 — 손절을 걸고도
    통과하는 신호만 남긴다."""
    need = ["significant_bh", "significant_dsr"] + (["tb_significant_dsr"] if require_barrier else [])
    missing = [c for c in need if c not in summary_train.columns]
    if missing:
        raise ValueError(f"요약표에 {missing} 컬럼이 없습니다 — compare_all에 deflate=True"
                         f"{', barrier=True' if require_barrier else ''}가 필요합니다.")
    row = summary_train[summary_train["horizon"] == judge_horizon]
    ok = row["significant_bh"].fillna(False).astype(bool) & row["significant_dsr"].fillna(False).astype(bool)
    if require_barrier:
        ok &= row["tb_significant_dsr"].fillna(False).astype(bool)
    ok &= ~row["signal"].isin(list(exclude))
    return row[ok]


def rank_stability(joined: pd.DataFrame, min_n: int = 5, min_rows: int = 10):
    """학습/검증 평균수익의 순위상관(스피어만). 양쪽 모두 관측 min_n 이상인 행만 쓰고,
    행이 min_rows 미만이면 계산하지 않는다(NaN). 반환 (상관, 사용 행 수)."""
    d = joined[(joined["tr_n_obs"] >= min_n) & (joined["te_n_obs"] >= min_n)]
    d = d.dropna(subset=["tr_mean_return_pct", "te_mean_return_pct"])
    if len(d) < min_rows:
        return np.nan, int(len(d))
    return float(d["tr_mean_return_pct"].rank().corr(d["te_mean_return_pct"].rank())), int(len(d))


def _join_train_test(sum_tr: pd.DataFrame, sum_te: pd.DataFrame) -> pd.DataFrame:
    keep = ["signal", "horizon", "n_obs", "mean_return_pct", "win_rate_pct", "t_stat", "p_value"]
    extra_tr = [c for c in ("significant_bh", "deflated_sr", "significant_dsr", "tb_significant_dsr",
                            "tb_mean_ret_pct", "tb_expectancy_r") if c in sum_tr.columns]
    extra_te = [c for c in ("tb_mean_ret_pct", "tb_expectancy_r") if c in sum_te.columns]
    tr = sum_tr[keep + extra_tr].rename(columns=lambda c: c if c in ("signal", "horizon") else "tr_" + c)
    te = sum_te[keep + extra_te].rename(columns=lambda c: c if c in ("signal", "horizon") else "te_" + c)
    return tr.merge(te, on=["signal", "horizon"], how="outer")


def oos_study(dfs_by_indicator: dict, compare_kwargs: dict, train_end, test_start,
              judge_horizon: int | None = None, warmup_bars: int = 60, min_test_n: int = 10,
              alpha: float = 0.05, blocked=None, exclude_signals=(), require_barrier: bool = False,
              min_n_rank: int = 5, min_rows_rank: int = 10) -> OOSResult:
    """
    dfs_by_indicator: run_backtest.run()이 돌려주는 {지표: 신호가 붙은 df}(전체 기간).
    compare_kwargs: compare_all()에 그대로 넘길 인자(horizons, entry_lag, cost_bps, deflate=True 등).
        알림·CLI와 같은 설정을 쓰려면 run_backtest.build_compare_kwargs(args, slippage)를 넘긴다.
    blocked: 신호를 제외할 날짜들(예: 데이터 소스 간 종가가 어긋난 날과 앞뒤 봉).
    """
    from backtest.engine import SIGNAL_SPECS, compare_all

    horizons = tuple(compare_kwargs["horizons"])
    if judge_horizon is None:
        judge_horizon = horizons[len(horizons) // 2]
    if judge_horizon not in horizons:
        raise ValueError(f"judge_horizon={judge_horizon}이 horizons {horizons}에 없습니다.")
    entry_lag = int(compare_kwargs.get("entry_lag", 0))

    any_df = next(iter(dfs_by_indicator.values()))
    win = purged_windows(any_df.index, train_end, test_start, max_horizon=max(horizons),
                         entry_lag=entry_lag, warmup_bars=warmup_bars)

    masked_tr, rm_tr = mask_events(dfs_by_indicator, SIGNAL_SPECS, win.train_lo, win.train_hi, blocked)
    masked_te, rm_te = mask_events(dfs_by_indicator, SIGNAL_SPECS, win.test_lo, win.test_hi, blocked)
    sum_tr, res_tr = compare_all(masked_tr, **compare_kwargs)
    sum_te, res_te = compare_all(masked_te, **compare_kwargs)
    if sum_tr.empty:
        raise ValueError("학습 구간에 이벤트가 하나도 없습니다.")
    sum_tr, sum_te = attach_n_obs(sum_tr, res_tr), (attach_n_obs(sum_te, res_te) if not sum_te.empty else sum_te)

    sel = select_train_signals(sum_tr, judge_horizon, exclude=exclude_signals, require_barrier=require_barrier)
    k = int(len(sel))
    joined = _join_train_test(sum_tr, sum_te) if not sum_te.empty else pd.DataFrame(
        columns=["signal", "horizon", "tr_n_obs", "tr_mean_return_pct", "te_n_obs", "te_mean_return_pct"])

    rows = []
    for r in sel.itertuples():
        te = joined[(joined["signal"] == r.signal) & (joined["horizon"] == judge_horizon)]
        te = te.iloc[0] if len(te) else None
        n = te["te_n_obs"] if te is not None else np.nan
        mean = te["te_mean_return_pct"] if te is not None else np.nan
        t = te["te_t_stat"] if te is not None else np.nan
        p2 = te["te_p_value"] if te is not None else np.nan
        verdict, p1, alpha_adj = judge_oos(n, mean, p2, k, min_n=min_test_n, alpha=alpha)
        rows.append(dict(signal=r.signal, tr_n_obs=int(r.n_obs), tr_mean_return_pct=r.mean_return_pct,
                         tr_t_stat=r.t_stat, tr_deflated_sr=getattr(r, "deflated_sr", np.nan),
                         te_n_obs=n, te_mean_return_pct=mean, te_t_stat=t, te_one_sided_p=p1, verdict=verdict))
    selected = pd.DataFrame(rows)

    corr, n_rows = rank_stability(joined, min_n=min_n_rank, min_rows=min_rows_rank) if len(joined) else (np.nan, 0)
    return OOSResult(windows=win, judge_horizon=judge_horizon, summary_train=sum_tr, summary_test=sum_te,
                     selected=selected, joined=joined, k=k, alpha_adj=(alpha / k if k else np.nan),
                     rank_corr=corr, n_rank_rows=n_rows, removed_train=rm_tr, removed_test=rm_te,
                     params=dict(train_end=str(pd.Timestamp(train_end).date()),
                                 test_start=str(pd.Timestamp(test_start).date()), warmup_bars=warmup_bars,
                                 min_test_n=min_test_n, alpha=alpha, require_barrier=require_barrier,
                                 min_n_rank=min_n_rank))


def format_oos_report(res: OOSResult, explore_top: int = 6) -> str:
    """OOSResult를 사람이 읽는 보고서로. 결론에는 항상 근거(표본 수, t값, 기준)를 같이 적는다."""
    w, p = res.windows, res.params
    L = []
    L.append("=== 표본 외(OOS) 검증 ===")
    L.append(f"학습 이벤트 구간 {w.train_lo.date()} ~ {w.train_hi.date()} ({w.n_train_bars}봉, "
             f"워밍업 {p['warmup_bars']}봉 제외, 마지막 {w.purged_bars}봉 퍼지)")
    L.append(f"검증 이벤트 구간 {w.test_lo.date()} ~ {w.test_hi.date()} ({w.n_test_bars}봉)")
    if res.removed_train or res.removed_test:
        L.append(f"제외한 신호 이벤트(차단 날짜): 학습 {res.removed_train}건, 검증 {res.removed_test}건")
    L.append(f"판정 보유기간 {res.judge_horizon}봉 | 선택 규칙: 학습에서 BH 통과 AND DSR 통과"
             f"{' AND 배리어 DSR 통과' if p['require_barrier'] else ''}")
    L.append(f"학습에서 선택된 신호: {res.k}개")
    cols = ["signal", "tr_n_obs", "tr_mean_return_pct", "tr_t_stat", "te_n_obs",
            "te_mean_return_pct", "te_t_stat", "te_one_sided_p", "verdict"]
    if res.k:
        L.append(f"검증 기준: 이벤트 >= {p['min_test_n']}, 평균 > 0, 단측 p < {p['alpha']}/{res.k} = {res.alpha_adj:.4f}")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            L.append(res.selected[cols].to_string(index=False))
        kept = int((res.selected["verdict"] == "유지됨").sum())
        cant = int((res.selected["verdict"] == "판정불가").sum())
        L.append(f"결론: 학습 통과 {res.k}개 중 검증에서 유지 {kept}개, 판정불가 {cant}개.")
    else:
        elig = int((res.summary_train[res.summary_train["horizon"] == res.judge_horizon]["n_obs"] >= 20).sum())
        L.append(f"결론: 학습 구간에서 기준을 통과한 신호가 없어 검증 대상이 없습니다 "
                 f"(이 보유기간에서 이벤트 20건 이상이라 DSR 판정이 가능했던 신호 {elig}개).")
        L.append("      '신호가 무의미하다'는 증거가 아니라 '이 표본으로는 판단할 수 없다'는 결과입니다.")
        j = res.joined[(res.joined["horizon"] == res.judge_horizon) & (res.joined["tr_n_obs"] >= 5)]
        if len(j) and "te_n_obs" in j.columns:
            j = j.sort_values("tr_t_stat", ascending=False).head(explore_top)
            L.append("[탐색용 후보 — 통과 기준 미달, 검증이 아님]")
            with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
                L.append(j[["signal", "tr_n_obs", "tr_mean_return_pct", "tr_t_stat", "te_n_obs",
                            "te_mean_return_pct", "te_t_stat"]].to_string(index=False))
    if np.isfinite(res.rank_corr):
        L.append(f"[안정성] 학습 vs 검증 평균수익 순위상관 {res.rank_corr:+.2f} (양쪽 이벤트 {p['min_n_rank']}건 이상인 "
                 f"신호x보유기간 {res.n_rank_rows}개). 0에 가까우면 학습 성적이 검증을 예측하지 못한다는 뜻.")
    else:
        L.append(f"[안정성] 비교 가능한 행이 {res.n_rank_rows}개뿐이라 순위상관을 계산하지 않았습니다(표본 부족).")
    return "\n".join(L)
