"""
반복 Walk-forward(확장창) 검증 — 표본 외 검증을 한 번이 아니라 여러 기간에 걸쳐 반복한다.

[왜 필요한가]
OOS 분할 한 번은 '운 좋은 검증 구간 하나'일 수 있다. 검증 구간이 하나면 통과해도 그 기간의 시장
환경 덕일 수 있고, 탈락해도 그 기간이 유독 나빴을 수 있다. 학습 끝을 앞으로 옮기며 같은 절차
(학습에서 BH+DSR로 신호 선택 -> 그 뒤 구간에서 유지되는지)를 반복하면 "이 신호가 시기와 무관하게
반복해서 통하는가"를 볼 수 있다.

[구조 — oos.oos_study 재사용]
확장창: 학습 시작은 고정, 학습 끝만 test_bars(기본 step)만큼 전진. 매 fold마다
  학습 = [워밍업 이후, train_end - 퍼지],  검증 = [train_end+1, train_end+test_bars]
검증 조각은 step=test_bars일 때 서로 겹치지 않는다. 마지막에 test_bars가 안 채워지는 조각은 쓰지 않는다
(짧은 검증은 판정이 불안정하다). 어떤 fold가 실패(학습 이벤트 없음 등)해도 나머지는 계속한다.

[집계 — 신호별로만 합친다]
같은 fold 안의 서로 다른 신호는 같은 날 같은 시장 움직임을 공유해 독립이 아니므로, 신호 간에는 합치지
않는다. 한 신호의 검증 결과는 fold 간에는(검증 구간이 겹치지 않으므로) 독립에 가깝다 — 단 fold 경계 부근
이벤트의 순방향 수익률 창이 다음 구간에 걸칠 수 있어 완전한 독립은 아니다. 이를 이용해 신호별로
  합산 평균 = Σ n_i·m_i / Σ n_i,  합산 SE² = Σ n_i²·se_i² / (Σ n_i)²   (se_i = |m_i / t_i|)
로 합쳐 단측 z·p를 구한다(se_i는 compare_all이 낸 NW·겹침 보정 t에서 역산).
다중검정: 한 번이라도 선택된 서로 다른 신호 수 m으로 Bonferroni(alpha/m).
fold 간 이질성(Cochran Q): 합산 평균이 유의해도 우위가 한 기간에만 있고 나머지에선 사라졌다면 합산이 그
사실을 가린다. 그래서 Q = Σ (m_i − m̄)² / se_i² (m̄는 역분산 가중평균)를 구해 자유도 K−1의 카이제곱
꼬리확률 p_het를 본다. p_het < het_alpha(기본 0.05)이면 fold마다 성과가 유의하게 다르다는 뜻이다.
판정: '안정' = 합산 단측 p < alpha/m 이고 p_het >= het_alpha 이고 선택된 fold >= min_folds.
      '불균질' = 합산은 유의하지만 fold 간 성과가 유의하게 다름(특정 기간 덕일 수 있음).
      '불안정' = 충분히 관측됐는데 합산이 유의하지 않음.
      '판정불가' = 선택된 fold 수 또는 합산 표본이 부족.
주의: fold가 적으면 이질성 검정의 검출력이 낮아 '안정'이 곧 '균질함의 증명'은 아니다.

합성 데이터 결과는 실제 시장 성과가 아니다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame            # fold별 요약(상태·구간·선택 수·판정 개수)
    selected: pd.DataFrame         # (fold, 신호) 행: oos_study의 selected + fold
    by_signal: pd.DataFrame        # 신호별 fold 간 합산
    n_folds_planned: int
    n_folds_ok: int
    params: dict = field(default_factory=dict)


def make_folds(index, first_train_end, test_bars: int, step_bars: int | None = None) -> list:
    """
    [(train_end, test_start, test_end), ...] 날짜 목록. 확장창.
    test_bars 봉이 다 채워지는 검증 조각만 만든다. step 기본값=test_bars(겹침 없음).
    """
    idx = pd.DatetimeIndex(index)
    step = int(step_bars) if step_bars else int(test_bars)
    if test_bars < 1 or step < 1:
        raise ValueError("test_bars와 step_bars는 1 이상이어야 합니다.")
    e0 = int(idx.searchsorted(pd.Timestamp(first_train_end), side="right")) - 1
    if e0 < 0:
        raise ValueError("첫 학습 종료일이 데이터 시작보다 앞입니다.")
    folds, e = [], e0
    while e + test_bars <= len(idx) - 1:
        folds.append((idx[e], idx[e + 1], idx[e + test_bars]))
        e += step
    if not folds:
        raise ValueError(f"fold를 하나도 만들 수 없습니다: 첫 학습 종료 이후 데이터가 {len(idx) - 1 - e0}봉인데 "
                         f"검증 조각은 {test_bars}봉이 필요합니다.")
    return folds


def _se_from_t(mean, t):
    if mean is None or t is None or pd.isna(mean) or pd.isna(t) or t == 0:
        return np.nan
    return abs(mean / t)


from backtest.stats import chi2_sf as _chi2_sf   # 카이제곱 꼬리확률, 구현은 stats.py 한 곳
from backtest.stats import norm_sf as _p1_from_z   # 단측 p = P(Z > z), 구현은 stats.py 한 곳


def pool_signal(rows: pd.DataFrame) -> dict:
    """한 신호의 fold별 검증 행(te_n_obs, te_mean_return_pct, te_t_stat)을 합친다(위 docstring 공식)."""
    d = rows[(rows["te_n_obs"] > 0) & rows["te_mean_return_pct"].notna()].copy()
    d["se"] = [_se_from_t(m, t) for m, t in zip(d["te_mean_return_pct"], d["te_t_stat"])]
    n_tot = float(d["te_n_obs"].sum())
    out = dict(n_folds_selected=int(len(rows)), n_folds_with_data=int(len(d)), pooled_n=n_tot,
               pooled_mean_pct=np.nan, pooled_se_pct=np.nan, pooled_z=np.nan, pooled_p1=np.nan,
               frac_folds_positive=np.nan, het_q=np.nan, het_p=np.nan)
    if n_tot <= 0 or d.empty:
        return out
    out["pooled_mean_pct"] = float((d["te_n_obs"] * d["te_mean_return_pct"]).sum() / n_tot)
    out["frac_folds_positive"] = float((d["te_mean_return_pct"] > 0).mean())
    if d["se"].notna().all():
        se = math.sqrt(float(((d["te_n_obs"] * d["se"]) ** 2).sum())) / n_tot
        out["pooled_se_pct"] = se
        if se > 0:
            out["pooled_z"] = out["pooled_mean_pct"] / se
            out["pooled_p1"] = _p1_from_z(out["pooled_z"])
        if len(d) >= 2 and (d["se"] > 0).all():       # Cochran Q: 역분산 가중평균 기준 fold 간 이질성
            w = 1.0 / d["se"] ** 2
            mbar = float((w * d["te_mean_return_pct"]).sum() / w.sum())
            q = float((w * (d["te_mean_return_pct"] - mbar) ** 2).sum())
            out["het_q"], out["het_p"] = q, _chi2_sf(q, len(d) - 1)
    return out


def walk_forward(dfs_by_indicator: dict, compare_kwargs: dict, first_train_end, test_bars: int,
                 step_bars: int | None = None, max_folds: int | None = None, alpha: float = 0.05,
                 min_folds: int = 3, min_pooled_n: int = 30, het_alpha: float = 0.05,
                 **oos_kwargs) -> WalkForwardResult:
    """
    oos_study()를 fold마다 호출한다. oos_kwargs(judge_horizon, warmup_bars, min_test_n,
    require_barrier, blocked, exclude_signals 등)는 그대로 전달된다. max_folds로 최근 N개 fold만 쓸 수 있다.
    """
    from backtest.oos import oos_study

    any_df = next(iter(dfs_by_indicator.values()))
    folds = make_folds(any_df.index, first_train_end, test_bars, step_bars)
    if max_folds:
        folds = folds[-int(max_folds):]

    fold_rows, sel_parts = [], []
    for i, (tr_end, te_start, te_end) in enumerate(folds, start=1):
        rec = dict(fold=i, train_end=tr_end, test_start=te_start, test_end=te_end, status="ok", reason="",
                   k=0, n_kept=0, n_failed=0, n_undetermined=0)
        try:
            res = oos_study(dfs_by_indicator, compare_kwargs, tr_end, te_start, test_end=te_end,
                            alpha=alpha, **oos_kwargs)
        except ValueError as e:                    # 학습 이벤트 없음 / 구간 소멸 등 — 이 fold만 건너뛴다
            rec.update(status="skipped", reason=str(e))
            fold_rows.append(rec)
            continue
        rec.update(k=res.k, rank_corr=res.rank_corr)
        if res.k:                                  # 선택 0개면 selected에 verdict 열이 없다
            v = res.selected["verdict"]
            rec.update(n_kept=int((v == "유지됨").sum()), n_failed=int((v == "유지 안 됨").sum()),
                       n_undetermined=int((v == "판정불가").sum()))
        fold_rows.append(rec)
        if res.k:
            sel_parts.append(res.selected.assign(fold=i, train_end=tr_end, test_start=te_start, test_end=te_end))

    folds_df = pd.DataFrame(fold_rows)
    selected = pd.concat(sel_parts, ignore_index=True) if sel_parts else pd.DataFrame()

    rows = []
    if not selected.empty:
        for sig, g in selected.groupby("signal"):
            pooled = pool_signal(g)
            rows.append(dict(signal=sig, **pooled))
    by_sig = pd.DataFrame(rows)
    if not by_sig.empty:
        m = len(by_sig)
        by_sig["alpha_adj"] = alpha / m
        def verdict(r):
            if r["n_folds_selected"] < min_folds or r["pooled_n"] < min_pooled_n or pd.isna(r["pooled_p1"]):
                return "판정불가"
            if r["pooled_p1"] >= r["alpha_adj"]:
                return "불안정"
            if pd.notna(r["het_p"]) and r["het_p"] < het_alpha:
                return "불균질"
            return "안정"
        by_sig["verdict"] = by_sig.apply(verdict, axis=1)
        by_sig = by_sig.sort_values(["verdict", "n_folds_selected"], ascending=[True, False]).reset_index(drop=True)
    return WalkForwardResult(folds=folds_df, selected=selected, by_signal=by_sig,
                             n_folds_planned=len(folds), n_folds_ok=int((folds_df["status"] == "ok").sum()),
                             params=dict(test_bars=test_bars, step_bars=step_bars or test_bars, alpha=alpha,
                                         min_folds=min_folds, min_pooled_n=min_pooled_n, het_alpha=het_alpha,
                                         first_train_end=str(pd.Timestamp(first_train_end).date())))


def format_walk_forward_report(res: WalkForwardResult) -> str:
    p = res.params
    L = ["=== 반복 Walk-forward 검증 (확장창) ===",
         f"첫 학습 종료 {p['first_train_end']}, 검증 조각 {p['test_bars']}봉씩, 전진 {p['step_bars']}봉 | "
         f"fold {res.n_folds_planned}개 중 실행 {res.n_folds_ok}개"]
    f = res.folds
    skipped = f[f["status"] != "ok"]
    if len(skipped):
        L.append(f"건너뛴 fold {len(skipped)}개: " + "; ".join(f"#{r.fold}({r.reason[:40]})" for r in skipped.itertuples()))
    ok = f[f["status"] == "ok"]
    if len(ok):
        L.append(f"신호가 하나라도 선택된 fold: {int((ok['k'] > 0).sum())}/{len(ok)}개 | "
                 f"선택 신호 판정 합계: 유지됨 {int(ok['n_kept'].sum())}, 유지 안 됨 {int(ok['n_failed'].sum())}, "
                 f"판정불가 {int(ok['n_undetermined'].sum())}")
        with pd.option_context("display.width", 200):
            L.append(ok[["fold", "train_end", "test_start", "test_end", "k", "n_kept", "n_failed",
                         "n_undetermined"]].assign(
                train_end=ok["train_end"].dt.date, test_start=ok["test_start"].dt.date,
                test_end=ok["test_end"].dt.date).to_string(index=False))
    if res.by_signal.empty:
        L.append("어느 fold에서도 학습 단계(BH+DSR)를 통과한 신호가 없어 반복 검증으로 판단할 수 있는 신호가 "
                 "없습니다. 신호가 없다는 것은 '안정적이다'가 아니라 '확인할 대상이 없다'는 뜻입니다.")
        return "\n".join(L)
    L.append(f"[신호별 합산] 검증 구간이 겹치지 않으므로 같은 신호의 fold 결과를 합칩니다(신호 간 합산은 하지 않음). "
             f"서로 다른 신호 {len(res.by_signal)}개 -> Bonferroni 유의수준 {res.by_signal['alpha_adj'].iloc[0]:.4f}, "
             f"'안정'은 합산 단측 p<유의수준, fold 간 이질성 p_het>={p['het_alpha']}, "
             f"선택된 fold>={p['min_folds']}, 합산 표본>={p['min_pooled_n']}일 때만")
    cols = ["signal", "n_folds_selected", "frac_folds_positive", "pooled_n", "pooled_mean_pct",
            "pooled_z", "pooled_p1", "het_p", "verdict"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        L.append(res.by_signal[cols].to_string(index=False))
    L.append("해석: '안정'은 여러 기간에서 선택됐고 합산 검증 성과가 유의하며 fold 간 차이도 유의하지 않다는 뜻입니다. "
             "'불균질'은 합산은 유의하지만 기간마다 성과가 크게 달라 특정 기간 덕일 수 있다는 경고입니다. "
             "'판정불가'는 무효가 아니라 증거 부족입니다. fold가 적으면 이질성 검정의 검출력이 낮으니 "
             "frac_folds_positive(양수였던 fold 비율)도 함께 보세요.")
    return "\n".join(L)
