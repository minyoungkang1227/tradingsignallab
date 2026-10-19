"""
공통 '실행 조건·검증 요약' 블록 — OOS·Walk-forward·국면·포트폴리오 리포트 맨 위에 같은 형식으로 붙인다.

[왜 필요한가]
리포트마다 출력이 달라서 "이 결과가 실제 데이터인지 합성인지, 수수료·슬리피지를 뺐는지, 표본이 몇 개인지,
검증을 통과했는지"를 매번 본문에서 찾아야 했다. 같은 항목을 같은 순서로 맨 위에 두면 서로 다른 분석
(IS/OOS, 반복 검증, 국면별, 포트폴리오)을 나란히 비교하기 쉽고, 비용 미반영 같은 조건을 빠뜨린 결과를
알아보기 쉽다. 이 블록은 출력만 추가하며 계산·CSV·기존 리포트 본문은 바꾸지 않는다.

[항목 — 모든 블록이 이 순서와 이름을 따른다]
  분석     어떤 종류의 분석인지
  데이터   실제 시장 데이터 / 합성 데이터(실제 시장 성과가 아님) / 미표시. 코드는 데이터가 실제인지 합성인지
           스스로 알 수 없으므로 실행자가 --data-label real|synthetic으로 알려야 한다. 안 알리면 '미표시'로
           표시해 실제 성과로 오인되지 않게 한다.
  기간     분석에 쓴 구간(학습/검증 등)
  체결     진입 시점, 수수료, 슬리피지, 겹침 처리, 보정 방식
  표본     분석에 쓰인 표본 수(구간별)
  기준     검증 통과의 정의(어떤 통계 기준으로 판정했는지)
  결과     검증 통과 여부 요약. 판정 개념이 없는 분석(포트폴리오)은 '해당 없음'이라고 명시한다.
"""
from __future__ import annotations

import pandas as pd

DATA_LABELS = {
    "real": "실제 시장 데이터",
    "synthetic": "합성 데이터 — 실제 시장 성과가 아님",
    None: "미표시 (실제/합성 여부를 --data-label real|synthetic으로 지정하세요)",
}
FIELDS = ("분석", "데이터", "기간", "체결", "표본", "기준", "결과")


def data_label_text(label) -> str:
    if label not in DATA_LABELS:
        raise ValueError(f"data_label은 'real', 'synthetic', None 중 하나여야 합니다: {label!r}")
    return DATA_LABELS[label]


def describe_execution(ck: dict) -> str:
    """compare_all 인자(dict)에서 체결 가정을 한 줄로 요약한다."""
    lag = int(ck.get("entry_lag", 0))
    col = ck.get("entry_col") or "종가"
    col = {"open": "시가", "close": "종가"}.get(col, col)
    entry = "신호 당일 종가 진입(look-ahead 가능)" if lag == 0 else f"신호 {lag}봉 뒤 {col} 진입"
    cost = float(ck.get("cost_bps", 0.0))
    slip = ck.get("slippage_bps", 0.0)
    if isinstance(slip, pd.Series):
        slip_txt = f"동적(평균 {float(slip.mean()):.1f}bp, 최대 {float(slip.max()):.1f}bp)"
    else:
        slip_txt = f"{float(slip):g}bp(왕복, 고정)"
    parts = [entry, f"수수료·세금 {cost:g}bp(왕복)", f"슬리피지 {slip_txt}"]
    if cost == 0 and (isinstance(slip, pd.Series) is False and float(slip) == 0):
        parts.append("※비용 미반영")
    fix = []
    if ck.get("drop_overlapping"):
        fix.append("겹침 제거")
    if ck.get("newey_west"):
        fix.append("Newey-West")
    if ck.get("deflate"):
        fix.append(f"DSR≥{ck.get('dsr_threshold', 0.95):g}")
    if ck.get("barrier"):
        fix.append(f"배리어 익절{ck.get('pt_mult', 2.0):g}σ/손절{ck.get('sl_mult', 1.0):g}σ"
                   + ("·순차" if ck.get("barrier_sequential") else ""))
    if fix:
        parts.append("보정 " + "·".join(fix))
    return ", ".join(parts)


def format_block(title_kind: str, data_label, period: str, execution: str, samples: str,
                 criteria: str, result: str) -> str:
    rows = [("분석", title_kind), ("데이터", data_label_text(data_label)), ("기간", period),
            ("체결", execution), ("표본", samples), ("기준", criteria), ("결과", result)]
    assert tuple(k for k, _ in rows) == FIELDS
    body = "\n".join(f"  {k}  {v}" for k, v in rows)
    return "[실행 조건·검증 요약]\n" + body


def _d(ts) -> str:
    return str(pd.Timestamp(ts).date())


def oos_header(res, ck: dict, data_label=None) -> str:
    w, p, sel = res.windows, res.params, res.selected
    h = res.judge_horizon
    ok = bool(res.k)
    if ok:
        v = sel["verdict"]
        kept, bad, und = int((v == "유지됨").sum()), int((v == "유지 안 됨").sum()), int((v == "판정불가").sum())
        n_te = int(pd.to_numeric(sel["te_n_obs"], errors="coerce").fillna(0).sum())
        result = f"선택 {res.k}개 중 검증 통과(유지됨) {kept}개, 유지 안 됨 {bad}개, 판정불가 {und}개"
        samples = (f"학습 {w.n_train_bars}봉(이벤트 구간), 검증 {w.n_test_bars}봉, 선택 신호 {res.k}개, "
                   f"검증 표본 합계 {n_te}건(신호별 최소 {p['min_test_n']}건 필요)")
    else:
        result = "통과 신호 없음 — 학습 단계(BH+DSR)를 통과한 신호가 없어 검증할 대상이 없음"
        samples = f"학습 {w.n_train_bars}봉(이벤트 구간), 검증 {w.n_test_bars}봉, 선택 신호 0개"
    criteria = (f"학습 선택: BH(FDR {ck.get('fdr', 0.10):g}) AND DSR≥{ck.get('dsr_threshold', 0.95):g}"
                f"(보유 {h}봉, 최소 {ck.get('min_events', 20)}건)"
                + (" AND 배리어 DSR" if p.get("require_barrier") else "")
                + (f"; 검증: 단측 p < α/k = {res.alpha_adj:.4f}" if ok else ""))
    period = (f"IS(학습) {_d(w.train_lo)}~{_d(w.train_hi)} (퍼지 {w.purged_bars}봉) / "
              f"OOS(검증) {_d(w.test_lo)}~{_d(w.test_hi)}")
    return format_block("표본 외(OOS) 검증", data_label, period, describe_execution(ck), samples, criteria, result)


def walk_forward_header(res, ck: dict, data_label=None) -> str:
    p, f, b = res.params, res.folds, res.by_signal
    ok = f[f["status"] == "ok"]
    period = (f"첫 학습 종료 {p['first_train_end']}부터 검증 {p['test_bars']}봉씩 전진, "
              f"fold {res.n_folds_planned}개 중 실행 {res.n_folds_ok}개 (확장창)")
    if len(ok):
        period += f", 검증 {_d(ok['test_start'].min())}~{_d(ok['test_end'].max())}"
    if b.empty:
        samples = f"fold별 선택 신호 합계 {int(ok['k'].sum()) if len(ok) else 0}건, 합산 가능 신호 0개"
        result = "통과 신호 없음 — 어느 fold에서도 학습 단계를 통과한 신호가 없어 판정할 대상이 없음"
    else:
        vc = b["verdict"].value_counts()
        samples = f"신호 {len(b)}개, 합산 검증 표본 합계 {int(b['pooled_n'].sum())}건"
        result = (f"안정 {int(vc.get('안정', 0))}개, 불균질 {int(vc.get('불균질', 0))}개, "
                  f"불안정 {int(vc.get('불안정', 0))}개, 판정불가 {int(vc.get('판정불가', 0))}개 (신호 {len(b)}개 중)")
    criteria = (f"학습 선택: BH AND DSR≥{ck.get('dsr_threshold', 0.95):g}; 판정: 합산 단측 p < α/m"
                f"(α={p['alpha']:g}), fold 간 이질성 p≥{p['het_alpha']:g}, 선택 fold≥{p['min_folds']}, "
                f"합산 표본≥{p['min_pooled_n']}")
    return format_block("반복 Walk-forward 검증(확장창)", data_label, period, describe_execution(ck),
                        samples, criteria, result)


def regime_header(res, ck: dict, judge_horizon: int, data_label=None, lo=None, hi=None) -> str:
    s = res.summary
    sh = res.share
    period = ("분석 구간 전체" if lo is None and hi is None else f"{lo or '처음'}~{hi or '끝'}")
    period += f" (국면 분류 불가 워밍업 {res.n_unclassified}봉 제외; 비중 " + ", ".join(
        f"{k} {sh[k]:.0%}" if pd.notna(sh[k]) else f"{k} -" for k in ("bull", "bear", "sideways")) + ")"
    if s.empty:
        samples, result = "국면별 이벤트 없음", "판정할 조합 없음"
    else:
        sub = s[(s["horizon"] == judge_horizon) & (s["n_obs"] > 0)]
        per = ", ".join(f"{r} {int(sub.loc[sub['regime'] == r, 'n_obs'].sum())}건"
                        for r in ("bull", "bear", "sideways"))
        samples = f"보유 {judge_horizon}봉 기준 국면별 관측 수: {per}"
        n_sig = int(sub["significant_bh_global"].sum()) if "significant_bh_global" in sub else 0
        result = f"전역 BH 유의 {n_sig}개 / 관측 있는 (국면×신호) {len(sub)}개 조합"
    criteria = f"BH(FDR {ck.get('fdr', 0.10):g}) 전체 국면 합산 재보정(significant_bh_global)" + \
               (f", DSR 시도 횟수에 다른 국면분 가산" if ck.get("deflate") else "")
    return format_block("시장 국면별 검증", data_label, period, describe_execution(ck), samples, criteria, result)


def portfolio_header(res, ck: dict, n_candidates: int, data_label=None) -> str:
    s, e = res.stats, res.equity
    if s.get("n_trades", 0) == 0:
        period, samples = "기간 없음(체결 거래 없음)", f"후보 거래 {n_candidates}건 중 체결 0건"
    else:
        period = f"{_d(e.index[0])}~{_d(e.index[-1])} ({s['n_days']}거래일)"
        samples = f"후보 거래 {n_candidates}건 중 체결 {s['n_trades']}건"
    return format_block("포트폴리오 시뮬레이션(다종목 동시 운용)", data_label, period, describe_execution(ck), samples,
                        "해당 없음 — 통계 검증 판정이 아니라 가정(위험·한도·우선순위)에 따른 시뮬레이션",
                        "해당 없음 — '검증 통과' 개념이 없음. 신호를 같은 구간으로 골랐다면 표본 내 결과")
