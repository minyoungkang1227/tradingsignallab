"""(v29) 선택지 탐색: 주식 비중 경로 · 국민연금 개시 나이 · 인출 순서.

모든 후보를 같은 난수(시나리오)로 돌려 짝지은 차이와 95% 신뢰구간을 낸다.
여러 후보 중 가장 좋은 것을 고르면 '승자의 저주'(우연히 좋게 나온 것을 고름)가 생기므로,
선택에 쓰지 않은 새 난수로 한 번 더 돌려 확인한다(표본 밖 검증).
"""
from dataclasses import replace
import numpy as np

from . import engine_tax, metrics


def _clone(hh, **kw):
    members = [replace(m) for m in hh.members]
    return replace(hh, members=members, **kw)


def glide_candidates(hh, ends=(-0.3, -0.2, -0.1, 0.0, 0.1)):
    out = []
    for d in ends:
        w_end = float(np.clip(hh.stock_weight + d, 0.0, 1.0))
        if d == 0.0:
            out.append(("주식 비중 유지", _clone(hh, glide_end=None)))
        else:
            out.append((f"주식 비중을 85세까지 {hh.stock_weight*100:.0f}%→{w_end*100:.0f}%", _clone(hh, glide_end=w_end, glide_ages=(max(hh.members[0].age, 60), 85))))
    return out


def nps_age_candidates(hh, ages=(60, 63, 65, 67, 70)):
    out = []
    for a in ages:
        h = _clone(hh)
        for m in h.members:
            m.nps_start_age = a
        out.append((f"국민연금 {a}세부터", h))
    return out


def order_candidates(hh):
    return [("인출: 과세계좌 먼저", _clone(hh, draw_order="taxable_first")),
            ("인출: ISA 먼저", _clone(hh, draw_order="isa_first"))]


def evaluate(cands, cfg, base_idx=0, **run_kw):
    """cands=[(label, Household)]. 반환: 후보별 지표 + 기준(base_idx) 대비 짝지은 차이."""
    results = [engine_tax.run(h, cfg, **run_kw) for _, h in cands]
    base = results[base_idx]
    rows = []
    for (label, _), res in zip(cands, results):
        s = engine_tax.summarize(res)
        cmp_ = metrics.compare(res, base)       # 고갈확률 차이 (후보 − 기준)
        rows.append({"선택지": label, "기본생활 유지 확률": s["기본생활 유지 확률"], "고갈확률": s["고갈확률"],
                     "CTE90 부족액(실질)": s["CTE90 부족액(실질)"], "고갈확률 차이(기준 대비)": cmp_["차이"],
                     "±": cmp_["±"], "유의": cmp_["유의"]})
    return rows


def search(hh, cfg, kind, base_idx=None, seed_check=12345, **run_kw):
    """kind: 'glide' | 'nps' | 'order'. 탐색(cfg.seed)에서 고른 뒤 새 난수(seed_check)로 재확인."""
    cands = {"glide": glide_candidates, "nps": nps_age_candidates, "order": order_candidates}[kind](hh)
    if base_idx is None:                                   # 기준: 현재 설정과 같은 후보
        base_idx = {"glide": 3, "nps": 2, "order": 0}[kind]
    rows = evaluate(cands, cfg, base_idx=base_idx, **run_kw)
    best = int(np.argmax([r["기본생활 유지 확률"] for r in rows]))
    cfg2 = replace(cfg, seed=seed_check)
    chk = evaluate([cands[base_idx], cands[best]], cfg2, base_idx=0, **run_kw)
    return {"rows": rows, "best": rows[best]["선택지"], "best_idx": best,
            "check": {"기준": chk[0], "추천 후보": chk[1]}}
