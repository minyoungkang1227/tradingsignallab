"""노후나침반 — 목표 기반 확률 은퇴자금 진단 (Streamlit 웹 앱, v12)

흐름: 단계별 입력(Task) → 최종 확인 → 목표별 달성 결과 → '무엇을 바꾸면' 비교
실행: streamlit run streamlit_app.py
입력값은 계산에만 쓰이며 서버에 저장하지 않습니다(계산 결과는 10분간 메모리 캐시).
"""
import json
from datetime import date
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from retire_sim.config import Person, Household, SimConfig, CareMarkov, CareShock, AddOns, NPSRules
from retire_sim.economy_v2 import EconomyV2
from retire_sim import engine_tax, mortality, optimize, report
from retire_sim.session import calc_id, save_blob, load_blob
from retire_sim import __version__ as MODEL_VERSION
from retire_sim.tax import TaxConfig, HouseConfig
from retire_sim.pension import estimate_nps_monthly

st.set_page_config(page_title="노후나침반 · 은퇴 목표 진단", page_icon="🧭", layout="centered")
TEAL, CORAL, INK, MUTED = "#1F6F78", "#C8553D", "#1C2B33", "#5B6B73"
PROFILES = {"보수형": 0.3, "균형형": 0.5, "성장형": 0.7}
# 국민연금연구원 「2024년 국민노후보장패널조사(10차 부가조사)」 50세 이상이 생각하는 노후 생활비 (월, 만원)
AVG_SPEND = {"부부_최소": 217, "부부_적정": 298, "개인_최소": 139, "개인_적정": 198}


def avg_essential(spouse: bool) -> int:
    """모르는 사람을 위한 기본생활비 기본값: 최소생활비 기준(여행·취미는 따로 입력)."""
    return AVG_SPEND["부부_최소"] if spouse else AVG_SPEND["개인_최소"]


def avg_lifestyle(spouse: bool) -> int:
    """여행·취미 기본값: (적정 − 최소) × 12개월."""
    d = AVG_SPEND["부부_적정"] - AVG_SPEND["부부_최소"] if spouse else AVG_SPEND["개인_적정"] - AVG_SPEND["개인_최소"]
    return int(round(d * 12 / 100) * 100)
st.markdown(f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
html, body, [class*="css"], .stMarkdown, button, input, label {{ font-family: 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif; }}
.hero {{ font-size: 1.8rem; line-height: 1.35; font-weight: 700; color: {INK}; margin: .3rem 0 .5rem; }}
.hero .n {{ color: {CORAL}; }}
.sub {{ color: {MUTED}; font-size: 1rem; }}
.gap {{ background: #E6EDEF; border-radius: 14px; padding: 1rem 1.2rem; margin: .4rem 0 1rem; }}
.gap .big {{ font-size: 2rem; font-weight: 700; color: {TEAL}; }}
.gap .eq {{ color: {MUTED}; font-size: .95rem; }}
.dots {{ display: flex; gap: 10px; margin: .8rem 0 .4rem; flex-wrap: wrap; }}
.dot {{ width: 32px; height: 32px; border-radius: 50%; }}
.dot.bad {{ background: {CORAL}; }} .dot.ok {{ background: {TEAL}; opacity: .85; }}
.legend, .note {{ color: {MUTED}; font-size: .88rem; line-height: 1.6; }}
.card {{ border: 1px solid #CBD7DB; border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; background: white; }}
.card h4 {{ margin: 0 0 .3rem; font-size: 1rem; color: {TEAL}; }}
.card p {{ margin: 0; line-height: 1.6; }}
.goal {{ border: 1px solid #CBD7DB; border-radius: 14px; padding: .9rem 1rem; background: white; height: 100%; }}
.goal .h {{ font-size: 1.15rem; font-weight: 800; color: {TEAL}; margin-bottom: .1rem; }}
.goal .t {{ color: {MUTED}; font-size: .92rem; margin-bottom: .35rem; }}
.goal .v {{ font-size: 1.45rem; font-weight: 700; color: {INK}; line-height: 1.3; }}
.goal .v .hl {{ color: {TEAL}; }}
.tip {{ background: #F1F6F7; border-left: 4px solid {TEAL}; border-radius: 8px; padding: .8rem 1rem; margin: .8rem 0; line-height: 1.7; }}
.act {{ display: flex; gap: .7rem; align-items: flex-start; border: 1px solid #CBD7DB; border-radius: 10px; padding: .6rem .8rem; margin-bottom: .4rem; background: white; }}
.act .r {{ flex: 0 0 26px; height: 26px; border-radius: 50%; background: {TEAL}; color: white; text-align: center; font-weight: 700; line-height: 26px; }}
.act .s {{ color: {MUTED}; font-size: .9rem; margin-top: .15rem; }}
.key {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin: 0 4px -1px 0; }}
.stMarkdown p, .stMarkdown li, .hero, .gap, .goal, .tip, .legend {{ word-break: keep-all; overflow-wrap: anywhere; }}
@media (max-width: 640px) {{
  .hero {{ font-size: 1.45rem; }}
  .gap .big {{ font-size: 1.3rem; }}
  .goal .v {{ font-size: 1.2rem; }}
  .dots {{ gap: 6px; flex-wrap: nowrap; }}
  .dot {{ width: 24px; height: 24px; }}
  .card, .goal {{ padding: .7rem .8rem; }}
  .goal {{ margin-bottom: .5rem; }}
  .tip, .gap {{ padding: .7rem .8rem; }}
  .legend, .note {{ font-size: .82rem; }}
  [data-testid="stTabs"] button p {{ font-size: .9rem; }}
}}
</style>""", unsafe_allow_html=True)

ss = st.session_state
if "p" not in ss:
    ss.p = dict(age=52, sex="M", retire_age=60, spouse=True, s_age=50, s_sex="F",
                deposit=1.5, invest=2.0, ret_acct=1.5, other=0.0, saving=150, salary=0, s_salary=0, save_mode="월 금액", save_rate=15, wage_g=1.0, ret_contrib=False, nps_auto=False, nps_years=10, s_nps_years=10,
                essential=217, lifestyle=1000, nps=110, s_nps=50, nps_start=65,
                priv=0, priv_start=60, priv_years=20, legacy=1.0, children=2, profile="균형형",
                region="대도시", h_official=0.0, h_market=0.0, h_n=1, h_joint=False, h_years=10, h_hi=0.0, rent=0,
                biz=0, biz_until=65, biz_work=False, rm_age=0, ds_age=0, ds_ratio=50, events=[], ev_ltv=60, ev_loan_years=30, ev_jeonse=0.0, ev_rent=0, overseas=20, own_equal=True, survivor=70,
                enhanced=True, paths=10000)
    ss.step = 0
p = ss.p


# ─────────────────────────── 계산 ───────────────────────────
def total_assets(p):
    return p["deposit"] + p["invest"] + p["ret_acct"] + p["other"]


def nps_values(p):
    """국민연금 월액(본인, 배우자): 자동 계산이면 소득·가입기간으로 추정."""
    if p.get("nps_auto"):
        n1 = estimate_nps_monthly(p.get("salary", 0), p.get("nps_years", 0), p["age"], p["retire_age"])
        n2 = estimate_nps_monthly(p.get("s_salary", 0), p.get("s_nps_years", 0), p["s_age"], p["retire_age"] - (p["age"] - p["s_age"])) if p["spouse"] else 0.0
        return n1, n2
    return float(p["nps"]), float(p["s_nps"]) if p["spouse"] else 0.0


def monthly_saving(p):
    if p["age"] >= p["retire_age"]: return 0.0
    if p.get("save_mode") == "소득의 %":
        return (p.get("salary", 0) + (p.get("s_salary", 0) if p["spouse"] else 0)) * p.get("save_rate", 0) / 100 / 12
    return float(p["saving"])


def build(p: dict, v: dict):
    nps_start = v.get("nps_start", p["nps_start"])
    retire = p["retire_age"] + v.get("retire_delta", 0)
    mk = lambda age, sex, nps, priv=0, ps=0, py=0: Person(
        age=age, sex=sex, nps_monthly=nps, nps_start_age=nps_start,
        private_pension_annual=priv, private_pension_start=ps, private_pension_years=py)
    n1, n2 = nps_values(p)
    members = [mk(p["age"], p["sex"], n1, p["priv"], p["priv_start"], p["priv_years"] if p["priv"] > 0 else 0)]
    members[0].salary = float(p.get("salary", 0))
    if p["spouse"]:
        members.append(mk(p["s_age"], p["s_sex"], n2)); members[1].salary = float(p.get("s_salary", 0))
    m = v.get("spend_mult", 1.0)
    liquid = max(v.get("assets", total_assets(p)) - p["ret_acct"], 0)
    use_rate = p.get("save_mode") == "소득의 %" and "saving_monthly" not in v
    events = []
    for ev in p.get("events", []):
        if ev["kind"] == "주택 구입":
            events.append(dict(label=ev["kind"], age=int(ev["age"]), amount=float(ev["amount"]), kind="house",
                               ltv=p.get("ev_ltv", 60) / 100, loan_years=int(p.get("ev_loan_years", 30)),
                               deposit_back=float(p.get("ev_jeonse", 0)) * 10000, rent_saving=float(p.get("ev_rent", 0)) * 12))
        else:
            events.append(dict(label=ev["kind"], age=int(ev["age"]), amount=float(ev["amount"]), years=int(ev["years"]), kind="cost"))
    hh = Household(members=members, liquid_assets=liquid * 10000, pension_balance=p["ret_acct"] * 10000, events=events,
                   saving_rate=(v.get("save_rate_override", p.get("save_rate", 0) / 100) + v.get("saving_delta", 0) * 12 / max((p.get("salary", 0) + (p.get("s_salary", 0) if p["spouse"] else 0)), 1)) if use_rate else 0.0, wage_growth=p.get("wage_g", 1.0) / 100,
                   retirement_contrib=bool(p.get("ret_contrib")) and p["age"] < p["retire_age"],
                   stock_weight=v.get("stock_weight", PROFILES[p["profile"]]),
                   annual_spending=p["essential"] * 12 * m, essential=p["essential"] * 12 * m,
                   lifestyle=p["lifestyle"] * m, legacy_target=p["legacy"] * 10000,
                   retire_age=retire if p["age"] < retire else None,
                   annual_saving=(v.get("saving_monthly", p["saving"]) + v.get("saving_delta", 0)) * 12,
                   survivor_spending_ratio=p["survivor"] / 100,
                   floor_method="fixed95" if p.get("floor") == "보수적" else "actuarial")
    care = CareMarkov() if p["enhanced"] else CareShock()
    if v.get("care_mult"):
        care = CareMarkov(incidence=tuple((a, x * v["care_mult"]) for a, x in CareMarkov().incidence)) \
            if p["enhanced"] else CareShock(lam=0.03 * v["care_mult"])
    cfg = SimConfig(n_paths=v.get("paths", p["paths"]), seed=42, care=care,
                    nps=NPSRules(basic_region=p.get("region", "대도시")))
    ekw = dict(v.get("eco", {}))
    for key in ("stock_shocks", "infl_shocks"):
        if key in ekw: ekw[key] = {int(t): x for t, x in ekw[key].items()}
    eco = EconomyV2.enhanced(**ekw) if p["enhanced"] else EconomyV2(**ekw)
    tkw = dict(overseas_share=p["overseas"] / 100, n_children=p["children"],
               ownership=None if (p["own_equal"] or not p["spouse"]) else (1, 0))
    tkw.update(v.get("tax", {}))
    k = len(members)
    hv = v.get("house", {})
    rm_age = hv.get("rm_age", p.get("rm_age", 0)); ds_age = hv.get("ds_age", p.get("ds_age", 0))
    house = HouseConfig(official=p["h_official"] * 10000, market=p["h_market"] * 10000, n_houses=p["h_n"],
                        owner_share=(tuple([1 / k] * k) if p["h_joint"] else None), years_held=p["h_years"],
                        hi_property_monthly=p["h_hi"] or None, rent_annual=p["rent"],
                        reverse_mortgage_age=rm_age or None, downsize_age=ds_age or None,
                        downsize_ratio=hv.get("ds_ratio", p.get("ds_ratio", 50)) / 100)
    biz = [dict(income=p["biz"], until_age=p["biz_until"], workplace=p["biz_work"])] if p["biz"] > 0 else [None]
    biz += [None] * (k - 1)
    q = None
    if v.get("q_mult"):
        q = {s: np.minimum(a * v["q_mult"], 1) for s, a in mortality.default_qx().items()}
        for a in q.values(): a[-1] = 1
    return hh, cfg, eco, TaxConfig(**tkw), house, biz, q, AddOns(**v.get("addons", {}))


@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def explore(p_json: str, kind: str) -> dict:
    """(v29) 주식 비중 경로 / 국민연금 개시 나이 / 인출 순서를 같은 시나리오 위에서 비교."""
    p = json.loads(p_json)
    hh, cfg, eco, tc, house, biz, q, ad = build(p, {})
    return optimize.search(hh, cfg, kind, e=eco, tc=tc, house=house, qx_table=q, biz=biz, addons=ad)


@st.cache_data(ttl=600, max_entries=128, show_spinner=False)
def simulate(p_json: str, v_json: str) -> dict:
    p, v = json.loads(p_json), json.loads(v_json)
    hh, cfg, eco, tc, house, biz, q, ad = build(p, v)
    res = engine_tax.run(hh, cfg, e=eco, tc=tc, house=house, qx_table=q, biz=biz, addons=ad)
    s = engine_tax.summarize(res)
    d = res["depleted_at"]; y = res["youngest_age"]; ages = np.arange(res["T"] + 1) + y
    W, alive = res["W_real"], res["hh_alive"]
    fan = [[ages[t], *np.percentile(W[alive[:, t], t], [5, 25, 50, 75, 95])]
           for t in range(W.shape[1]) if alive[:, t].sum() > 200]
    curve = [(int(ages[t]), float(((d >= 0) & (d <= t)).mean())) for t in range(res["T"] + 1)]
    t85 = max(0, min(85 - y, res["T"]))
    w85 = W[alive[:, t85], t85]
    corr = None
    if np.any(~np.isnan(res["stock_ret"])):
        sr, pr = res["stock_ret"].ravel(), res["port_ret"].ravel(); ok = ~np.isnan(sr)
        corr = float(np.corrcoef(sr[ok], pr[ok])[0, 1])
    # 은퇴 후 '살아 있는 해'의 평균 세금·건보료·보유세 (리포트의 1년 흐름용, 실질)
    r_age = p.get("retire_age", 0)
    t0 = max(0, min(r_age - p["age"], res["T"]))
    live = res["hh_alive"][:, t0:res["T"]]
    def _avg(key):
        v = res[key][:, t0:res["T"]]
        return float(v[live].mean()) if live.any() else 0.0
    post = {"세금": _avg("tax_y"), "건보료": _avg("hi_y"), "보유세": _avg("prop_y")}
    return {"s": s, "post": post, "dep": (d >= 0), "anti": bool(res.get("antithetic")),
            "dep_age": float(np.median(d[d >= 0] + y)) if (d >= 0).any() else None, "fan": fan, "curve": curve,
            "p50_85": float(np.percentile(w85, 50)) if w85.size else 0.0, "annuity_pay": res.get("annuity_pay", 0.0),
            "care_premium": res.get("care_premium", 0.0), "stock_corr": corr,
            "rm_monthly": res.get("rm_monthly", 0.0), "ds_cash": res.get("ds_cash", 0.0),
            "debt_share": float(np.mean(np.nan_to_num(res.get("debt_retire", np.zeros(1))) > 1))}


@st.cache_data(ttl=86400, max_entries=64, show_spinner=False)
def stock_lookup(code: str):
    from retire_sim.market_data import stock_stats
    try:
        key = st.secrets.get("ECOS_AUTH_KEY", None)
    except Exception:
        key = None
    return stock_stats(code, years=10, ecos_key=key)


ERP_SD = 0.014          # 주식 위험프리미엄 추정 표준오차 (KOSPI 2000~2026)


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def assumption_band(p_json: str) -> tuple:
    """가정(주식 위험프리미엄)을 ±1 표준오차만큼 움직였을 때 기본생활 지킬 확률의 범위.
    계산 횟수를 늘려도 줄지 않는 불확실성이라, 결과를 한 숫자로 읽지 않게 하려고 보여준다."""
    p = json.loads(p_json)
    base = EconomyV2.enhanced().erp
    out = []
    for d in (-ERP_SD, ERP_SD):
        r = simulate(p_json, json.dumps({"eco": {"erp": base + d}}, sort_keys=True))
        out.append(r["s"]["기본생활 유지 확률"])
    return min(out), max(out)


def run(p, v=None):
    return simulate(json.dumps(p, sort_keys=True), json.dumps(v or {}, sort_keys=True))


def solve_for(f, lo: float, hi: float, target: float, tol: float = 0.004, max_iter: int = 6):
    """f(x)(증가 함수)가 target이 되는 x를 찾음. 선형 보간(가성법) + 구간 보정 → 이분법보다 호출 수가 적다.
    반환: (해, None) / (0.0, None) 이미 충분 / (None, hi) 최대값으로도 미달."""
    y_lo, y_hi = f(lo), f(hi)
    if y_lo >= target: return lo, None
    if y_hi < target: return None, hi
    for _ in range(max_iter):
        w = (target - y_lo) / max(y_hi - y_lo, 1e-9)
        x = lo + (hi - lo) * min(max(w, 0.15), 0.85)        # 끝점 쏠림 방지
        y = f(x)
        if y >= target: hi, y_hi = x, y
        else: lo, y_lo = x, y
        if y_hi - y_lo < tol or hi - lo < (hi + lo) * 0.02: break
    return hi, None


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def required_assets(p_json: str, target: float = 0.9):
    """기본생활 유지 확률이 target이 되는 금융자산(억 원) — 같은 시나리오 위에서 보간 탐색."""
    p = json.loads(p_json); paths = min(p["paths"], 5000)
    f = lambda a: run(p, {"assets": a, "paths": paths})["s"]["기본생활 유지 확률"]
    return solve_for(f, 0.0, max(total_assets(p) * 3, 20.0), target)


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def required_saving(p_json: str, target: float = 0.9):
    """은퇴 전 월 저축액(만원) 역산: 기본생활 유지 확률이 target이 되는 최소 저축액."""
    p = json.loads(p_json); paths = min(p["paths"], 5000)
    inc = (p.get("salary", 0) + (p.get("s_salary", 0) if p["spouse"] else 0)) / 12
    if p.get("save_mode") == "소득의 %" and inc > 0:        # 저축률 방식: 소득과 함께 늘어나는 저축률을 역산 → 첫해 월액으로 표시
        f = lambda r: run(p, {"save_rate_override": r, "paths": paths})["s"]["기본생활 유지 확률"]
        r_, over_ = solve_for(f, 0.0, 0.9, target)
        return (None, 0.9 * inc) if r_ is None else (r_ * inc, None)
    f = lambda m: run(p, {"saving_monthly": m, "paths": paths})["s"]["기본생활 유지 확률"]
    hi, lo = max(monthly_saving(p) * 3, 500.0), 0.0
    return solve_for(f, lo, hi, target)


@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def action_ranking(p_json: str):
    """바꿀 수 있는 선택지들을 같은 시나리오로 돌려 효과(기본생활 지킬 확률 상승) 순으로 정렬."""
    p = json.loads(p_json); paths = min(p["paths"], 3000)   # 후보가 많아 가볍게 (순위만 보면 충분)
    b0 = run(p, {"paths": paths})
    cands = []
    pre = p["age"] < p["retire_age"]
    if pre:
        cands += [("은퇴를 2년 늦추기", f"{p['retire_age']}세 → {p['retire_age'] + 2}세", {"retire_delta": 2}),
                  ("매달 저축 30만 원 늘리기", f"월 {monthly_saving(p):.0f}만 원 → {monthly_saving(p) + 30:.0f}만 원", {"saving_delta": 30})]
    cands += [("생활비를 10% 줄이기", f"기본 월 {p['essential']}만 원 → {p['essential'] * 0.9:.0f}만 원", {"spend_mult": 0.9})]
    for k, w in PROFILES.items():
        if k != p["profile"]:
            cands.append((f"투자 성향을 {k}으로", f"{p['profile']} → {k} (주식 비중 {w * 100:.0f}%)", {"stock_weight": w}))
    for a in (60, 65, 68, 70):
        if a != p["nps_start"] and a >= min(p["age"], 70):
            cands.append((f"국민연금을 {a}세부터 받기", f"{p['nps_start']}세 → {a}세", {"nps_start": a}))
    if p["h_market"] > 0 and p["h_n"] == 1 and p["h_official"] <= 12 and not p.get("rm_age"):
        young = min(p["age"], p["s_age"]) if p["spouse"] else p["age"]
        for ra in sorted({max(young, 55, p["retire_age"] - (p["age"] - young)), max(young, 70)}):
            cands.append((f"{ra}세에 주택연금 가입", "집에 살면서 매달 연금 받기", {"house": {"rm_age": int(ra)}}))
    if p["h_market"] > 0 and not p.get("ds_age"):
        da = max(p["age"], 70)
        cands.append((f"{da}세에 집을 줄여 이사", f"시세 {p['h_market']:.1f}억 → 절반 가격 집", {"house": {"ds_age": int(da), "ds_ratio": 50}}))
    out = []
    for lab, how, v in cands:
        v = dict(v); v["paths"] = paths
        r = run(p, v); dlt, ci = paired(r, b0)
        gain = -dlt
        if gain > ci: out.append((lab, how, gain, ci))
    return sorted(out, key=lambda x: -x[2])


def paired(a, b):
    x = a["dep"].astype(float) - b["dep"].astype(float); n = len(x)
    if a["anti"] and b["anti"]:
        h = n // 2; x = (x[:h] + x[h:2 * h]) / 2; n = h
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(n)


def pct(x): return f"{x * 100:.0f}%"


def income_gap(p):
    need = p["essential"] + p["lifestyle"] / 12
    n1, n2 = nps_values(p)
    pension = n1 + n2 + p["priv"] / 12
    return need, pension, need - pension


# ─────────────────────────── 단계별 입력 ───────────────────────────
STEPS = ["은퇴 시점", "현재 자산", "기본생활", "여행·취미", "연금·소득", "남길 자산", "투자 성향", "추가 정보", "최종 확인"]


def nav(i, form_key):
    c1, c2 = st.columns(2)
    back = c1.form_submit_button("이전", disabled=(i == 0), width="stretch")
    nxt = c2.form_submit_button("다음", type="primary", width="stretch")
    return back, nxt


def go(delta):
    ss.step = max(0, min(len(STEPS) - 1, ss.step + delta)); st.rerun()


st.markdown("### 🧭 노후나침반")
if ss.step < len(STEPS) and not ss.get("done"):
    st.markdown(f'<p class="sub">내가 원하는 은퇴생활을 지키려면 무엇이 필요한지, 1만 가지 미래로 확인합니다. '
                f'단계 {ss.step + 1}/{len(STEPS)} · {STEPS[ss.step]}</p>', unsafe_allow_html=True)
    st.progress((ss.step + 1) / len(STEPS))

    with st.sidebar:
        st.markdown("**지금까지 확인한 내용**")
        if ss.step > 0: st.caption(f"은퇴: 현재 {p['age']}세 → {p['retire_age']}세 은퇴")
        if ss.step > 1: st.caption(f"금융자산: {total_assets(p):.1f}억 원" + (f", 은퇴 전 월 {monthly_saving(p):.0f}만 원 저축" if p['age'] < p['retire_age'] else ""))
        if ss.step > 2: st.caption(f"기본생활비: 월 {p['essential']}만 원")
        if ss.step > 3: st.caption(f"여행·취미: 연 {p['lifestyle']:,}만 원")
        if ss.step > 4: st.caption(f"연금: 월 {income_gap(p)[1]:.0f}만 원")
        if ss.step > 5: st.caption(f"남길 자산: {p['legacy']:.1f}억 원")
        if ss.step > 6: st.caption(f"투자 성향: {p['profile']}")

    if ss.step == 0:
        st.markdown("**입력이 번거로우시면, 예시 가구로 먼저 결과를 보세요.** 나중에 '입력 수정하기'로 내 정보로 바꿀 수 있어요.")
        e1, e2 = st.columns(2)
        if e1.button("60대 초반 부부 예시 보기", key="ex_60", width="stretch"):
            ss.p.update(age=60, sex="M", retire_age=60, spouse=True, s_age=58, s_sex="F", deposit=1.5, invest=2.5, ret_acct=1.0, other=0.0,
                        essential=250, lifestyle=1000, nps_auto=False, nps=110, s_nps=50, nps_start=65, priv=600, priv_start=60, priv_years=20,
                        legacy=1.0, profile="균형형", salary=0, s_salary=0, saving=0)
            ss.done = True; st.rerun()
        if e2.button("30대 맞벌이 부부 예시 보기", key="ex_30", width="stretch"):
            ss.p.update(age=34, sex="M", retire_age=60, spouse=True, s_age=32, s_sex="F", deposit=0.4, invest=0.4, ret_acct=0.2, other=0.0,
                        essential=300, lifestyle=1200, nps_auto=True, nps_years=5, s_nps_years=4, nps_start=65, priv=0, legacy=1.0, profile="균형형",
                        salary=5000, s_salary=4000, save_mode="소득의 %", save_rate=15, wage_g=1.0, ret_contrib=True)
            ss.done = True; st.rerun()
        with st.expander("저장해 둔 고객 불러오기"):
            up = st.file_uploader("고객 파일(.json)", type="json", key="load_file", label_visibility="collapsed")
            if up is not None and st.button("불러오기", key="load_go", width="stretch"):
                try:
                    inp, cl, adv, warn = load_blob(up.getvalue())
                    ss.p.update(inp); ss["client"] = cl; ss["advisor"] = adv
                    if warn: ss["load_warn"] = warn
                    ss.done = True; st.rerun()
                except Exception as ex:
                    st.error(f"불러오지 못했습니다: {ex}")
        st.divider()

    i = ss.step
    with st.form(f"step{i}"):
        if i == 0:
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("현재 나이", 20, 90, p["age"]); sex = c2.radio("성별", ["남", "여"], index=0 if p["sex"] == "M" else 1, horizontal=True)
            retire = c3.number_input("은퇴 (예정) 나이", 40, 90, p["retire_age"], help="이미 은퇴했다면 현재 나이 이하로 입력")
            spouse = st.checkbox("배우자와 함께", value=p["spouse"])
            c1, c2 = st.columns(2)
            s_age = c1.number_input("배우자 나이", 20, 90, p["s_age"]); s_sex = c2.radio("배우자 성별", ["남", "여"], index=0 if p["s_sex"] == "M" else 1, horizontal=True)
            st.caption("계획은 95세까지를 기준으로 기본생활비를 보호합니다. 실제 수명은 생명표로 확률적으로 계산합니다.")
            back, nxt = nav(i, "s0")
            if nxt or back:
                p.update(age=int(age), sex="M" if sex == "남" else "F", retire_age=int(retire), spouse=bool(spouse),
                         s_age=int(s_age), s_sex="M" if s_sex == "남" else "F")
                go(1 if nxt else -1)
        elif i == 1:
            st.markdown("집은 빼고, 꺼내 쓸 수 있는 금융자산만 적어주세요. (단위: 억 원)")
            c1, c2 = st.columns(2)
            dep = c1.number_input("예금·적금", 0.0, 500.0, p["deposit"], step=0.1)
            inv = c2.number_input("주식·펀드·ETF", 0.0, 500.0, p["invest"], step=0.1)
            c1, c2 = st.columns(2)
            ret = c1.number_input("퇴직연금·연금저축 적립금", 0.0, 500.0, p["ret_acct"], step=0.1)
            oth = c2.number_input("기타 금융자산", 0.0, 500.0, p["other"], step=0.1)
            pre = p["age"] < p["retire_age"]
            if pre:
                st.markdown("**은퇴 전 소득과 저축**")
                c1, c2, c3 = st.columns(3)
                sal = c1.number_input("본인 연 소득 (세전, 만 원)", 0, 100000, int(p.get("salary", 0)), step=100)
                ssal = c2.number_input("배우자 연 소득 (세전, 만 원)", 0, 100000, int(p.get("s_salary", 0)), step=100, disabled=not p["spouse"])
                wg = c3.number_input("실질 임금상승률 (%/년)", 0.0, 5.0, float(p.get("wage_g", 1.0)), step=0.5)
                mode = st.radio("저축 입력 방식", ["월 금액", "소득의 %"], index=0 if p.get("save_mode", "월 금액") == "월 금액" else 1, horizontal=True)
                c1, c2 = st.columns(2)
                sav = c1.number_input("한 달 저축·투자액 (만 원)", 0, 5000, int(p["saving"]), step=10)
                rate = c2.slider("저축률 (세전 소득 대비 %)", 0, 60, int(p.get("save_rate", 15)))
                rc = st.checkbox("직장 퇴직연금 적립 중 (매년 연봉의 1/12이 퇴직연금으로)", value=bool(p.get("ret_contrib")))
                st.caption("퇴직연금·연금저축 적립금은 55세 이후 연금으로 꺼내 쓰는 계좌로 따로 계산합니다.")
            back, nxt = nav(i, "s1")
            if nxt or back:
                p.update(deposit=dep, invest=inv, ret_acct=ret, other=oth)
                if pre:
                    p.update(saving=int(sav), salary=float(sal), s_salary=float(ssal), wage_g=float(wg), save_mode=mode,
                             save_rate=int(rate), ret_contrib=bool(rc))
                else:
                    p.update(saving=0, salary=0, s_salary=0, ret_contrib=False)
                go(1 if nxt else -1)
        elif i == 2:
            st.markdown("**꼭 필요한 기본생활비**는 얼마인가요? 식비·주거·의료·보험료 등 줄이기 어려운 지출입니다.")
            a_ess = avg_essential(p["spouse"])
            ess = st.number_input("월 기본생활비 (만 원, 오늘 가치)", 0, 5000, p["essential"], step=10,
                                  help="재산세·건강보험료는 빼고 입력하세요 (따로 계산)")
            st.caption(f"잘 모르시면 평균값을 쓰셔도 됩니다 — {'부부' if p['spouse'] else '혼자'} 기준 월 **{a_ess}만 원** "
                       f"(50세 이상이 생각하는 최소 생활비, 국민연금연구원 2024년 조사). 지금 쓰는 생활비를 떠올려 넣으면 더 정확합니다.")
            st.caption("이 금액은 가장 먼저 지키는 목표입니다. 자산이 부족해지면 다른 지출부터 줄입니다.")
            back, nxt = nav(i, "s2")
            if nxt or back:
                p.update(essential=int(ess)); go(1 if nxt else -1)
        elif i == 3:
            st.markdown("**여행·취미·외식** 등 원하는 은퇴생활에 1년에 얼마를 쓰고 싶으신가요?")
            a_life = avg_lifestyle(p["spouse"])
            life = st.number_input("연 여행·취미 예산 (만 원, 오늘 가치)", 0, 50000, p["lifestyle"], step=100)
            st.caption(f"잘 모르시면 평균값을 쓰셔도 됩니다 — 연 **{a_life:,}만 원** "
                       f"(같은 조사에서 '적정 생활비 − 최소 생활비'에 해당하는 금액). 여행 계획이 없으면 0으로 두셔도 됩니다.")
            st.caption("자산이 충분하면 모두 쓰고, 부족해지면 기본생활비를 지키기 위해 이 예산을 먼저 줄입니다.")
            back, nxt = nav(i, "s3")
            if nxt or back:
                p.update(lifestyle=int(life)); go(1 if nxt else -1)
        elif i == 4:
            auto = st.checkbox("국민연금 예상액 자동 계산 (소득·가입기간으로 추정)", value=bool(p.get("nps_auto")),
                               help="2026 개혁(소득대체율 43%) 기준 근사. 정확한 값은 국민연금공단 '내 연금 알아보기'")
            c1, c2, c3 = st.columns(3)
            nps = c1.number_input("국민연금 예상 월액 (만 원, 직접 입력 시)", 0, 500, int(p["nps"]))
            s_nps = c2.number_input("배우자 국민연금 (만 원, 직접 입력 시)", 0, 500, int(p["s_nps"]), disabled=not p["spouse"])
            nps_start = c3.slider("받기 시작할 나이", 60, 70, p["nps_start"])
            c1, c2 = st.columns(2)
            ny = c1.number_input("본인 지금까지 가입 기간 (년, 자동 계산 시)", 0, 45, int(p.get("nps_years", 10)))
            sny = c2.number_input("배우자 가입 기간 (년, 자동 계산 시)", 0, 45, int(p.get("s_nps_years", 10)), disabled=not p["spouse"])
            if p.get("nps_auto"):
                a1, a2 = nps_values(p)
                st.caption(f"현재 입력 기준 자동 추정: 본인 월 {a1:.0f}만 원" + (f", 배우자 월 {a2:.0f}만 원" if p["spouse"] else "") + " (오늘 가치, 정상수령 기준)")
            c1, c2, c3 = st.columns(3)
            priv = c1.number_input("사적연금 연 수령액 (만 원)", 0, 10000, p["priv"], help="연금저축·IRP·연금보험")
            priv_start = c2.number_input("사적연금 시작 나이", 40, 90, p["priv_start"])
            priv_years = c3.number_input("받는 기간 (년)", 0, 50, p["priv_years"])
            back, nxt = nav(i, "s4")
            if nxt or back:
                p.update(nps=int(nps), s_nps=int(s_nps), nps_start=int(nps_start), nps_auto=bool(auto), nps_years=int(ny), s_nps_years=int(sny), priv=int(priv),
                         priv_start=int(priv_start), priv_years=int(priv_years)); go(1 if nxt else -1)
        elif i == 5:
            c1, c2 = st.columns(2)
            leg = c1.number_input("가족에게 남기고 싶은 금액 (억 원, 오늘 가치)", 0.0, 500.0, p["legacy"], step=0.5,
                                  help="없으면 0. 집도 포함해 계산합니다")
            ch = c2.number_input("자녀 수", 0, 10, p["children"])
            back, nxt = nav(i, "s5")
            if nxt or back:
                p.update(legacy=float(leg), children=int(ch)); go(1 if nxt else -1)
        elif i == 6:
            st.markdown("금융자산을 어떤 성향으로 운용하고 계신가요?")
            prof = st.radio("투자 성향", list(PROFILES), index=list(PROFILES).index(p["profile"]), horizontal=True,
                            captions=["주식 약 30%, 안정 위주", "주식 약 50%, 균형", "주식 약 70%, 장기 성장"])
            st.caption("감당할 수 있는 위험은 자산 규모와 연금 비중에 따라 다릅니다. 결과 화면에서 세 성향을 모두 비교할 수 있습니다.")
            back, nxt = nav(i, "s6")
            if nxt or back:
                p.update(profile=prof); go(1 if nxt else -1)
        elif i == 7:
            st.markdown("해당되는 것만 입력하세요. 없으면 그대로 **다음**을 누르면 됩니다.")
            with st.expander("주택", expanded=p["h_official"] > 0):
                c1, c2, c3 = st.columns(3)
                h_off = c1.number_input("공시가격 합계 (억 원)", 0.0, 500.0, p["h_official"], step=0.5)
                h_mkt = c2.number_input("시세 합계 (억 원)", 0.0, 800.0, p["h_market"], step=0.5)
                h_n = c3.number_input("주택 수", 0, 10, p["h_n"])
                c1, c2, c3 = st.columns(3)
                h_joint = c1.checkbox("부부 공동명의", value=p["h_joint"])
                h_years = c2.number_input("보유 기간 (년)", 0, 60, p["h_years"])
                h_hi = c3.number_input("건보료 재산분 (월, 만 원)", 0.0, 500.0, p["h_hi"], help="공단 '지역보험료 모의계산'에서 소득 0, 재산만 넣은 값")
                region = st.radio("사시는 곳", ["대도시", "중소도시", "농어촌"],
                                  index=["대도시", "중소도시", "농어촌"].index(p.get("region", "대도시")), horizontal=True,
                                  help="기초연금 재산 공제가 지역마다 다릅니다. 대도시 1억 3,500만 원, 중소도시 8,500만 원, 농어촌 7,250만 원까지 재산에서 빼고 계산합니다. "
                                       "대도시 = 특별시·광역시, 중소도시 = 도의 '시', 농어촌 = 도의 '군'입니다.")
                rent = st.number_input("연 임대수입 (만 원)", 0, 100000, p["rent"])
                st.markdown("주택 활용 계획 (선택)")
                c1, c2, c3 = st.columns(3)
                rm_age = c1.number_input("주택연금 가입 나이 (0 = 안 함)", 0, 90, p.get("rm_age", 0),
                                         help="부부 중 나이 적은 사람 기준, 55세 이상·공시가격 12억 이하 1주택")
                ds_age = c2.number_input("집 줄이기 나이 (0 = 안 함)", 0, 90, p.get("ds_age", 0), help="본인 나이 기준, 팔고 작은 집으로 이사")
                ds_ratio = c3.slider("새 집 가격 (기존 대비 %)", 20, 90, p.get("ds_ratio", 50), step=10)
            with st.expander("큰 지출 계획 (자녀 교육비·결혼 자금·주택 구입 등)", expanded=bool(p.get("events"))):
                st.caption("나이는 본인 기준, 금액은 오늘 가치(만 원). 주택 구입은 집값 전체, 나머지는 1년 금액 × 기간(년).")
                kinds = ["없음", "자녀 교육비", "결혼·독립 지원", "주택 구입", "기타"]
                old_ev = p.get("events", []) + [{}] * 3
                new_ev = []
                for j in range(3):
                    e0 = old_ev[j] or {}
                    c1, c2, c3, c4 = st.columns([1.3, 0.8, 1, 0.8])
                    kd = c1.selectbox(f"지출 {j + 1}", kinds, index=kinds.index(e0.get("kind", "없음")), key=f"evk{j}")
                    ea = c2.number_input("나이", 20, 90, int(e0.get("age", max(p["age"] + 5, 30))), key=f"eva{j}")
                    em = c3.number_input("금액 (만 원)", 0, 500000, int(e0.get("amount", 0)), step=100, key=f"evm{j}")
                    ey = c4.number_input("기간 (년)", 1, 30, int(e0.get("years", 1)), key=f"evy{j}")
                    if kd != "없음" and em > 0:
                        new_ev.append(dict(kind=kd, age=int(ea), amount=float(em), years=int(ey)))
                st.markdown("주택 구입 시 대출 조건")
                c1, c2, c3 = st.columns(3)
                ltv = c1.slider("대출 비율 (%)", 0, 80, int(p.get("ev_ltv", 60)), step=5)
                lyr = c2.number_input("대출 기간 (년)", 5, 40, int(p.get("ev_loan_years", 30)))
                jb = c3.number_input("전세 보증금 회수 (억 원)", 0.0, 50.0, float(p.get("ev_jeonse", 0.0)), step=0.5)
                er = st.number_input("집을 사면 더 이상 안 내도 되는 월세·전세대출 이자 (월, 만 원)", 0, 1000, int(p.get("ev_rent", 0)), step=10,
                                     help="집을 사기 전에 매달 내던 주거비입니다. 은퇴 전에는 대출 상환에 보태고 남는 만큼 더 저축하고, 은퇴 후에는 생활비에서 빠집니다.")
                st.caption("대출 원리금은 은퇴 전에는 근로소득(생활비)에서 낸다고 보고, 은퇴 후 남은 원리금만 노후자금에서 나갑니다. "
                           "은퇴 전 자산이 부족한 지출은 대출(금리 + 1.5%p)로 처리하고, 남은 빚은 은퇴 시점에 자산으로 갚습니다.")
            with st.expander("사업", expanded=p["biz"] > 0):
                c1, c2, c3 = st.columns(3)
                biz = c1.number_input("연 사업소득 (만 원)", 0, 100000, p["biz"])
                biz_until = c2.number_input("그만둘 나이", 40, 90, p["biz_until"])
                biz_work = c3.checkbox("직원 있음 (직장가입자)", value=p["biz_work"])
            with st.expander("세금·명의·모델"):
                c1, c2 = st.columns(2)
                overseas = c1.slider("주식 중 해외주식 (%)", 0, 100, p["overseas"], step=10)
                own_equal = c2.checkbox("금융자산 부부 균등 명의", value=p["own_equal"])
                survivor = st.slider("한 명만 남았을 때 생활비 (%)", 40, 100, p["survivor"], step=5)
                st.markdown("**생활비를 지키는 방식**")
                st.caption("통장에 '이만큼은 꼭 남겨둬야 기본생활비를 끝까지 낼 수 있다'는 선을 정하고, "
                           "그보다 돈이 많을 때만 여행·취미에 씁니다. 이 선을 얼마나 넉넉하게 잡을지 고르세요.")
                fopts = {"균형형 (추천)": "계리적", "안심형": "보수적"}
                fsel = st.radio("생활비를 지키는 방식", list(fopts), index=0 if p.get("floor", "계리적") == "계리적" else 1,
                                horizontal=True, label_visibility="collapsed",
                                captions=["실제로 몇 살까지 살 가능성이 있는지, 국민연금을 언제부터 받는지 따져서 남겨둘 돈을 정합니다. "
                                          "생활비를 지키면서 여행도 적당히 즐길 수 있습니다.",
                                          "'95세까지 꼭 산다, 지금 모자란 돈이 평생 계속된다'고 넉넉하게 잡습니다. "
                                          "생활비는 더 안전하지만, 여행·취미에 쓸 돈은 많이 줄어듭니다."])
                floor = fopts[fsel]
                st.caption("어떤 걸 고를까요? 여행도 즐기고 싶다면 **균형형**, 여행을 줄이더라도 생활비만큼은 꼭 지키고 싶다면 **안심형**.")
                st.markdown("**계산 방식**")
                enhanced = st.toggle("현실 반영 계산 (추천)", value=p["enhanced"],
                                     help="켜두시면 됩니다. 수수료, 가끔 오는 주가 폭락, 물가와 금리가 함께 움직이는 점, 요양·간병 비용, "
                                          "통계청·한국은행 실제 자료까지 모두 반영합니다. 끄면 이런 요소가 빠져 결과가 실제보다 좋게 나옵니다.")
                st.caption("켜두시면 됩니다. 끄면 수수료·주가 폭락·간병비 같은 현실 요소가 빠져서 결과가 실제보다 좋게 나옵니다 (예전 결과와 비교할 때만 사용).")
                paths = st.select_slider("시나리오 수", [2000, 5000, 10000], value=p["paths"])
            back, nxt = nav(i, "s7")
            if nxt or back:
                p.update(region=region, h_official=h_off, h_market=h_mkt, h_n=int(h_n), h_joint=bool(h_joint), h_years=int(h_years),
                         h_hi=float(h_hi), rent=int(rent), rm_age=int(rm_age), ds_age=int(ds_age), ds_ratio=int(ds_ratio), events=new_ev, ev_ltv=int(ltv), ev_loan_years=int(lyr), ev_jeonse=float(jb), ev_rent=int(er), biz=int(biz), biz_until=int(biz_until), biz_work=bool(biz_work),
                         overseas=int(overseas), own_equal=bool(own_equal), survivor=int(survivor), floor=floor,
                         enhanced=bool(enhanced), paths=int(paths)); go(1 if nxt else -1)
        else:
            need, pension, gap = income_gap(p)
            st.markdown("앞에서 입력한 내용입니다. 이후 계산은 모두 이 가정을 바탕으로 합니다.")
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="card"><h4>은퇴</h4><p>현재 {p["age"]}세 · {p["retire_age"]}세 은퇴'
                        + (f'<br>배우자 {p["s_age"]}세' if p["spouse"] else '') + '</p></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><h4>자산</h4><p>금융자산 {total_assets(p):.1f}억 원'
                        + (f'<br>은퇴 전 월 {monthly_saving(p):.0f}만 원 저축' if p["age"] < p["retire_age"] else '')
                        + ('<br>퇴직연금 적립 중' if p.get("ret_contrib") and p["age"] < p["retire_age"] else '')
                        + (f'<br>주택 시세 {p["h_market"]:.1f}억 원' if p["h_market"] > 0 else '') + '</p></div>', unsafe_allow_html=True)
            ev_txt = "".join(f'<br>{e_["age"]}세 {e_["kind"]} {e_["amount"]:,.0f}만 원' + ("" if e_["kind"] == "주택 구입" else f' × {e_["years"]}년') for e_ in p.get("events", []))
            c1.markdown(f'<div class="card"><h4>목표</h4><p>기본생활: 월 {p["essential"]}만 원<br>여행·취미: 연 {p["lifestyle"]:,}만 원'
                        f'<br>남길 자산: {p["legacy"]:.1f}억 원' + (f'<br><b>큰 지출</b>{ev_txt}' if ev_txt else '') + '</p></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><h4>소득 · 성향</h4><p>연금: 월 {pension:.0f}만 원 ({p["nps_start"]}세부터'
                        + (', 국민연금 자동 추정' if p.get("nps_auto") else '') + ')'
                        + (f'<br>근로소득: 연 {(p.get("salary", 0) + (p.get("s_salary", 0) if p["spouse"] else 0)):,.0f}만 원' if p["age"] < p["retire_age"] and (p.get("salary", 0) or p.get("s_salary", 0)) else '')
                        + f'<br>투자 성향: {p["profile"]}</p></div>', unsafe_allow_html=True)
            st.markdown(f"**이 조건으로 계산해 볼까요?** 투자자산이 매달 메워야 할 금액은 약 **{max(gap, 0):.0f}만 원**입니다.")
            c1, c2 = st.columns(2)
            back = c1.form_submit_button("이전", width="stretch")
            calc = c2.form_submit_button("이 조건으로 계산하기", type="primary", width="stretch")
            if back: go(-1)
            if calc:
                ss.done = True; st.rerun()
    st.markdown('<p class="note">입력하신 정보는 계산에만 쓰이고 저장되지 않습니다.</p>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────── 결과 ───────────────────────────
with st.spinner("1만 가지 미래를 계산하는 중"):
    base = run(p)
s = base["s"]
if st.button("입력 수정하기"):
    ss.done = False; ss.step = len(STEPS) - 1; st.rerun()

tab1, tab2, tab6, tab3, tab4, tab8, tab5, tab7 = st.tabs(["결과", "바꿔보기", "상품 추가", "위기 상황", "세금", "리포트", "가정과 한계", "자세한 설명"])

with tab1:
    need, pension, gap = income_gap(p)
    ok = s["기본생활 유지 확률"]; n_ok = int(round(ok * 10)); n_bad = 10 - n_ok

    # 1) 한 문장 결론
    if n_bad == 0:
        st.markdown('<div class="hero">지금 계획대로면<br>10번 중 <span style="color:#1F6F78">9번 이상</span> 생활비를 끝까지 낼 수 있어요</div>', unsafe_allow_html=True)
    else:
        when = f"{base['dep_age']:.0f}세쯤 " if base["dep_age"] else ""
        st.markdown(f'<div class="hero">지금 계획대로면<br>10번 중 <span class="n">{n_bad}번</span>은 {when}생활비가 모자라요</div>', unsafe_allow_html=True)
    dots = "".join(f'<div class="dot {"bad" if k < n_bad else "ok"}"></div>' for k in range(10))
    st.markdown(f'<div class="dots">{dots}</div><div class="legend"><span class="key" style="background:{CORAL}"></span>모자라는 경우 &nbsp; '
                f'<span class="key" style="background:{TEAL}"></span>끝까지 괜찮은 경우 &nbsp;·&nbsp; 앞으로 있을 수 있는 여러 미래를 10번으로 줄여 표시했어요</div>',
                unsafe_allow_html=True)

    # 2) 매달 부족한 돈: 생활비 − 연금 = 부족분, 그리고 채우는 방법
    short = max(gap, 0)
    if short > 0:
        fill = "이 부족분은 <b>모아둔 돈에서 꺼내 채워요.</b>" + (f" 국민연금을 받기 전({p['nps_start']}세 전)에는 생활비 {need:.0f}만 원 전부를 모아둔 돈에서 씁니다." if pension > 0 else "")
        big = f'매달 <span class="n">{short:.0f}만 원</span>이 부족해요'
    else:
        fill = "연금만으로 생활비가 충분해요. 모아둔 돈은 여행·취미와 남길 돈에 쓸 수 있어요."
        big = "매달 연금으로 생활비가 충분해요"
    st.markdown(f'<div class="gap"><div class="eq">필요한 생활비 <b>월 {need:.0f}만 원</b><br>받는 연금 <b>월 {pension:.0f}만 원</b> (오늘 돈 가치 기준)</div>'
                f'<div class="big">{big}</div><div class="eq">{fill}</div></div>', unsafe_allow_html=True)

    # 3) 목표별 결과: 기본 / 여행·취미 / 자녀
    cards = [("🏠 기본 생활", "먹고 사는 데 꼭 필요한 돈", f'10번 중 <span class="hl">{n_ok}번</span> 끝까지 가능해요')]
    if p["lifestyle"] > 0:
        cards.append(("✈️ 여행·취미", "원하는 예산 중 쓸 수 있는 정도", f'<span class="hl">{s.get("여행·취미 평균 충족률", 0) * 100:.0f}%</span> 정도'))
    if p["legacy"] > 0:
        cards.append(("👨‍👩‍👧 자녀", f"{p['legacy']:.1f}억 원 남기기", f'10번 중 <span class="hl">{int(round(s.get("유산 목표 달성 확률", 0) * 10))}번</span> 가능해요'))
    for col, (h_, t_, v_) in zip(st.columns(len(cards)), cards):
        col.markdown(f'<div class="goal"><div class="h">{h_}</div><div class="t">{t_}</div><div class="v">{v_}</div></div>', unsafe_allow_html=True)

    # 3-2) 가정의 불확실성 — 난수를 늘려도 줄지 않는 폭
    with st.spinner("가정이 달라질 때의 폭을 계산 중"):
        lo, hi = assumption_band(json.dumps(p, sort_keys=True))
    st.markdown(f'<div class="legend">주식이 앞으로 얼마나 오를지는 아무도 모릅니다. 과거 자료로 추정한 범위 안에서만 봐도 '
                f'위 <b>{pct(ok)}</b>는 <b>{pct(lo)} ~ {pct(hi)}</b> 사이에서 움직입니다. '
                f'계산을 더 많이 해도 이 폭은 줄지 않으니, <b>숫자 하나보다 선택을 바꿨을 때의 차이</b>를 보세요.</div>',
                unsafe_allow_html=True)

    # 4-2) 무엇부터 하면 되나 — 효과 순위
    if n_ok < 9:
        with st.spinner("효과가 큰 방법을 찾는 중"):
            acts = action_ranking(json.dumps(p, sort_keys=True))
        if acts:
            st.markdown("**효과가 큰 순서대로 세 가지**")
            for rank, (lab, how, gain, ci_) in enumerate(acts[:3], 1):
                st.markdown(f'<div class="act"><div class="r">{rank}</div><div><b>{lab}</b>'
                            f'<div class="s">{how}<br>기본 생활을 지킬 가능성 {ok * 100:.0f}% → <b>{min(ok + gain, 1) * 100:.0f}%</b> '
                            f'({gain * 100:+.0f}%p)</div></div></div>', unsafe_allow_html=True)
            st.caption("권유가 아니라 '어떤 선택이 얼마나 효과가 큰지' 계산한 결과입니다. 자세한 조정은 '바꿔보기' 탭에서 하세요.")
    if p.get("events"):
        ds_ = base.get("debt_share", 0)
        st.caption("큰 지출 반영: " + ", ".join(f'{e_["age"]}세 {e_["kind"]}' for e_ in p["events"])
                   + (f" · 은퇴 때 갚을 빚이 남는 경우 10번 중 {ds_ * 10:.0f}번" if ds_ > 0.05 else ""))

    # 5) 나이별 남는 돈 (3개 선)
    fan = pd.DataFrame(base["fan"], columns=["나이", "p5", "p25", "p50", "p75", "p95"])
    lines_df = pd.concat([pd.DataFrame({"나이": fan["나이"], "억 원": fan[c] / 10000, "경우": lab})
                          for c, lab in (("p75", "운이 좋을 때"), ("p50", "보통일 때"), ("p25", "운이 나쁠 때"))])
    st.markdown("**나이별로 남는 돈** (오늘 돈 가치)")
    st.altair_chart(alt.Chart(lines_df).mark_line(strokeWidth=3).encode(
        x=alt.X("나이:Q", title="나이"),
        y=alt.Y("억 원:Q", title="억 원", axis=alt.Axis(titleAngle=0, titleAlign="left", titleY=-10, titleX=-5)),
        color=alt.Color("경우:N", sort=["운이 좋을 때", "보통일 때", "운이 나쁠 때"],
                        scale=alt.Scale(range=["#7FB8BE", TEAL, CORAL]),
                        legend=alt.Legend(title=None, orient="bottom", direction="horizontal", labelLimit=0, columns=3)),
        tooltip=["경우", "나이", alt.Tooltip("억 원:Q", format=".1f")]).properties(height=280), width="stretch")
    st.markdown('<div class="legend">선이 0에 닿는 때가 모아둔 돈을 다 쓰는 때예요. 운이 나쁘면(주가 하락·물가 상승 등) 더 빨리, '
                '운이 좋으면 더 늦게 닿습니다. 집은 포함하지 않았어요.</div>', unsafe_allow_html=True)

def make_report(p, base, client, advisor, cid) -> str:
    """현재 결과를 상담 리포트(HTML)로 묶는다. 화면에 보여주는 숫자와 같은 값을 쓴다."""
    s_ = base["s"]; ok = s_["기본생활 유지 확률"]; n_ok = int(round(ok * 10))
    need, pension, gap = income_gap(p)
    short = max(gap, 0)
    when = f"{base['dep_age']:.0f}세쯤 " if base["dep_age"] else ""
    hero = (f'지금 계획대로면 10번 중 <span class="n">{10 - n_ok}번</span>은 {when}생활비가 모자랍니다.'
            if n_ok < 10 else "지금 계획대로면 10번 중 9번 이상 생활비를 끝까지 낼 수 있습니다.")
    goals = [("🏠 기본 생활", "먹고 사는 데 꼭 필요한 돈", f"10번 중 {n_ok}번 가능")]
    if p["lifestyle"] > 0:
        goals.append(("✈️ 여행·취미", "원하는 예산 중 쓸 수 있는 정도", f"{s_.get('여행·취미 평균 충족률', 0) * 100:.0f}% 정도"))
    if p["legacy"] > 0:
        goals.append((f"👨‍👩‍👧 자녀", f"{p['legacy']:.1f}억 원 남기기",
                      f"10번 중 {int(round(s_.get('유산 목표 달성 확률', 0) * 10))}번 가능"))
    lo, hi = assumption_band(json.dumps(p, sort_keys=True))
    band = (f"주식이 앞으로 얼마나 오를지는 아무도 모릅니다. 과거 자료로 추정한 범위 안에서만 봐도 위 "
            f"<b>{pct(ok)}</b>는 <b>{pct(lo)} ~ {pct(hi)}</b> 사이에서 움직입니다. "
            f"계산을 더 많이 해도 이 폭은 줄지 않으니, 숫자 하나보다 <b>선택을 바꿨을 때의 차이</b>를 보셔야 합니다.")
    acts = [(lab, how, f"{ok * 100:.0f}% → {min(ok + g, 1) * 100:.0f}% ({g * 100:+.0f}%p)")
            for lab, how, g, _ in action_ranking(json.dumps(p, sort_keys=True))[:3]] if n_ok < 9 else []
    # 100세까지만 그린다 — 그 뒤는 살아 있는 경로가 적어 분위수가 들쭉날쭉해 읽기 어렵다
    rows = [(int(r[0]), r[4] / 10000, r[3] / 10000, r[2] / 10000) for r in base["fan"] if r[0] <= 100]
    po = base.get("post", {})
    flow = [("연금 수입", pension * 12), ("생활비", -(p["essential"] * 12 + p["lifestyle"])),
            ("세금", -po.get("세금", 0.0)), ("건보료", -po.get("건보료", 0.0))]
    if po.get("보유세", 0) > 0:
        flow.append(("보유세", -po["보유세"]))
    assume = ("· 금리·물가·주가가 서로 연결되어 움직이는 1만 가지 미래를 계산했습니다(한국은행 ECOS 2000~2026, KOSPI).<br>"
              "· 수명은 통계청 2024 완전생명표에 앞으로의 수명 연장을 반영했고, 간병은 건강보험공단 장기요양 통계로 추정했습니다.<br>"
              "· 국민연금·유족연금·기초연금·건강보험료·세금을 2026년 기준으로 반영했습니다.<br>"
              f"· 기본생활비 월 {p['essential']:,}만 원, 여행·취미 연 {p['lifestyle']:,}만 원, "
              f"국민연금 개시 {p['nps_start']}세로 계산했습니다.<br>"
              "· 결과는 가정에 따라 달라집니다. 절대값보다 선택지 사이의 차이를 보시기 바랍니다.")
    return report.build(dict(
        client=client or "고객", advisor_line=f" · 상담 {advisor}" if advisor else "",
        date=f"{date.today():%Y년 %m월 %d일}", calc_id=cid, version=MODEL_VERSION,
        hero=hero, goals=goals,
        gap_eq=f"필요한 생활비 월 {need:.0f}만 원 − 받는 연금 월 {pension:.0f}만 원 (오늘 돈 가치 기준)",
        gap_big=(f"매달 {short:.0f}만 원을 모아둔 돈에서 꺼내 써야 합니다" if short > 0
                 else "연금만으로 생활비가 충분합니다"),
        band=band, actions=acts, fan_svg=report.line_chart_svg(rows),
        flow_svg=report.waterfall_svg(flow), assumptions=assume))


def whatif(rows):
    out = []
    for label, v in rows:
        r = run(p, v); rs = r["s"]; dlt, ci = paired(r, base)
        verdict = "지금 계획" if not v else ("차이 없음" if abs(dlt) <= ci else ("나빠짐" if dlt > 0 else "좋아짐"))
        row = {"바꾸는 것": label, "기본생활 지킬 확률": pct(rs["기본생활 유지 확률"]),
               "변화 (%p)": "—" if not v else f"{-dlt * 100:+.1f} (±{ci * 100:.1f})", "판단": verdict}
        if p["lifestyle"] > 0: row["여행·취미 예산 충족"] = pct(rs.get("여행·취미 평균 충족률", 0))
        if p["legacy"] > 0: row["남길 자산 달성 확률"] = pct(rs.get("유산 목표 달성 확률", 0))
        note = []
        if r.get("rm_monthly"): note.append(f"주택연금 월 {r['rm_monthly']:.0f}만 원")
        if r.get("ds_cash"): note.append(f"확보 현금 {r['ds_cash'] / 10000:.1f}억")
        if p["h_market"] > 0: row["비고"] = ", ".join(note) or "—"
        out.append(row)
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")

with tab2:
    pre = p["age"] < p["retire_age"]
    st.markdown("조건을 **직접 바꿔 보고** 결과가 어떻게 달라지는지 확인하세요. 같은 1만 가지 미래 위에서 비교하므로, 달라진 만큼이 순수하게 그 선택의 효과입니다.")
    # 기본 생활을 10번 중 9번 지키려면 (첫 화면에서 옮겨 옴)
    with st.spinner("계산 중"):
        req, over = required_assets(json.dumps(p, sort_keys=True))
        rs_, over_ = required_saving(json.dumps(p, sort_keys=True)) if p["age"] < p["retire_age"] else (None, None)
    lines = []
    if n_ok >= 9:
        lines.append("지금 계획으로도 기본생활비를 <b>10번 중 9번 이상</b> 지킬 수 있어요.")
    else:
        if req is not None:
            lines.append(f"모아둔 돈이 지금 {total_assets(p):.1f}억 원에서 <b>약 {req:.1f}억 원</b>이면 됩니다.")
        if p["age"] < p["retire_age"]:
            if rs_ is not None and rs_ > 0:
                lines.append(f"또는 은퇴 전까지 매달 <b>약 {rs_:.0f}만 원</b>씩 모으면 됩니다 (지금 월 {monthly_saving(p):.0f}만 원).")
            elif rs_ is None:
                lines.append("저축만 늘려서는 어려워요. 은퇴 시기나 생활비도 함께 바꿔 보세요.")
        if req is None and p["age"] >= p["retire_age"]:
            lines.append("모아둔 돈만 늘려서는 어려워요. 생활비나 연금 받는 시기를 함께 바꿔 보세요.")
        lines.append("👉 <b>'바꿔보기'</b> 탭에서 은퇴 시기·저축·생활비를 바꿔 보면 결과가 어떻게 달라지는지 바로 볼 수 있어요.")
    st.markdown('<div class="tip"><b>기본 생활을 10번 중 9번 지키려면</b><br>' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
    with st.expander("숫자 읽는 법", expanded=not ss.get("tweak")):
        st.markdown(f"""
- **기본생활 지킬 확률 {pct(s["기본생활 유지 확률"])}** = 10번 중 약 {s["기본생활 유지 확률"] * 10:.0f}번은 생활비를 끝까지 낼 수 있다는 뜻이에요.
- **+5%p** = 확률이 5만큼 오른다는 뜻이에요 (예: 48% → 53%).
- 더 자세한 뜻은 **'자세한 설명'** 탭에 있어요.""")

    with st.form("tweak_form"):
        st.markdown("**바꿔 볼 조건**")
        c1, c2 = st.columns(2)
        rd = c1.slider("은퇴 시기", -5, 10, int(ss.get("tweak", {}).get("rd", 0)),
                       format="%+d년", disabled=not pre, help=f"지금 계획 {p['retire_age']}세. +3이면 3년 늦게 은퇴")
        sd = c2.slider("월 저축", -100, 300, int(ss.get("tweak", {}).get("sd", 0)),
                       step=10, format="%+d만 원", disabled=not pre, help=f"지금 월 {monthly_saving(p):.0f}만 원")
        c1, c2 = st.columns(2)
        sm = c1.slider("생활비", -30, 20,
                       int(ss.get("tweak", {}).get("sm", 0)), step=5, format="%+d%%",
                       help=f"지금 기본 월 {p['essential']}만 원 + 여행 연 {p['lifestyle']:,}만 원")
        ns = c2.slider("국민연금 시작 나이", 60, 70, int(ss.get("tweak", {}).get("ns", p["nps_start"])))
        prof_list = list(PROFILES)
        pf = st.radio("투자 성향", prof_list, index=prof_list.index(ss.get("tweak", {}).get("pf", p["profile"])), horizontal=True)
        go_tw = st.form_submit_button("이 조건으로 다시 계산", type="primary", width="stretch")
    if go_tw:
        ss.tweak = dict(rd=rd, sd=sd, sm=sm, ns=ns, pf=pf)

    tw = ss.get("tweak")
    if tw:
        v, desc = {}, []
        if tw["rd"] and pre: v["retire_delta"] = tw["rd"]; desc.append(f"은퇴 {p['retire_age'] + tw['rd']}세({tw['rd']:+d}년)")
        if tw["sd"] and pre: v["saving_delta"] = tw["sd"]; desc.append(f"월 저축 {tw['sd']:+d}만 원")
        if tw["sm"]: v["spend_mult"] = 1 + tw["sm"] / 100; desc.append(f"생활비 {tw['sm']:+d}%")
        if tw["ns"] != p["nps_start"]: v["nps_start"] = tw["ns"]; desc.append(f"국민연금 {tw['ns']}세부터")
        if tw["pf"] != p["profile"]: v["stock_weight"] = PROFILES[tw["pf"]]; desc.append(f"투자 성향 {tw['pf']}")
        if not v:
            st.info("바꾼 조건이 없습니다. 슬라이더를 움직인 뒤 '이 조건으로 다시 계산'을 누르세요.")
        else:
            with st.spinner("다시 계산하는 중"):
                r = run(p, v)
            rs = r["s"]; dlt, ci = paired(r, base)
            new_ok, old_ok = rs["기본생활 유지 확률"], s["기본생활 유지 확률"]
            st.markdown(f"**{', '.join(desc)}** 로 바꾸면")
            c1, c2, c3 = st.columns(3)
            c1.metric("기본생활 지킬 확률", pct(new_ok), f"{(new_ok - old_ok) * 100:+.1f}%p")
            if p["lifestyle"] > 0:
                c2.metric("여행·취미 예산 충족", pct(rs.get("여행·취미 평균 충족률", 0)),
                          f"{(rs.get('여행·취미 평균 충족률', 0) - s.get('여행·취미 평균 충족률', 0)) * 100:+.1f}%p")
            if p["legacy"] > 0:
                c3.metric("남길 자산 달성 확률", pct(rs.get("유산 목표 달성 확률", 0)),
                          f"{(rs.get('유산 목표 달성 확률', 0) - s.get('유산 목표 달성 확률', 0)) * 100:+.1f}%p")
            n_old, n_new = int(round(old_ok * 10)), int(round(new_ok * 10))
            dots = lambda k: "".join(f'<div class="dot {"ok" if i < k else "bad"}"></div>' for i in range(10))
            st.markdown(f'<div class="legend">지금 계획: 10번 중 {n_old}번 지킴</div><div class="dots">{dots(n_old)}</div>'
                        f'<div class="legend">바꾼 뒤: 10번 중 {n_new}번 지킴</div><div class="dots">{dots(n_new)}</div>', unsafe_allow_html=True)
            if abs(dlt) <= ci:
                st.caption(f"변화({-dlt * 100:+.1f}%p)가 계산 오차 범위(±{ci * 100:.1f}%p) 안이라 사실상 차이가 없습니다.")
            else:
                st.caption(f"계산 오차 범위 ±{ci * 100:.1f}%p를 넘는 변화라, 이 선택의 실제 효과로 볼 수 있습니다.")

    st.divider()
    if st.checkbox("변화 곡선 보기 (조건을 조금씩 바꿨을 때 확률이 어떻게 움직이는지)", key="curves"):
        cp = min(p["paths"], 5000)
        charts = []
        if pre:
            charts.append(("은퇴 시기 조정 (년)", [(d, {"retire_delta": d, "paths": cp}) for d in range(0, 6)]))
            charts.append(("월 저축 조정 (만 원)", [(d, {"saving_delta": d, "paths": cp}) for d in (0, 50, 100, 150, 200)]))
        charts.append(("생활비 조정 (%)", [(d, {"spend_mult": 1 + d / 100, "paths": cp}) for d in (-30, -20, -10, 0, 10)]))
        cols = st.columns(len(charts))
        with st.spinner("곡선 계산 중"):
            for col, (title, pts) in zip(cols, charts):
                df = pd.DataFrame([(x, run(p, v)["s"]["기본생활 유지 확률"] * 100) for x, v in pts], columns=["x", "y"])
                col.markdown(f"**{title}**")
                col.altair_chart(alt.Chart(df).mark_line(point=True, color=TEAL).encode(
                    x=alt.X("x:Q", title=None), y=alt.Y("y:Q", title="기본생활 지킬 확률 (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["x", alt.Tooltip("y:Q", format=".0f")]).properties(height=200), width="stretch")
        st.caption("곡선이 가파를수록 그 조건이 결과에 큰 영향을 줍니다 (계산 속도를 위해 5천 개 시나리오 기준).")

    with st.expander("자주 쓰는 선택지 한 번에 비교"):
        rows = [("지금 계획", {})]
        if pre:
            rows += [("은퇴 2년 늦추기", {"retire_delta": 2}), ("월 저축 50만 원 늘리기", {"saving_delta": 50})]
        rows += [("생활비 10% 줄이기", {"spend_mult": 0.9})]
        rows += [(f"투자 성향 {k}", {"stock_weight": w}) for k, w in PROFILES.items() if k != p["profile"]]
        for a in (p["nps_start"] - 5, p["nps_start"] + 3):
            if 60 <= a <= 70 and a >= min(p["age"], 70): rows.append((f"국민연금 {a}세부터", {"nps_start": a}))
        if p["h_market"] > 0 and not p.get("rm_age") and not p.get("ds_age"):
            young = min(p["age"], p["s_age"]) if p["spouse"] else p["age"]
            if p["h_n"] == 1 and p["h_official"] <= 12:
                for ra in sorted({max(young, 55, p["retire_age"] - (p["age"] - young)), max(young, 70)}):
                    rows.append((f"주택연금 {ra}세 가입 (연소자 기준)", {"house": {"rm_age": int(ra)}}))
            da = max(p["age"], 70)
            rows.append((f"{da}세에 절반 가격 집으로 이사", {"house": {"ds_age": int(da), "ds_ratio": 50}}))
        whatif(rows)
        if p["h_market"] > 0:
            st.caption("주택연금: 한국주택금융공사 2026-03 종신지급 정액형 월지급금 기준(명목 정액), 대출이자(금리+1.1%p)·보증료 0.75%가 쌓여 사망 시 집값에서 상환(집값 초과분은 청구 안 함), 재산세 25% 감면. "
                       "집 줄이기: 매도 비용 0.6%, 1주택 양도세(12억 초과분), 새 집 취득세 반영.")

    st.markdown("---")
    with st.expander("선택지 비교해 보기 (주식 비중·국민연금 개시 나이·인출 순서)"):
        st.markdown("같은 1만 가지 미래에서 여러 선택지를 한꺼번에 비교합니다. **추천이 아니라 비교**입니다. 여러 개 중 제일 좋은 걸 고르면 우연히 좋게 나온 것이 뽑힐 수 있어서, 새 미래로 한 번 더 확인한 결과도 같이 보여드려요.")
        kinds = {"주식 비중을 나이 들며 줄이기": "glide", "국민연금 받기 시작하는 나이": "nps", "돈 꺼내 쓰는 순서 (ISA 먼저?)": "order"}
        pick = st.radio("무엇을 비교할까요?", list(kinds), horizontal=True, key="explore_kind")
        if st.button("비교 계산하기", key="explore_go"):
            with st.spinner("여러 선택지를 계산 중입니다 (약 30초)…"):
                ss["explore"] = (kinds[pick], explore(json.dumps(p, sort_keys=True), kinds[pick]))
        if ss.get("explore") and ss["explore"][0] == kinds[pick]:
            out = ss["explore"][1]
            df = pd.DataFrame([{"선택지": r["선택지"], "기본생활 지킬 확률": pct(r["기본생활 유지 확률"]),
                                "지금 설정 대비": f"{-r['고갈확률 차이(기준 대비)'] * 100:+.1f}%p ± {r['±'] * 100:.1f}"} for r in out["rows"]])
            st.dataframe(df, hide_index=True)
            c = out["check"]
            st.caption(f"가장 높게 나온 선택지: **{out['best']}**. 새 미래로 다시 확인하면 지금 설정 {pct(c['기준']['기본생활 유지 확률'])} → 이 선택지 {pct(c['추천 후보']['기본생활 유지 확률'])}. "
                       "단, 국민연금을 늦게 받는 것은 그만큼 건강·자산 여건이 받쳐줘야 하고, 주식 비중은 마음의 편안함도 중요합니다.")

with tab6:
    st.markdown("주식·예금·연금·보험을 **더했을 때** 목표 달성과 위험이 어떻게 바뀌는지 봅니다. "
                "주식·예금·연금 일시납은 지금 금융자산에서 옮기는 것으로 계산합니다. **매수·가입 권유가 아닙니다.**")
    with st.form("addons"):
        st.markdown("**주식 (종목코드)**")
        c1, c2 = st.columns(2)
        code = c1.text_input("종목코드 6자리", value=ss.get("ad_code", ""), placeholder="예: 005930")
        s_amt = c2.number_input("투자 금액 (억 원)", 0.0, 100.0, float(ss.get("ad_samt", 0.0)), step=0.1)
        with st.expander("종목 통계를 직접 입력 (자동 조회가 안 될 때)"):
            c1, c2, c3 = st.columns(3)
            m_beta = c1.number_input("시장 베타", -1.0, 3.0, 1.0, step=0.1)
            m_idio = c2.number_input("고유 변동성 (%)", 0.0, 100.0, 25.0, step=1.0)
            m_rate = c3.number_input("금리 +1%p 때 수익률 변화 (%)", -20.0, 20.0, 0.0, step=0.5)
            manual = st.checkbox("직접 입력값 사용")
        st.markdown("**예금**")
        d_amt = st.number_input("예금으로 옮길 금액 (억 원)", 0.0, 100.0, 0.0, step=0.1)
        st.markdown("**연금보험 (일시납)**")
        c1, c2, c3 = st.columns(3)
        a_amt = c1.number_input("일시납 보험료 (억 원, 오늘 가치)", 0.0, 50.0, 0.0, step=0.1)
        a_age = c2.number_input("받기 시작할 나이", max(p["age"], 45), 95, max(p["age"], 65),
                                help="늦게 받을수록 같은 돈으로 받는 금액이 커집니다. 80세부터 받는 '거치형'은 오래 사는 위험만 막는 방식입니다.")
        a_joint = c3.checkbox("부부형 (한 명이라도 살아있으면 지급)", disabled=not p["spouse"])
        c1, c2, c3 = st.columns(3)
        a_sr = c1.slider("한 명 사망 후 지급 비율 (%)", 50, 100, 100, step=10, disabled=not a_joint,
                         help="부부형에서 한 분이 돌아가신 뒤 받는 비율. 낮출수록 평소 받는 금액이 커집니다.")
        a_gt = c2.number_input("보증기간 (년, 0 = 없음)", 0, 30, 0, step=5,
                               help="일찍 돌아가셔도 이 기간까지는 유족에게 지급됩니다. 길수록 받는 금액이 줄어듭니다.")
        a_esc = c3.selectbox("체증형", ["없음 (정액)", "연 2% 증액", "연 3% 증액"],
                             help="물가를 따라가도록 매년 늘어나는 방식. 처음 받는 금액은 적습니다.")
        st.markdown("**간병보험**")
        c1, c2 = st.columns(2)
        c_ben = c1.number_input("간병 시 연 보장액 (만 원)", 0, 10000, 0, step=100)
        c_who = c2.radio("피보험자", ["본인", "배우자"], horizontal=True, disabled=not p["spouse"])
        c_trig = st.radio("지급 조건", ["중증(1~2등급)", "모든 장기요양 등급"], horizontal=True)
        pc = 0
        if p["age"] < p["retire_age"]:
            st.markdown("**연금저축·IRP 추가 납입 (은퇴 전)**")
            pc = st.number_input("월 납입액 (만 원) — 저축액 중 일부를 연금계좌로", 0, 500, 0, step=10)
        go_cmp = st.form_submit_button("비교하기", type="primary", width="stretch")

    if go_cmp:
        ss.ad_code, ss.ad_samt = code.strip(), s_amt
        items, info = [], {}
        if s_amt > 0:
            if manual or not code.strip():
                stt = {"beta": m_beta, "idio_sigma": m_idio / 100, "rate_beta": m_rate, "name": "직접 입력"}
            else:
                try:
                    with st.spinner("종목 데이터 조회 중"):
                        stt = stock_lookup(code.strip())
                    stt["name"] = code.strip()
                except Exception as ex:
                    st.error(f"종목 데이터를 가져오지 못했습니다: {ex} — '직접 입력값 사용'으로 계산할 수 있습니다.")
                    stt = None
            if stt:
                info["stock"] = stt
                items.append((f"주식 {stt['name']} {s_amt:.1f}억", {"stock_amount": s_amt * 10000, "stock_beta": stt["beta"],
                              "stock_idio_sigma": stt["idio_sigma"], "stock_rate_beta": stt["rate_beta"]}))
        if d_amt > 0: items.append((f"예금 {d_amt:.1f}억", {"deposit_amount": d_amt * 10000}))
        if a_amt > 0:
            esc = {"없음 (정액)": 0.0, "연 2% 증액": 0.02, "연 3% 증액": 0.03}[a_esc]
            tag = f"{a_age}세" + (f", 부부형{a_sr}%" if a_joint else "") + (f", 보증 {a_gt}년" if a_gt else "") + ("" if esc == 0 else f", 체증 {esc:.0%}")
            items.append((f"연금보험 {a_amt:.1f}억 ({tag})",
                          {"annuity_premium": a_amt * 10000, "annuity_start_age": int(a_age), "annuity_joint": bool(a_joint),
                           "annuity_survivor_ratio": a_sr / 100, "annuity_guarantee_years": int(a_gt), "annuity_escalation": esc}))
        if c_ben > 0: items.append((f"간병보험 연 {c_ben:,}만 원 ({c_who})", {"care_benefit": float(c_ben), "care_member": 0 if c_who == "본인" else 1,
                                                                    "care_trigger": "severe" if c_trig.startswith("중증") else "any"}))
        if pc > 0: items.append((f"연금저축 월 {pc}만 원", {"pension_contrib_annual": float(pc * 12)}))
        if not items:
            st.info("추가할 상품의 금액을 하나 이상 입력하세요.")
        else:
            if len(items) > 1:
                allv = {}
                for _, v in items: allv.update(v)
                items.append(("모두 추가", allv))
            rows, results = [], {}
            with st.spinner("비교 계산 중"):
                for label, v in [("지금 계획", {})] + items:
                    r = run(p, {"addons": v} if v else {}); results[label] = r; rs = r["s"]
                    dlt, ci = paired(r, base) if v else (0.0, 0.0)
                    rows.append({"구성": label, "기본생활 유지": pct(rs["기본생활 유지 확률"]),
                                 "변화": "—" if not v else f"{-dlt * 100:+.1f}%p (±{ci * 100:.1f})",
                                 "가장 나쁜 30%의 부족액": f"{rs.get('CTE70 부족액(실질)', 0) / 10000:.2f}억",
                                 "안정적 생활 수준(1인 월)": f"{rs.get('확실성등가 소비(연, 1인)', 0) / 12:,.0f}만",
                                 "85세 금융자산 (중간값)": f"{r['p50_85'] / 10000:.2f}억",
                                 "여행·취미 충족": pct(rs.get("여행·취미 평균 충족률", 0)) if p["lifestyle"] > 0 else "—",
                                 "남길 자산 달성": pct(rs.get("유산 목표 달성 확률", 0)) if p["legacy"] > 0 else "—"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("가장 나쁜 30%의 부족액: 결과가 나쁜 쪽 30% 경우만 모아 평균 낸 부족액(오늘 가치) — 작을수록 최악의 경우가 덜 심각합니다. "
                       "안정적 생활 수준: 들쭉날쭉한 미래를 '매달 똑같이 이만큼 쓰는 것과 같다'로 환산한 금액 — 클수록 좋습니다. "
                       "연금·보험은 확률보다 이 두 지표에서 가치가 드러납니다.")
            if "stock" in info:
                stt = info["stock"]; r = next(v for k, v in results.items() if k.startswith("주식"))
                st.markdown(f"**종목 {stt['name']} 통계**" + (f" ({stt.get('start','')} ~ {stt.get('end','')}, {stt.get('months','')}개월)" if "months" in stt else ""))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("시장 베타", f"{stt['beta']:.2f}")
                c2.metric("기존 자산과 상관", f"{r['stock_corr']:.2f}" if r["stock_corr"] is not None else "—")
                c3.metric("연 변동성", f"{stt.get('sigma', np.sqrt(stt['beta']**2*0.18**2+stt['idio_sigma']**2)) * 100:.0f}%")
                c4.metric("금리 +1%p 때", f"{stt['rate_beta']:+.1f}%" if abs(stt["rate_beta"]) > 1e-9 else "자료 없음")
                st.caption("기대수익은 과거 수익률이 아니라 '금리 + 베타 × 위험프리미엄'으로 계산합니다. "
                           "금리 민감도는 ECOS 키가 설정된 경우에만 추정됩니다.")
            for k, r in results.items():
                if k.startswith("연금보험") and r["annuity_pay"] > 0:
                    st.caption(f"{k}: 예상 연금액 연 약 {r['annuity_pay']:,.0f}만 원 (처음 받는 금액, 사업비 5% 가정)")
                if k.startswith("간병보험") and r["care_premium"] > 0:
                    st.caption(f"{k}: 모델로 산출한 보험료 연 약 {r['care_premium']:,.0f}만 원 (80세까지 납입, 부가보험료 30% 가정, 지급 조건: {c_trig})")

with tab3:
    st.markdown("정해진 위기가 온다고 가정했을 때 목표가 얼마나 흔들리는지 봅니다.")
    ra = max(0, p["retire_age"] - p["age"])
    whatif([("평소", {}),
            ("은퇴 첫해 주가 −40%", {"eco": {"stock_shocks": {str(ra): -0.40}}}),
            ("은퇴 10년 뒤 주가 −40%", {"eco": {"stock_shocks": {str(ra + 10): -0.40}}}),
            ("3년 연속 물가 6%", {"eco": {"infl_shocks": {"0": .06, "1": .06, "2": .06}}}),
            ("예상보다 오래 삶 (사망률 −20%)", {"q_mult": 0.8}),
            ("간병 위험 2배", {"care_mult": 2.0})])
    st.caption("같은 폭락도 은퇴 직후에 올수록 타격이 큽니다 (수익률 순서 위험).")

with tab4:
    st.markdown("계좌와 명의를 바꾸면 세금·건보료와 목표 달성이 어떻게 달라지는지 비교합니다.")
    rows = [("지금 그대로", {}), ("ISA 활용", {"tax": {"use_isa": True}}), ("연금계좌 활용", {"tax": {"use_pension": True}}),
            ("ISA + 연금계좌", {"tax": {"use_isa": True, "use_pension": True}})]
    if p["children"] > 0:
        rows.append(("+ 자녀당 5천만 원씩 10년마다 증여", {"tax": {"use_isa": True, "use_pension": True, "gift_per_child_10y": 5000}}))
    out = []
    for label, v in rows:
        r = run(p, v); rs = r["s"]
        out.append({"방법": label, "기본생활 유지": pct(rs["기본생활 유지 확률"]),
                    "20년 세금+건보료": f"{(rs['20년 세금'] + rs['20년 건보료']) / 10000:.2f}억",
                    "가족에게 남는 돈(중앙값)": f"{rs['가족 이전 총액(중앙값)'] / 10000:.1f}억"})
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
    st.caption("증여는 가족에게 남는 돈을 늘리지만 내 기본생활 유지 확률을 낮출 수 있습니다.")

with tab8:
    st.markdown("상담 결과를 **고객에게 남길 리포트**로 만듭니다. 내려받아 열고 브라우저에서 인쇄(Ctrl+P) → "
                "'PDF로 저장'을 고르면 PDF가 됩니다.")
    if ss.get("load_warn"):
        st.warning(ss["load_warn"])
    c1, c2 = st.columns(2)
    client = c1.text_input("고객 성함", value=ss.get("client", ""), placeholder="예: 김철수", key="rep_client")
    advisor = c2.text_input("상담자 (선택)", value=ss.get("advisor", ""), placeholder="예: 홍길동 FP", key="rep_advisor")
    ss["client"] = client; ss["advisor"] = advisor
    cid = calc_id(p)
    st.caption(f"계산 ID **{cid}** · 모델 v{MODEL_VERSION} — 같은 입력과 같은 버전이면 언제 다시 계산해도 같은 결과입니다.")

    d1, d2 = st.columns(2)
    d1.download_button("고객 파일 저장 (.json)", save_blob(p, client, advisor),
                       file_name=f"노후나침반_{client or '고객'}_{date.today():%Y%m%d}.json",
                       mime="application/json", width="stretch",
                       help="다음 상담 때 첫 화면에서 이 파일을 올리면 입력이 그대로 복원됩니다.")

    if st.button("리포트 만들기", type="primary", key="report_go", width="stretch"):
        with st.spinner("리포트를 만드는 중"):
            ss["report_html"] = make_report(p, base, client, advisor, cid)
    if ss.get("report_html"):
        d2.download_button("리포트 내려받기 (.html)", ss["report_html"],
                           file_name=f"노후점검리포트_{client or '고객'}_{date.today():%Y%m%d}.html",
                           mime="text/html", width="stretch")
        st.markdown("**미리보기**")
        # HTML 문자열을 그대로 띄우려면 이 함수여야 한다 (st.iframe 은 주소만 받는다).
        # 앱 전체 스타일과 섞이지 않도록 iframe 안에서 그린다.
        st.components.v1.html(ss["report_html"], height=760, scrolling=True)

with tab5:
    st.markdown("""
**계산 방식**
- 금리·물가·주가가 서로 연결되어 움직이는 1만 가지 미래와, 통계청 생명표 기반 부부 각자의 수명·간병을 계산합니다.
- 목표는 **기본생활 → 여행·취미 → 남길 자산** 순으로 지킵니다. 매년 '기본생활비를 끝까지 내려면 남겨둘 돈'을 먼저 따로 두고, 그보다 많을 때만 여행·취미 예산을 씁니다. (균형형: 실제 수명 가능성과 연금 개시를 반영해 계산, 조금 더 오래 산다고 보수적으로 가정 / 안심형: 95세까지 지금 부족분이 계속된다고 가정)
- 국민연금, 사적연금 분리과세, 금융소득종합과세, 건강보험료, 재산세·종부세, 상속·증여세를 반영합니다.
- **유족연금**: 배우자가 먼저 돌아가시면 '유족연금(고인 연금의 60%, 가입 20년 이상 가정)'과 '내 국민연금 + 유족연금의 30%' 중 더 큰 쪽을 받는 것으로 계산합니다.
- **기초연금(2026, 월 약 35만 원)**: 65세 이상이면 소득·재산을 합친 '소득인정액'이 기준(단독 247만 원, 부부 395.2만 원) 이하일 때 받는 것으로 계산합니다. 부부 모두 받으면 각 20% 감액하고, 국민연금이 많으면 법에 정해진 산식대로 깎습니다(기준연금액의 50%가 하한). 재산 공제는 사시는 곳(대도시·중소도시·농어촌)에 따라 다릅니다. 기본생활 보호선에는 넣지 않았습니다(자격이 바뀔 수 있어 보수적으로 처리).
- 주택연금(한국주택금융공사 2026 월지급금표)과 집 줄이기(매도 비용·양도세·취득세)를 선택지로 비교할 수 있습니다.
- 집값도 오르내립니다. 물가 대비 연 1.25%씩 오르되 해마다 ±10% 정도 흔들리는 것으로 계산합니다(한국부동산원 2013~2026 자료 기준).
- **오래 살 위험**: 앞으로 수명이 얼마나 늘지 모른다는 점(사망률 불확실성)과, 배우자를 먼저 보낸 뒤 생존자의 사망 위험이 잠시 높아지는 점(사별 효과)을 넣었습니다. 후자는 해외 연구 기반 가정입니다.
- 20~40대는 소득·저축률·퇴직연금으로 은퇴 전 적립을 계산하고, 국민연금은 2026 개혁(소득대체율 43%) 기준으로 추정할 수 있습니다(근사치, 정확한 값은 국민연금공단 조회).

**알아두실 점**
- 사망률은 통계청 2024 완전생명표에 **앞으로의 수명 연장(코호트 개선)**을 반영, 금리·물가는 한국은행 ECOS 2000~2026년 데이터로 추정했습니다. 간병은 건강보험공단 장기요양 통계와 2026 본인부담·간병비 시세로, 주식은 KOSPI 2000~2026년 데이터로, 집값은 한국부동산원 주택가격지수(2013~2026)로 추정했습니다. 다만 **주식 위험프리미엄은 추정 오차가 커서**(5.2% ± 1.4%p) 결과의 절대값이 크게 움직일 수 있습니다. 결과의 절대값보다 **선택지 사이의 차이**를 보세요.
- 세법은 2026년 9월 기준으로 단순화했습니다. **투자 추천이나 세무 자문이 아니며**, 실제 결정 전 전문가와 상의하세요.
- 입력하신 정보는 계산에만 쓰이고 서버에 저장하지 않습니다.
""")
    st.markdown("모델 수식과 검증: [GitHub 저장소](https://github.com/minyoungkang1227/retirement-simulator)")

with tab7:
    st.markdown("""
**결과 숫자의 정확한 뜻**
- **기본생활 지킬 확률:** 1만 가지 미래(금리·물가·주가·수명·간병이 서로 다르게 흘러가는 경우)를 계산해, 살아있는 동안 기본생활비를 끝까지 낼 수 있었던 비율입니다. '10번 중 ○번'은 이 비율을 반올림한 값입니다.
- **몇 세쯤 모자라요:** 모자라게 되는 경우들만 모았을 때 그 가운데(중앙값) 나이입니다. 부부는 나이 적은 분 기준입니다.
- **여행·취미 예산:** 은퇴 후 살아있는 해마다 원하는 예산 중 실제로 쓴 비율의 평균입니다. 기본생활비가 위험해지면 이 예산부터 줄이도록 계산합니다.
- **남길 돈:** 마지막 분이 돌아가신 뒤 상속세를 빼고 남는 재산(집 포함, 주택연금·대출 상환 후)이 목표 이상일 확률입니다.
- **매달 꺼내 써야 하는 돈:** (기본생활비 + 여행 예산 ÷ 12) − (국민연금 + 사적연금 ÷ 12). 세금·건강보험료는 따로 계산해 결과에 반영합니다.

**가장 나쁜 30%의 부족액 (CTE70)**
- 1만 가지 미래를 결과가 나쁜 순서로 줄 세워, 나쁜 쪽 30%만 모아 평균 낸 부족액입니다. 미국 보험사 자본규제에서 쓰는 CTE와 같은 방식으로, '평균'보다 최악의 경우를 잘 보여줍니다.

**안정적 생활 수준 (확실성등가 소비)**
- 해마다 들쭉날쭉한 소비를 '매년 똑같이 이만큼 쓰는 것과 같다'로 바꾼 금액입니다. 경제학의 CRRA 효용(위험회피계수 3, 할인율 2%)으로 계산합니다. 소비가 불안정할수록 이 값이 낮아져서, 연금처럼 평생 일정하게 들어오는 돈의 가치가 드러납니다.

**계산 오차 (±)**
- 1만 가지 미래도 한정된 표본이라 약 ±1%p의 계산 오차가 있습니다. '바꿔보기'의 변화는 같은 1만 가지 미래를 공유해 비교하므로 오차가 ±0.5%p 안팎으로 더 작고, 변화가 이보다 작으면 '차이 없음'으로 판정합니다.

**10번 중 9번 지키려면**
- 모아둔 돈(또는 월 저축액)을 바꿔 가며 기본생활 지킬 확률이 90%가 되는 값을 찾은 것입니다(같은 미래 위에서 반복 계산). 권장 금액이 아니라 참고 수치입니다.

**사망률 개선**
- 의학 발달로 사망률은 해마다 낮아집니다. 연령대별로 연 0.3~2.5%씩 낮아지는 것으로 보고(통계청 생명표의 2013~2024 추이에 맞춤), 지금 60세가 80세가 될 때는 오늘의 80세보다 오래 사는 것으로 계산합니다. 이 때문에 기대여명이 60세 남성 23.7년 → 25.7년으로 늘어납니다.

**집값**
- 명목 집값 = 실질 집값 지수 × 물가. 실질 지수는 연 1.25%(추정 오차 ±1.18%p) 상승에 변동성 10%, 주식과 상관 0.1인 확률 과정입니다. 전국 지수의 연간 변동성 6.6%에 개별 주택의 고유 위험을 더한 값입니다.

**나이별로 남는 돈 그래프**
- 1만 가지 미래를 남은 금융자산 순서로 줄 세웠을 때 가운데(보통일 때), 아래에서 25% 지점(운이 나쁠 때), 위에서 25% 지점(운이 좋을 때)입니다. 오늘 돈 가치(물가를 뺀 값)이고, 집은 포함하지 않습니다. 가로축 나이는 부부 중 나이 적은 분 기준입니다.

**생활비를 지키는 방식**
- 균형형: 해마다 앞으로 모자랄 생활비 × 그때까지 살아있을 확률(조금 더 오래 산다고 보수적으로 가정)을 더해 '남겨둘 돈'을 정합니다. 국민연금 개시 시점도 반영합니다.
- 안심형: 95세까지 지금 모자라는 금액이 매년 계속된다고 보고 '남겨둘 돈'을 정합니다.

**현실 반영 계산**
- 물가와 금리가 함께 움직이는 관계, 펀드 수수료(연 0.5%), 가끔 오는 큰 주가 하락, 앞으로의 수익률 자체가 불확실하다는 점, 경증·중증으로 나뉜 간병, 통계청·한국은행·건강보험공단·KOSPI 실제 자료를 반영합니다.
""")
    st.markdown("수식과 검증 결과: [GitHub 저장소](https://github.com/minyoungkang1227/retirement-simulator)")

st.markdown('<p class="note">추천이 아닌 비교 결과입니다 · 제도·데이터 기준 2026-08 · © 2026 강민영</p>', unsafe_allow_html=True)
