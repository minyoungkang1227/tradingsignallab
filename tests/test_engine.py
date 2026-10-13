"""회귀 테스트: 코드를 고쳐도 대표 가구의 결과가 크게 달라지지 않는지 자동 확인.
실행: pytest -q    (커밋할 때 GitHub Actions가 자동 실행)
"""
import numpy as np
import pytest

from retire_sim.config import Person, Household, SimConfig, CareMarkov, AddOns
from retire_sim.economy_v2 import EconomyV2, generate, gaussian2_price, vasicek_price, integrated_cov
from retire_sim import engine_tax, metrics, mortality
from retire_sim.tax import TaxConfig, HouseConfig, reverse_mortgage_monthly
from retire_sim.pension import estimate_nps_monthly

ECO = EconomyV2.enhanced()
CFG = SimConfig(care=CareMarkov())


def _run(hh, **kw):
    kw.setdefault("e", ECO); kw.setdefault("tc", TaxConfig(overseas_share=0.2))
    return engine_tax.run(hh, CFG, **kw)


def couple_60():
    return Household(members=[Person(60, "M", nps_monthly=110, private_pension_annual=600, private_pension_start=60),
                              Person(58, "F", nps_monthly=50)],
                     liquid_assets=50_000, stock_weight=0.4, annual_spending=3600, essential=3000, lifestyle=600,
                     legacy_target=10_000)


def couple_30():
    n1 = estimate_nps_monthly(5000, 3, 30, 60); n2 = estimate_nps_monthly(4000, 2, 29, 59)
    return Household(members=[Person(30, "M", nps_monthly=n1, salary=5000), Person(29, "F", nps_monthly=n2, salary=4000)],
                     liquid_assets=5000, stock_weight=0.5, annual_spending=3600, essential=3600, lifestyle=1200,
                     legacy_target=10_000, retire_age=60, saving_rate=0.15, pension_balance=1000, retirement_contrib=True)


# ── 1. 대표 가구 결과 (범위로 고정: 분산 감소·리팩터링은 허용, 로직 변화는 잡아냄) ──
@pytest.mark.parametrize("hh_fn,key,lo,hi", [
    (couple_60, "기본생활 유지 확률", 0.66, 0.77),
    (couple_30, "기본생활 유지 확률", 0.78, 0.88),
    (couple_30, "여행·취미 평균 충족률", 0.85, 0.97),
])
def test_headline(hh_fn, key, lo, hi):
    s = engine_tax.summarize(_run(hh_fn()))
    assert lo <= s[key] <= hi, f"{key}={s[key]:.3f} (허용 {lo}~{hi})"


# ── 2. 방향성: 당연히 이래야 하는 관계 ──
def test_directions():
    base = _run(couple_30())
    b = engine_tax.summarize(base)["기본생활 유지 확률"]
    more_save = engine_tax.summarize(_run(couple_30().__class__(**{**couple_30().__dict__, "saving_rate": 0.25})))["기본생활 유지 확률"]
    less_spend = engine_tax.summarize(_run(couple_30().__class__(**{**couple_30().__dict__, "essential": 3000, "annual_spending": 3000})))["기본생활 유지 확률"]
    later = engine_tax.summarize(_run(couple_30().__class__(**{**couple_30().__dict__, "retire_age": 65})))["기본생활 유지 확률"]
    assert more_save > b and less_spend > b and later > b


def test_shock_order():
    """같은 폭락도 은퇴 직후가 더 치명적이어야 한다 (수익률 순서 위험)."""
    hh = couple_60()
    early = engine_tax.summarize(_run(hh, e=EconomyV2.enhanced(stock_shocks={0: -0.4})))["기본생활 유지 확률"]
    late = engine_tax.summarize(_run(hh, e=EconomyV2.enhanced(stock_shocks={10: -0.4})))["기본생활 유지 확률"]
    assert early < late


# ── 3. 이론값 검증 ──
def test_economy_moments():
    d = generate(EconomyV2.enhanced(erp_sd=0, pi_theta_sd=0, q_theta_sd=0, jump_lambda=0, fee=0), 60, 20_000,
                 np.random.default_rng(0))
    e = EconomyV2.enhanced()
    assert abs(d["rate"][:, -1].mean() - (e.q_theta + e.pi_theta)) < 0.004
    assert abs(d["infl"][:, -1].mean() - e.pi_theta) < 0.004


def test_bond_price_reduces_to_vasicek():
    e = EconomyV2()
    v = vasicek_price(5.0, np.array([0.025]), e)[0]
    g = gaussian2_price(5.0, np.array([0.025]), np.array([0.0]), e.r_kappa, e.r_theta, e.r_sigma, 1.0, 0.0, 1e-9, 0.0)[0]
    assert abs(v - g) < 1e-9


def test_integrated_cov_symmetric_psd():
    M = integrated_cov(0.2, 0.008, 0.4, 0.01, 0.3, -0.2, -0.1)
    assert np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > -1e-12


def test_life_expectancy_matches_table():
    q = mortality.default_qx()
    assert abs(mortality.life_expectancy(q["M"], 60) - 23.7) < 0.2
    assert abs(mortality.life_expectancy(q["F"], 60) - 28.4) < 0.2


def test_care_preserves_mortality():
    q = mortality.default_qx()
    al, ic, sv = mortality.simulate_life(q["M"], 60, 45, 40_000, np.random.default_rng(0), CareMarkov(), return_severe=True)
    assert abs((al.sum(1).mean() - 0.5) - 23.7) < 0.5
    assert 0.05 < sv[:, 5:].sum() / max(ic[:, 5:].sum(), 1) < 0.25       # 중증 비율 ≈ 12.7%


def test_reverse_mortgage_table():
    assert abs(reverse_mortgage_monthly(70, 30_000) - 92.3) < 1.5        # 공사 표: 70세·3억 → 월 92.3만원
    assert reverse_mortgage_monthly(80, 50_000) > reverse_mortgage_monthly(70, 50_000)


# ── 4. 계산 신뢰성 ──
def test_paired_ci_smaller_than_independent():
    hh = couple_60()
    a, b = _run(hh), _run(Household(**{**hh.__dict__, "stock_weight": 0.6}))
    c = metrics.compare(b, a)
    assert c["±"] < c["독립 가정 시 ±"]


def test_reproducible():
    s1 = engine_tax.summarize(_run(couple_60()))["기본생활 유지 확률"]
    s2 = engine_tax.summarize(_run(couple_60()))["기본생활 유지 확률"]
    assert s1 == s2


# ── 5. 기능이 꺼져 있을 때 결과가 바뀌지 않아야 ──
def test_addons_noop():
    hh = couple_60()
    a = engine_tax.summarize(_run(hh))["기본생활 유지 확률"]
    b = engine_tax.summarize(_run(hh, addons=AddOns()))["기본생활 유지 확률"]
    assert a == b


def test_house_strategies_help_cash_poor():
    hh = Household(members=[Person(65, "M", nps_monthly=130), Person(63, "F", nps_monthly=60)],
                   liquid_assets=30_000, stock_weight=0.5, annual_spending=3000, essential=3000, lifestyle=1200)
    H = HouseConfig(official=50_000, market=70_000, n_houses=1, years_held=20, hi_property_monthly=15)
    base = engine_tax.summarize(_run(hh, house=H))["기본생활 유지 확률"]
    rm = engine_tax.summarize(_run(hh, house=HouseConfig(**{**H.__dict__, "reverse_mortgage_age": 70})))["기본생활 유지 확률"]
    assert rm > base + 0.2


# ── 6. 집값 위험 (v26) ──
def test_house_index_distribution():
    from retire_sim.economy_v2 import generate
    d = generate(EconomyV2.enhanced(), 30, 20_000, np.random.default_rng(0))
    H = d["house"]
    assert H.shape[1] == 31 and np.allclose(H[:, 0], 1.0)
    lr = np.diff(np.log(H), axis=1)
    assert 0.08 < lr.std() < 0.12                                   # 연 변동성 ≈ 10%
    assert 0.0 < np.log(np.median(H[:, 30])) / 30 < 0.03            # 실질 상승률 ≈ 1.25%
    assert np.percentile(H[:, 30], 5) < 0.8 < 1.3 < np.percentile(H[:, 30], 95)


def test_house_risk_off_reproduces_v25():
    """집값 파라미터를 끄면 v25와 같아야 한다(집값 = 물가만큼)."""
    flat = EconomyV2.enhanced(house_sigma=0, house_real_growth=0, house_real_sd=0)
    d = generate(flat, 20, 5000, np.random.default_rng(0))
    assert np.allclose(d["house"], 1.0)


# ── 7. 사망률 개선·지표·연금 상품 (v27) ──
def test_mortality_improvement():
    q = mortality.default_qx()
    assert mortality.life_expectancy(q["M"], 60, improve=True) > mortality.life_expectancy(q["M"], 60) + 1.0
    p1 = mortality.project_qx(q["M"], 10)
    assert (p1[30:100] < q["M"][30:100]).all() and p1[-1] == 1.0


def test_cte_monotone_and_definition():
    from retire_sim.engine_tax import cte
    x = np.arange(100.0)
    assert cte(x, 0.9) == pytest.approx(x[90:].mean())
    s = engine_tax.summarize(_run(couple_60()))
    assert s["평균 부족액(실질)"] <= s["CTE70 부족액(실질)"] <= s["CTE90 부족액(실질)"]


def test_certainty_equivalent_positive_and_responsive():
    rich = Household(**{**couple_60().__dict__, "liquid_assets": 200_000})
    a = engine_tax.summarize(_run(couple_60()))["확실성등가 소비(연, 1인)"]
    b = engine_tax.summarize(_run(rich))["확실성등가 소비(연, 1인)"]
    assert 0 < a < b


def test_annuity_variants_pricing_order():
    """같은 보험료라면: 보증기간·체증·부부형이 붙을수록 처음 받는 금액이 작아야 한다."""
    hh = couple_60()
    def pay(**kw):
        kw.setdefault("annuity_start_age", 65)
        return _run(hh, addons=AddOns(annuity_premium=10_000, **kw))["annuity_pay"]
    plain = pay()
    assert pay(annuity_guarantee_years=20) < plain
    assert pay(annuity_escalation=0.02) < plain
    assert pay(annuity_joint=True) < plain
    assert plain < pay(annuity_start_age=80)                       # 늦게 받을수록 금액은 커진다
    assert pay(annuity_joint=True) < pay(annuity_joint=True, annuity_survivor_ratio=0.7)


# ── v28 제도 정밀화: 유족연금 · 기초연금 ──
def test_survivor_rules():
    from retire_sim.pension import survivor_pct, survivor_split, survivor_annual
    assert (survivor_pct(5), survivor_pct(10), survivor_pct(19.9), survivor_pct(20), survivor_pct(35)) == (0.4, 0.5, 0.5, 0.6, 0.6)
    own, sv = survivor_split(np.array([100.0, 10.0, 0.0]), np.array([60.0, 60.0, 60.0]), 0.30)
    # own 100: 100+18=118 > 60 → 본인연금+30%; own 10: 10+18=28 < 60 → 유족연금만; own 0 → 유족연금만
    assert np.allclose(own, [100, 0, 0]) and np.allclose(sv, [18, 60, 60])
    assert abs(survivor_annual(Person(65, "M", nps_monthly=100, nps_years=25)) - 720) < 1e-9


def test_basic_pension_rules():
    """기초연금 산정: 자격 판정 → 연계감액 → 부부감액 → 소득역전방지."""
    from retire_sim.pension import basic_pension_annual, basic_pension_one
    from retire_sim.config import NPSRules
    r = NPSRules(); one = np.ones(1); z = np.zeros(1); B = r.basic_amount
    yes, no = one > 0, one < 0

    def call(income=0.0, n=1, nps=0.0, a=0.0, sv=False, asset=0.0, fin=0.0, rules=r):
        inc = np.array([float(income)])
        return basic_pension_annual(inc, np.array([float(asset)]), np.array([float(fin)]),
                                    [yes] * n, [np.array([float(nps)])] * n, [np.array([float(a)])] * n,
                                    [sv * one > 0] * n, rules, one)[0]

    assert abs(call() - B * 12) < 1e-6                                   # 단독·국민연금 없음 → 기준연금액
    assert abs(call(n=2) - 2 * B * 0.8 * 12) < 1e-6                      # 부부 모두 수급 → 각 20% 감액
    assert call(income=300.0) == 0                                       # 선정기준액(단독 247만) 초과 → 0
    assert basic_pension_annual(z, z, z, [no], [z], [z], [no], r, one)[0] == 0   # 64세는 대상 아님

    # 연계감액: 국민연금이 기준연금액의 150% 이하면 A급여와 무관하게 전액
    assert basic_pension_one(np.array([1.5 * B]), np.array([3 * B]), B)[0] == B
    # A급여가 0이면 산식②가 기준연금액을 넘어 상한에 걸리므로 국민연금이 아무리 많아도 감액 없음
    assert basic_pension_one(np.array([5 * B]), np.array([0.0]), B)[0] == B
    # 두 산식 중 큰 값: A=1.2B, 국민연금=1.9B → 산식① 0.6B, 산식② 0.7B → 0.7B
    mid = basic_pension_one(np.array([1.9 * B]), np.array([1.2 * B]), B)[0]
    assert abs(mid - 0.7 * B) < 1e-9 and 0.5 * B < mid < B
    # 하한은 부가연금액(기준연금액의 50%)
    assert abs(basic_pension_one(np.array([5 * B]), np.array([5 * B]), B)[0] - 0.5 * B) < 1e-9
    assert call(nps=5 * B, a=5 * B, sv=True) == B * 12                   # 유족연금 수급자는 연계감액 없음

    # 지역별 기본재산액: 공제가 작은 농어촌은 같은 재산이라도 소득인정액이 커져 탈락할 수 있다
    # 재산 8.5억: 대도시는 (8.5 − 1.35)억 환산 월 238만 < 247만 → 수급, 농어촌은 (8.5 − 0.725)억 → 259만 → 탈락
    assert call(asset=85000.0) > 0
    assert call(asset=85000.0, rules=NPSRules(basic_region="농어촌")) == 0
    assert call(asset=85000.0, rules=NPSRules(basic_region="중소도시")) == 0


def test_a_benefit_and_region():
    from retire_sim.pension import a_benefit_monthly, NPS_A, NPS_C
    from retire_sim.config import NPSRules
    assert a_benefit_monthly(Person(65, "M", nps_years=9)) == 0                  # 10년 미만은 0
    assert abs(a_benefit_monthly(Person(65, "M", nps_years=20)) - NPS_C * NPS_A / 12) < 1e-9
    assert a_benefit_monthly(Person(65, "M", nps_years=40)) > a_benefit_monthly(Person(65, "M", nps_years=20))
    assert a_benefit_monthly(Person(65, "M", nps_years=50)) == a_benefit_monthly(Person(65, "M", nps_years=40))
    assert (NPSRules().base_asset(), NPSRules(basic_region="중소도시").base_asset(),
            NPSRules(basic_region="농어촌").base_asset()) == (13500.0, 8500.0, 7250.0)


def test_v28_directions():
    from retire_sim.config import NPSRules
    global CFG
    old = CFG
    try:
        res = {}
        for sv, bs in [(False, False), (True, False), (False, True)]:
            CFG = SimConfig(care=CareMarkov(), nps=NPSRules(survivor_on=sv, basic_on=bs))
            res[(sv, bs)] = engine_tax.summarize(_run(couple_60()))["기본생활 유지 확률"]
        assert res[(True, False)] > res[(False, False)]
        assert res[(False, True)] > res[(False, False)]
    finally:
        CFG = old


# ── v29 최적화: 글라이드 경로 · 인출 순서 · 탐색 ──
def test_glide_and_order():
    from dataclasses import replace
    hh = couple_60()
    flat = engine_tax.summarize(_run(hh))
    same = engine_tax.summarize(_run(replace(hh, glide_end=hh.stock_weight)))      # 시작=끝이면 고정 비중과 동일
    assert abs(flat["기본생활 유지 확률"] - same["기본생활 유지 확률"]) < 1e-9
    low = engine_tax.summarize(_run(replace(hh, glide_end=0.1)))
    assert low["기본생활 유지 확률"] != flat["기본생활 유지 확률"]
    isa = engine_tax.summarize(_run(replace(hh, draw_order="isa_first")))
    assert 0 <= isa["고갈확률"] <= 1


def test_optimize_search_structure():
    from retire_sim import optimize
    out = optimize.search(couple_60(), CFG, "nps", e=ECO, tc=TaxConfig(overseas_share=0.2))
    assert len(out["rows"]) == 5 and out["rows"][2]["고갈확률 차이(기준 대비)"] == 0.0
    assert set(out["check"]) == {"기준", "추천 후보"}
    # 연금 개시를 미루면 (같은 시나리오에서) 이 가구는 기본생활 유지 확률이 높아진다
    assert out["rows"][4]["기본생활 유지 확률"] > out["rows"][0]["기본생활 유지 확률"]


# ── v30 장수위험: 확률적 사망률 · 사별 효과 ──
def test_longevity_multiplier():
    m = mortality.longevity_multiplier(20000, 30, 0.05, 0.005, 42)
    assert m.shape == (20000, 30) and abs(m.mean() - 1) < 0.02          # 평균 1 근처(분산 보정)
    assert m[:, 25].std() > m[:, 1].std()                                # 시간이 갈수록 불확실성 증가
    assert mortality.longevity_multiplier(10, 5, 0, 0, 42) is None
    assert np.allclose(mortality.longevity_multiplier(100, 5, 0.05, 0.005, 7), mortality.longevity_multiplier(100, 5, 0.05, 0.005, 7))


def test_widowhood_effect():
    q = mortality.default_qx(110)
    n, T = 40000, 30
    rng = np.random.default_rng(0)
    al = np.stack([mortality.simulate_alive(q["M"], 70, T, n, rng), mortality.simulate_alive(q["F"], 68, T, n, rng)])
    a2, _, _ = mortality.apply_widowhood(al, None, None, [q["M"], q["F"]], [70, 68], ["M", "F"], (0.3, 0.15), 3.0, False, 1)
    assert (a2 <= al).all()                                               # 추가 사망만 생길 수 있음
    both = al[0].sum(1) + al[1].sum(1); both2 = a2[0].sum(1) + a2[1].sum(1)
    assert both2.mean() < both.mean()
    off, _, _ = mortality.apply_widowhood(al, None, None, [q["M"], q["F"]], [70, 68], ["M", "F"], (0, 0), 3.0, False, 1)
    assert (off == al).all()


def test_v30_tail_risk_up():
    global CFG
    old = CFG
    try:
        CFG = SimConfig(care=CareMarkov(), longevity_sigma_level=0, longevity_sigma_trend=0, widow_excess=(0, 0))
        off = engine_tax.summarize(_run(couple_30()))["CTE90 부족액(실질)"]
        CFG = SimConfig(care=CareMarkov(), widow_excess=(0, 0))
        on = engine_tax.summarize(_run(couple_30()))["CTE90 부족액(실질)"]
        assert on > off                                                   # 장수 불확실성은 꼬리 위험을 키운다
    finally:
        CFG = old


# ── v31 해석해 대조: 결정론적 극한에서 닫힌식과 정확히 일치해야 함 ──
def _deterministic_setup(R=0.03, spend=3000.0, infl=0.0):
    """변동성·세금·사망·간병·연금을 모두 끈 세계. 유일한 움직임은 수익률과 지출."""
    from retire_sim.config import NPSRules
    eco = EconomyV2(r0=R, r_theta=R, r_sigma=0.0, q0=R - infl, q_theta=R - infl, q_sigma=0.0,
                    pi0=infl, pi_theta=infl, pi_sigma=0.0, erp=0.0, s_sigma=0.0, fee=0.0,
                    jump_lambda=0.0, erp_sd=0.0, pi_theta_sd=0.0, q_theta_sd=0.0, antithetic=False,
term_premium=0.0, house_real_sd=0.0, house_sigma=0.0)
    qx = {"M": np.zeros(111), "F": np.zeros(111)}
    qx["M"][100] = qx["F"][100] = 1.0                       # 100세까지 아무도 죽지 않음
    hh = Household(members=[Person(60, "M")], liquid_assets=50_000, stock_weight=1.0,
                   annual_spending=spend, essential=spend, lifestyle=0.0, legacy_target=0.0,
                   survivor_spending_ratio=1.0, floor_method="fixed95")
    cfg = SimConfig(n_paths=64, seed=1, care=CareMarkov(enabled=False), mortality_improvement=False,
                    longevity_sigma_level=0.0, longevity_sigma_trend=0.0, widow_excess=(0.0, 0.0),
                    nps=NPSRules(basic_on=False, survivor_on=False))
    return hh, cfg, eco, qx


def test_analytic_benchmark_no_inflation():
    """W(t+1) = (W(t) − 지출) × e^r 와 상대오차 1e-7 안에서 같아야 한다 (지출은 연초, 수익은 연말)."""
    R, spend = 0.03, 3000.0
    hh, cfg, eco, qx = _deterministic_setup(R, spend)
    res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), qx_table=qx)
    W = res["W_real"][0]
    w = float(hh.liquid_assets)
    for t in range(15):
        w = (w - spend) * np.exp(R)
        assert abs(W[t + 1] - w) <= 1e-7 * abs(w), f"{t + 1}년차: 모형 {W[t + 1]:.4f} vs 닫힌식 {w:.4f}"


def test_analytic_benchmark_with_inflation():
    """물가가 2%여도 실질자산은 실질수익률 기준 닫힌식과 같아야 한다."""
    R, infl, spend = 0.05, 0.02, 3000.0
    hh, cfg, eco, qx = _deterministic_setup(R, spend, infl)
    res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), qx_table=qx)
    W = res["W_real"][0]
    w = float(hh.liquid_assets)
    for t in range(15):
        w = (w - spend) * np.exp(R) / (1 + infl)            # 실질 = 명목 ÷ 물가지수
        assert abs(W[t + 1] - w) <= 1e-7 * abs(w), f"{t + 1}년차: 모형 {W[t + 1]:.4f} vs 닫힌식 {w:.4f}"


def test_analytic_pension_offsets_spending():
    """연금이 들어오면 꺼내 쓰는 돈이 그만큼 줄어야 한다 (연 1,200만 원 = 월 100만 원)."""
    R, spend = 0.03, 3000.0
    hh, cfg, eco, qx = _deterministic_setup(R, spend)
    hh.members[0].nps_monthly = 100.0                      # 조기·연기 조정이 없도록 정상개시=개시=60세
    hh.members[0].nps_normal_age = hh.members[0].nps_start_age = 60
    res = engine_tax.run(hh, cfg, e=eco, tc=TaxConfig(enabled=False), qx_table=qx)
    W = res["W_real"][0]
    w = float(hh.liquid_assets)
    for t in range(15):
        w = (w - (spend - 1200.0)) * np.exp(R)
        assert abs(W[t + 1] - w) <= 1e-7 * abs(w), f"{t + 1}년차: 모형 {W[t + 1]:.4f} vs 닫힌식 {w:.4f}"


# ── v33 상담 세션: 저장·복원과 재현 ID ──
def test_session_roundtrip_and_calc_id():
    import json as _json
    from retire_sim import session, __version__

    inp = {"age": 60, "essential": 250, "spouse": True, "events": [{"age": 70, "amount": 1000}]}
    blob = session.save_blob(inp, "김철수", "홍길동 FP")
    back, client, advisor, warn = session.load_blob(blob)
    assert back == inp and client == "김철수" and advisor == "홍길동 FP" and warn == ""
    assert session.load_blob(blob.encode())[0] == inp                      # bytes 로 넣어도 같다

    # 계산 ID: 키 순서와 무관, 입력이 바뀌면 달라짐, 모델 버전이 바뀌어도 달라짐
    cid = session.calc_id(inp)
    assert len(cid) == 10 and cid == session.calc_id(dict(reversed(list(inp.items()))))
    assert cid != session.calc_id({**inp, "essential": 251})
    assert cid != session.calc_id(inp, version="999")
    assert _json.loads(blob)["calc_id"] == cid

    # 다른 모델 버전에서 저장된 파일은 경고와 함께 열린다
    d = _json.loads(blob); d["model_version"] = "1"
    back2, _, _, warn2 = session.load_blob(_json.dumps(d))
    assert back2 == inp and str(__version__) in warn2

    for bad in ('{"hello": 1}', '{"format": "retire-compass/1"}', '{"format": "other", "inputs": {}}'):
        with pytest.raises(ValueError):
            session.load_blob(bad)


def test_report_html_is_self_contained():
    """리포트는 외부 자료 없이 열려야 한다 — 그림은 SVG 문자열로 직접 그린다."""
    from retire_sim import report

    rows = [(60, 5.0, 4.0, 3.0), (70, 6.0, 4.0, 2.0), (80, 7.0, 3.0, 0.0)]
    svg = report.line_chart_svg(rows)
    assert svg.startswith("<svg") and svg.count("<polyline") == 3
    assert report.line_chart_svg([]) == ""
    # 폭포: 0인 항목은 빼고, 남는 부족분은 '모아둔 돈에서' 막대로 이어 붙인다
    wf = report.waterfall_svg([("연금", 2400), ("생활비", -3600), ("세금", 0.0)])
    assert wf.count("<rect") == 3 and "모아둔 돈에서" in wf
    assert report.waterfall_svg([("연금", 4000), ("생활비", -3600)]).count("<rect") == 2   # 남으면 안 붙는다
    assert report.waterfall_svg([]) == ""

    h = report.build(dict(client="김<철>수", advisor_line="", date="2026년 10월 13일", calc_id="ABC1234567",
                          version="33", hero="테스트", goals=[("A", "B", "C")], gap_eq="x", gap_big="y",
                          band="z", actions=[("무엇", "어떻게", "10% → 20%")], fan_svg=svg, flow_svg=wf,
                          assumptions="가정"))
    assert h.startswith("<!doctype html>") and "ABC1234567" in h and "무엇" in h
    assert "김&lt;철&gt;수" in h and "<철>" not in h                        # 고객명은 HTML 이스케이프
    assert "@media print" in h and "page-break-before" in h                # 인쇄 레이아웃
    assert "<img" not in h and "<script" not in h                          # 외부 이미지·스크립트 없음
