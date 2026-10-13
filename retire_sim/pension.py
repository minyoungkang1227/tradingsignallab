"""한국 연금 제도 모듈 (v1: 국민연금 조기/연기 조정)."""
from .config import Person, NPSRules


def nps_annual_amount(p: Person, r: NPSRules) -> float:
    """조기/연기 반영한 국민연금 연액(현재가치, 물가연동)."""
    diff = p.nps_start_age - p.nps_normal_age
    if diff < 0:
        diff = max(diff, -r.max_early_years)
        factor = 1 + r.early_cut_per_year * diff
    else:
        diff = min(diff, r.max_defer_years)
        factor = 1 + r.defer_add_per_year * diff
    return p.nps_monthly * 12 * factor


# ── v20 국민연금 예상액 자동 계산 ──
# 2026 연금개혁: 소득대체율 43% (비례상수 1.2 × 43/40 = 1.29), 기준소득월액 상·하한 659만·41만 원(2026.7~)
# A값(전체 가입자 평균소득월액): 2025년 약 309만 원에 최근 변동률 3.4%를 반영한 근사 319만 원
NPS_A = 319.0
NPS_B_CAP, NPS_B_FLOOR = 659.0, 41.0
NPS_C = 1.29


def estimate_nps_monthly(salary_annual: float, years_past: float, age: int, retire_age: int,
                         contrib_end_age: int = 60) -> float:
    """정상수령 기준 국민연금 월액(만원, 오늘 가치) 근사.

    기본연금액(연) = c × (A + B) × (1 + 0.05 × (가입연수 − 20)),  가입 10년 미만은 0.
    B는 현재 월 소득(상·하한 적용)을 오늘 가치로 유지한다고 가정(임금과 A값이 같이 오른다고 봄).
    과거 가입기간의 더 높은 소득대체율, 크레딧, 부양가족연금은 반영하지 않음(보수적).
    """
    if salary_annual <= 0 and years_past <= 0:
        return 0.0
    B = min(max(salary_annual / 12, NPS_B_FLOOR), NPS_B_CAP) if salary_annual > 0 else NPS_B_FLOOR
    years = years_past + max(0, min(retire_age, contrib_end_age) - age)
    if years < 10:
        return 0.0
    annual = NPS_C * (NPS_A + B) * (1 + 0.05 * (min(years, 40) - 20))
    return round(annual / 12, 1)


# ── v28 유족연금 · 기초연금 ──
def survivor_pct(years: float) -> float:
    """유족연금 지급률: 가입 10년 미만 40%, 10~20년 50%, 20년 이상 60%."""
    return 0.4 if years < 10 else (0.5 if years < 20 else 0.6)


def survivor_annual(dec: Person) -> float:
    """사망자 기준 유족연금 연액(오늘 가치). 기본연금액(조기·연기 조정 전) × 지급률."""
    return dec.nps_monthly * 12 * survivor_pct(dec.nps_years)


def survivor_split(own, S, overlap: float):
    """본인 노령연금(own)과 유족연금(S)을 함께 받을 때 선택: max(S, own + overlap×S).
    반환 (본인연금으로 받는 몫, 유족연금으로 받는 몫). 배열 가능."""
    import numpy as np
    own = np.asarray(own, float); S = np.asarray(S, float)
    combo = own + overlap * S
    only_s = (S > combo) | (own <= 0)
    return np.where(only_s, 0.0, own), np.where(only_s, S, overlap * S)


def a_benefit_monthly(p: Person, r=None) -> float:
    """기초연금 연계감액에 쓰는 A급여액(소득재분배급여) 월액 근사, 만원.

    기본연금액 ≈ c(A + B)(1 + 0.05(n − 20)) 중 전체 가입자 평균소득(A)에서 나오는 몫만 떼어낸 값:
    A급여 ≈ c·A·(1 + 0.05(n − 20)) / 12.  가입 10년 미만은 0.
    실제 값은 국민연금공단에서 조회할 수 있고, 여기서는 가입기간만으로 추정한다."""
    if p.nps_years < 10:
        return 0.0
    return NPS_C * NPS_A * (1 + 0.05 * (min(p.nps_years, 40) - 20)) / 12


def basic_pension_one(nps_m, a_m, B):
    """1인 기초연금액(월, 명목). 국민연금 연계감액까지 반영한 법정 산식. 배열 가능.

      · 국민연금 급여액 ≤ 기준연금액 150%  → 기준연금액 전액
      · 그 밖에는  min(기준연금액, max(산식①, 산식②))
          산식① = 기준연금액 × 250% − 국민연금 급여액
          산식② = max(기준연금액 − ⅔ × A급여액, 0) + 부가연금액(기준연금액의 ½)
    두 산식 모두 하한이 부가연금액이라 연계감액만으로는 기준연금액의 50% 아래로 내려가지 않는다."""
    import numpy as np
    nps_m = np.asarray(nps_m, float); a_m = np.asarray(a_m, float)
    f1 = 2.5 * B - nps_m
    f2 = np.maximum(B - 2.0 / 3.0 * a_m, 0.0) + 0.5 * B
    cut = np.minimum(B, np.maximum(f1, f2))
    return np.where(nps_m <= 1.5 * B, B, cut)


def basic_pension_annual(income_m, asset_gross, fin, elig, own_nps_m, a_m, survivor_flag, r, cpi):
    """기초연금 가구 합계(연, 명목 만원), 경로별 벡터.

    income_m   : 월 소득평가액(명목)
    asset_gross: 부동산 등 일반재산(명목, 주택연금·주담대 차감 후)
    fin        : 금융재산(명목)
    elig       : 구성원별 '65세 이상 생존' 마스크 리스트
    own_nps_m  : 구성원별 월 국민연금 급여액(명목)      a_m: 구성원별 A급여액(명목)
    survivor_flag: 유족연금 수급 여부 — 유족연금은 연계감액 대상이 아니라 기준연금액 전액

    순서: ① 소득인정액으로 자격 판정 → ② 1인별 산정(연계감액) → ③ 부부 동시수급 20% 감액
          → ④ 소득역전방지 감액(소득인정액 + 기초연금이 선정기준액을 넘는 만큼 깎되 최저선 유지)"""
    import numpy as np
    n_elig = sum(e.astype(int) for e in elig)
    prop = np.maximum(asset_gross - r.base_asset() * cpi, 0) + np.maximum(fin - r.basic_fin_deduct * cpi, 0)
    inc_m = income_m + prop * r.basic_conv_rate / 12                 # 소득인정액(월)
    lim = np.where(n_elig >= 2, r.basic_limit_couple, r.basic_limit_single) * cpi
    ok = (inc_m <= lim) & (n_elig > 0)
    B = r.basic_amount * cpi
    tot = np.zeros_like(income_m)
    for e, nm, am, sf in zip(elig, own_nps_m, a_m, survivor_flag):
        amt = np.where(sf, B, basic_pension_one(nm, am, B))          # ②
        tot += np.where(e, amt, 0.0)
    couple = n_elig >= 2
    tot = np.where(couple, tot * (1 - r.basic_couple_cut), tot)      # ③
    floor = np.where(couple, 0.2 * B, 0.1 * B) * n_elig              # ④ 최저연금액(수급자 수만큼)
    over = np.maximum(inc_m + tot - lim, 0.0)
    tot = np.maximum(tot - over, np.minimum(tot, floor))
    return np.where(ok, tot, 0.0) * 12
