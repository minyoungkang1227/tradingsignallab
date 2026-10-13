"""입력 스키마. 모든 금액은 '현재가치 기준 만원/년'."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Person:
    age: int
    sex: str                      # "M" | "F"
    nps_monthly: float = 0.0      # 국민연금 예상 월액(정상수령 기준, 만원)
    nps_normal_age: int = 65      # 정상 수령 개시 나이
    nps_start_age: int = 65       # 실제 수령 개시 나이(조기/연기 반영)
    nps_years: float = 20.0       # (v28) 국민연금 가입기간(년). 유족연금 지급률(40/50/60%) 결정
    private_pension_annual: float = 0.0   # 사적연금 연액(명목 고정, 만원)
    private_pension_start: int = 60
    private_pension_years: int = 20
    salary: float = 0.0           # (v20) 연 세전 근로소득(만원, 오늘 가치). 은퇴 전 저축·퇴직연금 계산용


@dataclass
class Household:
    members: List[Person]
    liquid_assets: float          # 금융자산 합계(만원)
    stock_weight: float = 0.5     # 주식 비중(연 1회 리밸런싱)
    glide_end: float = None       # (v29) 글라이드 경로: 종료 나이의 주식 비중 (None이면 고정 비중)
    glide_ages: tuple = (60, 85)  # (v29) 본인 나이 기준 (시작, 종료): 시작 이전엔 stock_weight, 종료 이후엔 glide_end
    draw_order: str = "taxable_first"  # (v29) 인출 순서: "taxable_first"(과세계좌→ISA) | "isa_first"(ISA→과세계좌)
    annual_spending: float = 3600 # 부부 기준 연 생활비(만원, 현재가치)
    survivor_spending_ratio: float = 0.7  # 한 명 사망 시 생활비 비율
    # ── v12 목표 기반(Goal-Based) 설정 ──
    essential: Optional[float] = None     # 기본생활비(연, 만원). None이면 annual_spending 사용
    lifestyle: float = 0.0                # 여행·취미 등 선택 지출(연, 만원). 자산이 부족하면 먼저 줄임
    legacy_target: float = 0.0            # 남기고 싶은 금액(만원, 현재가치)
    retire_age: Optional[int] = None      # 본인(첫 구성원) 은퇴 나이. None이면 이미 은퇴
    annual_saving: float = 0.0            # 은퇴 전 연 저축액(만원, 현재가치)
    planning_age: int = 95                # (v12 방식) 기본생활비 보호선 계산용 계획 나이
    # ── v20 적립기 강화 (20~40대) ──
    saving_rate: float = 0.0              # 세전 근로소득 대비 저축률. >0이면 annual_saving 대신 소득 × 저축률
    wage_growth: float = 0.01             # 실질 임금상승률(연)
    pension_balance: float = 0.0          # 퇴직연금·연금저축 적립금(만원) — 55세 이후 연금으로 인출, 인출 시 과세
    retirement_contrib: bool = False      # 직장 퇴직연금 적립(매년 연봉의 1/12, 은퇴 전)
    # ── v21 큰 지출 이벤트 ──
    # [{"label": "자녀 대학", "age": 48, "amount": 1000, "years": 4, "kind": "cost"},        (연 금액)
    #  {"label": "주택 구입", "age": 35, "amount": 60000, "kind": "house", "ltv": 0.6,       (집값)
    #   "loan_years": 30, "deposit_back": 30000, "rent_saving": 1200}]                      (전세 보증금 회수, 절감 주거비 연)
    # 본인 나이 기준, 금액은 만원·오늘 가치
    events: list = field(default_factory=list)
    floor_method: str = "actuarial"       # "actuarial"(v16): 생존확률 가중 현가 / "fixed95"(v12): 95세까지 확정
    floor_mort_mult: float = 0.8          # 보호선 산출용 사망률 배수 (<1이면 오래 사는 쪽으로 보수적)
    floor_real_rate: float = 0.02         # 보호선 할인율(실질)


@dataclass
class NPSRules:
    """국민연금 조기/연기 조정률 — 개발 전 최신 기준 재확인 필요."""
    early_cut_per_year: float = 0.06
    defer_add_per_year: float = 0.072
    max_early_years: int = 5
    max_defer_years: int = 5
    # (v28) 유족연금: 가입 10년 미만 40%, 10~20년 50%, 20년 이상 60% (기본연금액 기준)
    survivor_on: bool = True
    survivor_overlap: float = 0.30      # 본인 노령연금과 함께 받을 때 유족연금의 지급 비율(중복지급률)
    # (v28) 기초연금 (2026): 기준연금액 월 34.97만원, 선정기준액 단독 247만·부부 395.2만원(소득인정액)
    basic_on: bool = True
    basic_amount: float = 34.97         # 월, 만원 (오늘 가치, 물가연동)
    basic_limit_single: float = 247.0
    basic_limit_couple: float = 395.2
    basic_region: str = "대도시"         # 기본재산액 공제 지역: 대도시 | 중소도시 | 농어촌
    basic_base_assets: tuple = (13500.0, 8500.0, 7250.0)   # (대도시, 중소도시, 농어촌) 기본재산액(만원, 2026)
    basic_fin_deduct: float = 2000.0    # 금융재산 공제(만원)
    basic_conv_rate: float = 0.04       # 재산의 소득환산율(연)
    basic_couple_cut: float = 0.20      # 부부 모두 수급 시 각 20% 감액

    def base_asset(self) -> float:
        """거주 지역의 기본재산액 공제(만원). 대도시=특별시·광역시, 중소도시=도의 시, 농어촌=도의 군."""
        idx = {"대도시": 0, "중소도시": 1, "농어촌": 2}.get(self.basic_region, 0)
        return self.basic_base_assets[idx]


@dataclass
class CareShock:
    """간병비 점프: 나이 >= start_age인 생존자에게 연 lam 확률로 발생."""
    enabled: bool = True
    start_age: int = 75
    lam: float = 0.03
    cost_median: float = 2000     # 발생 시 연 비용 중앙값(만원)
    cost_log_sigma: float = 0.5
    duration_years: int = 3


@dataclass
class AddOns:
    """v15 '상품 추가해 보기': 기존 계획에 금융상품을 더했을 때 위험이 어떻게 바뀌는지 비교.
    금액 단위 만원(현재가치). 주식·예금·연금 일시납은 기존 금융자산에서 옮기는 것으로 가정."""
    # 개별 주식 (종목 통계는 market_data.stock_stats로 추정)
    stock_amount: float = 0.0
    stock_beta: float = 1.0          # 시장(KOSPI) 베타
    stock_idio_sigma: float = 0.25   # 고유 변동성(연)
    stock_rate_beta: float = 0.0     # 금리 1.0(=100%p) 변화당 로그수익률 변화 (예: -2.0 → 금리 +1%p에 -2%)
    stock_name: str = ""
    # 예금 (단기금리로 운용, 이자는 금융소득)
    deposit_amount: float = 0.0
    # 종신연금 (일시납 즉시·거치형, 연금보험 비과세 가정)
    annuity_premium: float = 0.0
    annuity_start_age: int = 65      # 본인 나이 기준 가입·개시 시점
    annuity_joint: bool = False      # 부부 중 한 명이라도 살아있으면 지급
    annuity_survivor_ratio: float = 1.0   # (v27) 부부형에서 한 명 사망 후 지급 비율 (0.7 = 70%)
    annuity_guarantee_years: int = 0      # (v27) 보증기간(년): 사망해도 이 기간까지는 지급
    annuity_escalation: float = 0.0       # (v27) 체증형: 매년 명목 증액률 (0.02 = 연 2%)
    annuity_loading: float = 0.05    # 사업비
    annuity_rate: float = 0.03       # 가격 산출 할인율(공시이율 가정)
    annuity_mort_mult: float = 0.8   # 가입자 선택효과: 사망률 80%로 가격 산출
    # 간병보험 (간병 상태에서 매년 정액 지급, 다중상태 모델 필요)
    care_benefit: float = 0.0        # 연 보장액(명목 정액)
    care_member: int = 0             # 피보험자 (0=본인, 1=배우자)
    care_pay_until: int = 80         # 보험료 납입 종료 나이
    care_loading: float = 0.3        # 부가보험료율
    care_rate: float = 0.03
    care_trigger: str = "severe"     # 지급 조건: "severe"(1~2등급 중증) / "any"(모든 장기요양 등급)
    # 연금저축·IRP 추가 납입 (은퇴 전, 세액공제 후 연금계좌로)
    pension_contrib_annual: float = 0.0
    pension_credit_rate: float = 0.132


@dataclass
class CareMarkov:
    """간병 다중상태 마르코프 모델.

    v17(기본): 건강(H) → 경증 간병(M, 장기요양 3~5등급·인지지원) → 중증 간병(S, 1~2등급) → 사망(D)
      매년: H→M (1-q_H)·i(x)·(1-e), H→S (1-q_H)·i(x)·e, M→S progress, M·S→H recovery,
            사망률 q_H·m_M(경증), q_H·m_S(중증)
      q_H는 생명표 q_x가 보존되도록 목표 유병률 p(x)와 중증 비율 s로 보정:
            q_H = q_x / (1 + (m_M−1)·p(x)(1−s) + (m_S−1)·p(x)·s)
      발생률 i(x)와 진행률은 건강보험공단 장기요양 통계(2025 인정자 123.5만 명, 등급 구성)와
      연령 분포(2022 장기요양실태조사), 2025 고령자 통계로 만든 연령별 유병률에 맞춰 보정
      (scripts/calibrate_care.py). 비용은 2026 장기요양 본인부담·비급여·간병비 시세 기준(오늘 가치, 연).
    v11~v16: CareMarkov.legacy() — 단일 간병 상태, 임시값.
    """
    enabled: bool = True
    two_level: bool = True
    incidence: tuple = ((65, 0.0015), (70, 0.0043), (75, 0.0075), (80, 0.0268), (85, 0.0693), (90, 0.0998), (999, 0.1260))
    prevalence: tuple = ((65, 0.003), (70, 0.015), (75, 0.039), (80, 0.099), (85, 0.240), (90, 0.423), (999, 0.554))
    severe_share: float = 0.127       # 인정자 중 1~2등급 비율 (2025: 4.5% + 8.2%)
    severe_entry: float = 0.10        # 처음 인정 시 중증으로 들어가는 비율
    progress: float = 0.0209          # 경증 → 중증 연 전이 확률 (보정값)
    recovery: float = 0.02            # 간병 → 건강 (등급외 판정 등)
    mort_mult: float = 1.8            # 경증 간병 사망률 배수
    mort_mult_severe: float = 4.0     # 중증 간병 사망률 배수
    cost_median: float = 720          # 경증 연 비용 중앙값(만원): 재가 본인부담 15% + 비급여·가사 지원
    cost_log_sigma: float = 0.6
    cost_severe_median: float = 1800  # 중증 연 비용 중앙값(만원): 요양원(월 60~75만) ~ 요양병원 개인간병(월 300만+) 혼합
    cost_severe_sigma: float = 0.8

    @classmethod
    def legacy(cls):
        """v11~v16 설정 (단일 간병 상태, 임시값)."""
        return cls(two_level=False, incidence=((65, 0.002), (75, 0.01), (85, 0.03), (999, 0.08)), prevalence=(),
                   mort_mult=2.5, recovery=0.10, cost_median=2000, cost_log_sigma=0.5)

    @staticmethod
    def _lookup(table, age):
        for lim, v in table:
            if age < lim:
                return v
        return table[-1][1]

    def inc(self, age):
        return self._lookup(self.incidence, age)

    def prev(self, age):
        return self._lookup(self.prevalence, age) if self.prevalence else None


@dataclass
class SimConfig:
    n_paths: int = 10_000
    max_age: int = 110
    seed: Optional[int] = 42
    nps: NPSRules = field(default_factory=NPSRules)
    care: object = field(default_factory=CareShock)   # CareShock(v2) 또는 CareMarkov(v11)
    mortality_improvement: bool = True    # (v27) 코호트 사망률 개선 반영 (False면 2024 기간생명표 그대로)
    risk_aversion: float = 3.0            # (v27) CRRA 위험회피계수 — 확실성등가 계산용
    longevity_sigma_level: float = 0.05   # (v30) 사망률 수준 불확실성 (로그 표준편차)
    longevity_sigma_trend: float = 0.005  # (v30) 사망률 개선 속도 불확실성 (연, 표준편차)
    # (v32) 사별 효과 — Moon et al.(2011) PLOS ONE 메타분석(226만 명, 전향 코호트 15편, 연령·성별 보정)
    #   성별 상대위험: 남 1.22, 여 1.03(여성은 통계적으로 유의하지 않음)
    #   시간 패턴: 사별 6개월 미만 1.41, 6개월 이상 1.14 → 초기 급등 배수 2.0, 감쇠 0.5년
    #   초과위험(s년 뒤) = widow_excess[성별] × (1 + widow_spike·exp(−s/widow_tau))
    widow_excess: tuple = (0.22, 0.03)    # 장기 초과 사망 위험 (남, 여). (0,0)이면 끔
    widow_spike: float = 2.0              # 사별 직후 추가 배수
    widow_tau: float = 0.5                # 급등 부분의 감쇠 시간(년)
