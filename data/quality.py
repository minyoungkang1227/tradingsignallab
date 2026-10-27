"""
데이터 품질 게이트 — 수집 직후 OHLCV가 "분석에 써도 되는 데이터인가"를 판정한다.

[왜 필요한가]
지금까지는 컬럼 구조(MultiIndex·중복 컬럼)만 수집 직후에 검사했고, 값 자체가
이상한지(high가 close보다 낮다, 날짜가 중복된다, 액면분할이 반영 안 돼 하루에
-50%가 찍힌다)는 아무도 보지 않았다. 그런 값은 에러를 내지 않고 조용히 통계로
흘러들어가 t-stat을 오염시킨다. "이상한 데이터가 보이면 그럴듯하게 설명하며
넘어가지 말고 멈춘다"는 원칙을 값 수준까지 확장한 것이 이 모듈이다.

[검사 다섯 개 — 서로 독립이다]
품질 체크는 하나로 뭉치지 않는다. 한 검사가 실패해도 나머지는 전부 돌려서
보고서에 함께 보여준다(원인이 겹쳐 있을 때 하나만 고치고 다시 걸리는 왕복을
줄이기 위해서다). 각 검사는 다른 검사의 결과를 입력으로 쓰지 않는다.

  1. ohlc_consistency : high >= max(open, close), low <= min(open, close),
                        가격 > 0, 거래량 >= 0.                          -> 실패
  2. index_integrity  : 날짜 인덱스가 중복 없이 오름차순인가.            -> 실패
  3. missing_values   : 가격 컬럼 NaN -> 실패 / 거래량 NaN -> 경고.
  4. date_gaps        : 일봉 이상에서 달력일 기준 최대 간격 초과.        -> 경고
  5. return_outliers  : 하루 종가 수익률 절대값이 한도 초과.             -> 경고

[심각도 두 단계]
  실패(중단) : 데이터 자체가 모순이라 어떤 통계도 믿을 수 없다. 그 종목은 분석에서
               제외하고 사유를 남긴다(단일 종목 실행은 멈춘다).
  경고(계속) : 실제 사건일 수도 있다(연휴 공백, 급등락). 판정을 사람에게 넘기고
               보고서에 남긴 채 진행한다. 그럴듯한 설명을 붙여 통과시키지 않는다 —
               경고도 반드시 출력된다.

[한계]
이 모듈은 "값이 서로 모순인가"를 본다. 값이 일관되게 틀린 경우(다른 종목 데이터,
수정주가가 일관되게 틀림)는 잡지 못한다. 그건 두 소스 교차검증이 필요하고, 아직 없다.
이상치 한도(기본 30%)는 KRX 일일 가격제한폭(±30%)에서 온 참고값이다. 가상자산은
이를 넘는 실제 변동이 있을 수 있어 실패가 아니라 경고로만 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 개별 검사 결과
PASS = "통과"
WARN = "경고"
FAIL = "실패"
SKIP = "해당없음"

# 보고서 전체 판정
V_OK = "통과"
V_WARN = "경고"
V_STOP = "중단"

PRICE_COLS = ("open", "high", "low", "close")


@dataclass
class QualityCheck:
    name: str
    status: str
    detail: str = ""


@dataclass
class QualityReport:
    symbol: str
    n_rows: int
    checks: list = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if any(c.status == FAIL for c in self.checks):
            return V_STOP
        if any(c.status == WARN for c in self.checks):
            return V_WARN
        return V_OK

    def by_name(self) -> dict:
        return {c.name: c for c in self.checks}

    def reasons(self, statuses=(FAIL,)) -> str:
        return "; ".join(f"{c.name}: {c.detail}" for c in self.checks if c.status in statuses)

    def format(self) -> str:
        head = f"[데이터 품질][{self.symbol or '-'}] 판정={self.verdict} (n={self.n_rows})"
        lines = [head]
        for c in self.checks:
            mark = {"통과": " ", "경고": "!", "실패": "X", "해당없음": "-"}[c.status]
            tail = f" — {c.detail}" if c.detail else ""
            lines.append(f"  [{mark}] {c.name} {c.status}{tail}")
        return "\n".join(lines)


def _fmt_dates(index, mask, k: int = 3) -> str:
    """위반 행의 처음 k개 날짜(또는 위치)를 사람이 읽을 형태로."""
    idx = index[np.asarray(mask)]
    shown = []
    for v in list(idx[:k]):
        shown.append(v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v))
    more = "" if len(idx) <= k else f" 외 {len(idx) - k}건"
    return ", ".join(shown) + more


def check_ohlc_consistency(df: pd.DataFrame, rel_tol: float = 1e-6) -> QualityCheck:
    name = "OHLC 정합성"
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        return QualityCheck(name, FAIL, f"필수 컬럼 없음: {missing}")
    px = df[list(PRICE_COLS)].astype(float)
    valid = px.notna().all(axis=1)     # NaN 행은 '결측' 검사의 몫이라 여기서는 제외
    o, h, l, c = (px[k][valid] for k in PRICE_COLS)

    nonpos = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    # 상대 허용오차: 부동소수점 반올림 때문에 high가 close보다 1e-12 낮은 정도는 모순이 아니다.
    hi_bad = h < np.maximum(o, c) * (1 - rel_tol)
    lo_bad = l > np.minimum(o, c) * (1 + rel_tol)
    vol_bad = pd.Series(False, index=o.index)
    if "volume" in df.columns:
        vol_bad = (df["volume"].astype(float)[valid] < 0)
    bad = nonpos | hi_bad | lo_bad | vol_bad
    n_bad = int(bad.sum())
    if n_bad == 0:
        return QualityCheck(name, PASS)
    parts = []
    for label, m in (("가격<=0", nonpos), ("high<max(open,close)", hi_bad),
                     ("low>min(open,close)", lo_bad), ("거래량<0", vol_bad)):
        if int(m.sum()):
            parts.append(f"{label} {int(m.sum())}행")
    return QualityCheck(name, FAIL, f"모순된 행 {n_bad}개 ({', '.join(parts)}); "
                                     f"처음: {_fmt_dates(o.index, bad.values)}")


def check_index_integrity(df: pd.DataFrame) -> QualityCheck:
    name = "인덱스(중복·정렬)"
    if not isinstance(df.index, pd.DatetimeIndex):
        return QualityCheck(name, FAIL, f"날짜 인덱스가 아님({type(df.index).__name__})")
    problems = []
    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        problems.append(f"중복 날짜 {n_dup}개")
    if not df.index.is_monotonic_increasing:
        problems.append("오름차순 아님(역순 또는 뒤섞임)")
    if problems:
        return QualityCheck(name, FAIL, ", ".join(problems))
    return QualityCheck(name, PASS)


def check_missing_values(df: pd.DataFrame) -> QualityCheck:
    name = "결측값"
    price_cols = [c for c in PRICE_COLS if c in df.columns]
    n_price = int(df[price_cols].isna().any(axis=1).sum()) if price_cols else 0
    if n_price:
        return QualityCheck(name, FAIL, f"가격 컬럼에 결측이 있는 행 {n_price}개 "
                                         f"({_fmt_dates(df.index, df[price_cols].isna().any(axis=1).values)})")
    if "volume" in df.columns:
        n_vol = int(df["volume"].isna().sum())
        if n_vol:
            return QualityCheck(name, WARN, f"거래량 결측 {n_vol}개 (가격은 온전)")
    return QualityCheck(name, PASS)


def check_date_gaps(df: pd.DataFrame, max_gap_days: int = 7) -> QualityCheck:
    name = "날짜 공백"
    if not isinstance(df.index, pd.DatetimeIndex):
        return QualityCheck(name, SKIP, "날짜 인덱스가 아니라 검사 불가")
    idx = df.index.sort_values().unique()    # 정렬·중복은 다른 검사의 몫 — 여기서는 독립적으로 정리해서 본다
    if len(idx) < 2:
        return QualityCheck(name, SKIP, "봉이 2개 미만")
    diffs = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
    if diffs.median() < pd.Timedelta(days=1):
        return QualityCheck(name, SKIP, "일봉보다 짧은 간격의 데이터라 공백 기준을 적용하지 않음")
    limit = pd.Timedelta(days=max_gap_days)
    big = diffs[diffs > limit]
    if big.empty:
        return QualityCheck(name, PASS)
    worst_end = big.idxmax()
    worst_days = big.max().days
    return QualityCheck(name, WARN, f"{max_gap_days}일 초과 공백 {len(big)}곳, 최대 {worst_days}일 "
                                     f"(끝: {worst_end.strftime('%Y-%m-%d')}) — 상장폐지·거래정지·수집 누락 확인 필요")


def check_return_outliers(df: pd.DataFrame, max_abs_return: float = 0.30) -> QualityCheck:
    name = "수익률 이상치"
    if "close" not in df.columns:
        return QualityCheck(name, SKIP, "close 컬럼 없음")
    close = df["close"].astype(float)
    close = close[close > 0].sort_index()    # 0 이하 가격은 OHLC 정합성 검사가 이미 잡는다
    if len(close) < 2:
        return QualityCheck(name, SKIP, "유효한 종가가 2개 미만")
    ret = close.pct_change().dropna()
    bad = ret.abs() > max_abs_return
    n_bad = int(bad.sum())
    if n_bad == 0:
        return QualityCheck(name, PASS)
    worst = ret[bad].abs().idxmax()
    return QualityCheck(name, WARN, f"일간 |수익률| > {max_abs_return:.0%} 인 날 {n_bad}개, "
                                     f"최대 {ret[worst]:+.1%} ({worst.strftime('%Y-%m-%d') if hasattr(worst, 'strftime') else worst})"
                                     f" — 액면분할·병합 미반영 가능성 확인 필요")


def run_quality_checks(df: pd.DataFrame, symbol: str = "", *, max_abs_return: float = 0.30,
                        max_gap_days: int = 7) -> QualityReport:
    """다섯 검사를 전부 실행해 보고서로 묶는다. 한 검사의 결과가 다른 검사를 건너뛰게 하지 않는다.
    검사 함수 자체가 예외를 내면 그 검사는 '통과'가 아니라 '실패'로 기록한다(검증 못 한 것을
    통과로 흘려보내지 않는다)."""
    report = QualityReport(symbol=symbol, n_rows=0 if df is None else len(df))
    if df is None:
        report.checks.append(QualityCheck("데이터 존재", FAIL, "None 반환"))
        return report
    runners = [
        ("OHLC 정합성", lambda: check_ohlc_consistency(df)),
        ("인덱스(중복·정렬)", lambda: check_index_integrity(df)),
        ("결측값", lambda: check_missing_values(df)),
        ("날짜 공백", lambda: check_date_gaps(df, max_gap_days)),
        ("수익률 이상치", lambda: check_return_outliers(df, max_abs_return)),
    ]
    for name, fn in runners:
        try:
            report.checks.append(fn())
        except Exception as e:  # noqa: BLE001
            report.checks.append(QualityCheck(name, FAIL, f"검사 중 오류: {type(e).__name__}: {e}"))
    return report
