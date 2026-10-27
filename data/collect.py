"""
데이터 "수집"(data/*.py의 fetch_* 함수들)과 "분석"(run_*.py)의 관심사 분리
(separation of concerns)를 위한 공통 계층.

원칙 세 가지 — 이 프로젝트의 모든 fetch_* 함수가 지켜야 하는 계약이다.

  1) 수집 함수는 심볼과 기간만 입력받아 표준화된 DataFrame
     (index=date 오름차순, columns=[open, high, low, close, volume])을
     반환한다. 분석 스크립트(run_backtest.py, run_advisor.py, ...)는 데이터가
     어디서 왔는지(업비트/KRX pykrx/KRX Open API) 몰라도 동작해야 한다 —
     이게 "관심사 분리"의 실질적 의미다.
  2) 수집 함수는 반환 직전 attach_fetch_meta()로 fetched_at(UTC 시각)·
     source·symbol을 df.attrs에 남긴다. 캐시가 오래됐는지, 어느 소스에서
     온 데이터인지 사후에 추적 가능해야 실전에서 신뢰할 수 있다.
  3) 여러 종목을 순회하며 수집할 때(유니버스 스캔, 파라미터 탐색용 다종목
     로딩 등) 종목 하나의 실패(네트워크 오류, 상장폐지, 데이터 부족)가
     전체 수집을 중단시켜서는 안 된다 — 실패를 기록하고 다음 종목으로
     넘어가는 견고성(robustness)이 기본 동작이어야 한다.

collect_universe()가 원칙 3)을 표준화한다. run_advisor.py의
_load_universe()와 run_param_search.py의 _load_symbols()에 거의 동일하게
중복돼 있던 "종목별 try/except + continue" 루프를 이 함수 하나로 대체한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

import pandas as pd


def attach_fetch_meta(obj: pd.DataFrame | pd.Series, symbol: str, source: str) -> pd.DataFrame | pd.Series:
    """수집 함수 말미에서 호출해 원칙 2)의 메타데이터를 attrs에 남긴다.

    주의: pandas의 attrs는 슬라이싱 등 일부 연산에서 보존되지 않을 수 있다
    (pandas 자체의 알려진 한계). 따라서 이 메타데이터는 "수집 직후 확인용"
    으로 취급하고, 장기 보관이 필요하면 별도 로그/DB에 옮겨 적어야 한다.
    """
    obj.attrs["symbol"] = symbol
    obj.attrs["source"] = source
    obj.attrs["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return obj


def assert_flat_columns(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """컬럼이 MultiIndex가 아니라 평범한(flat) Index인지 확인한다.

    assert_unique_columns()와는 별개의, 독립적인 검사다(원칙: 품질 체크는
    하나로 뭉치지 않고 각각 따로 본다) — 컬럼 "이름이 겹치는" 문제와 컬럼
    "구조 자체가 다단(MultiIndex)"인 문제는 원인도 증상도 다르다. 후자는
    컬럼 이름 자체는 중복되지 않아 assert_unique_columns가 못 잡는다
    (예: yfinance가 종목 1개 요청에도 (필드, 티커) 2단 컬럼을 반환하는
    경우). MultiIndex가 섞여 있으면 out["volume"] 같은 단일 컬럼 선택이
    Series 대신 DataFrame을 반환해, 그보다 한참 뒤(지표 계산 단계)에서
    알아보기 힘든 pandas 내부 에러로 번진다. 레벨이 1개뿐이면(모든 하위
    레벨이 실질적으로 비어있거나 전부 같은 값이면) 안전하게 평탄화하고
    경고만 남기고, 그 외(레벨 안에 여러 값이 섞여 있는 경우)에는 어느
    레벨을 표준 컬럼으로 써야 할지 스스로 판단하지 않고 예외를 던진다.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    extra_levels = df.columns.droplevel(0)
    if extra_levels.nunique(dropna=False) > 1:
        raise RuntimeError(
            f"[데이터 품질 오류][{context}] 컬럼이 다단(MultiIndex)이고 그 "
            f"안에 서로 다른 값이 섞여 있습니다({extra_levels.unique().tolist()}) "
            f"— 어떤 하위 레벨을 써야 할지 임의로 고를 수 없으니 원본 응답 "
            f"구조를 확인하세요."
        )
    print(f"[데이터 품질 경고][{context}] 컬럼이 MultiIndex로 반환됐습니다"
          f"(하위 레벨은 전부 동일값) — 첫 번째 레벨만 남기고 평탄화합니다.")
    df = df.copy()
    df.columns = df.columns.get_level_values(0)
    return df


def assert_unique_columns(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """표준 OHLCV 컬럼 중복을 수집 직후에 즉시 잡아낸다(원칙: 이상한 데이터가
    보이면 그럴듯하게 설명하며 넘어가지 말고 멈춘다).

    소스 API가 같은 필드를 두 번 반환하면(예: yfinance/pykrx 응답 구조가
    버전에 따라 바뀌는 경우) df["volume"]처럼 한 컬럼만 골라야 하는 자리에서
    Series 대신 DataFrame이 나와, 그보다 훨씬 뒤(지표 계산 단계)에서
    `ValueError: Cannot set a DataFrame with multiple columns to the single
    column ...` 같은 알아보기 힘든 pandas 내부 에러로 터진다. 이 함수를 수집
    함수 말미에서 호출해 그 자리에서 바로 잡는다.

    중복된 두 컬럼의 값이 완전히 같으면(단순 중복 응답) 첫 컬럼만 남기고
    경고만 출력한다. 값이 다르면 원본 응답 자체를 신뢰할 수 없다는 뜻이므로
    예외를 던져 파이프라인을 멈춘다 — 조용히 하나를 골라 쓰면 안 된다.
    """
    if not df.columns.duplicated().any():
        return df
    dup_names = df.columns[df.columns.duplicated()].unique().tolist()
    for name in dup_names:
        block = df.loc[:, df.columns == name]
        first = block.iloc[:, 0]
        if not all(block.iloc[:, i].equals(first) for i in range(1, block.shape[1])):
            raise RuntimeError(
                f"[데이터 품질 오류][{context}] '{name}' 컬럼이 중복 수집됐고 "
                f"값도 서로 달라 신뢰할 수 없습니다 — 원본 API 응답을 확인하세요."
            )
        print(f"[데이터 품질 경고][{context}] '{name}' 컬럼이 중복 수집됐습니다"
              f"(값은 동일) — 첫 번째 컬럼만 사용합니다.")
    return df.loc[:, ~df.columns.duplicated()]


@dataclass
class CollectResult:
    ok: dict = field(default_factory=dict)      # symbol -> DataFrame
    failed: dict = field(default_factory=dict)  # symbol -> 실패 사유 문자열
    quality: dict = field(default_factory=dict)  # symbol -> QualityReport (quality_gate=True일 때만 채워짐)

    def __len__(self) -> int:
        return len(self.ok)

    def summary(self) -> str:
        return f"성공 {len(self.ok)}개, 실패 {len(self.failed)}개"


def collect_universe(symbols: Iterable[str], fetch_one: Callable[..., pd.DataFrame],
                      min_len: int = 60, verbose: bool = True, quality_gate: bool = False,
                      **fetch_kwargs) -> CollectResult:
    """
    symbols: 종목코드 iterable
    fetch_one: (symbol, **fetch_kwargs) -> pd.DataFrame 시그니처의 단일 종목
        수집 함수. data.krx.fetch_krx_ohlcv, data.upbit.fetch_upbit_ohlcv_range
        등을 functools.partial로 감싸 넘긴다(뒤 인자를 symbol 하나만 받는
        형태로 고정).
    min_len: 이보다 짧은 데이터는 이후 통계 검정에서 의미가 없다고 보고
        실패로 처리한다(run_advisor.py/run_param_search.py가 기존에 개별
        구현했던 것과 동일한 기준, 기본값 60).
    verbose: True면 실패마다 경고를 출력한다(기존 스크립트들의 동작과 동일).
    quality_gate: True면 종목마다 data.quality.run_quality_checks()를 돌린다. '중단' 판정
        (OHLC 모순·날짜 중복/역순·가격 결측)이면 그 종목을 failed에 사유와 함께 넣고 제외하며,
        '경고' 판정(날짜 공백·수익률 이상치·거래량 결측)이면 보고서를 출력하되 포함한다.
        기본 False는 이전 동작과 완전히 같다. 검사 결과는 result.quality[symbol]에 남는다.

    반환: CollectResult. 종목 하나가 예외를 던지거나 데이터가 min_len
    미만이어도 나머지 종목 수집은 계속된다(원칙 3의 구현) — 이 함수 자체는
    예외를 전파하지 않는다.
    """
    result = CollectResult()
    for symbol in symbols:
        try:
            df = fetch_one(symbol, **fetch_kwargs)
        except Exception as e:  # noqa: BLE001 - 종목 하나의 실패가 전체 수집을 죽이면 안 됨
            result.failed[symbol] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"[경고] {symbol} 로딩 실패, 건너뜁니다: {e}")
            continue

        n = 0 if df is None else len(df)
        if df is None or n < min_len:
            result.failed[symbol] = f"데이터 부족(n={n} < {min_len})"
            if verbose:
                print(f"[경고] {symbol}: 데이터가 {n}봉밖에 없어 건너뜁니다.")
            continue

        if quality_gate:
            from data.quality import run_quality_checks, V_STOP, V_WARN
            report = run_quality_checks(df, symbol=str(symbol))
            result.quality[symbol] = report
            if report.verdict == V_STOP:
                result.failed[symbol] = f"데이터 품질 실패 — {report.reasons()}"
                if verbose:
                    print(f"[경고] {symbol}: 데이터 품질 검사에 걸려 제외합니다.\n{report.format()}")
                continue
            if report.verdict == V_WARN and verbose:
                print(report.format())

        result.ok[symbol] = df

    if verbose:
        print(f"[수집 완료] {result.summary()}")
    return result
