# CLAUDE.md — Trading Signal Lab

시장리스크·운영리스크 관리 중심의 파이썬 검증 도구. 신호(`indicators/`, `decision/`)는 포지션을
공급하는 쪽이고, 본론은 그 포지션의 위험 측정·검증·통제(`risk/`)다. 신용리스크는 다루지 않는다.

배경과 전체 파이프라인은 `README.md`, `OVERVIEW.md`, `PIPELINE.md`에 있다. 작업 전에 해당 영역
문서를 먼저 읽는다.

## 명령

| 용도 | 명령 |
|---|---|
| 리스크 측정·백테스트 | `python run_risk.py --market krx --symbols 005930,000660 --weights 0.5,0.5 --start 20150101 --end 20260930 --data-label real` |
| 한도 통제 포함 | `python run_risk.py --controls` |
| 신호 백테스트 | `python run_backtest.py` |
| 포트폴리오 시뮬레이션 | `python run_portfolio.py` |
| 대시보드 | `streamlit run app.py` |
| 전체 검증 | `pytest` (또는 `run_tests.bat`) |
| 외부 의존 제외 | `pytest -m "not external"` — CI가 돌리는 하위집합 |
| 단일 검증 | `pytest -k <이름>` 또는 `python _smoke_test_<이름>.py` |
| 커버리지 | `pytest --cov=. --cov-report=term-missing` |

의존성은 운영이 `requirements.txt`, 검증이 `requirements-dev.txt`다.
`pip install -r requirements.txt -r requirements-dev.txt`로 둘 다 깐다. 린터·타입체커는 아직 없다.

검증 구조는 루트의 `_smoke_test*.py` 39개가 본체이고, `tests/test_smoke_suite.py`가 그것들을
하위 프로세스로 실행해 종료 코드로 판정하는 브리지다. 스모크 테스트는 모듈 레벨 assert
스크립트라서 pytest가 직접 수집하지 않는다. 네트워크·외부 API·환경변수 의존이 의심되는
파일은 브리지가 내용을 스캔해 `external` 마커를 자동으로 붙이므로 목록을 손으로 관리하지 않는다.

개별 스모크 테스트를 진짜 pytest 함수로 옮기려면 `tests/`로 이동시키면 된다. 루트에서
빠지면 브리지 수집 대상에서 자연히 제외된다.

## 구조

- `data/` 수집(`krx.py`, `upbit.py`, `ecos.py`)과 품질 게이트(`quality.py`)
- `indicators/` Pine Script 지표 6종의 파이썬 재구현
- `decision/` 공통 판정 게이트(`gate.py`: 표본수 >= 10, t >= 3.0, BH, DSR)
- `risk/` 포지션 인터페이스, VaR·ES, 모델 백테스트, 한도 통제
- `backtest/` 이벤트 스터디, 표본 외·Walk-forward, 변동성 국면
- `alerts/` 텔레그램 알림과 리스크 일일 요약
- `finagent/` 공시·재무 조회

## 이 저장소의 규칙

1. **룩어헤드 금지.** 수익률은 t-1 비중 × t 수익률로 계산한다. 포지션·신호 코드를 수정할 때
   시점 정렬을 반드시 확인한다.
2. **합성 데이터 결과는 성과가 아니다.** 문서의 상태 표기에서 "합성 검증"과 "실데이터 검증"을
   섞지 않는다. 실데이터로 검증을 통과한 신호는 아직 없다.
3. **입증되지 않은 것을 입증된 것처럼 쓰지 않는다.** 리스크 모델이 "적합"이라는 말은 이 표본에서
   틀렸다는 증거를 못 찾았다는 뜻이다. 미래 손실이 VaR 안에 머문다는 보장이 아니다.
4. **통계 검정은 다중검정 보정을 거친다.** 신호 유의성 주장에는 Newey-West, 겹침 제거, BH, DSR이
   적용되어야 한다. `decision/gate.py`를 우회하는 판정을 만들지 않는다.
5. **비밀값은 환경변수로만.** API 키와 토큰은 `os.environ`으로 읽는다(`TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_CHAT_ID`, `KRX_OPEN_API_KEY`, `ECOS_API_KEY`). 코드·문서·커밋에 실제 값을 넣지 않는다.
6. **문서를 코드와 함께 고친다.** `README.md`의 상태 표와 `OVERVIEW.md`·`PIPELINE.md`가
   실제 코드와 어긋나지 않게 유지한다. `_smoke_test_docs_sync.py`가 이를 검사한다.
7. **기능을 바꾸면 해당 `_smoke_test_*.py`도 갱신한다.** 새 모듈에는 대응 스모크 테스트를 추가한다.

## 추천 ECC 스킬

- `/ecc:python-review` 통계·리스크 코드 리뷰, 특히 시점 정렬과 조용한 실패
- `/ecc:python-testing`, `/ecc:tdd-workflow` 스모크 테스트를 pytest로 이관할 때
- `/ecc:test-coverage` pytest 이관 이후 커버리지 측정
- `/ecc:security-review` 알림 파이프라인과 외부 API 연동
- `/ecc:docs-governance` 문서 드리프트 점검
