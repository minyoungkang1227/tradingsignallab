# finagent — DART·ECOS·FRED·Signal MCP 기반 Multi-Agent 하네스

BDAI 13기 5주차 복습 과제(2026-10-19 제출)로 만든 것을 리포에 합친 폴더.
기존 퀀트 코어(`advisor/`, `decision/`, `data/`)를 **MCP 서버 4종**으로 감싸고, 에이전트 3개(Macro / Fundamental / Signal)와 위험 게이트(G1~G7)로 묶었다.
원칙: LLM은 판단하지 않는다, 구조가 판단한다. 주문 도구는 없고 결과는 `PENDING_HUMAN_APPROVAL`(제안)까지만 간다.

| 파일 | 역할 |
|---|---|
| `finagent_mcp_servers.py` | DART / ECOS / FRED / Signal MCP 서버(stdio). Signal은 `advisor.evaluator.evaluate_symbol`을 그대로 호출 |
| `finagent_orchestrator.py` | MCP 클라이언트 + 에이전트 + Harness(G1~G7) + 리포트 |
| `finagent_demo.ipynb` | fixture 모드 시나리오 7종 실행본(현재 리포 기준으로 재실행) |
| `finagent_report_submitted_2026-10-19.pdf` | 제출 당시 설명서. 부호 버그 수정 **이전** 기준이라 3절의 "저장소 미수정" 서술은 현재와 다름 |
| `live_check.py` | live(실제 API) 경로를 소스별로 하나씩 점검하는 스크립트 |
| `_smoke_test_live_paths.py` | live 보조 함수(금액 파싱·키 누락·키 가림) 스모크 테스트. 네트워크 불필요 |

## 실행

```
pip install mcp pandas numpy requests
cd finagent
python finagent_orchestrator.py --symbol FIX_LONG --corp FIX001 --scenario risk_off
```

- 코어 경로(`FINAGENT_CORE`) 기본값은 이 폴더의 상위(리포 루트)다. zip은 필요 없다.
- 키(`DART_API_KEY`, `ECOS_API_KEY`, `FRED_API_KEY`)가 없으면 fixture 모드다. fixture 결과는 가상 값과 합성 OHLCV라 **실제 성과가 아니다**.
- live 모드(실제 API) 경로는 아직 실제 응답으로 끝까지 돌려본 적이 없다. 아래 'live 점검' 참고.
- Harness 수치(비중 상한 5%, VIX 25 등)는 가정값이다.

## 리포 본체와의 관계

- G5(부호 재검증)는 제출 당시 evaluator의 `abs(t)` 버그를 막기 위한 장치였다. 지금은 `decision/gate.py`가 `t ≥ +임계`로 고쳐져 있어 이중 안전장치로 남는다.
- 리포 코드를 고치면 이 폴더의 Signal 서버가 같이 영향을 받는다. `evaluate_symbol`의 인자를 바꾸면 `finagent_mcp_servers.py`의 호출부도 확인할 것.

## live 점검

실제 API로 바꾸기 전에 소스별로 먼저 점검한다. 오케스트레이터 전체를 돌리면 어디서 막혔는지 알기 어렵다.

```
set DART_API_KEY=...
set ECOS_API_KEY=...
set FRED_API_KEY=...
pip install pykrx          # 또는 yfinance (KRX 시세용, 키 불필요)
cd finagent
python live_check.py --corp 00126380 --symbol 005930 --year 2024
```

키가 없는 소스는 SKIP으로 건너뛴다. 각 줄의 `!` 메시지가 실패 원인이다.

발급처: DART `opendart.fss.or.kr`, ECOS `ecos.bok.or.kr`, FRED `fred.stlouisfed.org/docs/api/api_key.html`.

live 경로에서 지켜지는 것:
- 키가 없는데 live면 예외 대신 `ok=False` 응답을 돌려준다. 하네스 G2가 fail-closed로 받아 '보유'가 된다.
- 네트워크/HTTP 오류도 같은 방식으로 처리된다. 서버가 죽지 않는다.
- 경고 문구에 섞여 나오는 API 키는 `***`로 가린다(리포트에 그대로 실리기 때문).
- DART 금액의 결측 표기(`-`, 빈 문자열)는 0으로 읽는다. 음수 금액(`-1,234`)은 그대로 음수다.

아직 확인 못 한 것(모름): 실제 DART 계정과목명이 `매출액/영업이익/당기순이익/부채총계/자본총계`와 정확히 일치하는지는 기업·보고서마다 다를 수 있다. 계정이 안 잡히면 `계정 누락: [...]` 경고가 뜬다.
