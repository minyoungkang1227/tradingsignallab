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

## 실행

```
pip install mcp pandas numpy requests
cd finagent
python finagent_orchestrator.py --symbol FIX_LONG --corp FIX001 --scenario risk_off
```

- 코어 경로(`FINAGENT_CORE`) 기본값은 이 폴더의 상위(리포 루트)다. zip은 필요 없다.
- 키(`DART_API_KEY`, `ECOS_API_KEY`, `FRED_API_KEY`)가 없으면 fixture 모드다. fixture 결과는 가상 값과 합성 OHLCV라 **실제 성과가 아니다**.
- live 모드(실제 API) 경로는 제출 환경에서 네트워크가 막혀 실행해 검증하지 못했다.
- Harness 수치(비중 상한 5%, VIX 25 등)는 가정값이다.

## 리포 본체와의 관계

- G5(부호 재검증)는 제출 당시 evaluator의 `abs(t)` 버그를 막기 위한 장치였다. 지금은 `decision/gate.py`가 `t ≥ +임계`로 고쳐져 있어 이중 안전장치로 남는다.
- 리포 코드를 고치면 이 폴더의 Signal 서버가 같이 영향을 받는다. `evaluate_symbol`의 인자를 바꾸면 `finagent_mcp_servers.py`의 호출부도 확인할 것.
