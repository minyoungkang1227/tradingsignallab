# Trading Signal Lab → 리스크 관리 중심으로

이 저장소는 **시장리스크·운영리스크 관리**를 중심으로 재구성 중입니다(2026-10). 처음에는 Pine Script 지표를
파이썬으로 재구현해 이벤트 스터디로 검증하는 '신호 찾기' 도구였고, 그 코드는 그대로 남아 있습니다. 다만 역할이
바뀌었습니다 — 신호(`indicators/`, `decision/`)는 **포지션을 공급하는 쪽**이고, 그 포지션에 얼마의 위험이
실려 있는지, 그 위험 추정이 맞았는지, 한도를 넘으면 무엇을 멈추는지가 이 저장소의 본론입니다(`risk/`).
신용리스크는 다루지 않습니다.

방향을 바꾼 이유는 하나입니다. 신호가 돈이 되는지는 아직 실데이터로 입증된 것이 없고, 입증하기도 어렵습니다.
반면 리스크 측정은 포지션이 돈이 되는지와 무관하게 **가격 데이터만으로 객관적으로 검증**할 수 있습니다 — VaR
모델이 맞는지는 손실 초과 횟수(Kupiec)와 초과의 뭉침(Christoffersen)으로, 한도·손절이 작동하는지는
시뮬레이션으로 판정합니다. 즉 '입증되지 않은 일' 목록을 줄일 수 있는 쪽이 리스크입니다.

**이 도구는 "신호가 앞으로도 돈을 벌어준다"를 입증하지 않으며, 리스크 모델이 "적합"으로 나왔다고 해서 미래
손실이 VaR 안에 머문다는 보장도 아닙니다.** '적합'은 이 표본에서 틀렸다는 증거를 못 찾았다는 뜻입니다.

## 리스크 관리 구성 (진행 중)

| 층 | 무엇을 | 상태 | 코드 |
|---|---|---|---|
| 포지션 인터페이스 | '날짜 × 종목 비중' 테이블 하나로 통일. 지정 보유 포트폴리오·신호 전략·외부 계좌 모두 같은 입력. t-1 비중 × t 수익률(룩어헤드 없음) | 완료 | `risk/positions.py` |
| 시장리스크 측정 | 1일 VaR·ES — 역사적 / 파라메트릭(EWMA σ) / 필터드 역사적(FHS, 권장) | 완료(합성 검증) | `risk/var.py` |
| 리스크 모델 백테스트 | Kupiec 비율, Christoffersen 독립성, 조건부 커버리지, 바젤 신호등. 하나라도 기각이면 '부적합' | 완료(합성 검증) | `risk/var_backtest.py`, `run_risk.py` |
| 데이터 리스크(운영) | 수집 직후 OHLC 정합성·중복·결측(중단), 공백·이상치(경고) | 완료 | `data/quality.py` |
| 변동성 국면 | 고변동/저변동(확장 분위수 + 히스테리시스) — 국면별 한도 조정의 입력 | 완료(분류만) | `backtest/vol_regime.py` |
| 스트레스 테스트 | 과거 시나리오 재적용(2008, 2020-03, 2022)·가상 충격 | 예정 | — |
| 포지션 사이징 | 변동성 목표, 분수 켈리(추정 불확실성 반영), 리스크 예산 | 예정 | — |
| 포트폴리오 리스크 | 공분산 축소 추정, 종목별 리스크 기여도, 집중도, 스트레스 시 상관 붕괴 | 예정 | — |
| 통제 계층(운영) | 네 한도를 ES 97.5%에서 **도출**: ES 목표(비중 축소), 종목 리스크 예산(종목 축소), 일일손실(다음 날 증가 차단), 낙폭 서킷(부트스트랩 MDD 분위, 노출 절반). 리스크를 못 재는 날은 비중 0. 통제 전/후 비교와 통제 후 VaR 백테스트 | 완료(합성 검증) | `risk/limits.py`, `run_risk.py --controls` |
| 모니터링(운영) | 평일 장 마감 후 감시 포트폴리오의 일일 리스크 요약(내일 VaR/ES, 오늘 손익과 ES 초과, 모델 판정, 한도 상태)과 한도 위반(⚠)을 텔레그램으로. 조용한 날도 보내는 하트비트. 품질 게이트 중단 시 VaR 대신 중단 알림 | 완료(합성 검증) | `alerts/check_risk.py`, `.github/workflows/signal_alert.yml` |
| 대조(운영) | 의도 포지션 vs 실제 포지션 vs 잔고 대조, 불일치 시 중단. 거래소 장애·출금중단은 거래상대방 운영리스크로 포함 | 예정 | — |
| 모델 리스크 | 파라미터 민감도 표, 두 구현 교차검증(지금 스모크테스트가 하는 일의 체계화) | 예정 | — |

실행 예: `python run_risk.py --market krx --symbols 005930,000660 --weights 0.5,0.5 --start 20150101 --end 20260930 --data-label real`

## 신호 검증 도구 (포지션 공급원)

원래 이 저장소가 하던 일이고, 지금은 '리스크를 걸 포지션의 후보를 만들고 거르는 것'이 역할입니다. 지표 6종의
강세·약세 이벤트 12개를 이벤트 스터디로 검증하고(비용·체결지연 반영, Newey-West·겹침 제거, BH·DSR 다중검정 보정),
표본 외·Walk-forward·국면별로 유지되는지 보며, 공통 판정 게이트(`decision/gate.py`: 표본수 ≥ 10, t ≥ 3.0, BH, DSR)를
통과한 신호만 채택합니다. 지표 정의, Pine Script와의 차이, 팩터 조정 알파, 비용 반영, 다중검정 보정의 상세는
**`OVERVIEW.md` 2~7절**에 있습니다.

2026-10 재구성에서 매수/보유/매도 어드바이저(`advisor/`), IS/OOS 파라미터 탐색(`research/`), 한국시장 팩터
구성(`factors/`)을 제거했습니다. 리스크 방향에서 쓸 일이 없고(어드바이저·탐색), 실행이 검증된 적이 없어서(팩터)입니다.
팩터 조정 알파 계산 자체(`backtest/engine.py::factor_adjust`)는 남아 있어 외부 팩터 CSV를 `--factor-csv`로 주면
동작하지만, 기본은 '해당없음'으로 건너뜁니다. 제거된 코드는 git 이력에 있습니다.

## 이 프로젝트가 하는 일 / 아직 입증하지 못한 일

### 현재 할 수 있는 일

| 하는 일 | 내용 | 실행 |
|---|---|---|
| 과거 신호 성과 분석 | 신호 발생 후 N봉 평균수익률·승률·t값. 다음 봉 시가 진입, 수수료, 슬리피지 반영 가능 | `run_backtest.py` |
| 통계 검증 | Newey-West·겹침 제거, Benjamini-Hochberg 다중검정 보정, 디플레이티드 샤프비율(DSR), t ≥ 3.0 규칙 판정 | `run_backtest.py --deflate --newey-west --fdr` 등 |
| 표본 외 검증 | 학습 구간에서 고른 신호가 이후 구간에서 유지되는지(퍼지 적용 단일 분할), 학습 끝을 옮겨가며 반복하는 Walk-forward(신호별 합산·fold 간 이질성 검정) | `--oos-train-end`, `--wf-first-train-end` |
| 시장 국면 분석 | 종목 자체 가격으로 상승·하락·횡보(또는 `--regime-kind vol`로 고변동·저변동)를 나눠 국면별 성과·국면 간 부호 일관성 확인 | `--regime`, `--regime-kind` |
| 위험·경제적 성과 측정 | 트리플 배리어(손절 반영) 성과, R 단위 기대값·손익분기 승률·비용 대비 배수, 다종목 동시 운용 시 MDD·자본·포지션 제약 | `--barrier --barrier-economics`, `run_portfolio.py` |
| 데이터 품질 검사 | 수집 직후 OHLC 정합성·날짜 중복/정렬·결측(중단), 날짜 공백·수익률 이상치(경고)를 서로 독립적으로 판정. 모순 데이터는 분석 전에 멈춤 | 기본 켜짐, `--skip-quality-gate`로 끔 |
| 시장리스크 측정·백테스트 | 지정 포트폴리오의 1일 VaR/ES(역사적·파라메트릭·FHS)와 Kupiec·Christoffersen·바젤 백테스트 | `run_risk.py` |
| 자동 알림 | 평일 16:10(KST) GitHub Actions. 기본은 **일일 리스크 요약·한도 위반**(`alerts/check_risk.py`). 예전 신호 알림(`alerts/check_signals.py`, 공통 판정 게이트 통과 신호)은 기본 꺼짐이고 watchlist 항목에 `"signal_alerts": true`를 적은 종목만. 감시리스트(비중 포함)는 대시보드(관리자 전용)에서 편집 | `alerts/`, `app.py` |

위 기능의 **계산 정확성**은 합성 데이터 스모크테스트(손으로 계산한 기대값과 대조)로 검증했습니다.
그러나 **실제 시장 데이터로 돌린 결과는 이 저장소에 포함돼 있지 않습니다** — 아래 "검증 현황" 참고.

### 현재 입증되지 않은 일 (이 저장소가 주장하지 않는 일)

- **새로운 시장 환경에서의 안정적인 수익.** OOS·Walk-forward·국면 검증 도구는 만들어져 있지만, 이 도구들을
  실제 시장 데이터로 돌려 통과한 기록은 아직 없습니다. 합성 데이터 테스트 통과는 "코드가 설계대로
  계산한다"는 뜻일 뿐 "그 신호가 돈이 된다"는 뜻이 아닙니다.
- **포트폴리오 단위 실전 성과.** 시뮬레이터(`run_portfolio.py`)는 구현과 합성 데이터 검증만 마쳤습니다.
  결과는 비중·한도·우선순위 같은 가정에 좌우되며, 신호를 시뮬레이션 구간과 같은 데이터로 골랐다면 표본
  내(in-sample) 결과입니다. 실거래의 체결 불가·갭·유동성 부족은 일봉 시뮬레이션으로 다 반영되지 않습니다.
- **머신러닝에 의한 개별 신호 성공확률 예측.** 구현돼 있지 않습니다. 판정은 통계 기준에 따른 규칙(if/else)입니다.
- **자동 매매.** 알림만 보내며 주문 실행과 연결돼 있지 않습니다. 알림은 매매 추천이 아니고, 리스크 요약의 VaR/ES는 미래 손실의 상한이 아닙니다.

## 검증 현황: 합성 데이터 테스트와 실제 시장 데이터 검증은 다릅니다

두 가지는 답하는 질문이 다르므로 섞어서 인용하지 않습니다.

| 구분 | 무엇을 확인하나 | 현재 상태 |
|---|---|---|
| **합성 데이터 스모크테스트** | 코드가 설계한 공식대로 계산하는가(손계산 기대값 일치, 경계·예외 처리) | 저장소의 `_smoke_test*.py` 전부(개수는 늘어나므로 고정 표기 안 함)를 `python <파일>`로 각각 실행. 네트워크 불필요 |
| **실제 시장 데이터 검증** | 신호가 실제 시장에서 통계적·경제적으로 유효한가 | **기록된 통과 결과 없음.** 아래 표 참고 |

합성 데이터로 나온 수익률·t값·MDD는 어떤 경우에도 "성과"로 인용하지 않습니다. 합성 결과가 나오는 출력과
리포트에는 `실제 시장 성과가 아님`을 명시합니다(`--data-label synthetic`).

**실제 시장 데이터 검증 진행 상황** (이 표는 결과가 생길 때마다 갱신합니다):

| 항목 | 상태 |
|---|---|
| VaR 모델 백테스트(실데이터) | 미실시. `run_risk.py`를 삼성전자·SK하이닉스·KRW-BTC로 돌려 방법별 적합/부적합과 초과 횟수를 기록할 것 — 이 항목은 신호의 수익성과 무관하게 채울 수 있다 |
| 신호별 이벤트 스터디(실데이터) | 사용자 환경(pykrx/업비트 접속 가능)에서 실행 필요. 저장소에 결과 기록 없음 |
| OOS·Walk-forward·국면·포트폴리오 실데이터 실행 | 미실시. 지앤비에스 에코(382800) Colab 노트북은 데이터 품질 점검(OHLC·결측·소스 간 종가 불일치) 단계에서 멈춰 있고 OOS 결과는 없음 |
| KRX Open API·ECOS 수집 경로 | 실제 응답 대조 미검증 |
| 텔레그램 알림 운영 | 구성됨. 실제 발송 이력은 저장소 Actions 로그에서 확인 |

실데이터 결과를 기록할 때는 종목·기간·실행 조건(`--profile`, 비용)·표본 수·검증 통과 여부를 위 표에
날짜와 함께 남깁니다. 아래 "성과 리포트 표준 형식"이 그 항목을 한 블록으로 보여줍니다.

## 성과 리포트 표준 형식

OOS·Walk-forward·국면·포트폴리오 리포트는 맨 위에 같은 형식의 요약 블록을 출력합니다(계산·CSV는
바뀌지 않고 출력 맨 위에 줄만 추가됩니다). 항목과 순서는 모두 같습니다: **분석 / 데이터 / 기간 / 체결 /
표본 / 기준 / 결과**. 데이터가 실제인지 합성인지는 코드가 알 수 없어 `--data-label real|synthetic`으로
알려야 하며, 생략하면 `미표시`로 표시됩니다. 비용이 0이면 체결 줄에 `※비용 미반영`이 붙습니다. 포트폴리오는
통계 검증이 아니라 시뮬레이션이라 기준·결과를 `해당 없음`으로 명시합니다. 아래는 **합성 데이터**로 만든 예시입니다.

```
[실행 조건·검증 요약]
  분석  표본 외(OOS) 검증
  데이터  합성 데이터 — 실제 시장 성과가 아님
  기간  IS(학습) 2019-03-26~2021-04-09 (퍼지 6봉) / OOS(검증) 2021-04-20~2023-08-07
  체결  신호 1봉 뒤 시가 진입, 수수료·세금 36bp(왕복), 슬리피지 10bp(왕복, 고정), 보정 겹침 제거·Newey-West·DSR≥0.95
  표본  학습 534봉(이벤트 구간), 검증 600봉, 선택 신호 1개, 검증 표본 합계 74건(신호별 최소 10건 필요)
  기준  학습 선택: BH(FDR 0.1) AND DSR≥0.95(보유 3봉, 최소 20건); 검증: 단측 p < α/k = 0.0500
  결과  선택 1개 중 검증 통과(유지됨) 1개, 유지 안 됨 0개, 판정불가 0개
```

## 확장 분석 빠른 실행

```bash
# 실전에 가까운 조건 한 번에: 다음 봉 시가 진입, 시장별 왕복비용(KRX 36bp/업비트 10bp), 슬리피지 10bp,
# 겹침 제거, Newey-West, DSR, 손절 반영 배리어(순차·경제성). 비용 수치는 현재 세율로 --cost-bps로 덮어쓸 것
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 \
  --profile realistic --data-label real

# 표본 외 검증: 2024-06-28까지 학습 -> 이후 검증 (학습 끝 퍼지, DSR 자동 켜짐)
python run_backtest.py --market krx --symbol 005930 --start 20200101 --end 20260901 \
  --profile realistic --oos-train-end 2024-06-28 --data-label real

# 반복 Walk-forward(확장창): 검증 조각 60봉씩 전진, 신호별 합산과 fold 간 이질성 판정
python run_backtest.py --market krx --symbol 005930 --start 20200101 --end 20260901 \
  --profile realistic --wf-first-train-end 2023-12-29 --wf-test-bars 60 --data-label real

# 시장 국면(상승/하락/횡보)별 검증
python run_backtest.py --market krx --symbol 005930 --start 20200101 --end 20260901 \
  --profile realistic --regime --data-label real

# 감시리스트 여러 종목을 한 계좌로 동시 운용(1거래 위험 0.5%, 최대 동시 5개, 총노출 100%)
python run_portfolio.py --market krx --profile realistic --sim-start 2025-01-01 --data-label real
```

모든 옵션과 기본값은 `OVERVIEW.md`의 "옵션·기본값 대조표"에 있습니다. 그 표는 실제 `argparse` 정의에서
자동 생성하며, 코드와 어긋나면 `_smoke_test_docs_sync.py`가 실패합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
# 업비트 코인 (일봉 1000개)
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000

# 업비트 코인 (60분봉)
python run_backtest.py --market upbit --symbol KRW-BTC --unit minute --minute-unit 60 --count 1500

# 국내 주식 (SK하이닉스, 2022~2026 일봉)
python run_backtest.py --market krx --symbol 000660 --start 20220101 --end 20260901

# 삼성전자
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901

# 기관 변동 캔들의 거래량 이상치 판정을 "평균의 2배"(원본) 대신
# Kyle(1985) 기반 z-스코어(종목 고유 변동성 대비 95% 유의수준)로 바꿔서 비교
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000 \
  --id-vol-mode zscore --id-z-threshold 1.96

# Jegadeesh & Titman(1993) 모멘텀 필터: 롱 이벤트는 상승모멘텀 상위 30%,
# 숏 이벤트는 하락모멘텀 상위 30% 구간에서만 인정하도록 5개 지표 전부 필터링
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000 \
  --mom-filter --mom-threshold 0.7
```

## 결과물 (output/ 폴더)

- `summary.csv` — 지표 × 방향(LONG/SHORT) × 보유기간(봉수)별 평균수익률·승률·t-stat·p-value·
  `significant_bh` 표. 엑셀로 열어서 피벗해보시면 편합니다.
- `event_curves.png` — 신호 발생 후 시간이 지날수록 평균 누적수익률이 어떻게 움직이는지
  지표별로 겹쳐 그린 그래프. 곡선이 꾸준히 우상향(롱)/우하향(숏)하면 신호에 실제로
  예측력이 있다는 뜻이고, 0 근처에서 지그재그하면 노이즈일 가능성이 높습니다.

확장 분석을 켜면 추가로 `oos_train_summary.csv`·`oos_test_summary.csv`·`oos_selected.csv`(OOS),
`wf_folds.csv`·`wf_selected.csv`·`wf_by_signal.csv`(Walk-forward), `regime_summary.csv`(국면)이 생기고,
`run_portfolio.py`는 `portfolio_trades.csv`·`portfolio_equity.csv`·`portfolio_candidates.csv`를 만듭니다.

터미널에도 실행 즉시 "보유기간 N봉 기준 t-stat 랭킹" 표가 출력됩니다. t-stat 절댓값이
대략 2 이상이면 통계적으로 눈여겨볼 만하지만, `n_events`(표본 수)가 한 자릿수면 우연일
가능성이 커서 별도로 걸러서 보셔야 합니다. 여러 지표·방향·보유기간을 한꺼번에 비교할
때는 개별 t-stat만 보지 말고 `significant_bh` 컬럼(OVERVIEW.md 4절 다중 검정 보정 참고)을
우선 확인하세요.

## 파일 구조

```
risk/                 시장리스크·운영리스크 계층 (이 저장소의 본론)
  positions.py        공통 포지션 인터페이스(날짜 × 비중), 포트폴리오 수익률(t-1 비중 × t 수익률)
  var.py              1일 VaR/ES — 역사적·파라메트릭(EWMA σ)·필터드 역사적(FHS)
  var_backtest.py     Kupiec·Christoffersen·조건부 커버리지·바젤 신호등
  limits.py           통제 계층 — ES에서 한도 도출(출처 기록), 비중 적용(룩어헤드 없음), 전/후 비교
data/                 시세 수집 (업비트 REST API, pykrx/KRX Open API/yfinance, 한국은행 ECOS)
  quality.py          데이터 품질 게이트(운영리스크: 데이터 리스크 통제)
indicators/           지표 6종 재구현(5종 + 콤보 필터) + 모멘텀 필터 + 유동성 필터(Amihud)
backtest/
  engine.py           이벤트 스터디 엔진 (forward_return, event_study, compare_all, BH, 팩터 조정 알파)
  deflated.py         디플레이티드 샤프비율(DSR)
  triple_barrier.py   트리플 배리어(손절 반영), 순차 거래
  economics.py        경제적 유의성(R 기대값, 손익분기 승률, 비용 대비 배수)
  oos.py              표본 외(OOS) 검증 (퍼지 분할)
  walkforward.py      반복 Walk-forward 검증(확장창)
  regime.py           시장 국면별 검증
  portfolio.py        포트폴리오 시뮬레이터(다종목 동시 운용)
  profiles.py         실행 조건 프리셋(--profile realistic)
  report_common.py    성과 리포트 표준 요약 블록
decision/             공통 판정 게이트(채택/보류/기각)
alerts/               평일 자동 점검·텔레그램 알림 — check_risk.py(일일 리스크 요약·한도 위반, 기본),
                      check_signals.py(신호 알림, 기본 꺼짐), 감시리스트(watchlist.json, weight 포함), 상태 저장
app.py                Streamlit 대시보드(결과 조회, 감시리스트 편집은 관리자 전용)
tools/                문서 보조 도구(gen_option_table.py: 옵션·기본값 표 생성)
run_risk.py           시장리스크 측정·VaR 백테스트 CLI
run_backtest.py       신호 검증 파이프라인 CLI
run_portfolio.py      포트폴리오 시뮬레이션 CLI
OVERVIEW.md           상세 설계·근거·옵션 전체 대조표
PIPELINE.md           파이프라인 단계와 구현 상태
_smoke_test*.py, smoke_test_*.py   합성 데이터 스모크테스트(아래 "테스트")
```

## 테스트

모든 테스트는 네트워크 없이 합성 데이터로 돌며, 파일 하나씩 `python <파일명>`으로 실행합니다. 전부 한
번에 돌리려면 저장소 루트에서 `for f in _smoke_test*.py smoke_test_*.py; do python $f || echo FAIL $f; done`.

**이 테스트들이 증명하는 것:** 코드가 설계한 공식대로 계산한다(손계산 기대값 일치, 입력 오류·경계 처리,
기본값이 기존 동작을 보존). **증명하지 않는 것:** 신호가 실제 시장에서 통한다는 것. 테스트 출력의 수익률·t값은
인위적으로 만든 데이터에서 나온 값이며 성과가 아닙니다. 실제 시장 데이터 검증은 위 "검증 현황"에서 따로
관리합니다. 테스트 파일별 검증 대상은 `OVERVIEW.md`의 "파일 구조 & 테스트" 절에 있습니다.

## 다음에 고려해볼 만한 것

- 가장 먼저 필요한 것은 **`run_risk.py`를 실제 시장 데이터로 돌려 VaR 모델의 적합/부적합을 기록하는 일**입니다.
  신호와 달리 이 결과는 수익성과 무관하게 채울 수 있고, 리스크 중심 재구성의 첫 실데이터 근거가 됩니다.
- 그다음은 위 "리스크 관리 구성" 표의 예정 항목 순서대로: 스트레스 테스트 → 포지션 사이징 → 포트폴리오 리스크 →
  통제 계층(한도를 VaR에서 도출하고 수익률 비용을 측정) → 모니터링·대조.
- 신호 쪽은 **실제 시장 데이터로 OOS·Walk-forward·포트폴리오를 한 번씩 돌려 결과를 기록하는 일**이 남아 있습니다.
  그 전까지 이 저장소의 신호는 "검증 도구를 통과시켜 본 적이 없는 후보"입니다.
- 여기서 t-stat이 괜찮게 나온 신호가 있으면, 그걸 지난번 만든 Pine Script
  `Delta_Institutional_Combo_Alert.pine`처럼 실제 트레이딩뷰 알림용 코드로 다시 옮겨서
  TVExtBot 웹훅에 연결하는 흐름으로 이어갈 수 있습니다.
- 지금은 단일 신호 이벤트 스터디이므로, 여러 신호를 AND/OR로 조합했을 때 t-stat이
  더 좋아지는지도 다음 단계로 검증해볼 만합니다.
- 확률미분방정식 공부와 연결짓고 싶으시면, `forward_return`의 분포를 정규분포 가정 없이
  부트스트랩으로 재검정하거나, GBM 대비 실제 수익률 분포의 두꺼운 꼬리(fat tail)를
  비교하는 방향으로 확장할 수도 있습니다.
