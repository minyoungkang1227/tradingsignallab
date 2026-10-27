# trading_signal_lab — 전체 파이프라인 (큰 틀)

지금까지 조각조각 만든 걸 하나의 파이프라인으로 놓고 보면, 데이터가 들어와서
"이 신호를 믿어도 되는가"라는 결론이 나올 때까지 [0]~[7]단계를 거치고, 그 뒤에
표본 외·국면·포트폴리오 검증([8]~[9])과 알림([10])이 이어집니다. 실거래 주문
자동화는 이 파이프라인의 산출물(검증된 신호)을 넘겨받는 별도 단계로, 아직
이 저장소 범위 밖입니다(알림은 있고 주문 연결은 없음).

> **구현 상태 표기 원칙**: "완료"는 코드가 있고 합성 데이터 스모크테스트로 계산 정확성을 검증했다는 뜻입니다.
> 실제 시장 데이터로 결과를 검증했다는 뜻이 아닙니다(실데이터 검증 현황은 README의 "검증 현황" 참고).

```
[0] 데이터 수집
      │  data/upbit.py, data/krx.py (+ data/quality.py 품질 게이트)
      │  → OHLCV DataFrame (open, high, low, close, volume)
      ▼
[1] 지표 신호 생성 (6종: 원본 Pine Script 재구현 5종 + 직접 설계한 콤보 필터)
      │  indicators/institutional_displacement.py, delta_trading.py,
      │  big_sales.py, ultimate_rsi.py, vp_box.py, combo_filter.py
      │  → 원본 df + 이벤트 컬럼(bull_shift, crossover_confirmed, ...)
      ▼
[2] 신호 필터링 (원 신호를 더 엄격하게 거르는 레이어)
      │  indicators/institutional_displacement.py (vol_mode="zscore", Kyle 1985)
      │  indicators/momentum.py (모멘텀 필터, Jegadeesh & Titman 1993)
      │  indicators/liquidity.py (유동성 필터, Amihud 2002 — --liq-filter)
      │  → 이벤트 컬럼에 AND 마스크 적용, 이벤트 수는 줄어들거나 같음
      ▼
[3] 이벤트 스터디 백테스트  ← 거래비용·체결지연·다중검정 보정 반영 완료
      │  backtest/engine.py: event_study(), compare_all()
      │  entry_lag/entry_col로 t+1 시가 진입 강제(look-ahead 방지),
      │  cost_bps로 왕복 거래비용 차감(업비트 10bp / KRX 36bp 근사),
      │  slippage_bps(고정 또는 Amihud 기반 동적값), newey_west/drop_overlapping으로 겹치는 이벤트 보정,
      │  benjamini_hochberg()로 요약표 전체에 다중 검정 보정(p_value, significant_bh),
      │  backtest/deflated.py: 디플레이티드 샤프비율(--deflate → significant_dsr),
      │  backtest/triple_barrier.py: 손절 반영 배리어 통계(--barrier → tb_* 컬럼, --barrier-sequential),
      │  backtest/economics.py: R 기대값·손익분기 승률·비용 대비 배수(--barrier-economics)
      │  → 신호별 순방향수익률 평균·표준편차·t-stat·p-value·significant_bh(+DSR·배리어·경제성 컬럼), 누적수익률 곡선
      ▼
[4] 팩터 구성 (별도 데이터 소스 — KRX 전체 유니버스 필요)  ← 생존편향 부분 완화 완료
      │  factors/kr_fama_french.py: compute_smb_hml(), compute_wml()
      │  delisted_return 옵션으로 구간 중 상장폐지 종목을 전손 취급(기본은 이전과
      │  동일하게 조용히 제외 — 켜야만 적용됨)
      │  → SMB, HML, WML 월별 시계열 (Fama-French 1993 + Carhart 1997)
      ▼
[5] 팩터 조정 알파 검증  ← 구현 완료
      │  backtest/engine.py: factor_adjust(), event_alpha()
      │  [3]의 이벤트 수익률을 [4]의 SMB/HML/WML에 OLS 회귀 (merge_asof로 날짜 매칭)
      │  → alpha, alpha_t_stat (팩터로 설명 안 되는 순수 예측력)
      ▼
[6] 리포팅
      run_backtest.py: output/summary.csv, output/event_curves.png, output/judgment.csv
      (--factor-csv를 주면 [5]·[7]이 CLI에서 자동 실행됨. --market upbit이면 [4]~[5] 자동 스킵)
      ▼
[7] 판단 (규칙 기반) ← 구현 완료
      decision/rule_based.py: judge_signal(), judge_indicator()
      [3]의 원 신호 t-stat(기본 t≥+3.0, 부호 반영) + [5]의 알파·알파 t-stat + [3]의 다중검정 보정 결과를
      규칙(if/else)으로 판정
      → "채택"/"보류"/"기각" + 근거 문장. LLM 호출 없음(재현성·비용·지연 문제 회피)
      ▼
[8] 확장 검증 (표본 외·국면)  ← 구현 완료(합성 데이터로 계산 검증, 실데이터 결과 없음)
      backtest/profiles.py: --profile realistic (다음 봉 시가 진입·비용·슬리피지·보정을 한 번에)
      backtest/oos.py: --oos-train-end (퍼지 적용 표본 외 검증, 학습=BH+DSR 선택 → 검증=Bonferroni 단측 p)
      backtest/walkforward.py: --wf-first-train-end (확장창 반복 검증, 신호별 합산, fold 간 이질성 Cochran Q)
      backtest/regime.py: --regime (종목 자체 가격 기준 상승·하락·횡보, 전역 BH 재보정)
      → oos_*.csv, wf_*.csv, regime_summary.csv, 표준 "실행 조건·검증 요약" 블록(backtest/report_common.py)
      ▼
[9] 포트폴리오 시뮬레이션  ← 구현 완료(합성 데이터로 계산 검증, 실데이터 결과 없음)
      backtest/portfolio.py + run_portfolio.py: 감시리스트 다종목 동시 운용
      1거래 위험 고정(1R), 최대 동시 보유·총노출 제약, 롱숏 충돌, 평가자산 기준 MDD
      → portfolio_trades.csv, portfolio_equity.csv, portfolio_candidates.csv
      ▼
[10] 알림·대시보드  ← 운영 구성됨
      alerts/check_signals.py (평일 16:10 KST, GitHub Actions): 국내 주식은 공통 판정 게이트
      (decision/gate.py — 표본수·t(부호 반영)·BH·DSR) 통과 신호만,
      업비트는 신호 즉시 → 텔레그램. 배리어·경제성 결과는 메시지에 정보로만 첨부(게이트 아님)
      app.py (Streamlit): 결과 조회, 감시리스트 편집(관리자 전용, github_sync.py로 저장소에 반영)
```

## 단계별 상세

| 단계 | 파일 | 입력 | 출력 | 상태 |
|---|---|---|---|---|
| 0. 데이터 수집 | `data/upbit.py`, `data/krx.py`, 품질 게이트 `data/quality.py` | 종목코드, 기간 | OHLCV DataFrame (품질 검사 통과분) | 완료 (네트워크 필요) |
| 1. 지표 신호 생성 | `indicators/*.py` (6종) | OHLCV | 이벤트 컬럼 포함 df | 완료 |
| 2. 신호 필터링 | `institutional_displacement.py`(z-score), `momentum.py`, `liquidity.py` | df + 파라미터 | 필터링된 이벤트 컬럼 | 완료 |
| 3. 이벤트 스터디 | `backtest/engine.py`, `deflated.py`, `triple_barrier.py`, `economics.py` | 필터링된 df | t-stat 요약표, 수익률 곡선 | 완료 (거래비용·슬리피지·t+1 체결 옵션(기본값은 이전 동작과 동일), Newey-West·겹침 제거, Benjamini-Hochberg(p_value/significant_bh, 항상 계산됨), DSR·배리어·경제성(옵션, 기본 꺼짐)) |
| 4. 팩터 구성 | `factors/kr_fama_french.py` | KRX 전종목 시가총액·PBR·종가 | SMB/HML/WML 시계열 | 완료 (네트워크 필요, 실행 미검증; 생존편향 부분 완화 옵션(`delisted_return`) 추가) |
| 5. 팩터 조정 알파 | `backtest/engine.py::factor_adjust()`, `event_alpha()` | 3의 출력 + 4의 출력 | alpha, alpha_t_stat, beta별 t-stat, R² | 완료 (합성데이터로 회귀 정확성 검증, 실데이터 연동은 4단계 실행 이후 가능) |
| 6. 리포팅 | `run_backtest.py` | 3(+5)의 출력 | csv, png, judgment.csv | 완료 (5·7단계는 `--factor-csv`로 CLI에 연결됨) |
| 7. 판단 (규칙 기반) | `decision/rule_based.py` | 3+5의 출력 | 채택/보류/기각 + 근거 | 완료 (분기 전부 합성데이터로 검증, 다중검정 보정 결과 반영, LLM 미사용) |
| 8. 확장 검증 | `backtest/profiles.py`, `oos.py`, `walkforward.py`, `regime.py`, `report_common.py` | 3의 입력 | oos_*/wf_*/regime_summary.csv, 표준 요약 블록 | 완료 (합성데이터로 손계산 검증, **실데이터 결과 없음**) |
| 9. 포트폴리오 시뮬레이션 | `backtest/portfolio.py`, `run_portfolio.py` | 감시리스트 종목별 후보 거래 | 체결 거래, 평가자산 곡선, 성과 요약 | 완료 (합성데이터로 손계산 검증, **실데이터 결과 없음**, 결과가 가정에 의존) |
| 10. 알림·대시보드 | `alerts/`, `app.py` | 3의 summary | 텔레그램 메시지, 웹 화면 | 운영 구성됨 (실제 발송 이력은 Actions 로그로 확인) |

## 이 파이프라인이 답하는 질문 (단계별)

- 1~2단계: "언제 진입 신호가 뜨는가, 그중 믿을 만한 것만 추리면?"
- 3단계: "그 신호가 통계적으로 유의한 수익을 냈는가?"
- 4~5단계: "그 유의한 수익이 순수한 신호력인가, 아니면 이미 알려진 팩터(사이즈·가치·모멘텀) 프리미엄의 재포장인가?"
- 8단계: "학습에서 고른 신호가 다른 기간에도 유지되는가? 어떤 시장 국면에서 통하는가? 비용을 내고도 남는가?"
- 9단계: "여러 종목을 한 계좌로 운용하면 자본·포지션 제약 때문에 결과가 어떻게 달라지고, MDD는 얼마인가?"

## 지금 이 저장소 밖에 있는 것 (참고)

- **실거래 주문 자동화**: 검증된 신호를 실제 주문으로 연결하는 부분(업비트 API 주문 실행, TVExtBot 웹훅 연동)은 이 코드베이스에 없음. 신호 **알림**(텔레그램)까지만 있다.
- **머신러닝 기반 성공확률 예측**: 미구현. 판정은 통계 기준 규칙이고 어드바이저도 검증된 신호 수 다수결이다.
- **실제 시장 데이터 검증 결과**: 8~9단계를 포함해 이 저장소에는 실데이터로 돌린 결과 기록이 없다(README "검증 현황").
- **LLM 기반 판단**: TradingAgents류 멀티에이전트(bull/bear 논거 생성 등) — 7단계는 지금 규칙 기반으로만 구현했고, LLM 버전은 look-ahead bias·재현성 검증 절차가 먼저 설계돼야 해서 미착수. 나중에 추가하더라도 규칙 기반 판정을 대체하지 말고 나란히 두고 비교할 것(재현 가능한 대조군 확보).
- **생존 편향(survivorship bias) 완전 해소는 아님**: [4]단계는 `delisted_return` 옵션으로 "t0에는 있었는데 t1 조회에서 사라진 종목"을 전손 처리하는 부분 완화만 구현했다. `fetch_cross_section()`이 애초에 조회 시점에 pykrx가 반환하지 않는 종목(예: 조회 자체가 막힌 극단적 초기 상장폐지 사례)까지 포착하지는 못하므로, 완전한 해소는 아니고 "구간 중간에 사라진 종목을 조용히 빼지 않는다" 수준의 개선이다.
- **다중 검정 보정(Benjamini-Hochberg)의 적용 범위**: `compare_all()` 요약표(3단계) 차원에서는 구현했지만, `decision/rule_based.py`의 `judge_indicator()`는 지표를 하나씩 개별 호출하는 구조라 BH 절차를 적용할 "전체 묶음"이 그 함수 안에는 없음 — 최종 채택 여부는 `judge_indicator()`의 판정뿐 아니라 `run_backtest.py`가 출력하는 `significant_bh` 컬럼도 함께 봐야 함.

## 다음 실행 순서 제안

1. ~~`factor_adjust()` 구현 (5단계 채우기)~~ — 완료.
2. ~~규칙 기반 판단 레이어 구현 (7단계 채우기)~~ — 완료. `judge_signal()`(순수 판정 함수, 6가지 분기)과 `judge_indicator()`(3·5단계를 엮어 판정까지 한 번에 내는 통합 진입점) 구현, `_smoke_test_decision.py`로 전체 분기 검증 완료.
3. ~~[3]단계에 거래비용·t+1 체결 지연 반영~~ — 완료. `forward_return()`에 `entry_lag`/`entry_col`/`cost_bps` 추가, `event_study()`·`compare_all()`·`event_alpha()`·`judge_indicator()`까지 전부 관통 반영. 기본값(0, None, 0.0)은 이전 동작과 동일해서 기존 스모크 테스트 5개 전부 그대로 통과 확인, 신규 동작은 `_smoke_test_cost_entry.py`로 별도 검증(결정적 등비수열 가격으로 수치 정확성 확인). `run_backtest.py`에 `--entry-lag`/`--entry-col`/`--cost-bps` CLI 옵션 추가.
4. ~~[4]단계 생존 편향 부분 완화~~ — 완료. `_value_weighted_return()`에 `delisted_return` 옵션 추가 — t0에는 있었지만 t1 조회에서 사라진(상장폐지·거래정지 추정) 종목을 조용히 빼는 대신 지정한 수익률(예: -1.0=전손)로 강제 반영. `compute_smb_hml()`/`compute_wml()`/`build_kr_factors.py` CLI(`--delisted-return`)까지 관통 반영. 기본값 None은 이전 동작과 동일(기존 스모크 테스트 그대로 통과), 신규 동작은 `_smoke_test_survivorship.py`로 검증(저수준 함수 수치 정확성 + SMB/HML이 손실 반영 시 실제로 낮아지는 방향성 확인). 완전한 생존 편향 해소는 아니며 한계는 PIPELINE.md 및 README에 명시.
5. ~~[3]단계 요약표에 다중 검정 보정(Benjamini-Hochberg) 추가~~ — 완료. `backtest/engine.py`에 `p_value_from_t()`(정규분포 근사, scipy 불필요)와 `benjamini_hochberg()`(교과서적 BH 절차) 추가, `compare_all()`이 요약표에 `p_value`/`significant_bh` 컬럼을 항상 붙이도록 반영(`fdr` 파라미터, 기본 0.10). 새 컬럼 추가만이라 기존 컬럼을 읽는 호출부는 그대로 동작 — 기존 스모크 테스트 전부 재통과 확인. 수치 정확성은 `_smoke_test_bh.py`로 검증(p-value 기준값 대조, BH 절차 손 계산 재현, `compare_all()` 통합 시 논리적 정합성). `run_backtest.py`에 `--fdr` CLI 옵션 추가, 랭킹 출력에 `p_value`/`significant_bh` 컬럼과 해석 안내 문구 추가. 적용 범위의 한계(judge_indicator()는 미반영)는 위 "지금 이 저장소 밖에 있는 것" 절에 명시.
6. 반드시 보완해야 할 4가지 항목(거래비용/체결지연/생존편향/다중검정) 전부 반영 완료 — 로컬/서버에서 실제 pykrx 데이터로 파이프라인 전체(0→7)를 한 번 끝까지 돌려서, 지금까지 만든 필터·팩터·판정·비용반영·다중검정보정 조합이 5개 지표 각각에 대해 실제로 어떤 verdict을 내는지 첫 실측치 확인. (5·7단계를 잇는 CLI는 `--factor-csv`로 이미 연결돼 있음 — 남은 일은 실제 데이터로 한 번 끝까지 돌려 결과를 기록하는 것.)
7. ~~유동성 필터(Amihud)~~ — 완료(`--liq-filter`, `--dynamic-slippage`). 그 결과를 보고 LLM 기반 판단 도입 여부, 실거래 주문 자동화 착수 여부를 결정.
8. ~~DSR·트리플 배리어·경제적 유의성~~ — 완료. 알림 게이트는 국내 주식에서 BH → DSR, 배리어·경제성은 정보용.
9. ~~표본 외 검증(OOS)·반복 Walk-forward·시장 국면별 검증·실전 프로필~~ — 완료(합성데이터 검증). **다음: 실제 시장 데이터로 실행해 결과 기록.**
10. ~~포트폴리오 시뮬레이터~~ — 완료(합성데이터 검증). 다음: 학습 구간에서 고른 신호를 검증 구간에서만 시뮬레이션해 표본 외 포트폴리오 성과를 보는 연결.

## 방향 전환 (2026-10) — 리스크 관리 중심

위 파이프라인은 신호 검증 축입니다. 저장소의 본론은 `risk/`(시장리스크 측정·VaR 백테스트, 이후 스트레스·사이징·한도·모니터링)로 옮겨가며,
이 파이프라인의 출력(검증된 신호)은 `risk/positions.py`의 비중 테이블로 변환되어 리스크 계층의 입력이 됩니다. README의 "리스크 관리 구성" 표 참고.
