# Trading Signal Lab — 전체 개요

Pine Script 지표(기관 변동 캔들, Delta Trading, Big Sales, Ultimate RSI, VP Box + 조합 필터)를
파이썬으로 재구현하고, 각 신호가 통계적으로 유의미한 예측력을 갖는지 **이벤트
스터디(event study)** 로 검증한 뒤, 표본 외(OOS)·반복 Walk-forward·시장 국면·경제적 성과
(R 기대값, 손익분기 승률)·포트폴리오 시뮬레이션까지 확장 검증하는 **통계 검증 도구**입니다.
텔레그램 자동 알림과 Streamlit 대시보드도 포함합니다.

**현재 할 수 있는 일:** 과거 신호 성과 분석, 통계 검증(BH·DSR), 표본 외 검증, 시장 국면 분석,
위험·경제적 성과 측정, 자동 알림.
**현재 입증되지 않은 일:** 새로운 시장 환경에서의 안정적인 수익, 포트폴리오 단위 실전 성과,
머신러닝에 의한 개별 신호 성공확률 예측. 실제 시장 데이터로 이 검증을 통과했다고 기록된 신호는
아직 없습니다(15절의 검증 현황 참고). 합성 데이터 결과는 성과가 아닙니다.

---

## 1. 전체 파이프라인 (0 → 10단계)

```
[0] 데이터 수집
      data/upbit.py, data/krx.py
      → OHLCV DataFrame (open, high, low, close, volume)
      ▼
[1] 지표 신호 생성 (5종 + 조합 필터, 원본 Pine Script 재구현)
      indicators/institutional_displacement.py, delta_trading.py,
      big_sales.py, ultimate_rsi.py, vp_box.py, combo_filter.py(캔들 구조·거래량 델타·추세를 연속 점수로 결합, 6번째 신호)
      → 원본 df + 이벤트 컬럼(bull_shift, crossover_confirmed, ...)
      ▼
[2] 신호 필터링 (원 신호를 더 엄격하게 거르는 레이어)
      institutional_displacement.py (vol_mode="zscore", Kyle 1985)
      momentum.py (모멘텀 필터, Jegadeesh & Titman 1993)
      liquidity.py (유동성 필터, Amihud 2002 — --liq-filter)
      → 이벤트 컬럼에 AND 마스크 적용, 이벤트 수는 줄어들거나 같음
      ▼
[3] 이벤트 스터디 백테스트
      backtest/engine.py: event_study(), compare_all()
      entry_lag/entry_col로 t+1 시가 진입 강제(look-ahead 방지)
      cost_bps(수수료)·slippage_bps(슬리피지, 고정값 또는 Amihud 기반 동적값)로
      왕복 비용 차감(2원화)
      newey_west/drop_overlapping으로 겹치는 이벤트의 자기상관 보정
      benjamini_hochberg()로 다중 검정 보정 (p_value, significant_bh)
      → 신호별 순방향수익률 평균·표준편차·t-stat·p-value·significant_bh, 누적수익률 곡선
      ▼
[4] 팩터 구성 (별도 데이터 소스 — KRX 전체 유니버스 필요, 코인은 자동 스킵)
      factors/kr_fama_french.py: compute_smb_hml(), compute_wml()
      delisted_return 옵션으로 상장폐지 종목 전손 처리 (생존 편향 부분 완화)
      → SMB, HML, WML 월별 시계열 (Fama-French 1993 + Carhart 1997)
      ▼
[5] 팩터 조정 알파 검증 (코인은 자동 스킵)
      backtest/engine.py: factor_adjust(), event_alpha()
      [3]의 이벤트 수익률을 [4]의 SMB/HML/WML에 OLS 회귀 (merge_asof로 날짜 매칭)
      모멘텀 필터 사용 시 WML 자동 제외(이중차감 방지)
      → alpha, alpha_t_stat (팩터로 설명 안 되는 순수 예측력)
      ▼
[6] 리포팅
      run_backtest.py: output/summary.csv, output/event_curves.png, output/judgment.csv
      ▼
[7] 판단 (규칙 기반)
      decision/rule_based.py: judge_signal(), judge_indicator()
      [3]의 원 신호 t-stat + [5]의 알파·알파 t-stat + [3]의 다중검정 보정 결과를
      규칙(if/else)으로 판정
      → "채택"/"보류"/"기각" + 근거 문장. LLM 호출 없음(재현성·비용·지연 회피)
      ▼
[8] 확장 검증 (run_backtest.py의 옵션으로 켠다, 기본은 꺼짐)
      backtest/profiles.py (--profile realistic), oos.py (--oos-train-end),
      walkforward.py (--wf-*), regime.py (--regime), triple_barrier.py/economics
      → 표본 외 유지 여부, fold 간 안정성, 국면별 성과, R 기대값·손익분기 승률
      ▼
[9] 포트폴리오 시뮬레이션 (run_portfolio.py)
      backtest/portfolio.py: 1거래 위험 고정, 동시 포지션·총노출 상한, 신호 충돌 규칙
      → 누적수익률, MDD, 거래 수, 상한에 걸려 줄거나 건너뛴 횟수 (합성이면 성과 아님)
      ▼
[10] 알림·대시보드 (alerts/, app.py)
      GitHub Actions(평일 16:10 KST) → 신호 감지 → 텔레그램. 국내주식은 공통 판정
      게이트(decision/gate.py) 통과분만, 업비트 코인은 즉시 알림(watchlist 설정)
```

**--market upbit 자동 분기**: 가상자산은 4~5단계(KRX 유니버스 기반 SMB/HML/WML
팩터 구성·회귀)를 자동으로 건너뜁니다. 주식용 사이즈·가치·모멘텀 팩터를 코인에
적용하면 정의 자체가 안 맞아 회귀 결과가 무의미해지기 때문입니다 —
`run_backtest.py`가 `--market upbit`이면 `--factor-csv`를 실수로 줘도 무시하고
그 이유를 출력합니다.

**이 파이프라인이 답하는 질문**

| 단계 | 질문 |
|---|---|
| 1~2 | 언제 진입 신호가 뜨는가, 그중 믿을 만한 것만 추리면? |
| 3 | 그 신호가 통계적으로 유의한 수익을 냈는가? (거래비용·체결지연 반영, 다중검정 보정까지 마친 상태로) |
| 4~5 | 그 유의한 수익이 순수한 신호력인가, 이미 알려진 팩터 프리미엄의 재포장인가? |
| 7 | 결국 이 신호를 채택할지 말지 |

### 단계별 상태

| 단계 | 파일 | 상태 |
|---|---|---|
| 0. 데이터 수집 | `data/upbit.py`, `data/krx.py` | 완료 (네트워크 필요) |
| 1. 지표 신호 생성 | `indicators/*.py` (5종) | 완료 |
| 2. 신호 필터링 | z-score, momentum.py, liquidity.py | 완료 |
| 3. 이벤트 스터디 | `backtest/engine.py` | 완료 (거래비용·슬리피지(고정/동적)·체결지연·Newey-West·다중검정 보정 반영) |
| 4. 팩터 구성 | `factors/kr_fama_french.py` | 완료 (네트워크 필요, 실행 미검증; 생존편향 부분 완화; 코인은 자동 스킵) |
| 5. 팩터 조정 알파 | `backtest/engine.py::factor_adjust/event_alpha` | 완료 (합성데이터로 검증; 모멘텀 필터 시 WML 자동 제외; 코인은 자동 스킵) |
| 6. 리포팅 | `run_backtest.py` | 완료 (summary.csv/event_curves.png/judgment.csv) |
| 7. 판단 (규칙 기반) | `decision/rule_based.py` | 완료 (LLM 미사용, 다중검정 보정 결과까지 반영) |
| 8. 확장 검증 | `backtest/profiles.py`, `oos.py`, `walkforward.py`, `regime.py`, `triple_barrier.py` | 코드·합성 테스트 완료. **실데이터 결과 없음** |
| 9. 포트폴리오 | `backtest/portfolio.py`, `run_portfolio.py` | 코드·합성 테스트 완료. **실데이터 결과 없음** |
| 10. 알림·대시보드 | `alerts/`, `app.py` | 구성됨(운영 이력은 Actions 로그). 대시보드에 OOS 등 신규 열은 아직 없음 |

이 저장소에 없는 것: 실거래 주문 자동화(업비트 API·TVExtBot 웹훅 연동), LLM 기반 판단
레이어(도입해도 규칙 기반을 대체하지 않고 나란히 비교할 예정), 머신러닝 기반 신호 성공확률
예측, 실제 시장 데이터로 얻은 검증 결과 기록.

---

## 2. 지표별 신호

| 지표 | LONG 이벤트 | SHORT 이벤트 |
|---|---|---|
| 기관 변동 캔들 (Institutional Displacement) | 거래량 급등 + 몸통 큰 양봉 | 거래량 급등 + 몸통 큰 음봉 |
| Delta Trading | EMA(12/26) 골든크로스 + 거래량 확인 | EMA 데드크로스 + 거래량 확인 |
| Big Sales | 최근 N봉 내내 상승 마감이 없다가 거래량 폭증 | 최근 N봉 내내 하락 마감이 없다가 거래량 폭증 |
| Ultimate RSI | 과매도(20) 구간에서 위로 이탈(반등) | 과매수(80) 구간에서 아래로 이탈(반락) |
| VP Box | 직전 기간 POC(최다거래가) 가격까지 내려왔다 지지 확인 | 직전 기간 POC 가격까지 올라왔다 저항 확인 |
| 콤보 필터 (`combo_filter`) | 캔들 구조·거래량 델타·추세 점수를 연속값으로 결합한 composite_score가 임계 초과(상승 방향) | 같은 점수가 하락 방향 |

원본 Pine Script와의 근사 차이는 각 파일 상단 docstring에 명시했습니다(예: 기관
변동 캔들은 캔들 내부 매수/매도 분할 표시만 생략, VP Box는 실시간 박스 대신
"완료된 이전 기간 POC"를 사용).

---

## 3. [2]단계 — 신호 필터 (이론적 근거 보강)

### Kyle(1985) 기반 z-score 거래량 필터 (`--id-vol-mode zscore`)

원본은 "평균 거래량의 N배"라는 경험적 배수로 이상치를 판정하지만, 이는 종목마다
거래량 변동성이 다르다는 걸 무시합니다. Kyle(1985) 정보거래 모형의 균형식(주문
흐름과 가격충격의 관계)에서 착안해, 종목 고유의 거래량 변동성 대비 z-score
(`--id-z-threshold`, 기본 1.96 = 95% 유의수준)로 이상치를 정의하도록 옵션을
추가했습니다. 원본 방식(`multiple`)이 기본값으로 유지되어 기존 동작과 100%
호환됩니다.

### Jegadeesh & Titman(1993) 모멘텀 필터 (`--mom-filter`)

원 논문은 매달 전체 종목을 과거 J개월 수익률로 횡단면 정렬해 승자/패자 10분위를
만들지만, 여기서는 종목 하나씩 시계열로 백테스트하므로 그 정렬이 불가능합니다.
대신 "그 종목 자신의 과거 window 기간 모멘텀 분포 대비 지금이 몇 퍼센타일인가"라는
시계열 근사(`indicators/momentum.py`)를 씁니다.

- 롱 이벤트는 `mom_percentile >= threshold`(상승모멘텀 상위권)일 때만 인정
- 숏 이벤트는 `mom_percentile <= 1-threshold`(하락모멘텀 상위권)일 때만 인정
- `--mom-lookback`(기본 126봉≈6개월), `--mom-window`(기본 252봉≈1년),
  `--mom-threshold`(기본 0.7 = 상하위 30%)로 조절

### Amihud(2002) 유동성 필터 (`--liq-filter`, `indicators/liquidity.py`)

Amihud(2002)의 비유동성 척도 `ILLIQ_t = |R_t| / 거래대금_t`("적은 거래대금으로도
가격이 크게 움직인다"는 뜻)를 쓴다. 원 논문은 이 값을 종목별로 월평균 내어
여러 종목 간 횡단면 비교(비유동적인 종목일수록 기대수익률이 높다는 프리미엄
검증)에 쓰지만, 이 프로젝트는 종목 하나씩 시계열로만 다루므로(그리고 Amihud
비율의 절대 크기는 통화·가격대·거래관행마다 달라 종목 간 비교가 무의미하므로)
"지금이 이 종목 자기 자신의 역사에서 유동성이 좋은 편인가"라는 **시계열
퍼센타일**로만 정규화해서 쓴다 — `momentum.py`와 같은 설계 패턴이다.

- 신호가 뜬 시점의 `amihud_percentile`(0~1)이 `--liq-threshold`(기본 0.9)를
  초과하면(=그 종목 역사상 가장 비유동적이었던 상위 10% 구간) 그 이벤트는
  제외된다 — 유동성이 이 정도로 나쁘면 이벤트 스터디상 유의해 보여도 실제로는
  그만한 물량을 그 가격에 체결하기 어렵기 때문.
- 방향(롱/숏)과 무관하게 모든 이벤트 컬럼에 같은 마스크를 AND 적용한다(비유동성은
  방향성이 없는 문제).
- `--liq-window`(기본 20봉≈1개월, 일별 비유동성 평활화), `--liq-percentile-window`
  (기본 252봉≈1년, 퍼센타일 정규화 윈도우)로 조절. 윈도우가 덜 찬 구간(NaN)은
  보수적으로 통과시킨다(정보가 없다고 초반 표본을 부당하게 줄이지 않기 위해).

```bash
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 \
  --liq-filter --liq-threshold 0.9
```

---

## 4. [3]단계 — 이벤트 스터디 엔진 (`backtest/engine.py`)

### 거래비용·슬리피지·체결 시점 반영 (`--entry-lag`, `--entry-col`, `--cost-bps`, `--slippage-bps`)

"신호가 뜨자마자 그 봉 종가로 즉시, 비용 없이 체결된다"는 가정은 여러 문제를
낳습니다.

- **Look-ahead bias**: 신호는 그 봉이 마감돼야 확정되는데 같은 봉 종가로
  진입한다고 가정하면, 그 봉이 끝나기 전의 정보를 이미 알고 거래한 셈이 됩니다.
  현실적으로는 "t봉 종가에 신호 확정 → t+1봉에서 진입"이 맞습니다.
- **거래 수수료 미반영**: 업비트(편도 0.05%×2=10bp), KRX(편도 수수료+거래세 약
  0.18%×2=36bp) 같은 왕복 수수료를 빼지 않으면 짧은 보유기간 신호일수록 유의성이
  실제보다 부풀려집니다.
- **슬리피지 미반영**: 수수료와 별개로, 실제 체결가는 기대한 진입/청산가와
  벌어질 수 있습니다(특히 t+1 시가 진입이나 코인의 변동성 장세에서). `cost_bps`
  (확정 수수료)와 `slippage_bps`(불확실한 체결 미끄러짐)를 개념적으로 분리해
  둘 다 반영할 수 있게 했습니다.

```bash
# t봉 종가에 신호 확정 -> t+1봉 시가로 진입, KRX 수수료(36bp)+슬리피지(10bp) 반영
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 \
  --entry-lag 1 --entry-col open --cost-bps 36 --slippage-bps 10
```

`forward_return(df, horizon, entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0)`이
`exit_price/entry_price - 1`을 계산할 때 진입가를 `entry_lag`만큼 미래로 밀고,
`cost_bps + slippage_bps`는 direction(롱/숏)과 무관하게 항상 차감합니다(비용·
슬리피지는 방향과 상관없이 실제로 나가는 돈이므로). `event_study()`·
`compare_all()`·`event_alpha()`·`judge_indicator()`까지 전부 동일하게
관통되어, [3]단계와 [5]단계가 항상 같은 체결 가정을 씁니다. **기본값(0, None,
0.0, 0.0)은 이전 동작과 완전히 동일**하므로 명시적으로 켜야만 적용됩니다.

### Amihud(2002) 기반 동적 슬리피지 (`--dynamic-slippage`)

지금까지 `--slippage-bps`는 전 구간에 똑같은 고정값을 썼지만, 실제로는 유동성이
낮을수록 체결 시 슬리피지가 커진다. `--dynamic-slippage`를 켜면 위 [2]단계와
같은 Amihud 비유동성 퍼센타일을 이용해 봉(이벤트)마다 슬리피지 bp를 다르게
추정한다: `--slippage-bps` 값을 최소치(`base_bps`)로 삼고, 비유동성 퍼센타일에
비례해 `--max-extra-slippage-bps`(기본 50)까지 선형으로 더한다 — 유동성이
가장 나쁜 시점(percentile=1)이면 `base_bps + max_extra_bps`, 가장 좋은
시점(percentile=0 또는 윈도우 미달 NaN)이면 `base_bps`만 적용된다.

`forward_return(df, horizon, ..., slippage_bps=0.0)`의 `slippage_bps`는 이제
스칼라(고정값) 대신 `df.index`와 같은 인덱스의 `pd.Series`(봉별 슬리피지)도
받을 수 있도록 타입을 넓혔다 — `--dynamic-slippage`가 만드는 값이 바로 이
Series다. `event_study()`/`compare_all()`/`event_alpha()`/`judge_indicator()`
모두 이 Series를 그대로 받으며, `run_backtest.py`는 [3]단계(`compare_all`)와
[5]단계(`run_judgment`가 부르는 `judge_indicator`)에 **같은** 슬리피지 값(고정
또는 동적)을 넘겨 두 단계의 체결 가정이 어긋나지 않게 한다.

```bash
# 유동성 낮을수록 슬리피지가 커지도록 동적 추정 (최소 5bp, 최대 55bp)
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 \
  --dynamic-slippage --slippage-bps 5 --max-extra-slippage-bps 50
```

### 중복(겹치는) 이벤트 자기상관 보정 (`--newey-west`, `--drop-overlapping`)

보유기간(N)이 길어지면(예: 5봉, 20봉) 며칠 간격으로 뜬 신호들의 순방향수익률
추적 구간이 서로 겹치게(overlapping) 됩니다. 이 경우 관측치 간 자기상관이
생겨, 단순 `std/sqrt(n)` 방식의 t-stat이 실제보다 부풀려집니다. 두 가지
해결책을 모두 옵션으로 제공합니다.

- `--newey-west`: 표본을 그대로 두고, Newey-West(1987) HAC 표준오차(Bartlett
  커널, lag=보유기간-1)로 t-stat을 다시 계산합니다. 자기상관이 있는 데이터일수록
  t-stat이 더 보수적으로(작게) 나옵니다.
- `--drop-overlapping`: 표본 자체에서 겹치는 이벤트를 그리디하게 제거합니다
  (시간순으로 훑어 직전 채택 이벤트로부터 보유기간 이상 떨어진 것만 남김) —
  자기상관을 원천 차단하는 대신 표본 수가 줄어듭니다.

```bash
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 \
  --horizons 5,20 --newey-west
```

둘 다 `event_study()`/`compare_all()`의 옵션이며 기본 `False`(이전 동작과
동일)입니다. 동시에 켜도 되고(제거 후 남은 표본에 추가로 HAC 보정), 하나만
써도 됩니다.

### 다중 검정 보정 (`--fdr`, `benjamini_hochberg()`)

5개 지표 × 롱/숏 × 여러 보유기간을 한 번에 검증하면, 개별 t-stat은 정확해도
그중 몇 개는 순전히 우연으로 유의하게 나올 수 있습니다(p-hacking). Benjamini-
Hochberg(BH) 절차는 여러 p-value를 동시에 볼 때 이런 우연한 발견의 비율(false
discovery rate)을 목표 수준 이하로 통제합니다.

```bash
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 --fdr 0.10
```

`compare_all()`이 만드는 요약표에 자동으로 두 컬럼이 붙습니다.

- `p_value` — t-stat으로부터 정규분포 근사(scipy 없이 `math.erfc`)로 구한 양측검정 p-value
- `significant_bh` — 요약표 전체에 BH 절차를 적용해 목표 FDR(기본 10%) 이하로
  보정한 뒤에도 유의하다고 남는 신호만 `True`. 개별 t-stat이 |2| 이상이어도
  `significant_bh`가 `False`면 우연일 가능성을 배제할 수 없다는 뜻입니다.

이 `significant_bh`는 [7]단계 `judge_indicator()`/`judge_signal()`에도
`bh_significant` 파라미터로 전달되어 최종 판정("채택"/"보류"/"기각")에 직접
반영됩니다(아래 7절 참고) — `run_backtest.py`가 `compare_all()`의 요약표에서
해당 지표·보유기간의 `significant_bh` 값을 찾아 자동으로 넘겨줍니다.

---

## 5. [4]단계 — 한국시장 팩터 구성 (`factors/kr_fama_french.py`)

Fama & French(1993) SMB(사이즈)·HML(가치)와 Carhart(1997) WML(모멘텀) 팩터를
KRX 데이터로 직접 구성합니다. 지표들의 초과수익이 순수한 신호 예측력인지,
소형주·가치주·모멘텀 팩터 프리미엄의 재포장인지 가르는 데 씁니다.

- KOSPI≈NYSE, KOSDAQ≈NASDAQ 대응관계로 사이즈·BE/ME breakpoint를 KOSPI 종목만으로 계산
- BE/ME는 회계데이터 대신 PBR의 역수(1/PBR)로 근사
- 원 논문은 연 1회(6월 말) 리밸런싱하지만 여기서는 월말 리밸런싱으로 단순화
- WML은 사이즈×형성기간모멘텀(t-12개월~t-1개월 수익률, 최근 1개월은 단기반전
  회피를 위해 제외) 2×3 정렬로 승자-패자 롱숏 팩터 구성

```bash
python build_kr_factors.py --start 20220101 --end 20240101 --output output/kr_factors.csv
```

### 생존 편향 부분 완화 (`--delisted-return`)

리밸런싱 구간 도중 상장폐지·장기 거래정지된 종목은 다음 시점 조회에서 사라집니다.
이전 동작은 이런 종목을 조용히 빼고 남은 종목끼리만 가중치를 재정규화했는데,
상장폐지는 거의 항상 큰 손실로 끝나므로 이렇게 빼면 팩터 수익률이 실제보다
좋게(위로) 편향됩니다.

```bash
python build_kr_factors.py --start 20220101 --end 20240101 \
  --output output/kr_factors.csv --delisted-return -1.0
```

`_value_weighted_return(weight, close_t0, close_t1, delisted_return=None)`이
이걸 처리합니다. 기본값 `None`은 이전 동작(조용히 제외)과 동일하고, 숫자(예:
`-1.0`)를 주면 그 종목에 지정한 수익률을 강제로 대입합니다.

한계: "구간 중간에 사라진 종목을 조용히 빼지 않는다"는 부분 완화이지 완전한
해소는 아닙니다. `fetch_cross_section()`이 애초에 조회 시점에 pykrx가 반환하지
않는 종목까지 포착하지는 못합니다.

---

## 6. [5]단계 — 팩터 조정 알파 (`factor_adjust`, `event_alpha`)

이벤트 스터디 수익률을 SMB/HML/WML에 OLS 회귀시켜, 팩터로 설명되지 않는 순수
알파(및 t-통계량)를 뽑아냅니다.

- `factor_adjust(event_dates, event_returns, factor_df)` — 저수준 회귀 함수.
  이벤트 날짜를 월별 팩터의 가장 가까운 리밸런싱 시점과 매칭(`merge_asof`)한 뒤
  scipy 없이 numpy만으로 계수·t-통계량 직접 계산.
- `event_alpha(df, event_col, factor_df, horizon=20, entry_lag=..., entry_col=..., cost_bps=..., slippage_bps=...)` —
  `event_study()`와 같은 인터페이스로 감싼 편의 함수. [3]단계와 동일한 체결
  가정(entry_lag/entry_col/cost_bps/slippage_bps)을 넘겨야 두 단계가 일관됩니다.
- `combine_factors(smb_hml_df, wml_df)`로 SMB/HML과 WML을 하나의 `factor_df`로 결합.

### 회귀가 성립하지 않을 때 — 숫자를 지어내지 않는다

`factor_adjust()`는 계수가 유일하게 결정되지 않는 표본에서는 값을 전부 NaN으로 돌려주고
`note`에 사유를 담습니다(정상일 때 `note=None`). 성립하지 않는 경우는 두 가지입니다.

1. **표본 부족** — 유효 이벤트 수가 파라미터 수(절편 + 팩터 개수) 이하.
2. **설계행렬 특이(rank 부족)** — 이벤트에 붙은 팩터 행이 몇 종류뿐이라 팩터들을 서로
   구분할 수 없는 경우. `merge_asof(direction="nearest")`는 이벤트 날짜마다 가장 가까운
   월말 팩터를 붙이므로, 한 종목의 이벤트가 짧은 기간에 몰리면 서로 다른 팩터 행이
   1~2개밖에 안 붙어 rank가 파라미터 수에 못 미칩니다.

2번은 예전에 `np.linalg.inv`가 `LinAlgError`를 던져 파이프라인 전체를 중단시키던
지점입니다. 이제는 중단되지 않고 해당 신호만 판정 보류로 넘어갑니다
(`judge_signal()`은 `alpha_t_stat`이 NaN이면 판정하지 않습니다). 최소노름 해(`pinv`)로
숫자를 채우지 않는 이유는, 계수가 유일하지 않은 상태에서 고른 해는 알파를 임의로
바꿔 놓아 "팩터로 설명되지 않는 알파"라는 해석 자체가 성립하지 않기 때문입니다.

### 모멘텀 필터와 WML의 이중차감 방지

[2]단계에서 이미 `--mom-filter`(Jegadeesh & Titman 모멘텀)로 추세 신호만
추려낸 상태에서, [5]단계 회귀에 다시 WML(모멘텀 팩터)을 넣으면 모멘텀 효과가
두 단계에서 이중으로 차감되어 알파가 과소평가됩니다. `run_backtest.py`는
`--mom-filter`가 켜져 있으면 `--factor-cols`를 지정하지 않아도 자동으로 WML을
빼고 `("SMB", "HML")`만 회귀에 씁니다(명시적으로 `--factor-cols SMB,HML,WML`를
주면 자동 선택을 무시하고 그대로 따릅니다).

### 코인은 자동 스킵

`--market upbit`이면 4~5단계 전체(팩터 구성·조정 알파)를 건너뜁니다. KRX
상장 주식 유니버스로 정의한 사이즈·가치·모멘텀 팩터를 코인에 적용하면 두
시장의 팩터 정의 자체가 맞지 않아 회귀 결과가 의미 없는 숫자만 낳기
때문입니다. `--factor-csv`를 실수로 함께 줘도 무시되고 그 이유가 출력됩니다.

---

## 7. [7]단계 — 규칙 기반 판단 (`decision/rule_based.py`)

TradingAgents(Xiao et al. 2024)류 LLM 멀티에이전트 판단 구조를 참고했지만,
재현성·비용·지연·look-ahead bias 문제를 피하기 위해 지금은 순수 규칙(if/else)으로만
구현했습니다 — 같은 입력이면 항상 같은 판정이 나옵니다.

`judge_signal()`의 판정 기준(7가지 분기, `--t-threshold` 기본값 **3.0**):

1. 표본(`n_events`)이 너무 적음 → **보류**
2. 원 신호 t-stat이 유의하지 않음(|t| < `t_threshold`) → **기각**
3. `bh_significant`가 `False`로 명시됨 → **기각**(개별 t-stat은 유의해도 다중검정
   보정을 통과하지 못함 — p-hacking 가능성)
4. 알파 t-stat 계산 불가(회귀 표본 부족) → **보류**
5. 원 신호는 유의하나 알파는 유의하지 않음(팩터로 설명됨) → **기각**
6. 원 수익률과 알파의 부호가 다름 → **보류**(사람 재확인 필요)
7. 전부 통과 → **채택**

### `t_threshold` 기본값을 2.0 → 3.0으로 상향 (`--t-threshold`)

Harvey, Liu & Zhu(2016, "...and the Cross-Section of Expected Returns")는
학계가 지금까지 제시한 수백 개의 팩터 대부분이 다중 비교(multiple testing)
상황에서 우연히 유의하게 나온 것(p-hacking)임을 보이고, 단일 가설검정의
관행적 기준(|t|≥2.0)을 다중 비교 상황에 그대로 쓰면 가짜 신호를 걸러내지
못한다며 |t|≥3.0을 최소 기준으로 제안했습니다. 이 저장소는 5개 지표 × 롱/숏 ×
여러 보유기간을 동시에 검증하는 전형적인 다중 비교 상황이라 이 권고를 받아들여
`judge_signal()`/`judge_indicator()`의 `t_threshold` 기본값을 3.0으로 올렸습니다.

Benjamini-Hochberg(위 3번 분기, `significant_bh`)가 "사후에" 여러 검정 결과를
놓고 다중검정을 보정하는 것과 달리, 이건 "사전에" 개별 유의성 판정 기준
자체를 보수적으로 잡는 것이라 서로 대체재가 아니라 이중 방어선입니다. 기존
기준(2.0)으로 되돌리려면 `--t-threshold 2.0`을 지정하면 됩니다.

`judge_indicator(df, event_col, direction, factor_df, ..., bh_significant=None)`이
[3]단계(`event_study`)와 [5]단계(`event_alpha`)를 엮어 위 판정까지 한 번에
냅니다. `bh_significant`는 이 함수 혼자서는 계산할 수 없습니다 — 다중검정
보정은 "여러 지표를 동시에 비교했을 때"만 의미가 있는데, `judge_indicator()`는
지표 하나만 보기 때문입니다. 그래서 `run_backtest.py`가 `compare_all()`의
요약표에서 해당 지표·보유기간의 `significant_bh` 값을 찾아 호출부에서 넘겨줍니다
(기본 `None`이면 이 검사를 생략하고 이전 동작과 동일하게 판정).

나중에 LLM 기반 판단을 추가하더라도 이 규칙 기반 판정을 대체하지 말고 나란히
두고 비교하는 것을 권장합니다(재현 가능한 대조군 확보).

---

## 8. IS/OOS 파라미터 탐색 (`research/param_search.py`, `run_param_search.py`)

지금까지 `--mom-threshold`/`--liq-threshold` 같은 값은 논문 근거로 고른
고정 상수였습니다. 이 값을 데이터에서 "찾고" 싶다면 곧바로 세 가지 문제가
생깁니다: (1) 파라미터가 여러 개라 조합을 다 시도하면 폭발한다, (2) IS
구간에서 t-stat이 제일 높은 조합 하나를 고르는 순간 그 자체가 또 다른
다중검정(수백 개 조합 중 우연히 좋아 보인 걸 고른 위험)이 된다, (3) 종목
하나로만 보면 표본(`n_events`)이 몇십 개 수준이라 작다. `research/param_search.py`는
이 세 문제를 각각 겨냥한 도구를 제공하고, `run_param_search.py`가 이걸 하나의
검증 절차로 묶습니다.

- **좌표별 순차탐색(Anchor Search, `sequential_anchor_search`)** — 파라미터
  P개 × 후보 C개를 전부 조합하는 그리드서치(C^P번 평가)를 피하고, 한 번에
  하나씩만 바꿔가며 최선의 값을 찾는다(P×C번). 대신 파라미터 간 상호작용은
  놓칠 수 있다는 한계가 있다 — 표본이 작은 지금 상황에서는 조합폭발·소표본
  과최적화 쪽 위험이 더 크다고 보고 택한 절충.
- **Peak 대신 Plateau 탐색(`detect_plateau`)** — IS에서 성과가 가장 좋은
  후보 1개가 아니라, 성과 상위 `top_frac`(기본 20%) 후보들이 파라미터 격자
  상에서 이어진 구간(plateau)을 이루는지 본다. 이어져 있으면 그 구간의
  중앙값을 "강인한" 값으로 채택하고, 여러 구간으로 흩어져 있으면(peak, 전형적인
  과최적화 스파이크) 그 파라미터는 탐색하지 않은 것으로 보고 기본값을 유지한다.
  다만 이것도 IS 정보로 고른 것이라 편향이 완전히 사라지는 건 아니다 — 진짜
  확인은 여전히 OOS에서 한다.
- **시도 횟수(K)를 반영한 임계값(`bonferroni_t_threshold`,
  `deflated_t_threshold_sqrt2lnk`)** — Bailey, Borwein, López de Prado &
  Zhu(2014)/Harvey, Liu & Zhu(2016)가 쓰는 두 방식을 각각 구현했다. 본페로니는
  `p_value_from_t`를 이분법으로 역산해 정확하게 계산하고, `sqrt(2*ln K)`는
  관행적으로 많이 쓰이지만 K가 작을 때(대략 50 미만) 근사오차가 크다 — 그래서
  둘 다 계산해 나란히 보여준다. K는 그리드 전체 칸 수가 아니라 순차탐색이
  실제로 평가한 횟수(P×C + 최종 재평가 1회)를 쓴다.
- **횡단면 풀링 + 날짜 클러스터 강건 표준오차(`pool_event_returns`,
  `backtest.engine.cluster_robust_t_stat`)** — 여러 종목의 이벤트 수익률을
  하나의 표본으로 합쳐 n을 늘리되, 같은 날 여러 종목에서 동시에 이벤트가
  몰리는 횡단면 상관을 무시하면 t-stat이 다시 부풀려진다. Newey-West가
  "같은 종목 안에서 시간이 겹치는" 문제를 보정하는 것과 같은 원리를 "여러
  종목이 같은 날짜에 겹치는" 축에 적용한 1-way 클러스터 강건 표준오차를 쓴다.
- **IS/OOS 분할(`split_is_oos`)** — 지표(모멘텀·유동성 롤링윈도우 포함)는
  항상 전체 기간으로 미리 계산해두고, 이벤트 날짜만 `--split-date` 기준으로
  IS(≤)/OOS(>)로 자른다 — 그래야 OOS 구간 초반이 롤링윈도우 워밍업 때문에
  표본을 잃지 않는다.

전체 절차는 `run_param_search.py`가 실행한다: 여러 종목의 IS 구간에서
`mom_threshold`·`liq_threshold`를 순차탐색 → plateau 판정 → 확정된 조합의
IS t-stat이 Bonferroni/sqrt(2lnK) 중 더 엄격한 기준을 통과하는지 확인 →
통과하면(또는 통과하지 못해도 참고용으로) 탐색 과정이 전혀 들여다보지 않은
OOS 구간에서 같은 조합의 t-stat을 계산해 재현되는지 확인.

```bash
python run_param_search.py --market krx --symbols 005930,000660,035420,051910,207940 \
  --start 20180101 --end 20260901 --split-date 20240101 \
  --target-signal "델타트레이딩-골든크로스" --target-horizon 5
```

합성 데이터(자기상관을 넣은 가짜 종목 3~5개)로 배관 전체가 끝까지 도는 건
확인했지만(`_smoke_test_param_search.py`, `_smoke_test_run_param_search.py`),
진짜 유의미한 결과(IS/OOS 모두 통과하는 파라미터가 실제로 있는지)는 이
샌드박스에서 pykrx/업비트 네트워크 접속이 막혀 있어 확인할 수 없다 — 실제
KRX/업비트 데이터로는 사용자 환경에서 직접 돌려봐야 한다. 참고로 합성 데이터
실행 결과는 대체로 "기각"(IS 단계부터 시도 횟수 보정 기준 미달)이 나오는데,
이건 합성 데이터에 애초에 진짜 신호를 심어두지 않았으니 당연한 결과이지
배관 자체의 문제가 아니다.

**주의(범위)**: 이 스크립트는 `mom_threshold`/`liq_threshold` 두 값만 탐색한다.
`id_z_threshold`처럼 순서가 있는 다른 파라미터로 확장하고 싶으면
`sequential_anchor_search()`에 넘기는 `param_grid`만 늘리면 되지만,
`id_vol_mode`("multiple" vs "zscore")처럼 순서가 없는 범주형 파라미터는
plateau 개념 자체가 성립하지 않으므로 이 프레임워크 밖에서(논문 근거로) 그냥
고정해두는 걸 권장한다.

---

## 9. 질의응답형 매수/보유/매도 어드바이저 (`advisor/`, `run_advisor.py`)

지금까지의 [1]~[7]단계는 전부 "과거에 이 신호가 통계적으로 유의미했는가"를
검증하는 레이어였다. `advisor/`는 그 검증 결과를 "지금 이 순간"에 적용해,
"이 종목/코인 지금 사야 돼, 팔아야 돼, 들고 있어야 돼?"라는 질문에 답하고,
같은 기준으로 다른 유망 종목까지 추천한다. 투자자문이 아니라 "지금까지
검증한 통계가 이 순간 이 종목에 대해 뭐라고 말하는지"를 근거와 함께 요약해
보여주는 도구라는 점은 변하지 않는다 — verdict만 내지 않고 항상 어떤 신호가
어떤 근거·t-stat으로 검증됐는지를 같이 보여준다.

### 신호 합치기 — 보수적 다수결

한 종목에 여러 지표의 신호가 동시에(때로는 충돌 방향으로) 뜰 수 있다.
`advisor/evaluator.py`의 `evaluate_symbol()`은 가장 최근 봉에서 활성화된
신호들 중 "검증된"(historically `|t-stat| >= t_threshold`, 기본 3.0) 것만
추려서, 검증된 강세 신호 수와 검증된 약세 신호 수를 비교한다. 많은 쪽으로
판정하고, 동률이거나(0대0 포함) 애매하면 무조건 **보유**로 떨어진다 —
`decision/rule_based.py`의 "애매하면 보류" 철학과 같은 보수적 기본값이다.

### 신뢰도 판단 — 하이브리드(종목 자체 vs 횡단면 풀링)

"이 신호를 얼마나 믿을지"는 그 종목 자신의 과거 이벤트 수(`n_events`)가
`--min-own-events`(기본 10) 이상이면 그 종목 자체 통계를 쓰고, 부족하면(신규
상장주·알트코인처럼 역사가 짧은 경우) 같은 신호 유형을 여러 종목에서 풀링한
횡단면 통계(`advisor/universe_scan.py`의 `compute_pooled_stats()` —
[8]절과 같은 `cluster_robust_t_stat()`으로 계산)로 대체한다. 표본이 부족하면
"판단 불가"로 손 놓는 대신 "이 신호 유형은 다른 종목들에서는 대체로 어떠했는가"로
보수적으로 보강하되, 이건 "신호의 신뢰도가 종목을 넘어 전이된다"는 가정을
깔고 있다는 점이 있어 evidence에 항상 `own`/`pooled`/`미검증(표본부족)` 중
어느 근거를 썼는지 표시한다.

### 유니버스 스캔 — 배치 캐싱

"다른 유망 종목"을 추천하려면 KRX 300이나 업비트 상위 코인 전체를 훑어야
하는데, 질문이 들어올 때마다 수백 종목을 실시간으로 다 스캔하면 너무 느리고
API 호출량도 커진다. 그래서 `run_advisor.py --mode scan`을 배치로(하루 한 번
정도) 돌려 종목별 매수/보유/매도 판정과 신호유형별 풀링 통계를
`output/advisor_scan.csv`/`output/advisor_pooled_stats.csv`로 캐시해두고,
`--mode query`는 질문받은 종목 하나만 실시간으로 평가한 뒤 이 캐시에서
"다른 유망 종목"(검증된 신호가 뜬, verdict≠보유인 종목들을 강도 점수순으로)을
읽어와 같이 보여준다.

```bash
# 1) 유니버스 스캔(캐시 생성) — 하루에 한 번 정도
python run_advisor.py --mode scan --market krx \
  --symbols 005930,000660,035420,051910,207940,035720 \
  --start 20200101 --end 20260901

# 2) 개별 종목 질의 — 스캔 캐시가 있으면 다른 유망 종목도 같이 보여줌
python run_advisor.py --mode query --market krx --symbol 005930 \
  --start 20200101 --end 20260901
```

`--mode scan`/`--mode query` 모두 `run_backtest.py`와 같은 파라미터명
(`--mom-filter`, `--liq-filter`, `--entry-lag` 등)을 쓴다 — "지금 뜬 신호"가
과거에 검증했던 것과 정확히 같은 정의(같은 필터 임계값)를 쓰게 하기 위해서다.

합성 데이터로 전체 배관(종목 평가 → 하이브리드 신뢰도 → 다수결 → 유니버스
스캔 → 캐시 저장/재로딩 → CLI `scan`→`query` 순서 실행)까지
`_smoke_test_advisor.py`로 검증했다. 다만 실제로 검증된 신호가 정말 잘 통하는
종목을 찾아내는지는(진짜 신호 유무) 이 샌드박스에서 pykrx/업비트 네트워크가
막혀 있어 확인할 수 없다 — 실제 데이터로는 사용자 환경에서 직접 스캔을
돌려봐야 한다.

---

## 10. DSR·트리플 배리어·경제성 (`--deflate`, `--barrier*`)

- **DSR(Deflated Sharpe Ratio, 임계 0.95, 최소 20건)**: 같은 표에서 여러 신호·보유기간을 시도한 것을
  감안해 "많이 해보다 우연히 나온 최고값"을 걸러냅니다. BH(FDR 0.10)가 p값 기준이라면 DSR은 시도
  횟수를 반영한 샤프 기준이라 둘을 AND로 씁니다.
- **트리플 배리어(`--barrier`)**: 익절 2σ·손절 1σ·시간 한도로 거래를 끝냅니다. `--barrier-sequential`이면
  앞 거래가 끝나기 전의 신호는 건너뛰는 순차 거래, `--barrier-economics`는 R 기대값·손익분기 승률·
  비용/1R 배수를 계산합니다.
- 기본은 모두 꺼짐입니다. `--profile realistic`이 켭니다.

## 11. 실전 프로필·OOS·Walk-forward·국면 (`backtest/profiles.py`, `oos.py`, `walkforward.py`, `regime.py`)

- **`--profile realistic`**: 신호 다음 봉 시가 진입(entry_lag 1, open), 왕복 비용(KRX 36bp/업비트 10bp),
  슬리피지 10bp, Newey-West, 겹침 제거, DSR, 순차 배리어·경제성을 한 번에 켭니다. 직접 준 옵션이 프리셋보다
  우선합니다. 비용 수치는 이 저장소 도움말의 참고값이며 **현재 세율로 확인한 값이 아닙니다**. 실행 전에
  `--cost-bps`로 덮어쓰세요. 프로필을 안 주면 기존 동작 그대로입니다.
- **OOS(`--oos-train-end`)**: 학습 구간에서 BH AND DSR로 신호를 고르고, 학습 끝에서 `최대 보유기간 + 진입지연`만큼
  퍼지(purge)한 뒤 검증 구간에서 단측 p < α/k(Bonferroni)로 유지 여부를 봅니다.
- **Walk-forward(`--wf-first-train-end`, `--wf-test-bars` 등)**: 확장창(시작 고정, 끝만 전진)으로 fold마다 OOS를 반복합니다.
  신호별로 fold 결과를 역분산 가중 합산하고(z·p, Bonferroni α/m), Cochran Q로 fold 간 이질성을 검정해
  `안정 / 불균질 / 불안정 / 판정불가`를 냅니다. 한 fold에서만 엣지가 있으면 합산 z가 유의해도 `불균질`입니다.
- **국면(`--regime`)**: MA200과 60봉 수익률 z(임계 0.5)로 상승/하락/횡보를 나누고, 국면별 성과에 전역 BH를 적용합니다.

## 12. 포트폴리오 시뮬레이터 (`backtest/portfolio.py`, `run_portfolio.py`)

감시리스트 여러 종목의 후보 거래(비순차 배리어 거래)를 한 계좌에 올립니다. 비중은 1거래 위험 고정
(`w = risk_pct / (sl_mult·σ)`)이고, 종목당 최대 비중(0.25)·최대 동시 포지션(5)·총노출(1.0)·
최소 체결 비율(0.5) 상한에 걸리면 비중을 줄이거나 건너뜁니다. 같은 종목 충돌·보유 중 중복 신호는 규칙에
따라 처리하고, 우선순위는 `low_vol` 또는 `random(seed)`입니다. 일별 평가손익으로 에쿼티와 MDD를 계산하고
매수후보유와 비교합니다. 이것은 **시뮬레이션**이며 통계 검증이 아닙니다(리포트의 기준·결과는 "해당 없음").
실데이터 결과가 없으므로 포트폴리오 단위 실전 성과는 입증되지 않았습니다.

## 13. 알림·대시보드 (`alerts/`, `app.py`)

GitHub Actions가 평일 16:10 KST(cron `10 7 * * 1-5` UTC)에 `alerts/check_signals.py`를 실행해 감시리스트
(`alerts/watchlist.json`)의 신호를 확인하고 텔레그램으로 보냅니다. 국내주식은 공통 판정
게이트(13-1절)를 통과한 신호만, 업비트 코인은 `require_significance`가 기본 꺼짐이라 즉시 알림입니다.
동기화합니다. 토큰 등 인증정보는 GitHub Secrets에만 두며, 채팅이나 코드에 노출되면 즉시 재발급합니다.

### 13-1. 공통 판정 게이트 (`decision/gate.py`)

"이 신호를 믿을 것인가"에 세 진입점이 서로 다른 답을 내고 있었습니다.

| 진입점 | 예전 기준 |
|---|---|
| `decision/rule_based.py` | 표본 ≥ 10, \|t\| ≥ 3.0, BH, 알파 \|t\| ≥ 3.0, 부호 일치 |
| `alerts/check_signals.py` | BH, DSR (t 임계값도 표본수 기준도 없음) |
| `advisor/evaluator.py` | \|t\| ≥ 임계값 (BH도 DSR도 알파도 없음) |

세 축(t 임계값·다중검정·알파) 전부에서 엇갈려서 누가 더 엄격한지 줄 세울 수조차 없었습니다.
예를 들어 \|t\|=2.5이고 BH·DSR을 통과한 신호는 규칙판정에서는 기각인데 알림은 발송됐습니다.
특히 매수·보유·매도를 직접 추천하는 어드바이저가 가장 느슨한 기준을 쓰면서, 하필 유니버스를
스캔해 종목과 신호를 동시에 여럿 보는 구조라 다중검정 문제가 가장 큰 자리에서 보정이 가장
약했습니다.

지금은 세 진입점이 모두 `decision/gate.py::evaluate_gate()`를 부릅니다. 게이트는 순서 있는
다섯 층으로 되어 있고, 각 층은 통과·실패·해당없음·판정불가 중 하나를 냅니다.

| 층 | 기준 | 비고 |
|---|---|---|
| 표본수 | `n_events` ≥ 10 | 미달이면 보류 |
| 개별 t-stat | \|t\| ≥ 3.0 | Harvey, Liu & Zhu(2016) |
| BH 다중검정 | `significant_bh` | 묶음 차원 FDR 0.10 |
| DSR | `significant_dsr` | 비정규성 + 선택편의 |
| 팩터 알파 | \|alpha_t\| ≥ 3.0, 원 수익률과 부호 일치 | 팩터 데이터가 있을 때만 |

판정은 **순서상 처음 걸린 층**이 결정합니다. 실패면 기각, 판정불가면 보류, 끝까지 걸리는
층이 없으면 채택입니다. 모아서 보는 방식(실패 우선)이 아니라 순서 우선인 이유는, 표본이
모자란 신호에 "알파가 유의하지 않다"고 말해 봐야 조치로 이어지지 않기 때문입니다.

**해당없음과 판정불가를 구분하는 게 핵심입니다.** 가상자산은 KRX 사이즈·가치·모멘텀 팩터
정의가 적용되지 않아 알파 층이 구조적으로 해당없음이고, 이건 흠이 아닙니다. 반면 국내주식인데
팩터 표본이 모자라 알파를 못 구한 건 판정불가이고 보류 사유입니다. 둘을 뭉개면 코인
어드바이저가 영원히 보류로 떨어지거나, 반대로 검증하지 못한 항목을 통과로 흘려보냅니다.
같은 이유로 BH·DSR 결과가 **없으면** 통과가 아니라 판정불가입니다 — 검사하지 않은 것을
통과로 흘려보내는 게 애초에 이 게이트를 만든 이유였던 결함입니다.

**바뀐 동작**: 알림은 이제 표본수와 \|t\| ≥ 3.0도 요구하므로 예전보다 보수적입니다.
`significant_dsr` 컬럼이 없으면 예전에는 BH만으로 통과시켰지만 지금은 알림을 보내지 않습니다.
어드바이저는 BH·DSR을 적용하게 되어 추천 종목이 줄어듭니다. 업비트 알림의
`require_significance` 기본 꺼짐(원시 이벤트 즉시 알림)은 게이트 기준이 아니라 감시리스트
항목별 설정이라 그대로 두었습니다.

게이트는 "통계적으로 진짜인가"만 봅니다. "그래서 손절 걸고 매매하면 벌 수 있나"는 다른
질문이고 `backtest/economics.py`가 답합니다. 게이트 통과가 수익을 뜻하지 않습니다.

### 성과 리포트 표준 블록

OOS·Walk-forward·국면·포트폴리오 리포트는 맨 위에 같은 7개 항목(분석/데이터/기간/체결/표본/기준/결과)을
같은 순서로 출력합니다(`backtest/report_common.py`). `--data-label real|synthetic`을 생략하면 `미표시`이고
비용이 0이면 `※비용 미반영`이 붙습니다.

---

## 14. 실행 예시 모음

```bash
# 기본 실행 (업비트 코인, 일봉 1000개) — 4~5단계는 자동으로 건너뜀
python run_backtest.py --market upbit --symbol KRW-BTC --unit day --count 1000

# 국내 주식 (SK하이닉스, 2022~2026 일봉)
python run_backtest.py --market krx --symbol 000660 --start 20220101 --end 20260901

# 한국시장 팩터 구성 (생존 편향 부분 완화 포함, KRX 전용)
python build_kr_factors.py --start 20220101 --end 20240101 \
  --output output/kr_factors.csv --delisted-return -1.0

# 가장 현실적인 실행 예시: Kyle(1985) z-score 필터 + 모멘텀 필터 + 유동성 필터
# (Amihud 2002) + 거래비용/동적 슬리피지/체결지연 + Newey-West 보정 + 다중검정
# 보정 + 팩터 조정 알파·판단까지 전부 한 번에
# (--mom-filter가 켜져 있으므로 팩터 회귀에서 WML은 자동 제외됨)
python run_backtest.py --market krx --symbol 005930 --start 20220101 --end 20260901 \
  --id-vol-mode zscore --id-z-threshold 1.96 \
  --mom-filter --mom-threshold 0.7 \
  --liq-filter --liq-threshold 0.9 \
  --entry-lag 1 --entry-col open --cost-bps 36 \
  --dynamic-slippage --slippage-bps 5 --max-extra-slippage-bps 50 \
  --newey-west --fdr 0.10 \
  --factor-csv output/kr_factors.csv
```

```bash
# 확장 검증 (profile 한 줄로 실전 조건, 비용 수치는 현재 세율로 덮어쓸 것)
python run_backtest.py --market krx --symbol 005930 --start 20200101 --end 20260901 \
  --profile realistic --oos-train-end 2024-06-28 --data-label real
python run_backtest.py --market krx --symbol 005930 --start 20200101 --end 20260901 \
  --profile realistic --wf-first-train-end 2023-12-29 --wf-test-bars 60 --regime --data-label real
python run_portfolio.py --market krx --profile realistic --sim-start 2025-01-01 --data-label real
```

**결과물(`output/` 폴더)**: `summary.csv`(지표×방향×보유기간별 평균수익률·승률·
t-stat·p-value·significant_bh), `event_curves.png`(신호별 평균 누적수익률 곡선),
`judgment.csv`(`--factor-csv`를 줬을 때만 생성 — 지표별 "채택"/"보류"/"기각"
판정과 근거). 터미널에도 "보유기간 N봉 기준 t-stat 랭킹"과 "규칙 기반 판단"
표가 출력되며, 여러 조합을 비교할 때는 개별 t-stat보다 `significant_bh`
컬럼을 우선 참고해야 합니다.

---

## 15. 파일 구조 & 테스트

```
data/                시세·거시데이터 수집 (업비트 REST API, pykrx/KRX Open API, 한국은행 ECOS)
  collect.py          수집(data/*.py)과 분석(run_*.py)의 관심사 분리 계층 — 모든
                       fetch_* 함수가 지키는 계약(심볼+기간만 입력→표준 DataFrame,
                       fetched_at 등 attrs 메타데이터)과 collect_universe()(종목
                       하나 실패해도 나머지는 계속 수집)를 정의
  krx.py              pykrx/yfinance 폴백 경로 + 신규 KRX Open API(AUTH_KEY) 경로
  ecos.py             한국은행 ECOS Open API(원달러 환율 등, KRX/FDR 교차검증용)
indicators/          지표 6종 재구현(5종 + combo_filter) + 모멘텀 필터 + 유동성 필터(Amihud 2002)
backtest/engine.py   이벤트 스터디 엔진 (forward_return, event_study, event_returns,
                     compare_all, factor_adjust, event_alpha, benjamini_hochberg,
                     cluster_robust_t_stat)
backtest/stats.py    정규분포 꼬리확률(norm_sf/norm_cdf) 단일 구현 — engine·deflated·
                     walkforward가 전부 여기를 쓴다(모듈마다 복제하면 p값 기준이 갈림)
backtest/deflated.py, triple_barrier.py, economics.py   DSR, 트리플 배리어, R 기대값·손익분기
backtest/profiles.py     --profile realistic 프리셋
backtest/oos.py, walkforward.py, regime.py   표본 외, 반복 Walk-forward, 시장 국면
backtest/portfolio.py    포트폴리오 시뮬레이터
backtest/report_common.py  성과 리포트 표준 블록(분석/데이터/기간/체결/표본/기준/결과)
backtest/execution_log.py  신호 실행 로그(CSV)
alerts/              텔레그램 알림(check_signals, telegram, state, github_sync, config)
app.py               Streamlit 대시보드
tools/gen_option_table.py  argparse에서 옵션·기본값 표 생성(17절)
factors/kr_fama_french.py  SMB/HML/WML 팩터 구성
decision/gate.py           공통 판정 게이트 — 알림·어드바이저·규칙판정이 전부 여기를 쓴다
decision/rule_based.py     규칙 기반 판단 레이어(게이트에 팩터 알파까지 요구하는 래퍼)
research/param_search.py   IS/OOS 파라미터 탐색 (anchor search, plateau, 시도횟수
                     보정 임계값, 횡단면 풀링)
advisor/             매수/보유/매도 어드바이저 (evaluator.py: 종목 평가·다수결,
                     universe_scan.py: 풀링 통계·유니버스 스캔)
run_backtest.py      전체 파이프라인 CLI 진입점
run_param_search.py  IS/OOS 파라미터 탐색 CLI 진입점
run_advisor.py       매수/보유/매도 어드바이저 CLI 진입점
run_portfolio.py     포트폴리오 시뮬레이션 CLI 진입점
build_kr_factors.py  팩터 구성 전용 CLI
PIPELINE.md          파이프라인 아키텍처 문서(이 문서의 원본 중 하나)
README.md            프로젝트 소개, 할 수 있는 일/입증되지 않은 일, 검증 현황
```

아래 표는 **합성 데이터로 계산 로직이 맞는지**를 확인하는 테스트입니다. 손으로 계산한 기대값과 일치하는지,
경계·예외 처리, 기본값이 기존 동작을 보존하는지를 봅니다. 이것은 실제 시장에서 신호가 통한다는 증거가
아닙니다. 테스트가 출력하는 수익률·t값은 인위적으로 만든 데이터의 값이며 성과로 인용하지 않습니다.
개수는 늘어나므로 고정 표기하지 않고, 저장소의 `_smoke_test*.py`와 `smoke_test_*.py`를 파일별로
`python <파일>`로 실행합니다.

### 15-1. 합성 데이터 정확성 테스트

| 테스트 파일 | 검증 대상 |
|---|---|

| `_smoke_test_oos.py` | purge 구간 계산, 학습 선택→검증 유지 판정, test_end 창 |
| `_smoke_test_deflated.py`, `_smoke_test_dsr_power.py`, `_smoke_test_compare_all_dsr.py` | DSR 계산, 검정력, compare_all 통합 |
| `_smoke_test_triple_barrier.py`, `_smoke_test_barrier_study.py` | 트리플 배리어 청산 규칙·순차 거래·가격 포함 출력 |
| `_smoke_test_economics.py` | R 기대값, 손익분기 승률, 비용 배수 |
| `_smoke_test_regime.py` | 국면 분류와 국면별 BH |
| `_smoke_test_profiles.py` | 프로필 적용·사용자 옵션 우선·기본값 보존 |
| `_smoke_test_walkforward.py` | fold 생성, 신호별 합산, 이질성(Cochran Q) 판정, 경계 처리 |
| `_smoke_test_portfolio.py` | 1R 비중·상한·충돌 규칙·에쿼티·MDD 손계산 |
| `_smoke_test_run_portfolio.py` | run_portfolio.py CLI 배관과 리포트 헤더 |
| `_smoke_test_cli_oos_regime.py`, `_smoke_test_cli_help.py` | run_backtest.py 확장 옵션 배관, 도움말 |
| `_smoke_test_combo_filter.py` | 콤보 필터 점수·신호 |
| `_smoke_test_report_common.py` | 표준 리포트 블록 항목·순서·라벨·비용 미반영 표기 |
| `_smoke_test_stats.py` | 정규분포 꼬리확률 손계산 기준값, 항등식, 세 모듈이 단일 구현을 쓰는지 |
| `_smoke_test_gate.py` | 공통 판정 게이트 층별 판정, 순서 우선 규칙, 해당없음/판정불가 구분, 세 진입점 일치 |
| `_smoke_test_docs_sync.py` | OVERVIEW 옵션 대조표가 argparse 실제 정의와 일치하는지 |
| `_smoke_test_execution_log.py` | 실행 로그 손계산, signal_id 결정성, 재실행 시 중복 없음 |
| `_smoke_test_alerts.py`, `smoke_test_alerts.py`, `smoke_test_github_sync.py` | 알림 판정·상태 저장, GitHub 동기화(base64, 409 충돌), 네트워크 호출은 모킹 |

### 15-2. 실제 시장 데이터 검증 현황 (합성 테스트와 별개)

| 항목 | 상태 |
|---|---|
| 신호별 이벤트 스터디 (실데이터) | 사용자 환경에서 실행 필요. 기록된 결과 없음 |
| OOS·Walk-forward·국면·포트폴리오 (실데이터) | 미실시. 지앤비에스 에코(382800) Colab은 데이터 품질 점검에서 멈춰 있음 |
| 팩터 SMB/HML/WML | 실행 미검증 (pykrx 전종목 조회 필요) |
| KRX Open API·ECOS | 실제 응답 필드·날짜 형식 대조 미검증 |
| 프로필 비용 수치 (KRX 36bp, 업비트 10bp) | 저장소 참고값. 현재 세율로 확인되지 않음 |

실데이터 결과가 생기면 종목·기간·실행 조건·표본 수·검증 통과 여부를 날짜와 함께 이 표에 기록합니다.

---|---|
| `_smoke_test.py` | 5개 지표 + 필터 전체 파이프라인이 정상 동작하는지 |
| `_smoke_test_factors.py` | SMB/HML 계산식이 사이즈·가치효과를 정확히 분리하는지 |
| `_smoke_test_wml.py` | WML(모멘텀 팩터) 계산 정확성 |
| `_smoke_test_factor_adjust.py` | OLS 회귀가 알파·베타를 오차 1e-9 이내로 복원하는지, 표본 부족·설계행렬 특이 시 예외 없이 NaN+사유 |
| `_smoke_test_decision.py` | 규칙 기반 판정 7가지 분기 전부(bh_significant 관련 포함) |
| `_smoke_test_cost_entry.py` | 거래비용·t+1 체결지연 수치 정확성 |
| `_smoke_test_survivorship.py` | 생존 편향 완화(전손 반영) 수치·방향성 정확성 |
| `_smoke_test_bh.py` | Benjamini-Hochberg 다중검정 보정 정확성 |
| `_smoke_test_nw_slippage.py` | 슬리피지 합산 차감, Newey-West 보정, 겹치는 이벤트 제거 정확성 |
| `_smoke_test_run_judgment.py` | 코인 자동 스킵, 모멘텀 필터 시 WML 자동 제외 로직 |
| `_smoke_test_liquidity.py` | Amihud 비유동성·퍼센타일 계산, 유동성 필터, 동적 슬리피지 선형매핑, `pd.Series` 슬리피지 반영 정확성 |
| `_smoke_test_param_search.py` | plateau/peak 판정, anchor search 평가횟수·정확성, Bonferroni/sqrt(2lnK) 임계값, cluster-robust t-stat, event_returns/풀링/IS·OOS 분할 배관 |
| `_smoke_test_run_param_search.py` | run_param_search.py 전체 배관(여러 종목 로딩→탐색→plateau→임계값→IS/OOS 검증)이 끝까지 도는지 |
| `_smoke_test_advisor.py` | 활성 신호 감지, 하이브리드 신뢰도(own/pooled) 전환, 보수적 다수결, 유니버스 스캔 정렬, run_advisor.py CLI(scan→query 캐시 재사용)까지 전체 배관 |
| `_smoke_test_collect.py` | 수집 계약 3원칙(attrs 메타데이터, 종목 하나 실패해도 계속 수집, 표본부족도 실패로 분류) + run_advisor/run_param_search가 collect_universe로 정상 위임됐는지 |

---

## 16. 다음에 할 만한 것

0. `data/krx.py`의 `fetch_krx_open_api_ohlcv()`와 `data/ecos.py`는 실제 네트워크가
   열린 환경에서 처음 실행할 때 응답 필드명(`_KRX_OPEN_API_FIELDS`)·날짜 형식을
   실제 API 응답과 대조해 검증 필요(이 샌드박스는 네트워크가 막혀 있어 구조만
   맞춰두고 실제 호출로는 검증하지 못함).
1. 로컬/서버에서 실제 pykrx/업비트 데이터로 `run_param_search.py`를 KRX 종목
   여러 개(또는 업비트 상위 코인)에 돌려서, IS 단계 시도횟수 보정 기준을
   통과하고 OOS에서도 재현되는 `mom_threshold`/`liq_threshold` 조합이 실제로
   존재하는지 첫 실측치 확인. (합성 데이터로는 배관만 검증됐고, 진짜 신호
   유무는 확인 못한 상태.)
2. 그 결과를 보고 LLM 기반 판단 도입 여부, 실거래 자동화(업비트 API·TVExtBot
   웹훅) 착수 여부를 결정.
3. 종목을 KRX 300/업비트 상위 20개 수준으로 더 넓혀 완전한 횡단면 확장(지금은
   `run_param_search.py`가 사용자가 지정한 소수 종목만 풀링) — "몇 %의 종목에서
   유의한 알파가 났는가" 같은 요약 통계까지 내면 논문 완성도가 더 올라감.
4. `run_advisor.py --mode scan`을 KRX 300/업비트 상위 코인 전체로 실제 돌려서
   (지금은 합성 데이터로 배관만 검증됨) 매수/보유/매도 어드바이저가 실제로
   말이 되는 종목을 골라내는지 확인. 종목 자체 이력이 짧아 하이브리드
   신뢰도(pooled 근거)에 의존하는 비율이 얼마나 되는지도 같이 살펴볼 것.
5. 확률미분방정식 공부와 연결하고 싶다면, `forward_return`의 분포를 정규분포
   가정 없이 부트스트랩으로 재검정하거나, GBM 대비 실제 수익률 분포의 두꺼운
   꼬리(fat tail)를 비교하는 방향으로 확장 가능.

---

## 17. 옵션·기본값 대조표

`run_backtest.py`와 `run_portfolio.py`의 `argparse` 정의에서 자동 생성한 표입니다. 직접 고치지 말고
`python tools/gen_option_table.py`로 다시 만들어 붙여 넣으세요. 코드와 어긋나면 `_smoke_test_docs_sync.py`가 실패합니다.

<!-- OPTION_TABLE_START -->
| 스크립트 | 옵션 | 기본값 | 의미(도움말 첫 문장) |
|---|---|---|---|
| `run_backtest.py` | `--market` | `없음` |  |
| `run_backtest.py` | `--symbol` | `없음` | upbit: 'KRW-BTC' 형식 / krx: '005930' 형식 종목코드 |
| `run_backtest.py` | `--unit` | `day` | upbit 전용 |
| `run_backtest.py` | `--minute-unit` | `60` | upbit unit=minute일 때 분봉 단위 |
| `run_backtest.py` | `--count` | `1000` | upbit 캔들 개수 |
| `run_backtest.py` | `--start` | `없음` | krx 시작일 YYYYMMDD |
| `run_backtest.py` | `--end` | `없음` | krx 종료일 YYYYMMDD |
| `run_backtest.py` | `--horizons` | `1,3,5,10,20` | 검증할 보유기간(봉수), 콤마구분 |
| `run_backtest.py` | `--id-vol-mode` | `multiple` | 기관 변동 캔들의 거래량 이상치 판정 방식: multiple=평균의 배수(원본, 기본값), zscore=Kyle(1985) 기반 z-스코어 |
| `run_backtest.py` | `--id-vol-mult` | `2` | vol-mode=multiple일 때 임계 배수 |
| `run_backtest.py` | `--id-z-threshold` | `1.96` | vol-mode=zscore일 때 임계 z값 (기본 1.96 = 95% 유의수준) |
| `run_backtest.py` | `--mom-filter` | `꺼짐` | Jegadeesh & Titman(1993) 모멘텀 필터 활성화: 롱 이벤트는 상승 모멘텀 상위권, 숏 이벤트는 하락 모멘텀 상위권에서만 인정 |
| `run_backtest.py` | `--mom-lookback` | `126` | 모멘텀 측정 구간(봉수) |
| `run_backtest.py` | `--mom-window` | `252` | 모멘텀 퍼센타일을 매길 롤링 윈도우(봉수) |
| `run_backtest.py` | `--mom-threshold` | `0.7` | 0.5~1.0 |
| `run_backtest.py` | `--entry-lag` | `0` | 신호 확정(t봉 종가) 후 실제 진입까지 걸리는 봉 수 |
| `run_backtest.py` | `--entry-col` | `없음` | entry-lag>0일 때 실제 진입가로 쓸 컬럼명 (예: open) |
| `run_backtest.py` | `--cost-bps` | `0` | 왕복 거래 수수료(bp, 1bp=0.01%) |
| `run_backtest.py` | `--slippage-bps` | `0` | 왕복 슬리피지(bp) |
| `run_backtest.py` | `--newey-west` | `꺼짐` | 보유기간이 길어 이벤트 간 순방향수익률 추적 구간이 겹칠 때(overlapping events) 생기는 자기상관을 Newey-West HAC 보… |
| `run_backtest.py` | `--drop-overlapping` | `꺼짐` | 보유기간이 겹치는 이벤트를 애초에 표본에서 제거한다(직전 채택 이벤트로부터 보유기간 이상 떨어진 이벤트만 남김) |
| `run_backtest.py` | `--liq-filter` | `꺼짐` | Amihud(2002) 비유동성 필터 활성화: 그 종목 자기 역사 대비 비유동성이 높은(--liq-threshold 초과) 구간의 이벤트는 실제… |
| `run_backtest.py` | `--liq-window` | `20` | 일별 Amihud 비유동성을 평활화할 롤링 윈도우(봉수) |
| `run_backtest.py` | `--liq-percentile-window` | `252` | Amihud 값을 그 종목 자기 자신의 과거 분포 대비 퍼센타일로 정규화할 롤링 윈도우(봉수) |
| `run_backtest.py` | `--liq-threshold` | `0.9` | 0~1 |
| `run_backtest.py` | `--dynamic-slippage` | `꺼짐` | 슬리피지를 --slippage-bps 고정값 대신 Amihud(2002) 비유동성 퍼센타일에 비례해 봉마다 |
| `run_backtest.py` | `--max-extra-slippage-bps` | `50` | --dynamic-slippage일 때 비유동성이 가장 심한 시점(percentile=1)에 --slippage-bps에 추가로 더해질 최대 슬… |
| `run_backtest.py` | `--deflate` | `꺼짐` | 디플레이티드 샤프비율(DSR, Bailey & Lopez de Prado 2014)을 추가로 계산한다 |
| `run_backtest.py` | `--dsr-threshold` | `0.95` | --deflate일 때 significant_dsr 판정 임계값 |
| `run_backtest.py` | `--dsr-min-events` | `20` | --deflate일 때 DSR 판정에 요구하는 최소 이벤트 수 |
| `run_backtest.py` | `--dsr-extra-trials` | `0` | --deflate일 때, 이 실행 바깥에서 이미 시도한 검정 횟수 |
| `run_backtest.py` | `--barrier` | `꺼짐` | 같은 신호를 손절/익절을 걸고 매매했을 때의 실현 수익률을 tb_* 컬럼으로 나란히 보여준다(트리플 배리어) |
| `run_backtest.py` | `--barrier-pt-mult` | `2` | --barrier일 때 익절선을 실현변동성의 몇 배로 둘지 |
| `run_backtest.py` | `--barrier-sl-mult` | `1` | --barrier일 때 손절선을 실현변동성의 몇 배로 둘지 |
| `run_backtest.py` | `--barrier-vol-span` | `20` | --barrier일 때 배리어 폭 기준이 되는 실현변동성의 지수가중 구간 |
| `run_backtest.py` | `--barrier-no-intrabar` | `켜짐` | --barrier일 때 봉 내부 고가/저가 대신 종가만으로 배리어 터치를 판정한다(일봉 고저를 신뢰하기 어려울 때) |
| `run_backtest.py` | `--profile` | `없음` | 실전형 실행 조건 프리셋 |
| `run_backtest.py` | `--data-label` | `없음` | 이번에 쓴 데이터가 실제 시장 데이터인지 합성 데이터인지 리포트 머리글에 표시한다 |
| `run_backtest.py` | `--barrier-sequential` | `꺼짐` | --barrier일 때 앞 거래가 청산되기 전에 진입하는 거래를 배리어 통계에서 뺀다(순차 거래) |
| `run_backtest.py` | `--barrier-economics` | `꺼짐` | --barrier일 때 경제적 유의성 컬럼(R 단위 기대값, 손익분기 승률, 비용 대비 배수)을 요약표에 추가한다 |
| `run_backtest.py` | `--oos-train-end` | `없음` | 표본 외(OOS) 검증을 켠다 |
| `run_backtest.py` | `--oos-test-start` | `없음` | 검증 구간 첫 날(YYYY-MM-DD) |
| `run_backtest.py` | `--oos-warmup` | `60` | OOS: 지표가 안정되기 전 학습 초기 몇 봉의 이벤트를 뺄지 |
| `run_backtest.py` | `--oos-min-test-n` | `10` | OOS: 검증 구간에서 판정하려면 필요한 최소 이벤트 수 |
| `run_backtest.py` | `--oos-alpha` | `0.05` | OOS: 검증 구간 유의수준 |
| `run_backtest.py` | `--oos-require-barrier` | `꺼짐` | OOS: 학습 선택 조건에 배리어 기반 DSR 통과도 요구한다(손절을 걸고도 통과한 신호만) |
| `run_backtest.py` | `--wf-first-train-end` | `없음` | 반복 Walk-forward 검증을 켠다 |
| `run_backtest.py` | `--wf-test-bars` | `60` | Walk-forward: 검증 조각 길이(봉) |
| `run_backtest.py` | `--wf-step-bars` | `없음` | Walk-forward: 학습 끝을 한 번에 전진시키는 봉 수 |
| `run_backtest.py` | `--wf-max-folds` | `없음` | Walk-forward: 가장 최근 N개 fold만 사용 |
| `run_backtest.py` | `--wf-min-folds` | `3` | Walk-forward: 신호가 '안정/불균질/불안정'으로 판정되려면 선택돼야 하는 최소 fold 수 |
| `run_backtest.py` | `--regime` | `꺼짐` | 시장 국면(상승/하락/횡보, 종목 자체 가격 기준)별로 나눠 신호 성과를 검증한다 |
| `run_backtest.py` | `--regime-ma-len` | `200` | 국면 분류: 추세선(이동평균) 길이 |
| `run_backtest.py` | `--regime-ret-len` | `60` | 국면 분류: 최근 수익률 계산 구간 |
| `run_backtest.py` | `--regime-vol-len` | `60` | 국면 분류: 변동성 계산 구간 |
| `run_backtest.py` | `--regime-z-thr` | `0.5` | 국면 분류: 최근 수익률의 방향성(변동성 대비) 임계값 |
| `run_backtest.py` | `--regime-min-n` | `5` | 국면 일관성 판단에 쓸 국면별 최소 관측 수 |
| `run_backtest.py` | `--fdr` | `0.1` | 다중 검정 보정(Benjamini-Hochberg) 목표 false discovery rate |
| `run_backtest.py` | `--factor-csv` | `없음` | build_kr_factors.py로 만든 SMB/HML/WML csv 경로 |
| `run_backtest.py` | `--factor-cols` | `없음` | 팩터 회귀에 쓸 열 이름(콤마 구분, 예: 'SMB,HML') |
| `run_backtest.py` | `--t-threshold` | `3` | 7단계 규칙 기반 판단에서 '유의하다'고 볼 \|t-stat\| 최소값 |
| `run_backtest.py` | `--big-sales-len` | `7` |  |
| `run_backtest.py` | `--vp-bins` | `20` |  |
| `run_backtest.py` | `--vp-freq` | `W` | VP Box 구간 (W=주, M=월) |
| `run_backtest.py` | `--combo-body-min-pct` | `55` | 콤보필터: 몸통이 전체 고가-저가 범위에서 차지하는 최소 비율(%) |
| `run_backtest.py` | `--combo-wick-max-pct` | `25` | 콤보필터: 진행 방향 반대쪽 꼬리의 최대 허용 비율(%) |
| `run_backtest.py` | `--combo-vol-len` | `20` | 콤보필터: 거래량 SMA 길이 |
| `run_backtest.py` | `--combo-vol-mult` | `1.6` | 콤보필터: 거래량 스파이크 판정 배수 |
| `run_backtest.py` | `--combo-ma-len` | `50` | 콤보필터: 추세 필터 이동평균 길이 |
| `run_backtest.py` | `--combo-ma-type` | `EMA` | 콤보필터: 추세 필터 이동평균 종류 |
| `run_backtest.py` | `--combo-score-threshold` | `0.62` | 콤보필터: composite_score 신호 인정 임계값(0~1) |
| `run_backtest.py` | `--log-executions` | `꺼짐` | 이번 실행에서 뜬 신호 이벤트들을 signal_id 단위로 체결 로그 CSV에 기록한다(메타라벨링용 원재료) |
| `run_backtest.py` | `--execution-log-path` | `없음` | --log-executions일 때 기록할 CSV 경로 |
| `run_backtest.py` | `--output` | `output` |  |
| `run_portfolio.py` | `--watchlist` | `alerts/watchlist.json` | 종목 목록 JSON |
| `run_portfolio.py` | `--market` | `krx` | 한 번에 한 시장만 돌린다(달력·연환산 기준이 달라 섞으면 해석이 흐려진다) |
| `run_portfolio.py` | `--start` | `없음` | krx 데이터 시작(YYYYMMDD) |
| `run_portfolio.py` | `--end` | `없음` | krx 데이터 끝(YYYYMMDD) |
| `run_portfolio.py` | `--profile` | `realistic` | 실행 조건 프리셋 |
| `run_portfolio.py` | `--cost-bps` | `없음` | 프리셋의 왕복비용을 덮어씀(bp) |
| `run_portfolio.py` | `--slippage-bps` | `없음` | 프리셋의 슬리피지를 덮어씀(bp) |
| `run_portfolio.py` | `--signals` | `없음` | 쓸 신호 라벨을 쉼표로 |
| `run_portfolio.py` | `--hold-bars` | `없음` | 배리어 최대 보유 봉수 |
| `run_portfolio.py` | `--sim-start` | `없음` | 이 날짜 이후 진입만 시뮬레이션(YYYY-MM-DD) |
| `run_portfolio.py` | `--sim-end` | `없음` | 이 날짜 이전 진입만 시뮬레이션(YYYY-MM-DD) |
| `run_portfolio.py` | `--risk-pct` | `0.005` | 1거래 위험(1R)을 자본의 몇 %로 할지(소수) |
| `run_portfolio.py` | `--max-positions` | `5` | 최대 동시 보유 종목 수 |
| `run_portfolio.py` | `--max-gross` | `1` | 총 노출 상한(자본 대비) |
| `run_portfolio.py` | `--max-weight` | `0.25` | 종목당 비중 상한 |
| `run_portfolio.py` | `--min-fill-frac` | `0.5` | 남은 여력이 의도 비중의 이 비율 미만이면 건너뜀(이상이면 줄여서 매수) |
| `run_portfolio.py` | `--priority` | `low_vol` | 같은 날 후보가 많을 때 순서 |
| `run_portfolio.py` | `--seed` | `0` | --priority random의 시드 |
| `run_portfolio.py` | `--seeds` | `1` | --priority random일 때 시드 0..N-1로 반복해 결과 분포를 보여준다(우선순위 가정에 대한 민감도) |
| `run_portfolio.py` | `--data-label` | `없음` | 데이터가 실제 시장 데이터인지 합성인지 리포트 머리글에 표시 |
| `run_portfolio.py` | `--output` | `output` |  |
<!-- OPTION_TABLE_END -->
