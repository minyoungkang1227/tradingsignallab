"""
trading_signal_lab을 매번 CLI로 돌리지 않고 브라우저에서 바로 돌려보는 Streamlit 대시보드.

[설계 원칙] run_backtest.py의 실제 로직(load_data/run/run_judgment)을 그대로 import해서
재사용한다 — 이 파일 안에서 백테스트 로직을 다시 구현하지 않는다. CLI와 대시보드가
서로 다른 결과를 낼 위험(로직이 두 곳에 흩어져 하나만 고치는 실수)을 원천 차단하기 위함.

run_backtest.main()의 argparse 결과를 흉내낸 SimpleNamespace를 직접 만들어서
run_backtest.load_data()/run()/run_judgment()에 그대로 넘긴다 — CLI 플래그 이름과
1:1 대응이라 CLI에 새 옵션이 추가되면 build_args()의 DEFAULTS에도 같이 추가해야
깨지지 않는다(이 파일의 유일한 유지보수 포인트).

이 파일은 화면 문구(한글 라벨, 설명 문구)만 다루고 계산 로직에는 전혀 손대지 않는다
— 표시용 매핑(DIRECTION_LABELS, COLUMN_LABELS 등)은 전부 화면에 보여주기 직전에만
적용되고, run_backtest.py로 넘어가는 값(args)이나 CLI 동작에는 영향을 주지 않는다.

사용법: streamlit run app.py
"""
from __future__ import annotations

import types

import pandas as pd
import streamlit as st

import run_backtest as rb
from alerts.github_sync import (
    GitHubSyncError, STRENGTH_LABELS, fetch_watchlist, items_to_rows, rows_to_items,
    update_watchlist,
)

# alerts/check_signals.py(텔레그램 알림, GitHub Actions)가 읽는 바로 그 저장소/파일.
# 아래 "5. 감시리스트 관리" 섹션은 이 경로를 직접 고쳐 쓴다 — 자세한 이유는
# alerts/github_sync.py 모듈 docstring 참고.
WATCHLIST_REPO = "minyoungkang1227/tradingsignallab"
WATCHLIST_PATH = "alerts/watchlist.json"

# run_backtest.py main()의 argparse 기본값과 1:1로 맞춘 기본 설정.
# CLI에 새 --옵션이 추가되면 여기에도 같은 dest/default를 추가해야 함.
DEFAULTS = dict(
    unit="day", minute_unit=60, count=1000, start=None, end=None,
    horizons="1,3,5,10,20",
    id_vol_mode="multiple", id_vol_mult=2.0, id_z_threshold=1.96,
    mom_filter=False, mom_lookback=126, mom_window=252, mom_threshold=0.7,
    entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
    newey_west=False, drop_overlapping=False,
    liq_filter=False, liq_window=20, liq_percentile_window=252, liq_threshold=0.9,
    dynamic_slippage=False, max_extra_slippage_bps=50.0,
    fdr=0.10, factor_csv=None, factor_cols=None, t_threshold=3.0,
    big_sales_len=7, vp_bins=20, vp_freq="W",
    combo_body_min_pct=55.0, combo_wick_max_pct=25.0, combo_vol_len=20,
    combo_vol_mult=1.6, combo_ma_len=50, combo_ma_type="EMA", combo_score_threshold=0.62,
    log_executions=False, execution_log_path=None, output="output",
)

# 결과 화면에서만 쓰는 한글 라벨 매핑 (run_backtest.py/backtest/engine.py의 실제
# 컬럼명·값은 그대로 두고, 화면에 보여줄 때만 바꿔치기한다).
COLUMN_LABELS = {
    "signal": "신호",
    "direction": "방향",
    "n_events": "표본수",
    "mean_return_pct": "평균수익률(%)",
    "win_rate_pct": "승률(%)",
    "t_stat": "t값",
    "p_value": "p값",
    "significant_bh": "통계적 유의 (BH보정)",
}
DIRECTION_LABELS = {"LONG": "상승(롱)", "SHORT": "하락(숏)"}

# 사이드바 종목 선택 드롭다운에 쓸 예시 종목. 종목코드를 모르는 사람도 몇 번 눌러보며
# 바로 써볼 수 있게 하기 위함 — 여기 없는 종목은 '직접 입력'을 고르면 된다.
KRX_EXAMPLES = {
    "005930": "삼성전자 (005930)",
    "000660": "SK하이닉스 (000660)",
    "035420": "NAVER (035420)",
    "035720": "카카오 (035720)",
    "005380": "현대차 (005380)",
}
UPBIT_EXAMPLES = {
    "KRW-BTC": "비트코인 (KRW-BTC)",
    "KRW-ETH": "이더리움 (KRW-ETH)",
    "KRW-XRP": "리플 (KRW-XRP)",
    "KRW-SOL": "솔라나 (KRW-SOL)",
}
DIRECT_INPUT = "직접 입력"


def build_args(**overrides) -> types.SimpleNamespace:
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def is_owner() -> bool:
    """공개 배포된 주소에는 누구나 들어올 수 있어서, '체결 로그 기록' 같은 본인 전용
    기능은 기본적으로 숨긴다. Streamlit Cloud의 Secrets에 OWNER_TOKEN을 등록해두고
    (이 값은 .gitignore에 걸린 secrets.toml에만 있어 깃허브 공개 저장소에는 올라가지
    않는다), 주소 끝에 ?owner=<그 값>을 붙여서 접속했을 때만 본인으로 인식한다.
    로컬에서 secrets.toml 없이 돌릴 때는 조용히 False로 처리한다.
    """
    try:
        token = st.secrets.get("OWNER_TOKEN")
    except Exception:
        token = None
    if not token:
        return False
    try:
        return st.query_params.get("owner") == token
    except Exception:
        return False


def github_token() -> str | None:
    """감시리스트 관리 섹션이 GitHub 저장소에 쓰기 위한 토큰. OWNER_TOKEN과 같은 자리
    (Streamlit Cloud Secrets)에 GITHUB_TOKEN으로 등록해둔다. 없으면 None — 그 경우
    섹션 자체를 숨긴다(토큰 없이 시도했다가 애매한 에러를 보는 것보다 낫다)."""
    try:
        return st.secrets.get("GITHUB_TOKEN")
    except Exception:
        return None


IS_OWNER = is_owner()


st.set_page_config(page_title="trading_signal_lab 대시보드", layout="wide")
st.title("trading_signal_lab — 신호 검증 대시보드")
st.caption(
    "run_backtest.py를 매번 터미널에서 돌리는 대신 여기서 종목/파라미터만 바꿔가며 "
    "이벤트 스터디 결과를 바로 확인한다. 로직은 run_backtest.py와 완전히 동일 — "
    "이 페이지는 그 함수들을 그대로 호출하는 얇은 껍데기다."
)

with st.expander("이 페이지가 하는 일이 궁금하다면 (처음이면 한 번 읽어보세요)"):
    st.markdown(
        "- **무엇을 하나요**: 특정 조건(예: 거래량이 평소보다 많이 터졌다, 두 이동평균선이 "
        "교차했다 등)이 발생했을 때마다 그 시점을 '신호'로 표시해두고, 그 이후 일정 기간이 "
        "지났을 때 실제로 주가가 어떻게 움직였는지를 과거 데이터로 전부 모아서 통계적으로 "
        "요약해줍니다. 이걸 흔히 '이벤트 스터디'라고 부릅니다.\n"
        "- **왜 필요한가요**: \"이 신호, 왠지 잘 맞는 것 같다\"는 감이 아니라, 과거에 그 신호가 "
        "몇 번 나왔고(표본수) 그때마다 평균 몇 % 움직였고(평균수익률) 그게 우연이 아니라고 "
        "볼 만큼 통계적으로 믿을 만한지(t값·p값)를 숫자로 확인하기 위함입니다.\n"
        "- **주의할 점**: 과거에 잘 맞았다고 미래에도 그대로 맞는다는 보장은 없습니다. 표본수가 "
        "너무 적거나(10개 미만) 통계적 유의성을 통과하지 못한 신호는 아직 결론으로 쓰기엔 "
        "이릅니다. 이 화면은 투자 자문이 아니라 검증 도구입니다."
    )

with st.sidebar:
    st.header("1. 데이터 불러오기")
    st.caption("어떤 종목을, 어느 기간만큼 가져와서 검증할지 정합니다.")
    market = st.selectbox(
        "시장", ["krx", "upbit"], index=0,
        format_func=lambda v: {"krx": "국내 주식 (KRX)", "upbit": "업비트 코인"}[v],
    )
    examples = KRX_EXAMPLES if market == "krx" else UPBIT_EXAMPLES
    symbol_choice = st.selectbox(
        "종목 선택", list(examples.keys()) + [DIRECT_INPUT],
        format_func=lambda c: examples.get(c, "✏️ 직접 입력 (종목코드를 아는 경우)"),
        help="자주 찾는 종목 몇 개를 바로 고를 수 있게 모아뒀습니다. 목록에 없는 "
             "종목은 맨 아래 '직접 입력'을 고르세요.",
    )
    if symbol_choice == DIRECT_INPUT:
        symbol = st.text_input(
            "종목코드 직접 입력", value="005930" if market == "krx" else "KRW-BTC",
            help="국내 주식은 '005930'(삼성전자)처럼 6자리 종목코드, "
                 "업비트는 'KRW-BTC'처럼 '원화마켓-코인심볼' 형식으로 입력하세요.",
        )
    else:
        symbol = symbol_choice
    if market == "krx":
        start = st.text_input(
            "시작일 (YYYYMMDD)", value="20200101",
            help="데이터를 가져올 시작 날짜. 예: 2020년 1월 1일 → 20200101",
        )
        end = st.text_input(
            "종료일 (YYYYMMDD)", value="20260901",
            help="데이터를 가져올 마지막 날짜.",
        )
        count, unit, minute_unit = DEFAULTS["count"], DEFAULTS["unit"], DEFAULTS["minute_unit"]
    else:
        unit = st.selectbox(
            "봉 단위", ["day", "week", "minute"], index=0,
            format_func=lambda v: {"day": "일봉", "week": "주봉", "minute": "분봉"}[v],
        )
        minute_unit = st.number_input(
            "분봉 단위(분)", value=60, step=1,
            help="봉 단위를 '분봉'으로 선택했을 때만 사용됩니다. 예: 60이면 60분봉.",
        ) if unit == "minute" else 60
        count = st.number_input(
            "캔들 개수", value=1000, step=100,
            help="가장 최근 캔들부터 몇 개를 가져올지.",
        )
        start = end = None

    horizons = st.text_input(
        "검증할 보유기간 (봉 수, 콤마로 구분)", value="1,3,5,10,20",
        help="신호가 뜬 뒤 1봉, 3봉, 5봉... 뒤의 수익률을 각각 확인합니다. "
             "일봉 기준이면 '5'는 신호 발생 5거래일 뒤라는 뜻입니다.",
    )

    # 체결 로그 기록은 배포자 본인의 메타라벨링 작업용 기능이라, 공개 주소로 들어온
    # 일반 방문자에게는 아예 보이지 않게 한다 — 여러 사람이 같은 서버의 같은 파일에
    # 동시에 기록을 남기면 서로 내용이 섞여버리는 문제도 막을 수 있다.
    if IS_OWNER:
        st.header("2. 체결 기록 (선택, 관리자 전용)")
        st.caption(
            "메타라벨링(나중에 어떤 신호가 더 믿을 만한지 머신러닝으로 다시 거르는 작업)을 "
            "위한 준비 단계입니다. 지금 당장 필요 없다면 꺼둔 채로 실행해도 결과 확인에는 "
            "아무 지장이 없습니다."
        )
        log_executions = st.checkbox(
            "이번 실행의 신호를 체결 로그 파일에 기록하기", value=False,
            help="이번 실행에서 나온 신호 하나하나를 output/execution_log.csv 파일에 "
                 "계속 누적해서 쌓아둡니다. 나중에 메타라벨링 모델을 학습시킬 때 쓸 원재료를 "
                 "미리 모아두는 용도입니다. 참고: 청산 사유(exit_reason)는 전부 '보유기간 "
                 "만료(horizon_exit)'로 기록되며, 실제 손절/익절 로직이 적용된 건 아닙니다.",
        )
    else:
        log_executions = False

    with st.expander("3. 고급 설정 (선택) — 더 현실적으로, 더 엄격하게"):
        st.caption(
            "기본값 그대로 둬도 결과를 보는 데는 전혀 문제없습니다. 아래는 '실제로 거래했다면 "
            "어땠을까'를 더 현실에 가깝게 보거나, 결과를 더 깐깐하게 검증하고 싶을 때만 "
            "건드리면 되는 선택 항목입니다."
        )

        st.markdown("**① 실제로 거래했다면?**")
        entry_lag = st.number_input(
            "신호 발생 후 며칠(봉) 뒤에 살까요?", value=0, step=1,
            help="0이면 신호가 뜬 바로 그날 매수한다고 가정합니다. 예: 1이면 신호 다음날 매수.",
        )
        entry_col = st.selectbox(
            "그날 매수 가격은 어디 기준?", [None, "open", "close"], index=0,
            format_func=lambda v: {
                None: "기본값 (별도 지정 안 함)", "open": "시가 (장 시작가)", "close": "종가 (장 마감가)",
            }[v],
            help="매수 체결가를 시가로 볼지 종가로 볼지 고릅니다. 잘 모르면 기본값으로 둬도 됩니다.",
        )
        cost_col, slip_col = st.columns(2)
        with cost_col:
            cost_bps = st.number_input(
                "수수료 (bp, 왕복)", value=0.0, step=1.0,
                help="사고팔 때 나가는 수수료. 1bp = 0.01%예요 — 예를 들어 왕복 수수료가 "
                     "0.015%라면 1.5를 입력하면 됩니다. 모르면 0으로 둬도 됩니다.",
            )
        with slip_col:
            slippage_bps = st.number_input(
                "슬리피지 (bp, 왕복)", value=0.0, step=1.0,
                help="원하는 가격과 실제 체결가의 차이(미끄러짐). 수수료와 같은 bp 단위입니다.",
            )

        st.divider()
        st.markdown("**② 이 결과, 얼마나 믿을 수 있나요?**")
        newey_west = st.checkbox(
            "더 엄격한 통계 보정 적용하기 (Newey-West)", value=False,
            help="수익률이 하루하루 비슷하게 이어지는(자기상관) 경향이 있으면 t값이 실제보다 "
                 "커 보일 수 있습니다. 이 보정을 켜면 좀 더 보수적인(깐깐한) t값을 계산합니다.",
        )
        drop_overlapping = st.checkbox(
            "겹치는 신호는 한 번만 세기", value=False,
            help="신호가 연달아 떠서 보유기간이 서로 겹치면 사실상 같은 구간을 중복으로 "
                 "세는 셈입니다. 이를 막고 더 독립적인 표본만 남기고 싶을 때 켭니다.",
        )
        fdr = st.slider(
            "여러 신호를 동시에 비교할 때 허용할 오차 수준", 0.01, 0.50, 0.10, step=0.01,
            help="지표 여러 개를 한꺼번에 비교하다 보면 그중 일부는 순전히 운으로 좋아 "
                 "보일 수 있습니다(통계 용어로 FDR). 낮출수록 더 엄격하게, '진짜' 신호만 "
                 "통과시킵니다.",
        )
        t_threshold = st.number_input(
            "신호의 '합격선'으로 볼 t값", value=3.0, step=0.5,
            help="이 값보다 커야 믿을 만한 신호로 봅니다. 보통 통계에서는 절댓값 2 이상이면 "
                 "눈여겨볼 만하다고 보는데, 여기서는 더 엄격하게 3.0을 기본값으로 뒀습니다.",
        )

    with st.expander("4. 콤보필터 — 캔들 모양 + 거래량으로 신호 거르기"):
        st.caption(
            "캔들 하나하나의 생김새(몸통이 크고 꼬리는 짧은지), 그 순간 거래량이 평소보다 "
            "많이 터졌는지, 그리고 전체적인 추세 방향과 맞는지를 한꺼번에 점수로 매겨서, "
            "일정 점수를 넘을 때만 신호로 인정하는 필터입니다. (TradingView에 있던 원본 "
            "Pine Script 지표 'candle_volume_entry_filter'를 그대로 옮겨온 것입니다.)"
        )

        combo_advanced = st.checkbox(
            "세부 수치를 직접 조정하고 싶어요 (전문가용)", value=False,
            help="끄면 아래 설명대로의 검증된 기본값을 그대로 사용합니다.",
        )

        if not combo_advanced:
            st.caption(
                "지금은 기본값을 그대로 쓰고 있습니다 — 캔들 몸통이 전체 길이의 55% 이상, "
                "위아래 꼬리는 25% 이하, 최근 20봉 평균 거래량의 1.6배 이상 터지고, 50봉 "
                "지수이동평균(EMA) 방향과 일치하며, 이 조건들을 종합한 점수가 0.62 이상일 "
                "때만 신호로 인정합니다. 체크박스를 켜면 이 숫자들을 직접 바꿀 수 있습니다."
            )
            combo_body_min_pct = DEFAULTS["combo_body_min_pct"]
            combo_wick_max_pct = DEFAULTS["combo_wick_max_pct"]
            combo_vol_len = DEFAULTS["combo_vol_len"]
            combo_vol_mult = DEFAULTS["combo_vol_mult"]
            combo_ma_len = DEFAULTS["combo_ma_len"]
            combo_ma_type = DEFAULTS["combo_ma_type"]
            combo_score_threshold = DEFAULTS["combo_score_threshold"]
        else:
            st.markdown("**캔들 모양 조건**")
            combo_body_min_pct = st.number_input(
                "몸통은 전체 길이의 최소 몇 %?", value=55.0,
                help="캔들 전체 길이(고가-저가) 대비 몸통(시가-종가)이 이 비율 이상이어야 "
                     "'힘있게 밀어붙인' 캔들로 봅니다. 높일수록 더 확실한 모양만 인정합니다.",
            )
            combo_wick_max_pct = st.number_input(
                "꼬리는 전체 길이의 최대 몇 %까지 허용?", value=25.0,
                help="위/아래 꼬리가 이 비율을 넘으면 '망설이다 되돌아온' 모양으로 보고 "
                     "신호에서 제외합니다. 낮출수록 더 깔끔한 모양만 통과합니다.",
            )

            st.markdown("**거래량 조건**")
            combo_vol_len = st.number_input(
                "평균 거래량을 비교할 기간 (봉 수)", value=20, step=1,
                help="최근 거래량이 '평소보다 많다'고 판단할 때, 몇 봉 평균을 기준으로 "
                     "삼을지.",
            )
            combo_vol_mult = st.number_input(
                "평균 대비 몇 배 이상이어야 '거래량 터짐'으로 볼까요?", value=1.6,
                help="예: 1.6이면 위에서 정한 평균 거래량의 1.6배 이상일 때만 인정합니다. "
                     "높일수록 확실히 큰 거래량만 통과합니다.",
            )

            st.markdown("**추세 방향 조건**")
            combo_ma_len = st.number_input(
                "추세 판단용 이동평균 기간 (봉 수)", value=50, step=1,
                help="이 기간의 이동평균선 방향으로 전체 추세를 판단합니다.",
            )
            combo_ma_type = st.selectbox(
                "이동평균 종류", ["EMA", "SMA", "WMA"], index=0,
                format_func=lambda v: {
                    "EMA": "지수이동평균 (EMA) — 최근 가격에 더 민감하게 반응",
                    "SMA": "단순이동평균 (SMA) — 전체 구간을 동일한 비중으로",
                    "WMA": "가중이동평균 (WMA) — 최근으로 갈수록 비중을 더 크게",
                }[v],
            )

            st.markdown("**종합 판정**")
            combo_score_threshold = st.number_input(
                "종합 점수 기준 (0~1)", value=0.62, min_value=0.0, max_value=1.0,
                help="위 세 가지 조건을 모두 합쳐 0~1 사이 점수로 매긴 뒤, 이 값 이상일 "
                     "때만 신호로 인정합니다. 원본 Pine Script가 쓰던 값은 0.62이고, 이 "
                     "대시보드로 직접 재검증해보기 전까지는 '일단 원본 그대로 따라간다'는 "
                     "참고용 값 정도로 보는 게 안전합니다.",
            )

    # 텔레그램 알림(GitHub Actions, alerts/check_signals.py)이 감시할 종목 목록을
    # 여기서 직접 관리한다. 위쪽의 '검증 실행'과는 완전히 별개 기능이라 결과 화면에는
    # 영향을 주지 않는다 — 공개 방문자에게는 아예 안 보이게 owner 전용으로 숨긴다
    # (텔레그램 알림이 본인 한 명에게만 가는 개인용 기능이기 때문).
    if IS_OWNER:
        st.header("5. 감시리스트 관리 (관리자 전용, 텔레그램 알림용)")
        _token = github_token()
        if not _token:
            st.caption(
                "GITHUB_TOKEN이 Secrets에 등록돼 있지 않아 이 섹션을 쓸 수 없습니다 "
                "(alerts/github_sync.py 모듈 docstring에 발급 방법 설명)."
            )
        else:
            st.caption(
                "평일 16:10에 자동으로 도는 텔레그램 알림(GitHub Actions)이 감시하는 "
                "종목 목록입니다. 여기서 저장하면 저장소의 alerts/watchlist.json이 "
                "바로 바뀌고, 다음 실행부터 적용됩니다. '강도'는 시장별 기본값을 "
                "덮어쓸 때만 바꾸면 됩니다 — 국내 주식은 기본이 'BH보정 필요'(검증된 "
                "신호만 알림), 코인은 기본이 '즉시 알림'(검증 없이 원시 신호 바로 "
                "알림)입니다."
            )

            if "watchlist_sha" not in st.session_state or st.button("감시리스트 새로고침"):
                try:
                    _items, _sha = fetch_watchlist(_token, WATCHLIST_REPO, WATCHLIST_PATH)
                    st.session_state["watchlist_items"] = _items
                    st.session_state["watchlist_sha"] = _sha
                except GitHubSyncError as e:
                    st.error(str(e))
                    st.session_state.pop("watchlist_items", None)
                    st.session_state.pop("watchlist_sha", None)

            if "watchlist_items" in st.session_state:
                _df = pd.DataFrame(
                    items_to_rows(st.session_state["watchlist_items"]),
                    columns=["시장", "종목코드", "이름", "강도"],
                )

                _edited = st.data_editor(
                    _df,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "시장": st.column_config.SelectboxColumn(options=["krx", "upbit"], required=True),
                        "종목코드": st.column_config.TextColumn(required=True),
                        "이름": st.column_config.TextColumn(),
                        "강도": st.column_config.SelectboxColumn(
                            options=list(STRENGTH_LABELS.values()), required=True,
                        ),
                    },
                    key="watchlist_editor",
                )

                if st.button("감시리스트 저장", type="primary"):
                    _new_items = rows_to_items(_edited.to_dict("records"))
                    try:
                        new_sha = update_watchlist(
                            _token, WATCHLIST_REPO, _new_items, st.session_state["watchlist_sha"],
                            WATCHLIST_PATH,
                        )
                        st.session_state["watchlist_items"] = _new_items
                        st.session_state["watchlist_sha"] = new_sha
                        st.success(f"저장했습니다 ({len(_new_items)}개 종목). 다음 알림 실행부터 적용됩니다.")
                    except GitHubSyncError as e:
                        st.error(str(e))

    run_clicked = st.button("검증 실행", type="primary", use_container_width=True)

if not run_clicked:
    st.info("왼쪽 사이드바에서 종목과 기간을 정하고 [검증 실행]을 눌러주세요. "
            "나머지 설정은 기본값 그대로 둬도 괜찮습니다.")
    st.stop()

args = build_args(
    market=market, symbol=symbol, unit=unit, minute_unit=minute_unit, count=count,
    start=start, end=end, horizons=horizons,
    entry_lag=entry_lag, entry_col=entry_col, cost_bps=cost_bps, slippage_bps=slippage_bps,
    newey_west=newey_west, drop_overlapping=drop_overlapping, fdr=fdr, t_threshold=t_threshold,
    combo_body_min_pct=combo_body_min_pct, combo_wick_max_pct=combo_wick_max_pct,
    combo_vol_len=combo_vol_len, combo_vol_mult=combo_vol_mult, combo_ma_len=combo_ma_len,
    combo_ma_type=combo_ma_type, combo_score_threshold=combo_score_threshold,
    log_executions=log_executions,
)

with st.spinner(f"{market}/{symbol} 데이터 수집 중..."):
    try:
        df = rb.load_data(args)
    except SystemExit as e:
        st.error("데이터를 가져오지 못했습니다. 종목 선택과 시작일/종료일 형식을 "
                 "다시 확인해주세요 (예: 종목코드 005930, 날짜 20200101).")
        with st.expander("자세한 오류 내용 보기 (문제 해결용)"):
            st.code(str(e))
        st.stop()
    except Exception as e:
        st.error("데이터를 가져오는 중 문제가 발생했습니다. 종목코드가 맞는지, 혹은 "
                 "데이터 제공처가 일시적으로 불안정한 건 아닌지 확인한 뒤 다시 "
                 "시도해주세요.")
        with st.expander("자세한 오류 내용 보기 (문제 해결용)"):
            st.code(str(e))
        st.stop()

st.success(f"{len(df)}봉 수집 완료 ({df.index.min().date()} ~ {df.index.max().date()})")

with st.spinner("이벤트 스터디 계산 중..."):
    summary, results, dfs_by_indicator, slippage_arg = rb.run(df, args)

if summary.empty:
    st.warning("검증된 이벤트가 없습니다 (신호가 한 번도 발생하지 않았을 수 있습니다). "
               "기간을 늘리거나 종목을 바꿔서 다시 시도해보세요.")
    st.stop()

horizon_list = [int(h) for h in horizons.split(",")]
main_horizon = st.selectbox(
    "랭킹에 쓸 보유기간 (봉 수)", horizon_list, index=len(horizon_list) // 2,
    help="아래 표와 차트를 어느 보유기간 기준으로 볼지 고릅니다.",
)

sub = summary[summary["horizon"] == main_horizon].sort_values("t_stat", ascending=False)
st.subheader(f"보유기간 {main_horizon}봉 기준 t값 랭킹")
st.caption(
    "|t값| >= 2면 눈여겨볼 만한 신호, 표본수가 10개 미만이면 아직 표본이 부족합니다. "
    "'통계적 유의(BH보정)'는 이번 실행에서 나온 전체 신호 x 방향 x 보유기간 조합을 "
    "한꺼번에 보정한 뒤에도 유의하다고 판정된 것만 True입니다 — 개별 t값만 보고 "
    "판단하면 우연히 유의하게 나온 신호를 걸러내지 못할 수 있습니다."
)
with st.expander("표에 나오는 용어가 낯설다면"):
    st.markdown(
        "- **표본수**: 이 신호가 과거 데이터에서 몇 번 발생했는지. 적을수록(특히 10개 "
        "미만) 결과를 신뢰하기 어렵습니다.\n"
        "- **평균수익률 / 승률**: 신호 발생 후 해당 보유기간 동안의 평균 수익률과, "
        "수익이 났던 비율.\n"
        "- **t값**: 그 평균수익률이 우연(0%)과 통계적으로 얼마나 다른지를 나타내는 "
        "지표. 절댓값이 클수록 '우연이 아닐 가능성'이 높습니다. 보통 2 이상이면 "
        "눈여겨볼 만하다고 봅니다.\n"
        "- **p값**: t값을 '우연히 이런 결과가 나올 확률'로 환산한 값. 작을수록 "
        "우연이 아닐 가능성이 높습니다.\n"
        "- **통계적 유의 (BH보정)**: 여러 신호를 동시에 비교하면 그중 일부는 순전히 "
        "우연으로 유의하게 보일 수 있습니다. 이를 감안해 보정한 뒤에도 유의하다고 "
        "판정됐는지 여부입니다."
    )

display_sub = sub[["signal", "direction", "n_events", "mean_return_pct", "win_rate_pct",
                    "t_stat", "p_value", "significant_bh"]].reset_index(drop=True).copy()
display_sub["direction"] = display_sub["direction"].map(DIRECTION_LABELS).fillna(display_sub["direction"])
display_sub["significant_bh"] = display_sub["significant_bh"].map({True: "예", False: "아니오"})
display_sub = display_sub.rename(columns=COLUMN_LABELS)
st.dataframe(display_sub, use_container_width=True)

st.subheader("신호 발생 후 평균 누적수익률 경로")
st.caption("신호가 발생한 시점을 0으로 두고, 그 뒤 각 보유기간까지 평균적으로 수익률이 "
           "어떻게 누적되는지를 보여줍니다. (단위: %)")
curve_df = pd.DataFrame({
    label: res.curve for label, res in results.items() if res.n_events > 0
})
if not curve_df.empty:
    st.line_chart(curve_df * 100)
else:
    st.caption("플롯할 이벤트가 없습니다.")

if log_executions and IS_OWNER:
    from backtest.execution_log import log_all_signals
    from pathlib import Path

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.execution_log_path or str(out_dir / "execution_log.csv")
    n_new = log_all_signals(
        dfs_by_indicator, symbol=args.symbol, horizon=main_horizon, path=log_path,
        entry_lag=args.entry_lag, entry_col=args.entry_col, cost_bps=args.cost_bps,
        slippage_bps=slippage_arg, drop_overlapping=args.drop_overlapping,
    )
    st.success(
        f"체결 로그 {log_path}에 {n_new}건 신규 기록 (보유기간={main_horizon}봉 기준). "
        f"청산 사유는 전부 '보유기간 만료'로 기록되며, 실제 손절/익절 로직이 적용된 건 "
        f"아닙니다."
    )
    if Path(log_path).exists():
        log_df = pd.read_csv(log_path)
        with st.expander(f"누적 체결 로그 미리보기 ({len(log_df)}행)"):
            st.dataframe(log_df.tail(50), use_container_width=True)
        st.download_button(
            "체결 로그 파일(execution_log.csv) 다운로드", data=Path(log_path).read_bytes(),
            file_name="execution_log.csv", mime="text/csv",
        )

st.caption(
    "주의: 이 화면은 과거 통계 기반 신호 요약이지 투자자문이 아닙니다. "
    "표본수가 작거나(특히 10개 미만) 통계적 유의성이 '아니오'면 결론으로 쓰기엔 이릅니다."
)
