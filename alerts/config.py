"""
run_backtest.py의 argparse 기본값과 1:1로 맞춘 DEFAULTS/build_args.

app.py의 같은 이름 함수/딕셔너리와 패턴이 완전히 동일하다 — 의도적인 독립 복제다
(app.py는 streamlit 런타임 밖에서 import하면 st.set_page_config() 등이 즉시
실행돼 에러가 나므로, GitHub Actions처럼 streamlit 없이 도는 스크립트에서는
app.py를 그대로 import할 수 없다). run_backtest.py의 CLI에 새 --옵션이 추가되면
app.py의 DEFAULTS와 이 파일의 DEFAULTS 둘 다 같이 고쳐야 한다 — 이 파일 docstring과
app.py docstring 양쪽에 동일하게 남겨둔 유지보수 포인트.

watchlist.json 형식(리스트, 각 항목 하나가 감시할 종목 하나):
  krx  : {"market": "krx", "symbol": "005930", "label": "삼성전자",
          "start": "20230101" (생략 가능, 기본 2년 전), "end": 생략(기본 오늘)}
  upbit: {"market": "upbit", "symbol": "KRW-BTC", "label": "비트코인",
          "unit": "day" (생략 가능), "count": 1000 (생략 가능)}
"""
from __future__ import annotations

import json
import types
from datetime import datetime, timedelta
from pathlib import Path

DEFAULTS = dict(
    unit="day", minute_unit=60, count=1000, start=None, end=None,
    horizons="1,3,5,10,20",
    id_vol_mode="multiple", id_vol_mult=2.0, id_z_threshold=1.96,
    mom_filter=False, mom_lookback=126, mom_window=252, mom_threshold=0.7,
    entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
    newey_west=False,
    # [동작 변경] 알림 경로에서만 True로 바꿨다(CLI/대시보드 기본값은 그대로 False).
    # 보유기간이 겹치는 이벤트를 그대로 두면 서로 거의 같은 구간을 중복해서 세는
    # 셈이라 실질 표본 수가 n보다 적은데도 n으로 계산되고, 그만큼 t값이 부풀려진다
    # ("t값이 비정상적으로 크게 나오는" 전형적 원인 중 하나). 아래 deflate=True가
    # 쓰는 DSR은 관측치 독립을 가정하므로 이 보정 없이는 판정 자체가 낙관적으로
    # 치우친다.
    drop_overlapping=True,
    liq_filter=False, liq_window=20, liq_percentile_window=252, liq_threshold=0.9,
    dynamic_slippage=False, max_extra_slippage_bps=50.0,
    fdr=0.10, factor_csv=None, factor_cols=None, t_threshold=3.0,
    big_sales_len=7, vp_bins=20, vp_freq="W",
    combo_body_min_pct=55.0, combo_wick_max_pct=25.0, combo_vol_len=20,
    combo_vol_mult=1.6, combo_ma_len=50, combo_ma_type="EMA", combo_score_threshold=0.62,
    log_executions=False, execution_log_path=None, output="output",
    # --- 디플레이티드 샤프비율(backtest.deflated) ---
    # 알림 경로에서는 기본으로 켠다. 알림은 "유의하다"는 판정 하나로 곧장
    # 텔레그램을 쏘기 때문에, 눈으로 확인하며 보는 대시보드보다 거짓양성에
    # 훨씬 민감하다 — 틀린 알림이 반복되면 알림 자체를 무시하게 되고 그러면
    # 시스템이 있으나 마나다.
    #
    # 합성 랜덤워크(진짜 신호 0건) 30회 검증 기준:
    #   BH보정만  -> 1,800건 중 93건 통과 (거짓양성률 5.17%)
    #   BH + DSR  -> 1,800건 중  1건 통과 (거짓양성률 0.06%)
    # 대신 진짜 신호 검출력은 79% -> 50%로 떨어진다. 알림에서는 "놓친 신호"보다
    # "틀린 알림"의 비용이 크다고 보고 이 트레이드오프를 택했다.
    # (전부 합성 데이터 기준이며 실제 시장 성과가 아님.)
    deflate=True, dsr_threshold=0.95, dsr_min_events=20, dsr_extra_trials=0,
    # --- 트리플 배리어(backtest.triple_barrier) ---
    # 알림에 "이 신호를 손절 걸고 매매했으면 과거에 실제로 얼마였나"를 같이
    # 실어주기 위해 켠다. 게이트로는 쓰지 않는다 — 이미 DSR이 검출력을
    # 79%->50%로 깎았는데 게이트를 하나 더 직렬로 걸면 알림이 거의 안 뜨고,
    # 배리어 결과는 손절폭(sl_mult) 선택에 민감해서 판정 기준으로 쓰기엔
    # 불안정하다. "신호가 진짜인가"(DSR)와 "손절 걸고 벌 수 있나"(배리어)는
    # 다른 질문이므로, 앞은 게이트로 뒤는 정보로 쓴다.
    # require_barrier_profit=True로 두면 배리어 평균수익이 양수인 신호만
    # 알리도록 게이트를 추가할 수 있다(기본 꺼짐).
    barrier=True, barrier_pt_mult=2.0, barrier_sl_mult=1.0,
    barrier_vol_span=20, barrier_intrabar=True,
    # 순차 거래(앞 거래 청산 후에만 진입)로 겹침을 없애고, 경제적 유의성(R 기대값·손익분기
    # 승률·비용 배수)을 계산해 알림에 정보로 싣는다. 둘 다 알림 게이트가 아니다.
    barrier_sequential=True, barrier_economics=True,
    require_barrier_profit=False,
)

# krx 종목인데 watchlist.json에 start/end를 안 적었을 때 기본으로 쓸 조회 기간.
# 콤보필터 ma_len=50, vol_len=20 등 롤링 계산에 여유 있게 쌓이도록 2년으로 잡는다.
_DEFAULT_LOOKBACK_DAYS = 730


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def build_args(**overrides) -> types.SimpleNamespace:
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def args_for_watch_item(item: dict) -> types.SimpleNamespace:
    """watchlist.json의 항목 하나(dict)를 run_backtest.load_data/run에 넘길
    args로 변환한다. market별로 필요한 필드만 채우고 나머지는 DEFAULTS를 쓴다."""
    market = item["market"]
    symbol = item["symbol"]

    if market == "krx":
        end = item.get("end") or _today_str()
        if item.get("start"):
            start = item["start"]
        else:
            start_dt = datetime.strptime(end, "%Y%m%d") - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
            start = start_dt.strftime("%Y%m%d")
        return build_args(market="krx", symbol=symbol, start=start, end=end)

    if market == "upbit":
        return build_args(
            market="upbit", symbol=symbol,
            unit=item.get("unit", DEFAULTS["unit"]),
            minute_unit=item.get("minute_unit", DEFAULTS["minute_unit"]),
            count=item.get("count", DEFAULTS["count"]),
        )

    raise ValueError(f"알 수 없는 market: {market!r} (watchlist 항목: {item})")


def load_watchlist(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"감시 리스트 파일이 없습니다: {path} — alerts/watchlist.json을 먼저 만드세요."
        )
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        if "market" not in item or "symbol" not in item:
            raise ValueError(f"watchlist 항목에 market/symbol이 빠졌습니다: {item}")
    return items
