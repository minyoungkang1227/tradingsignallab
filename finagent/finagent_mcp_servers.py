"""
finagent_mcp_servers.py — DART / ECOS / FRED / Signal MCP 서버 4종 (단일 파일).

실행:  python finagent_mcp_servers.py <dart|ecos|fred|signal>      (stdio MCP 서버)
환경변수:
  FINAGENT_MODE      fixture(기본: 키 없으면) | live
  FINAGENT_SCENARIO  baseline | event_risk | risk_off | bad_data   (fixture 모드 전용)
  DART_API_KEY / ECOS_API_KEY / FRED_API_KEY / KRX 관련 키 (live 모드)
  FINAGENT_CORE      trading_signal_lab 코어 경로 (폴더 또는 zip). 기본: 이 파일의 상위 폴더(리포 루트),
                     없으면 옆의 trading_signal_lab_core.zip 또는 ./core

설계 원칙 (강의: "LLM은 판단하지 않는다, 구조가 판단한다")
  - MCP 서버는 '데이터 수집 + 결정론적 계산'만 한다. 매수/매도 주문 도구는 어디에도 없다.
  - 모든 응답은 JSON: {source, mode, as_of, ok, data, table(markdown), warnings}.
  - fixture 모드 결과는 mode="fixture"로 표시된다 — 실제 시장 데이터가 아니다.
"""
from __future__ import annotations

import json
import os
import sys
import types
from datetime import date, timedelta
from pathlib import Path

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ModuleNotFoundError:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server

HERE = Path(__file__).resolve().parent
SCENARIO = os.environ.get("FINAGENT_SCENARIO", "baseline")


def _mode(key_env: str) -> str:
    m = os.environ.get("FINAGENT_MODE")
    if m in ("fixture", "live"):
        return m
    return "live" if os.environ.get(key_env) else "fixture"


def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "(데이터 없음)"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _resp(source, mode, ok=True, data=None, rows=None, warnings=None, as_of=None) -> str:
    return json.dumps(dict(
        source=source, mode=mode, ok=ok, as_of=as_of or str(date.today()),
        data=data if data is not None else {}, table=_md_table(rows or []),
        warnings=warnings or []), ensure_ascii=False)


def _need_key(env: str, source: str):
    """live 모드인데 키가 없으면 깔끔한 실패 응답을 돌려준다(KeyError 대신)."""
    if not os.environ.get(env):
        return _resp(source, "live", ok=False, warnings=[f"{env} 환경변수가 없습니다 (live 모드)"])
    return None


def _err(e: Exception) -> str:
    """예외 메시지에서 API 키를 가린다(경고 문구는 리포트에 그대로 실린다)."""
    t = f"{type(e).__name__}: {e}"
    for env in ("DART_API_KEY", "ECOS_API_KEY", "FRED_API_KEY", "KRX_API_KEY"):
        k = os.environ.get(env)
        if k and len(k) >= 4:
            t = t.replace(k, "***")
    return t


def _num(x, default=0.0) -> float:
    """DART 금액 문자열 파싱. 결측은 '-' 또는 '' 또는 None으로 온다."""
    t = str(x if x is not None else "").replace(",", "").strip()
    if t in ("", "-", "None"):
        return default
    try:
        return float(t)
    except ValueError:
        return default


# ════════════════════════════════════════════════════════════════════
# DART
# ════════════════════════════════════════════════════════════════════
DART_BASE = "https://opendart.fss.or.kr/api"

# 가상 기업(FIX001) — 실제 기업 수치가 아니다.
_DART_FIX = {
    "baseline": dict(revenue=[100.0, 112.0], op_income=[10.0, 13.5], net_income=[7.0, 9.8],
                     liabilities=[60.0, 62.0], equity=[100.0, 110.0], disclosures=[
                         dict(date="최근", title="분기보고서", risk="low")]),
    "event_risk": dict(revenue=[100.0, 98.0], op_income=[10.0, 4.0], net_income=[7.0, -2.0],
                       liabilities=[90.0, 160.0], equity=[100.0, 80.0], disclosures=[
                           dict(date="최근", title="유상증자 결정", risk="high"),
                           dict(date="최근", title="감사의견 한정 관련 공시", risk="high")]),
}


def _dart_live(path: str, **params):
    import requests
    params["crtfc_key"] = os.environ["DART_API_KEY"]
    r = requests.get(f"{DART_BASE}/{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def dart_financials_impl(corp_code: str, year: int, reprt_code: str = "11011") -> str:
    """핵심 재무(매출·영업이익·순이익·부채·자본)와 파생 비율."""
    mode = _mode("DART_API_KEY")
    if mode == "fixture":
        if SCENARIO == "bad_data":
            return _resp("dart", mode, ok=False, warnings=["status=013: 조회된 데이터가 없습니다 (fixture bad_data)"])
        f = _DART_FIX["event_risk" if SCENARIO == "event_risk" else "baseline"]
        rev, op, ni, li, eq = f["revenue"], f["op_income"], f["net_income"], f["liabilities"], f["equity"]
    else:
        if (bad := _need_key("DART_API_KEY", "dart")):
            return bad
        try:
            j = _dart_live("fnlttSinglAcntAll.json", corp_code=corp_code, bsns_year=str(year),
                           reprt_code=reprt_code, fs_div="CFS")
        except Exception as e:
            return _resp("dart", mode, ok=False, warnings=[f"DART 호출 실패: {_err(e)}"])
        if j.get("status") != "000":
            return _resp("dart", mode, ok=False, warnings=[f"DART status={j.get('status')}: {j.get('message')}"])
        acc = {}
        for it in j["list"]:
            nm = it["account_nm"].replace(" ", "")
            for k, tgt in (("매출액", "rev"), ("영업이익", "op"), ("당기순이익", "ni"),
                           ("부채총계", "li"), ("자본총계", "eq")):
                if nm == k and tgt not in acc:
                    acc[tgt] = (_num(it.get("frmtrm_amount")), _num(it.get("thstrm_amount")))
        miss = {"rev", "op", "ni", "li", "eq"} - set(acc)
        if miss:
            return _resp("dart", mode, ok=False, warnings=[f"계정 누락: {sorted(miss)}"])
        rev, op, ni, li, eq = (list(acc[k]) for k in ("rev", "op", "ni", "li", "eq"))
    data = dict(corp_code=corp_code, year=year,
                revenue_growth=round(rev[1] / rev[0] - 1, 4) if rev[0] else None,
                op_margin=round(op[1] / rev[1], 4) if rev[1] else None,
                net_income=ni[1], debt_ratio=round(li[1] / eq[1], 4) if eq[1] else None)
    rows = [dict(항목="매출 증가율", 값=data["revenue_growth"]), dict(항목="영업이익률", 값=data["op_margin"]),
            dict(항목="당기순이익", 값=data["net_income"]), dict(항목="부채비율", 값=data["debt_ratio"])]
    return _resp("dart", mode, data=data, rows=rows)


def dart_disclosures_impl(corp_code: str, days: int = 30) -> str:
    """최근 공시 목록 + 위험 키워드 태깅(결정론적 키워드 규칙)."""
    mode = _mode("DART_API_KEY")
    if mode == "fixture":
        if SCENARIO == "bad_data":
            return _resp("dart", mode, ok=False, warnings=["공시 조회 실패 (fixture bad_data)"])
        items = _DART_FIX["event_risk" if SCENARIO == "event_risk" else "baseline"]["disclosures"]
    else:
        if (bad := _need_key("DART_API_KEY", "dart")):
            return bad
        end = date.today()
        try:
            j = _dart_live("list.json", corp_code=corp_code, bgn_de=(end - timedelta(days=days)).strftime("%Y%m%d"),
                           end_de=end.strftime("%Y%m%d"), page_count=50)
        except Exception as e:
            return _resp("dart", mode, ok=False, warnings=[f"DART 호출 실패: {_err(e)}"])
        if j.get("status") not in ("000", "013"):
            return _resp("dart", mode, ok=False, warnings=[f"DART status={j.get('status')}: {j.get('message')}"])
        RISK = ("유상증자", "감자", "횡령", "배임", "상장폐지", "감사의견", "회생", "부도", "불성실")
        items = [dict(date=x["rcept_dt"], title=x["report_nm"],
                      risk="high" if any(k in x["report_nm"] for k in RISK) else "low")
                 for x in j.get("list", [])]
    n_high = sum(1 for i in items if i["risk"] == "high")
    return _resp("dart", mode, data=dict(corp_code=corp_code, n_disclosures=len(items), n_high_risk=n_high, items=items),
                 rows=items)


# ════════════════════════════════════════════════════════════════════
# ECOS  (통계표/항목 코드는 repo data/ecos.py STAT_CODES 기준 — 국고채 등은 미검증이라 넣지 않음)
# ════════════════════════════════════════════════════════════════════
ECOS_CODES = {"base_rate": ("722Y001", "0101000", "M"), "usdkrw": ("731Y001", "0000001", "D")}
_ECOS_FIX = {
    "baseline": dict(base_rate=[2.50, 2.50, 2.50], usdkrw=[1380, 1385, 1378]),
    "event_risk": dict(base_rate=[2.50, 2.50, 2.50], usdkrw=[1380, 1385, 1382]),
    "risk_off": dict(base_rate=[2.50, 2.50, 2.75], usdkrw=[1400, 1470, 1565]),
}


def ecos_series_impl(name: str, n: int = 3) -> str:
    if name not in ECOS_CODES:
        return _resp("ecos", "n/a", ok=False, warnings=[f"허용되지 않은 시리즈 '{name}' (허용: {sorted(ECOS_CODES)})"])
    mode = _mode("ECOS_API_KEY")
    if mode == "fixture":
        if SCENARIO == "bad_data":
            return _resp("ecos", mode, ok=False, warnings=["ECOS 응답 없음 (fixture bad_data)"])
        vals = _ECOS_FIX.get(SCENARIO, _ECOS_FIX["baseline"])[name][-n:]
    else:
        if (bad := _need_key("ECOS_API_KEY", "ecos")):
            return bad
        sys.path.insert(0, str(_core_path()))
        stat, item, cyc = ECOS_CODES[name]
        end = date.today()
        fmt, span = ("%Y%m%d", 60) if cyc == "D" else ("%Y%m", 400)
        try:
            from data.ecos import fetch_ecos_series
            s = fetch_ecos_series(stat, item, (end - timedelta(days=span)).strftime(fmt), end.strftime(fmt), cycle=cyc)
        except Exception as e:
            return _resp("ecos", mode, ok=False, warnings=[f"ECOS 호출 실패: {_err(e)}"])
        vals = [float(v) for v in s.tail(n)]
    if not vals:
        return _resp("ecos", mode, ok=False, warnings=[f"'{name}' 구간에 데이터가 없습니다"])
    chg = vals[-1] / vals[0] - 1 if vals[0] else 0.0
    return _resp("ecos", mode, data=dict(series=name, values=vals, last=vals[-1], change_pct=round(chg * 100, 2)),
                 rows=[dict(시리즈=name, 최근값=vals[-1], 기간변화율_pct=round(chg * 100, 2))])


# ════════════════════════════════════════════════════════════════════
# FRED
# ════════════════════════════════════════════════════════════════════
FRED_ALLOWED = {"FEDFUNDS", "DGS10", "DGS2", "T10Y2Y", "VIXCLS"}
_FRED_FIX = {
    "baseline": dict(FEDFUNDS=[4.33], DGS10=[4.10], DGS2=[3.80], T10Y2Y=[0.30], VIXCLS=[15.0]),
    "event_risk": dict(FEDFUNDS=[4.33], DGS10=[4.10], DGS2=[3.80], T10Y2Y=[0.30], VIXCLS=[16.0]),
    "risk_off": dict(FEDFUNDS=[4.33], DGS10=[4.00], DGS2=[4.40], T10Y2Y=[-0.40], VIXCLS=[32.0]),
}


def fred_series_impl(series_id: str) -> str:
    if series_id not in FRED_ALLOWED:
        return _resp("fred", "n/a", ok=False, warnings=[f"허용되지 않은 시리즈 '{series_id}' (허용: {sorted(FRED_ALLOWED)})"])
    mode = _mode("FRED_API_KEY")
    if mode == "fixture":
        if SCENARIO == "bad_data":
            return _resp("fred", mode, ok=False, warnings=["FRED 응답 없음 (fixture bad_data)"])
        last = _FRED_FIX.get(SCENARIO, _FRED_FIX["baseline"])[series_id][-1]
        d = str(date.today())
    else:
        if (bad := _need_key("FRED_API_KEY", "fred")):
            return bad
        import requests
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=15, params=dict(
                series_id=series_id, api_key=os.environ["FRED_API_KEY"], file_type="json",
                sort_order="desc", limit=10))
            r.raise_for_status()
            obs = [o for o in r.json().get("observations", []) if o["value"] != "."]  # 결측은 "."
        except Exception as e:
            return _resp("fred", mode, ok=False, warnings=[f"FRED 호출 실패: {_err(e)}"])
        if not obs:
            return _resp("fred", mode, ok=False, warnings=["유효 관측치 없음"])
        last, d = float(obs[0]["value"]), obs[0]["date"]
    return _resp("fred", mode, as_of=d, data=dict(series=series_id, last=last, date=d),
                 rows=[dict(시리즈=series_id, 값=last, 날짜=d)])


# ════════════════════════════════════════════════════════════════════
# Signal (trading_signal_lab 래퍼)
# ════════════════════════════════════════════════════════════════════
def _core_path() -> Path:
    env = os.environ.get("FINAGENT_CORE")
    for c in ([Path(env)] if env else []) + [HERE.parent, HERE / "trading_signal_lab_core.zip", HERE / "core"]:
        if c.exists():
            return c
    raise FileNotFoundError("trading_signal_lab 코어를 찾지 못했습니다 (FINAGENT_CORE 설정).")


def _signal_cfg():
    # repo _smoke_test_advisor.make_cfg 와 동일한 값
    return types.SimpleNamespace(
        id_vol_mode="multiple", id_vol_mult=2.0, id_z_threshold=1.96, big_sales_len=7, vp_bins=20, vp_freq="W",
        mom_filter=False, mom_lookback=126, mom_window=252, mom_threshold=0.7,
        liq_filter=False, liq_window=20, liq_percentile_window=252, liq_threshold=0.9,
        entry_lag=0, entry_col=None, cost_bps=0.0, slippage_bps=0.0,
        target_horizon=5, t_threshold=3.0, min_own_events=10)


def _fixture_ohlcv(name: str):
    """합성 OHLCV (실제 시세 아님). 고정 시드로 repo의 실제 evaluator가 결정론적 결과를 낸다."""
    import numpy as np
    import pandas as pd

    def ou(seed, n=1500, kappa=0.35, sigma=0.02):
        rng = np.random.RandomState(seed)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] * (1 - kappa) + rng.normal(0, sigma)
        trend = np.cumsum(rng.normal(0.0003, 0.002, n))
        close = np.exp(np.log(100) + trend + x)
        o = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
        h = np.maximum(o, close) * (1 + rng.uniform(0, 0.006, n))
        l = np.minimum(o, close) * (1 - rng.uniform(0, 0.006, n))
        return pd.DataFrame(dict(open=o, high=h, low=l, close=close, volume=rng.uniform(1e5, 3e5, n)),
                            index=pd.bdate_range("2020-01-01", periods=n))

    def episodes(seed, n=1500, rebound=1.0, n_ep=28, sigma=0.008):
        rng = np.random.RandomState(seed)
        r = rng.normal(0.0003, sigma, n)
        starts = np.sort(rng.choice(np.arange(60, n - 60), n_ep, replace=False))
        last_end = 0
        for s in starts:
            if s < last_end + 30:
                continue
            r[s:s + 8] += -0.022
            r[s + 8:s + 14] += rebound * 0.012
            last_end = s + 14
        s = n - 12
        r[s:s + 8] += -0.022
        r[s + 8] += 0.01
        close = 100 * np.exp(np.cumsum(r))
        o = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.001, n))
        h = np.maximum(o, close) * (1 + rng.uniform(0, 0.004, n))
        l = np.minimum(o, close) * (1 - rng.uniform(0, 0.004, n))
        return pd.DataFrame(dict(open=o, high=h, low=l, close=close, volume=rng.uniform(1e5, 3e5, n)),
                            index=pd.bdate_range("2020-01-01", periods=n))

    def mirror(df):  # 가격을 k/price 로 뒤집어 FIX_POS의 숏 신호를 롱 신호로 변환
        k = 10000.0
        m = df.copy()
        m["open"], m["close"], m["high"], m["low"] = k / df["open"], k / df["close"], k / df["low"], k / df["high"]
        return m

    return {"FIX_POS": lambda: episodes(2, rebound=1.0),   # 검증된 양(+)의 t 신호 (VP박스-POC저항, 매도)
            "FIX_LONG": lambda: mirror(episodes(2, rebound=1.0)),  # 검증된 롱 신호 (VP박스-POC지지, 매수, t≈+5.5)
            "FIX_NEG": lambda: ou(1, kappa=0.35),          # |t|만 크고 t<0 인 신호 (evaluator는 '검증'으로 판정)
            "FIX_NONE": lambda: episodes(0, rebound=1.0),  # 활성 신호 없음
            }[name]()


FIXTURE_SYMBOLS = ("FIX_POS", "FIX_LONG", "FIX_NEG", "FIX_NONE")


def signal_evaluate_impl(symbol: str) -> str:
    """repo advisor.evaluator.evaluate_symbol 결과를 그대로 반환한다(수정 없음)."""
    core = _core_path()
    sys.path.insert(0, str(core))
    from advisor.evaluator import evaluate_symbol
    mode = "fixture" if symbol in FIXTURE_SYMBOLS else "live"
    if SCENARIO == "bad_data" and mode == "fixture":
        return _resp("signal", mode, ok=False, warnings=["OHLCV 로드 실패 (fixture bad_data)"])
    if mode == "fixture":
        df = _fixture_ohlcv(symbol)
    else:
        end = date.today()
        try:
            from data.krx import fetch_krx_ohlcv
            df = fetch_krx_ohlcv(symbol, (end - timedelta(days=3 * 365)).strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        except Exception as e:
            return _resp("signal", mode, ok=False, warnings=[f"OHLCV 수집 실패({symbol}): {_err(e)}"])
        if df is None or len(df) == 0:
            return _resp("signal", mode, ok=False, warnings=[f"OHLCV 데이터가 비어 있습니다({symbol})"])
    try:
        v = evaluate_symbol(df, symbol, _signal_cfg())
    except Exception as e:
        return _resp("signal", mode, ok=False, warnings=[f"evaluate_symbol 실패({symbol}): {_err(e)}"])
    sigs = [dict(label=s.label, direction=s.direction, evaluator_validated=bool(s.validated), basis=s.basis,
                 n_events=int(s.n_events), mean_return=None if s.mean_return != s.mean_return else round(s.mean_return, 5),
                 t_stat=None if s.t_stat != s.t_stat else round(float(s.t_stat), 3)) for s in v.active_signals]
    return _resp("signal", mode, as_of=v.as_of, data=dict(symbol=symbol, verdict=v.verdict,
                 strength=round(v.strength, 3), active_signals=sigs, t_threshold=3.0),
                 rows=[dict(신호=s["label"], 방향=s["direction"], evaluator검증=s["evaluator_validated"],
                            n=s["n_events"], 평균수익=s["mean_return"], t=s["t_stat"]) for s in sigs])


# ════════════════════════════════════════════════════════════════════
# MCP 서버 등록 — 서버마다 필요한 도구만 노출(인터페이스 분리). 주문/체결 도구는 없다.
# ════════════════════════════════════════════════════════════════════
def build(kind: str):
    srv = _Server(f"finagent-{kind}")
    if kind == "dart":
        srv.tool()(dart_financials_impl)
        srv.tool()(dart_disclosures_impl)
    elif kind == "ecos":
        srv.tool()(ecos_series_impl)
    elif kind == "fred":
        srv.tool()(fred_series_impl)
    elif kind == "signal":
        srv.tool()(signal_evaluate_impl)
    else:
        raise SystemExit("usage: finagent_mcp_servers.py <dart|ecos|fred|signal>")
    return srv


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "").run()
