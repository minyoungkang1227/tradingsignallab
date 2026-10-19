"""
live_check.py — MCP 서버 4종의 live 경로(실제 API)를 하나씩 점검한다.

실행:
    cd finagent
    python live_check.py                 # 환경변수에 있는 키만 점검
    python live_check.py --corp 00126380 --symbol 005930 --year 2024

환경변수(있는 것만 점검, 없으면 SKIP):
    DART_API_KEY  https://opendart.fss.or.kr  에서 발급
    ECOS_API_KEY  https://ecos.bok.or.kr      에서 발급
    FRED_API_KEY  https://fred.stlouisfed.org/docs/api/api_key.html
KRX(signal)는 키가 필요 없고 pykrx 또는 yfinance 패키지가 필요하다.

이 스크립트는 MCP를 거치지 않고 서버 구현 함수를 직접 부른다.
오케스트레이터 전체를 돌리기 전에 '어느 소스가 왜 실패하는지'만 빠르게 보기 위한 것이다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

os.environ["FINAGENT_MODE"] = "live"

import finagent_mcp_servers as S  # noqa: E402

OKMARK, NG, SKIP = "[OK]  ", "[FAIL]", "[SKIP]"


def show(title: str, raw: str) -> bool:
    out = json.loads(raw)
    ok = bool(out.get("ok"))
    print(f"{OKMARK if ok else NG} {title}  (mode={out.get('mode')}, as_of={out.get('as_of')})")
    if out.get("warnings"):
        for w in out["warnings"]:
            print(f"        ! {w}")
    if ok:
        d = out.get("data", {})
        brief = {k: v for k, v in d.items() if k != "items"}
        print(f"        {json.dumps(brief, ensure_ascii=False)[:300]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", default="00126380", help="DART corp_code (기본: 삼성전자)")
    ap.add_argument("--symbol", default="005930", help="KRX 6자리 종목코드")
    ap.add_argument("--year", type=int, default=2024)
    a = ap.parse_args()

    results = {}

    print("=== DART ===")
    if os.environ.get("DART_API_KEY"):
        results["dart.financials"] = show(
            f"dart_financials({a.corp}, {a.year})", S.dart_financials_impl(a.corp, a.year))
        results["dart.disclosures"] = show(
            f"dart_disclosures({a.corp})", S.dart_disclosures_impl(a.corp, 30))
    else:
        print(f"{SKIP} DART_API_KEY 없음")

    print("\n=== ECOS ===")
    if os.environ.get("ECOS_API_KEY"):
        for name in ("usdkrw", "base_rate"):
            results[f"ecos.{name}"] = show(f"ecos_series({name})", S.ecos_series_impl(name, 3))
    else:
        print(f"{SKIP} ECOS_API_KEY 없음")

    print("\n=== FRED ===")
    if os.environ.get("FRED_API_KEY"):
        for sid in ("VIXCLS", "T10Y2Y", "DGS10"):
            results[f"fred.{sid}"] = show(f"fred_series({sid})", S.fred_series_impl(sid))
    else:
        print(f"{SKIP} FRED_API_KEY 없음")

    print("\n=== Signal (KRX, 키 불필요) ===")
    results["signal"] = show(f"signal_evaluate({a.symbol})", S.signal_evaluate_impl(a.symbol))

    print("\n=== 요약 ===")
    if not results:
        print("점검한 항목이 없다. 키를 설정하고 다시 실행할 것.")
        return 1
    bad = [k for k, v in results.items() if not v]
    for k, v in results.items():
        print(f"  {'OK  ' if v else 'FAIL'}  {k}")
    if bad:
        print(f"\n실패: {', '.join(bad)}")
        print("위 '!' 줄의 메시지가 원인이다. 키·코드·네트워크 순으로 확인할 것.")
        return 1
    print("\n전부 통과. 이제 오케스트레이터를 live로 돌릴 수 있다:")
    print(f"  python finagent_orchestrator.py --symbol {a.symbol} --corp {a.corp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
