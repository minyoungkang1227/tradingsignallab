"""
CLI 옵션·기본값 대조표 생성기 — 문서(OVERVIEW.md)의 옵션 표를 실제 argparse에서 만든다.

왜 이렇게 하나: 옵션 이름·기본값을 문서에 손으로 적으면 코드가 바뀔 때마다 어긋난다(이미 PIPELINE.md가
그랬다). 표를 코드에서 생성하고, 문서에 붙은 표가 생성 결과와 다르면 스모크테스트(_smoke_test_docs_sync.py)가
실패하게 해서 '문서와 실제 옵션·기본값 일치'를 사람이 기억하지 않아도 유지한다.

사용: python tools/gen_option_table.py  → 표 본문을 출력. OVERVIEW.md의 마커 사이에 붙여 넣는다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

START, END = "<!-- OPTION_TABLE_START -->", "<!-- OPTION_TABLE_END -->"


def _default_text(a: argparse.Action) -> str:
    if isinstance(a, (argparse._StoreTrueAction,)):
        return "꺼짐"
    if isinstance(a, argparse._StoreFalseAction):
        return "켜짐"
    d = a.default
    if d is None:
        return "없음"
    if isinstance(d, float) and d == int(d) and abs(d) < 1e9:
        return f"{d:g}"
    return str(d)


def _first_sentence(h: str, limit: int = 80) -> str:
    h = " ".join((h or "").replace("%%", "%").split())
    m = re.search(r"(?<=[.다요음함임])\s", h)
    s = h[:m.start()] if m and m.start() <= limit * 2 else h
    s = s.rstrip(".")
    if len(s) > limit:
        s = s[:limit].rstrip() + "…"
    return s.replace("|", "\\|")


def rows_for(script: str, parser: argparse.ArgumentParser):
    rows = []
    for a in parser._actions:
        if not a.option_strings or a.dest == "help":
            continue
        opt = max(a.option_strings, key=len)
        rows.append((script, opt, _default_text(a), _first_sentence(a.help)))
    return rows


def build_table() -> str:
    import run_backtest
    import run_portfolio
    rows = rows_for("run_backtest.py", run_backtest.build_parser()) + \
           rows_for("run_portfolio.py", run_portfolio.build_parser())
    head = "| 스크립트 | 옵션 | 기본값 | 의미(도움말 첫 문장) |\n|---|---|---|---|"
    body = "\n".join(f"| `{s}` | `{o}` | `{d}` | {h} |" for s, o, d, h in rows)
    return head + "\n" + body


if __name__ == "__main__":
    print(build_table())
