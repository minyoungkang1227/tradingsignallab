"""
검증 프로필(프리셋) — 실전에 가까운 실행 조건을 옵션 한 개로 켠다.

[왜 필요한가]
기본 실행은 '신호 당일 종가에 진입, 비용 0'이라 look-ahead와 비용 무시로 결과가 낙관적이다.
실전 조건(다음 봉 시가 진입, 거래비용, 슬리피지, 겹치는 이벤트 제거, 다중검정 보정)을 매번
옵션 6~8개로 직접 맞추면 실행마다 조건이 달라져 결과를 서로 비교할 수 없다. 한 이름으로 묶으면
같은 조건에서 OOS(02)와 포트폴리오 시뮬레이션(03)이 돈다.

[원칙]
- 기본은 프로필 없음 = 기존 동작 그대로. --profile을 줄 때만 바뀐다.
- 사용자가 직접 준 옵션이 프리셋보다 우선한다(프리셋은 '기본값'을 대신 채울 뿐).
- 바뀐 항목은 호출자가 목록으로 받아 출력한다(어떤 조건으로 돌았는지 숨기지 않는다).

[비용 수치의 출처 — 확인 필요]
cost_bps는 이 저장소 run_backtest.py 도움말의 참고값(업비트 편도 0.05%x2=10bp, KRX 편도 약 0.18%x2=36bp)을
그대로 썼다. KRX 거래세·수수료율은 해마다 바뀌므로 실행 전에 현재 값을 직접 확인해 --cost-bps로 덮어써라.
슬리피지 10bp는 유동성 충분한 종목 기준 흔한 근사치(--slippage-bps 도움말과 동일)이며, 소형주는 부족할 수 있다.
"""
from __future__ import annotations

PROFILES = {
    "realistic": {
        "desc": "신호 확정 다음 봉 시가 진입 + 시장별 왕복비용 + 슬리피지 + 겹침 제거 + 다중검정 보정(DSR) + "
                "손절 반영 순차 거래",
        "fixed": {                       # 시장과 무관한 값
            "entry_lag": 1, "entry_col": "open", "slippage_bps": 10.0,
            "newey_west": True, "drop_overlapping": True,
            "deflate": True, "barrier": True, "barrier_sequential": True, "barrier_economics": True,
        },
        "cost_bps_by_market": {"krx": 36.0, "upbit": 10.0},
    },
}


def explicit_dests(parser, argv) -> set:
    """argv에 실제로 적힌 옵션의 dest 집합(argparse 기본값과 사용자가 같은 값을 직접 준 경우를 구분)."""
    out = set()
    for a in parser._actions:
        for opt in a.option_strings:
            if any(tok == opt or tok.startswith(opt + "=") for tok in argv):
                out.add(a.dest)
    return out


def apply_profile(args, name, explicit=frozenset()):
    """
    args에 프리셋을 적용하고 [(dest, 이전값, 새값)] 변경 목록을 돌려준다.
    name이 None이면 아무것도 하지 않는다(기존 동작). 사용자가 직접 준 옵션(explicit)은 건드리지 않는다.
    """
    if not name:
        return []
    if name not in PROFILES:
        raise ValueError(f"알 수 없는 프로필: {name!r} (가능: {sorted(PROFILES)})")
    prof = PROFILES[name]
    values = dict(prof["fixed"])
    market = getattr(args, "market", None)
    cb = prof["cost_bps_by_market"]
    if market not in cb:
        raise ValueError(f"프로필 {name!r}에 시장 {market!r}의 비용 값이 없다(가능: {sorted(cb)})")
    values["cost_bps"] = cb[market]
    changes = []
    for dest, new in values.items():
        if dest in explicit:
            continue
        old = getattr(args, dest, None)
        setattr(args, dest, new)
        if old != new:
            changes.append((dest, old, new))
    return changes
