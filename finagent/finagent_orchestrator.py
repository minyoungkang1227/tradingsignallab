"""
finagent_orchestrator.py — Multi-Agent 오케스트레이터 + Harness (DART/ECOS/FRED/Signal MCP 클라이언트).

흐름:  Macro Agent ─┐
       Fundamental ─┼→ Risk Harness(G1~G7) → Portfolio Agent → Explanation Agent → PENDING_HUMAN_APPROVAL
       Signal Agent ┘

"LLM은 판단하지 않는다, 구조가 판단한다": 모든 에이전트는 결정론적 규칙이며, LLM은 설명 단계에서만
(선택적으로) 쓸 수 있다. 이 파일은 LLM 호출 없이 동작한다. 주문 실행 도구는 없다(G7).

사용:  python finagent_orchestrator.py --symbol FIX_LONG --corp FIX001 --scenario risk_off
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
SERVER_FILE = HERE / "finagent_mcp_servers.py"

# ── Harness 정책 (모두 '가정'값이다 — 실거래 전 본인 기준으로 재설정할 것) ──
POLICY = dict(
    t_threshold=3.0,          # repo의 Harvey-Liu-Zhu 기준과 동일
    vix_risk_off=25.0, curve_inverted=0.0, usdkrw_spike_pct=5.0,   # 매크로 risk-off 조건
    max_debt_ratio=1.5,       # 부채총계/자본총계
    max_position=0.05,        # 단일 종목 비중 상한(가정)
    vix_haircut=20.0,         # VIX가 이 이상이면 비중 상한 절반
)

# G1: 에이전트별 도구 허용목록 (인터페이스 분리 — 필요한 도구만 노출)
ALLOWED_TOOLS = {
    "macro": {"ecos": {"ecos_series_impl"}, "fred": {"fred_series_impl"}},
    "fundamental": {"dart": {"dart_financials_impl", "dart_disclosures_impl"}},
    "signal": {"signal": {"signal_evaluate_impl"}},
}


@dataclass
class Gate:
    id: str
    name: str
    passed: bool
    detail: str
    effect: str = ""   # 비어있지 않으면 최종 행동에 영향을 준 것


@dataclass
class Trace:
    calls: list = field(default_factory=list)   # (agent, server, tool, ok)
    gates: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class MCPHub:
    """4개 MCP 서버를 stdio 서브프로세스로 띄우고, 에이전트별 허용목록(G1)을 강제한다."""

    def __init__(self, scenario: str, trace: Trace):
        self.scenario, self.trace, self.sessions, self._stack = scenario, trace, {}, None

    async def __aenter__(self):
        from contextlib import AsyncExitStack
        self._stack = AsyncExitStack()
        env = {**os.environ, "FINAGENT_SCENARIO": self.scenario}
        for kind in ("dart", "ecos", "fred", "signal"):
            params = StdioServerParameters(command=sys.executable, args=[str(SERVER_FILE), kind], env=env)
            # Jupyter/Colab의 sys.stderr는 fileno()가 없어 서브프로세스 stderr로 못 쓴다 → 실제 파일로 대체
            errlog = self._stack.enter_context(open(os.devnull, "w"))
            r, w = await self._stack.enter_async_context(stdio_client(params, errlog=errlog))
            s = await self._stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            self.sessions[kind] = s
        return self

    async def __aexit__(self, *a):
        await self._stack.aclose()

    async def list_tools(self):
        return {k: [t.name for t in (await s.list_tools()).tools] for k, s in self.sessions.items()}

    async def call(self, agent: str, server: str, tool: str, **args) -> dict:
        if tool not in ALLOWED_TOOLS.get(agent, {}).get(server, set()):
            self.trace.calls.append((agent, server, tool, "BLOCKED(G1)"))
            raise PermissionError(f"G1: {agent} 에이전트는 {server}.{tool} 호출 권한이 없습니다")
        res = await self.sessions[server].call_tool(tool, args)
        text = "".join(c.text for c in res.content if getattr(c, "text", None))
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            out = dict(source=server, mode="?", ok=False, data={}, warnings=[f"비JSON 응답: {text[:120]}"])
        self.trace.calls.append((agent, server, tool, "ok" if out.get("ok") else "FAIL"))
        return out


# ── Agents ───────────────────────────────────────────────────────────
async def macro_agent(hub: MCPHub) -> dict:
    ecos_fx = await hub.call("macro", "ecos", "ecos_series_impl", name="usdkrw")
    ecos_rate = await hub.call("macro", "ecos", "ecos_series_impl", name="base_rate")
    vix = await hub.call("macro", "fred", "fred_series_impl", series_id="VIXCLS")
    curve = await hub.call("macro", "fred", "fred_series_impl", series_id="T10Y2Y")
    outs = dict(usdkrw=ecos_fx, base_rate=ecos_rate, vix=vix, curve=curve)
    complete = all(o["ok"] for o in outs.values())
    reasons = []
    if complete:
        if vix["data"]["last"] >= POLICY["vix_risk_off"]:
            reasons.append(f"VIX {vix['data']['last']:.1f} ≥ {POLICY['vix_risk_off']}")
        if curve["data"]["last"] < POLICY["curve_inverted"]:
            reasons.append(f"10Y-2Y 스프레드 {curve['data']['last']:+.2f} (역전)")
        if ecos_fx["data"]["change_pct"] >= POLICY["usdkrw_spike_pct"]:
            reasons.append(f"원/달러 {ecos_fx['data']['change_pct']:+.1f}% 급등")
    return dict(complete=complete, regime="risk_off" if reasons else "neutral", reasons=reasons,
                vix=vix["data"].get("last"), modes={k: o["mode"] for k, o in outs.items()},
                warnings=[w for o in outs.values() for w in o["warnings"]])


async def fundamental_agent(hub: MCPHub, corp: str) -> dict:
    fin = await hub.call("fundamental", "dart", "dart_financials_impl", corp_code=corp, year=2025)
    dis = await hub.call("fundamental", "dart", "dart_disclosures_impl", corp_code=corp)
    complete = fin["ok"] and dis["ok"]
    flags = []
    if complete:
        d, e = fin["data"], dis["data"]
        if e["n_high_risk"] > 0:
            flags.append("고위험 공시: " + ", ".join(i["title"] for i in e["items"] if i["risk"] == "high"))
        if d["debt_ratio"] is not None and d["debt_ratio"] > POLICY["max_debt_ratio"]:
            flags.append(f"부채비율 {d['debt_ratio']:.2f} > {POLICY['max_debt_ratio']}")
        if d["net_income"] is not None and d["net_income"] < 0:
            flags.append("당기순손실")
    return dict(complete=complete, event_risk=bool(flags), flags=flags,
                modes=dict(fin=fin["mode"], dis=dis["mode"]), financials=fin["data"],
                warnings=fin["warnings"] + dis["warnings"])


async def signal_agent(hub: MCPHub, symbol: str) -> dict:
    out = await hub.call("signal", "signal", "signal_evaluate_impl", symbol=symbol)
    return dict(complete=out["ok"], mode=out["mode"], raw=out["data"], warnings=out["warnings"])


# ── Risk Harness ─────────────────────────────────────────────────────
def risk_harness(macro: dict, fund: dict, sig: dict, trace: Trace) -> dict:
    G = trace.gates
    action = None  # 최종 행동

    # G2 데이터 완전성 — fail-closed
    missing = [n for n, a in (("macro", macro), ("fundamental", fund), ("signal", sig)) if not a["complete"]]
    G.append(Gate("G2", "데이터 완전성", not missing,
                  "모든 소스 수신" if not missing else f"누락: {missing}",
                  "HOLD(데이터 불완전 → 판단 보류)" if missing else ""))
    if missing:
        action = "보유"

    # G3 데이터 출처 표시 (차단 아님)
    modes = {**macro.get("modes", {}), **{f"dart_{k}": v for k, v in fund.get("modes", {}).items()}, "signal": sig.get("mode")}
    fixture = [k for k, v in modes.items() if v == "fixture"]
    G.append(Gate("G3", "데이터 출처", True, "FIXTURE(가상 데이터) 사용: " + ", ".join(fixture) if fixture else "live 데이터"))

    # G5 신호 부호 재검증 — repo evaluator는 abs(t)만 보므로 t<0(손실 신호)도 '검증'으로 찍는다
    true_long = true_short = 0
    demoted = []
    if sig["complete"]:
        for s in sig["raw"]["active_signals"]:
            ok = s["evaluator_validated"] and s["t_stat"] is not None and s["t_stat"] >= POLICY["t_threshold"]
            if s["evaluator_validated"] and not ok:
                demoted.append(f"{s['label']}(t={s['t_stat']:+.2f})")
            if ok:
                true_long += s["direction"] == 1
                true_short += s["direction"] == -1
        raw_v = sig["raw"]["verdict"]
        sig_action = "매수" if true_long > true_short else "매도" if true_short > true_long else "보유"
        G.append(Gate("G5", "신호 부호 재검증(t≥+임계)", not demoted,
                      f"evaluator 판정={raw_v}, 재검증 판정={sig_action}" +
                      (f" / 방향조정 t<0 이라 무효화: {', '.join(demoted)}" if demoted else ""),
                      f"evaluator의 '{raw_v}'를 '{sig_action}'로 정정" if raw_v != sig_action else ""))
        if action is None:
            action = sig_action

    # G4 이벤트 리스크 — 신규 매수 차단
    if fund["complete"]:
        blocked = fund["event_risk"] and action == "매수"
        G.append(Gate("G4", "이벤트 리스크(공시·재무)", not fund["event_risk"],
                      "; ".join(fund["flags"]) if fund["flags"] else "이상 없음",
                      "신규 매수 차단 → 보유" if blocked else ""))
        if blocked:
            action = "보유"

    # G6 매크로 risk-off — 신규 매수 차단 (매도/축소는 허용)
    if macro["complete"]:
        blocked = macro["regime"] == "risk_off" and action == "매수"
        G.append(Gate("G6", "매크로 레짐", macro["regime"] != "risk_off",
                      "; ".join(macro["reasons"]) if macro["reasons"] else "neutral",
                      "신규 매수 차단 → 보유" if blocked else ""))
        if blocked:
            action = "보유"

    # G7 사람 승인 — 항상 마지막. 주문 실행 도구가 존재하지 않는다.
    G.append(Gate("G7", "Human-in-the-loop", True, "주문 도구 없음. 결과는 '제안'이며 사람 승인 대기", "PENDING_HUMAN_APPROVAL"))
    return dict(action=action or "보유")


def portfolio_agent(action: str, macro: dict) -> dict:
    cap = POLICY["max_position"]
    if macro.get("vix") and macro["vix"] >= POLICY["vix_haircut"]:
        cap /= 2
    if action == "매수":
        return dict(proposal="신규/추가 매수 제안", max_weight=cap)
    if action == "매도":
        return dict(proposal="비중 축소/청산 제안", max_weight=0.0)
    return dict(proposal="현 비중 유지(신규 거래 없음)", max_weight=None)


def explain(symbol, scenario, macro, fund, sig, final, port, trace) -> str:
    L = [f"## {symbol} · 시나리오 `{scenario}` → **{final['action']}** (상태: PENDING_HUMAN_APPROVAL)", "",
         f"제안: {port['proposal']}" + (f" (비중 상한 {port['max_weight']*100:.1f}%, 가정값)" if port["max_weight"] else ""), "",
         "| Gate | 항목 | 통과 | 근거 | 행동 영향 |", "|---|---|---|---|---|"]
    for g in trace.gates:
        L.append(f"| {g.id} | {g.name} | {'✅' if g.passed else '⛔'} | {g.detail} | {g.effect or '-'} |")
    L += ["", "호출 로그: " + ", ".join(f"{a}→{s}.{t}[{r}]" for a, s, t, r in trace.calls)]
    return "\n".join(L)


async def run(symbol: str, corp: str, scenario: str) -> dict:
    trace = Trace()
    async with MCPHub(scenario, trace) as hub:
        macro, fund, sig = await asyncio.gather(macro_agent(hub), fundamental_agent(hub, corp), signal_agent(hub, symbol))
        # G1 시연: 매크로 에이전트가 권한 밖(signal) 도구를 부르면 차단된다
        try:
            await hub.call("macro", "signal", "signal_evaluate_impl", symbol=symbol)
        except PermissionError as e:
            trace.gates.append(Gate("G1", "도구 허용목록", True, f"권한 밖 호출 차단 확인: {e}", ""))
    final = risk_harness(macro, fund, sig, trace)
    port = portfolio_agent(final["action"], macro)
    trace.gates.sort(key=lambda g: g.id)
    report = explain(symbol, scenario, macro, fund, sig, final, port, trace)
    return dict(symbol=symbol, scenario=scenario, action=final["action"], status="PENDING_HUMAN_APPROVAL",
                portfolio=port, macro=macro, fundamental=fund, signal=sig,
                gates=[g.__dict__ for g in trace.gates], calls=trace.calls, report=report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="FIX_LONG")
    ap.add_argument("--corp", default="FIX001")
    ap.add_argument("--scenario", default="baseline", choices=["baseline", "event_risk", "risk_off", "bad_data"])
    a = ap.parse_args()
    out = asyncio.run(run(a.symbol, a.corp, a.scenario))
    print(out["report"])


if __name__ == "__main__":
    main()
