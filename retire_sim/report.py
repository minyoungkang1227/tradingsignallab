"""상담 리포트: 고객에게 남길 한두 장짜리 요약을 HTML로 만든다.

왜 PDF가 아니라 HTML인가
  한글 PDF는 글꼴 파일을 함께 배포해야 하고(용량·라이선스), 배포 환경마다 글꼴이 달라 깨지기 쉽다.
  HTML은 어디서든 같게 보이고, 브라우저의 '인쇄 → PDF로 저장'으로 PDF가 된다.
  표지·여백·쪽 나눔은 @page 와 @media print 로 맞춰 두었다.

그림은 외부 라이브러리 없이 SVG 문자열로 직접 그린다(의존성 0, 인쇄 시 깨지지 않음).
"""
from __future__ import annotations

import html
from datetime import date

TEAL, CORAL, INK, MUTED, LINE = "#1F6F78", "#E2725B", "#1B2B2E", "#5A6B6E", "#CBD7DB"


def _esc(x) -> str:
    return html.escape(str(x))


def line_chart_svg(rows, w=640, h=240, pad=46, top=24) -> str:
    """rows: [(나이, 운이 좋을 때, 보통, 운이 나쁠 때)] — 억 원. 축과 0선을 가진 선 그래프."""
    if not rows:
        return ""
    xs = [r[0] for r in rows]
    ys = [v for r in rows for v in r[1:]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(0.0, min(ys)), max(ys)
    if y1 <= y0:
        y1 = y0 + 1
    sx = lambda x: pad + (x - x0) / max(x1 - x0, 1) * (w - pad - 14)
    sy = lambda y: h - pad - (y - y0) / (y1 - y0) * (h - pad - top)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="나이별 남는 돈">']
    # 가로 눈금
    steps = 4
    for i in range(steps + 1):
        v = y0 + (y1 - y0) * i / steps
        y = sy(v)
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w - 14}" y2="{y:.1f}" stroke="{LINE}" stroke-width="1"/>')
        out.append(f'<text x="{pad - 6}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{v:.0f}</text>')
    for lab, idx, color, dash in (("운이 좋을 때", 1, "#7FB8BE", "4 3"), ("보통일 때", 2, TEAL, ""), ("운이 나쁠 때", 3, CORAL, "")):
        pts = " ".join(f"{sx(r[0]):.1f},{sy(r[idx]):.1f}" for r in rows)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"{d}/>')
    # 나이 축
    for x in (x0, (x0 + x1) // 2, x1):
        out.append(f'<text x="{sx(x):.1f}" y="{h - 14}" font-size="11" fill="{MUTED}" text-anchor="middle">{x}세</text>')
    out.append(f'<text x="{pad - 34}" y="14" font-size="11" fill="{MUTED}">억 원</text>')
    out.append("</svg>")
    return "".join(out)


def waterfall_svg(items, w=640, h=250, pad=46) -> str:
    """은퇴 후 평균 1년, 연금이 들어와 생활비·세금으로 빠지고 얼마가 모자라는지 보여주는 폭포 그래프.

    items: [(항목, 연 금액)] — 첫 항목은 들어오는 돈(+), 나머지는 나가는 돈(−).
    마지막에 '모아둔 돈에서' 막대를 더해 부족분이 어디서 메워지는지 눈으로 잇는다."""
    items = [(k, float(v)) for k, v in items if abs(v) > 1e-9]
    if not items:
        return ""
    net = sum(v for _, v in items)
    bars, run = [], 0.0
    for k, v in items:
        bars.append((k, run, run + v, v))                 # (이름, 시작, 끝, 증감)
        run += v
    if net < 0:
        bars.append(("모아둔 돈에서", run, 0.0, -net))     # 부족분을 메우는 막대
    lo = min(min(b[1], b[2]) for b in bars); hi = max(max(b[1], b[2]) for b in bars)
    lo = min(lo, 0.0); hi = max(hi, 0.0)
    if hi <= lo:
        hi = lo + 1
    top, bot = 26, h - 46
    sy = lambda y: bot - (y - lo) / (hi - lo) * (bot - top)
    bw = (w - pad - 14) / len(bars)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="은퇴 후 1년 돈의 흐름">']
    out.append(f'<line x1="{pad}" y1="{sy(0):.1f}" x2="{w - 14}" y2="{sy(0):.1f}" stroke="{LINE}" stroke-width="1"/>')
    for i, (k, y0, y1, v) in enumerate(bars):
        x = pad + i * bw + bw * 0.2
        bwid = bw * 0.6
        ytop, ybot = sy(max(y0, y1)), sy(min(y0, y1))
        color = TEAL if v > 0 else CORAL
        if k == "모아둔 돈에서":
            color = "#8A6FB0"
        out.append(f'<rect x="{x:.1f}" y="{ytop:.1f}" width="{bwid:.1f}" height="{max(ybot - ytop, 1.5):.1f}" fill="{color}" rx="3"/>')
        out.append(f'<text x="{x + bwid / 2:.1f}" y="{ytop - 6:.1f}" font-size="11" fill="{INK}" text-anchor="middle">{v:+,.0f}</text>')
        out.append(f'<text x="{x + bwid / 2:.1f}" y="{bot + 17:.1f}" font-size="11" fill="{MUTED}" text-anchor="middle">{_esc(k)}</text>')
        if i + 1 < len(bars):                              # 막대를 잇는 점선
            xr = x + bwid; y = sy(y1)
            out.append(f'<line x1="{xr:.1f}" y1="{y:.1f}" x2="{x + bw:.1f}" y2="{y:.1f}" '
                       f'stroke="{LINE}" stroke-width="1" stroke-dasharray="3 2"/>')
    out.append(f'<text x="{pad}" y="{h - 8}" font-size="11" fill="{MUTED}">단위: 만 원 / 년 (오늘 돈 가치, 은퇴 후 평균)</text>')
    out.append("</svg>")
    return "".join(out)


CSS = f"""
@page {{ size: A4; margin: 14mm 13mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: Pretendard, -apple-system, 'Malgun Gothic', sans-serif; color: {INK};
        margin: 0; padding: 18px; background: #F4F7F8; line-height: 1.55; }}
.page {{ max-width: 820px; margin: 0 auto 18px; background: #fff; padding: 26px 30px;
         border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,.07); }}
h1 {{ font-size: 1.5rem; margin: 0 0 2px; }}
h2 {{ font-size: 1.05rem; margin: 22px 0 8px; color: {TEAL};
      border-bottom: 2px solid {LINE}; padding-bottom: 5px; }}
.meta {{ color: {MUTED}; font-size: .86rem; }}
.hero {{ font-size: 1.3rem; font-weight: 700; line-height: 1.45; margin: 14px 0 4px; }}
.hero .n {{ color: {CORAL}; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0; }}
.card {{ border: 1px solid {LINE}; border-radius: 9px; padding: 11px 13px; }}
.card .h {{ font-weight: 700; color: {TEAL}; font-size: .95rem; }}
.card .t {{ color: {MUTED}; font-size: .8rem; margin: 1px 0 5px; }}
.card .v {{ font-size: 1.2rem; font-weight: 700; }}
.gap {{ background: #E6EDEF; border-radius: 9px; padding: 12px 15px; margin: 12px 0; }}
.gap .big {{ font-size: 1.25rem; font-weight: 700; color: {TEAL}; }}
.gap .eq {{ color: {MUTED}; font-size: .88rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid {LINE}; }}
th {{ color: {MUTED}; font-weight: 600; font-size: .83rem; }}
td.r, th.r {{ text-align: right; }}
.note {{ color: {MUTED}; font-size: .8rem; line-height: 1.6; }}
.band {{ background: #FAFCFC; border-left: 3px solid {TEAL}; padding: 9px 13px; margin: 10px 0;
         font-size: .88rem; }}
.foot {{ color: {MUTED}; font-size: .74rem; margin-top: 16px; border-top: 1px solid {LINE}; padding-top: 9px; }}
@media print {{
  body {{ background: #fff; padding: 0; }}
  .page {{ box-shadow: none; border-radius: 0; margin: 0; padding: 0; max-width: none; }}
  .pagebreak {{ page-break-before: always; }}
  .noprint {{ display: none; }}
}}
"""


def build(ctx: dict) -> str:
    """ctx 키: client, advisor, calc_id, version, hero, goals, gap, band, actions, fan, flow, assumptions"""
    g = "".join(
        f'<div class="card"><div class="h">{_esc(h)}</div><div class="t">{_esc(t)}</div><div class="v">{_esc(v)}</div></div>'
        for h, t, v in ctx["goals"])
    acts = "".join(
        f'<tr><td>{i}</td><td>{_esc(lab)}</td><td>{_esc(how)}</td><td class="r">{_esc(eff)}</td></tr>'
        for i, (lab, how, eff) in enumerate(ctx["actions"], 1))
    act_block = (f'<h2>지금 할 수 있는 일 — 효과가 큰 순서</h2><table>'
                 f'<tr><th>순위</th><th>무엇을</th><th>어떻게</th><th class="r">기본 생활 지킬 가능성</th></tr>'
                 f'{acts}</table>') if ctx["actions"] else ""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>노후 점검 리포트 — {_esc(ctx['client'])}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>{CSS}</style></head><body>
<div class="page">
  <h1>🧭 노후 점검 리포트</h1>
  <div class="meta">{_esc(ctx['client'])} 님 · 작성 {_esc(ctx['date'])}{_esc(ctx['advisor_line'])}</div>
  <div class="hero">{ctx['hero']}</div>
  <div class="gap"><div class="eq">{ctx['gap_eq']}</div><div class="big">{ctx['gap_big']}</div></div>
  <h2>세 가지 목표</h2>
  <div class="grid">{g}</div>
  <div class="band">{ctx['band']}</div>
  <h2>나이별로 남는 돈 (오늘 돈 가치)</h2>
  {ctx['fan_svg']}
  <div class="note">선이 0에 닿는 때가 모아둔 돈을 다 쓰는 때입니다. 집은 포함하지 않았습니다.</div>
  {act_block}
  <div class="foot">
    계산 ID {_esc(ctx['calc_id'])} · 모델 v{_esc(ctx['version'])} · 1만 가지 미래 시뮬레이션<br>
    같은 입력과 같은 모델 버전이면 언제 다시 계산해도 같은 결과가 나옵니다.
  </div>
</div>
<div class="page pagebreak">
  <h2>은퇴 후 1년, 돈은 이렇게 움직입니다</h2>
  {ctx['flow_svg']}
  <div class="note">연금으로 들어온 돈에서 생활비·세금·건강보험료가 빠지고, 모자라는 만큼을 모아둔 돈에서 꺼내 씁니다.</div>
  <h2>이 계산이 가정한 것</h2>
  <div class="note">{ctx['assumptions']}</div>
  <div class="foot">이 리포트는 투자 권유나 세무 자문이 아닙니다. 실제 결정 전에 전문가와 상의하시기 바랍니다.<br>
  노후나침반 · retire-compass.streamlit.app</div>
</div>
</body></html>"""
