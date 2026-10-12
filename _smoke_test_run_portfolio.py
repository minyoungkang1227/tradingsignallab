"""run_portfolio.py CLI 통합 스모크테스트. 합성 데이터 결과는 실제 시장 성과가 아님."""
import io, json, tempfile, contextlib, zlib
from pathlib import Path
import numpy as np, pandas as pd
import run_portfolio as rp

def make_df(sym, n=500):
    rng = np.random.default_rng(zlib.crc32(sym.encode()))
    ix = pd.bdate_range("2022-01-03", periods=n)
    c = 100*np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    o = np.r_[c[0], c[:-1]]*(1 + rng.normal(0, 0.004, n))
    v = rng.integers(1_000, 5_000, n).astype(float)*(1 + 4*(rng.random(n) < 0.06))
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c)*1.006, "low": np.minimum(o, c)*0.994, "close": c, "volume": v}, index=ix)
    d.index.name = "date"; return d

def loader(args):
    if args.symbol == "FAIL": raise SystemExit("로딩 실패(테스트)")
    return make_df(args.symbol)

wl = [{"market": "krx", "symbol": s, "label": s} for s in ["AAA", "BBB", "CCC", "FAIL", "DDD"]] + \
     [{"market": "upbit", "symbol": "KRW-X", "label": "x"}]
tmp = Path(tempfile.mkdtemp()); wlp = tmp / "wl.json"; wlp.write_text(json.dumps(wl), encoding="utf-8")

def go(extra, out=None):
    ns = rp.build_parser().parse_args(["--watchlist", str(wlp)] + extra)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = rp.run_portfolio(ns, loader=loader, out_dir=out)
    return res, buf.getvalue()

out = tmp / "out"
(res, results, cand, fails), txt = go([], out)
assert "FAIL" in fails and "AAA" not in fails                       # 한 종목 실패는 격리되고 나머지는 계속
assert set(cand["symbol"]).issubset({"AAA", "BBB", "CCC", "DDD"}) and len(cand) > 0   # upbit 종목은 market=krx에서 제외
assert res.stats["n_trades"] > 0 and res.stats["max_concurrent"] <= 5
for f in ["portfolio_trades.csv", "portfolio_equity.csv", "portfolio_candidates.csv"]:
    assert (out / f).exists(), f
assert "포트폴리오 시뮬레이션" in txt and "참고 기준선" in txt and "[건너뜀]" in txt
# 불변식: 노출 한도, 거래 수 <= 후보 수, 비용 반영(profile realistic이 entry_lag=1·36bp)
assert res.trades["weight"].max() <= 0.25 + 1e-12 and len(res.trades) <= len(cand)
assert ((cand["gross_ret"] - cand["ret"]) > 0.0045).all() and ((cand["gross_ret"] - cand["ret"]) < 0.0047).all()   # 36+10bp
print(f"[PASS] 1) 감시리스트 4종목(1개 로딩 실패 격리) -> 후보 {len(cand)}건 중 {res.stats['n_trades']}건 체결, "
      f"비용 46bp(36+10) 반영, 한도 준수, 결과 파일 3개")

# 2) 같은 입력이면 결과 동일(결정적), 프로필 none이면 비용 0
(r2, *_), _ = go([])
assert np.allclose(r2.equity.values, res.equity.values) and list(r2.trades["symbol"]) == list(res.trades["symbol"])
(_, _, c0, _), _ = go(["--profile", "none"])
assert np.allclose(c0["gross_ret"] - c0["ret"], 0.0)
print("[PASS] 2) 재실행 결과 동일(결정적), --profile none이면 비용 0")

# 3) 신호 필터·구간 필터·제약 강화(슬롯 1)로 거래가 줄거나 같음
sig = cand["signal"].iloc[0]
(_, _, cs_, _), _ = go(["--signals", sig]); assert set(cs_["signal"]) == {sig}
mid = str(cand["entry_date"].iloc[len(cand)//2].date())
(_, _, cw, _), _ = go(["--sim-start", mid]); assert (cw["entry_date"] >= pd.Timestamp(mid)).all() and len(cw) < len(cand)
(r1, *_), _ = go(["--max-positions", "1"]); assert r1.stats["max_concurrent"] == 1 and r1.stats["n_trades"] <= res.stats["n_trades"]
print("[PASS] 3) 신호 필터, 진입 구간 필터, 슬롯 1개(동시 보유 1, 체결 수 감소) 동작")

# 4) 우선순위 민감도: random 시드 여러 개 -> 분포 출력
(_, rs, _, _), t4 = go(["--priority", "random", "--seeds", "6"])
assert len(rs) == 6 and "우선순위 민감도" in t4
print("[PASS] 4) random 시드 6개 반복 -> 누적수익·MDD 분포 출력")

# 5) 시장에 종목이 없으면 명확히 종료
wlp.write_text(json.dumps([wl[-1]]), encoding="utf-8")
try:
    go([]); raise AssertionError("종료해야 함")
except SystemExit as e:
    assert "종목이 없습니다" in str(e)
print("[PASS] 5) 선택한 시장에 종목이 없으면 조용히 진행하지 않고 종료")
print("\n모든 스모크테스트 통과. (합성 데이터 결과 — 실제 시장 성과가 아님)")
