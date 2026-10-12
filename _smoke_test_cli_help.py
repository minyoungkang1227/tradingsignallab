"""
CLI --help 무결성 테스트.

[왜 필요한가]
argparse는 help 문자열에 %-포맷을 적용한다. 그래서 리터럴 퍼센트를 그냥 '%'로
쓰면 `--help` 자체가 ValueError로 죽는다. 한글 도움말에서는 '%' 다음에 한글이
오는 경우가 많아('95% 유의수준', '상위 10% 제외') 이 함정에 걸리기 쉽고,
평소 실행 경로에서는 드러나지 않아 한참 모르고 지나가게 된다
(실제로 --deflate 추가 때 이 문제로 --help가 깨진 채 배포될 뻔했다).

이 테스트는 옵션을 추가할 때마다 자동으로 그걸 잡는다.
"""
import argparse
import subprocess
import sys

import run_backtest as rb

# --- 1) 모든 help 문자열이 %-포맷을 통과하는가 ------------------------------
helps = []
_orig_add = argparse.ArgumentParser.add_argument


def _spy(self, *a, **k):
    h = k.get("help")
    if h:
        helps.append((a[0] if a else "?", h))
    return _orig_add(self, *a, **k)


argparse.ArgumentParser.add_argument = _spy
try:
    rb.main()
except BaseException:
    pass
finally:
    argparse.ArgumentParser.add_argument = _orig_add

assert len(helps) > 20, f"도움말을 가진 옵션이 너무 적게 수집됨: {len(helps)}"

bad = []
for name, h in helps:
    try:
        h % {"default": None, "prog": "run_backtest.py"}
    except Exception as e:
        bad.append((name, type(e).__name__, str(e), h[:80]))

if bad:
    print(f"[FAIL] %-포맷을 통과하지 못하는 help 문자열 {len(bad)}개:")
    for n, t, e, h in bad:
        print(f"   {n}: {t}: {e}")
        print(f"      -> {h}")
    print("\n   리터럴 퍼센트는 '%%'로 써야 합니다(예: '기본 0.95(5%% 유의수준)').")
    sys.exit(1)

print(f"[PASS] 1) help 문자열 {len(helps)}개 전부 %-포맷 통과")


# --- 2) 실제로 `--help`를 실행해봐도 죽지 않는가 ----------------------------
r = subprocess.run([sys.executable, "run_backtest.py", "--help"],
                    capture_output=True, text=True, timeout=120)
assert r.returncode == 0, f"--help 종료코드 {r.returncode}\n{r.stderr[-600:]}"
assert "usage:" in r.stdout
print(f"[PASS] 2) `run_backtest.py --help` 정상 종료 (출력 {len(r.stdout):,}자)")


# --- 3) 퍼센트가 화면에 제대로 한 개로 렌더링되는가 -------------------------
# 이스케이프를 과하게 해서 '%%'가 그대로 보이면 안 된다.
assert "%%" not in r.stdout, [
    ln.strip() for ln in r.stdout.split("\n") if "%%" in ln][:5]
print("[PASS] 3) 도움말 출력에 '%%'가 그대로 새어나오지 않음(이스케이프 과잉 없음)")


# --- 4) 이번에 추가한 옵션들이 실제로 등록돼 있는가 -------------------------
EXPECTED = ["--deflate", "--dsr-threshold", "--dsr-min-events", "--dsr-extra-trials",
            "--barrier", "--barrier-pt-mult", "--barrier-sl-mult",
            "--barrier-vol-span", "--barrier-no-intrabar"]
missing = [o for o in EXPECTED if o not in r.stdout]
assert not missing, f"도움말에 없는 옵션: {missing}"
print(f"[PASS] 4) 신규 옵션 {len(EXPECTED)}개 전부 도움말에 등록됨")

print("\n모든 스모크테스트 통과.")
