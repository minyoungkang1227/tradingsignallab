"""루트 `_smoke_test*.py`를 pytest 케이스로 실행하는 브리지.

기존 스모크 테스트는 모듈 레벨에서 assert하고 print하는 스크립트다. 손계산 기대값과
그 근거를 주석으로 남겨둔 자산이므로 내용을 다시 쓰지 않고, 실행과 판정만 pytest로
옮긴다. 각 파일은 하위 프로세스에서 돌고 종료 코드로 판정한다.

이렇게 하는 이유는 이전 `run_tests.bat`이 판정을 신뢰할 수 없었기 때문이다. cmd에서
`>/dev/null`은 유효한 경로가 아니라 리디렉션 자체가 실패했고, 루프가 종료 코드를
남기지 않아 CI에서 쓸 수 없었다.

실행:
    pytest                          전체
    pytest -m "not external"        외부 의존 제외
    pytest -k gate                  이름으로 선택
    pytest -q --no-header           조용히

개별 파일을 진짜 pytest 테스트 함수로 옮기는 작업은 이 브리지와 무관하게 진행할 수
있다. 옮긴 파일은 루트에서 `tests/`로 이동하면 브리지 수집 대상에서 자연히 빠진다.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# 하위 프로세스 하나당 상한. 부트스트랩·Walk-forward가 있는 테스트는 수십 초 걸린다.
TIMEOUT_SEC = 600

# 이 토큰이 파일 안에 있으면 네트워크·외부 API·환경변수에 의존할 수 있다고 보고
# `external` 마커를 붙인다. 목록을 손으로 관리하지 않기 위해 수집 시점에 스캔한다.
EXTERNAL_HINTS = (
    "requests.",
    "urlopen",
    "pykrx",
    "yfinance",
    "upbit",
    "os.environ",
    "getenv",
)


def _discover() -> list[Path]:
    return sorted(REPO_ROOT.glob("_smoke_test*.py"))


def _is_external(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    return any(hint in text for hint in EXTERNAL_HINTS)


def _params() -> list:
    params = []
    for path in _discover():
        marks = [pytest.mark.smoke]
        if _is_external(path):
            marks.append(pytest.mark.external)
        params.append(pytest.param(path, id=path.stem, marks=marks))
    return params


_PARAMS = _params()

if not _PARAMS:
    # 브리지가 아무것도 못 찾았다는 건 경로가 틀렸다는 뜻이다. 조용히 0건 통과로
    # 넘어가면 전체 검증이 사라진 것을 모르게 되므로 수집 단계에서 실패시킨다.
    raise RuntimeError(
        f"_smoke_test*.py를 찾지 못했습니다. 탐색 경로: {REPO_ROOT}. "
        "브리지가 0건 통과로 넘어가면 검증이 사라진 것을 알 수 없으므로 실패로 처리합니다."
    )


@pytest.mark.parametrize("script", _PARAMS)
def test_smoke_script(script: Path) -> None:
    """스모크 스크립트 하나를 실행하고 종료 코드 0을 요구한다."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{script.name}이 {TIMEOUT_SEC}초 안에 끝나지 않았습니다.\n"
            f"--- stdout ---\n{exc.stdout or ''}\n"
            f"--- stderr ---\n{exc.stderr or ''}"
        )

    if proc.returncode != 0:
        pytest.fail(
            f"{script.name} 종료 코드 {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )


def test_bridge_sees_every_script() -> None:
    """수집된 케이스 수가 디스크의 스크립트 수와 같은지 확인한다.

    파라미터화가 조용히 일부를 빠뜨리면 통과 건수만 줄고 아무도 눈치채지 못한다.
    """
    assert len(_PARAMS) == len(_discover())
