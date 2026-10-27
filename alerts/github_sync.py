"""
Streamlit 대시보드(app.py)의 '감시리스트 관리' 섹션이 GitHub 저장소의
alerts/watchlist.json을 직접 읽고 쓰기 위한 최소 래퍼.

[왜 필요한가] Streamlit Community Cloud는 서버 파일시스템을 영구 저장하지
않는다(재배포하면 날아가고, 여러 방문자가 같은 서버를 공유해서 "내 감시리스트"로
안전하게 쓸 수도 없다). 그래서 대시보드에서 고른 감시리스트는 로컬 파일이 아니라
GitHub 저장소의 alerts/watchlist.json 자체를 직접 고쳐서 저장한다 — 그래야
저장소를 체크아웃해서 도는 GitHub Actions(alerts/check_signals.py)가 다음
실행부터 바로 그 내용을 읽는다.

GitHub Contents API만 사용한다(PyGithub 같은 무거운 의존성 추가 안 함) —
requirements.txt에 이미 있는 requests만으로 충분하다.

인증: 이 저장소에 쓰기 권한이 있는 GitHub 토큰이 필요하다(Fine-grained PAT를
이 저장소 하나로만 Contents: Read and write 권한으로 발급하는 걸 권장 — 토큰이
새더라도 피해 범위를 이 저장소로 한정하기 위함). Streamlit Cloud의 Secrets에
GITHUB_TOKEN으로 등록해서 쓴다 — OWNER_TOKEN과 마찬가지로 코드/깃허브 공개
저장소에는 절대 직접 적지 않는다(secrets.toml은 .gitignore에 걸려 있음).
"""
from __future__ import annotations

import base64
import json

import requests

_API_BASE = "https://api.github.com"
_TIMEOUT = 10.0


class GitHubSyncError(RuntimeError):
    """GitHub Contents API 호출이 실패했을 때 — 원인을 사용자가 읽을 수 있는
    한국어 문장으로 감싸서 올린다(호출부가 st.error로 그대로 보여줄 수 있도록)."""


# app.py의 "감시리스트 관리" 섹션(st.data_editor)이 보여줄 한글 라벨. watchlist.json의
# require_significance(None/True/False)와 1:1 대응 — 여기서만 쓰는 표시용 매핑이고,
# 실제 저장되는 필드명/의미는 alerts/check_signals.py가 읽는 것과 동일하게 유지한다.
STRENGTH_LABELS = {
    None: "기본값 (시장별 자동)",
    True: "BH보정 필요",
    False: "즉시 알림",
}
_STRENGTH_VALUES = {v: k for k, v in STRENGTH_LABELS.items()}


def items_to_rows(items: list[dict]) -> list[dict]:
    """watchlist.json의 items(list[dict])를 st.data_editor에 바로 넣을 수 있는
    한글 컬럼명 rows로 바꾼다. app.py에서 이 함수를 쓰는 대신 직접 변환 로직을
    짜면, 테스트 없이 UI 코드 안에 로직이 숨어버려서 여기 별도 함수로 뺐다."""
    return [
        {
            "시장": it["market"],
            "종목코드": it["symbol"],
            "이름": it.get("label", ""),
            "강도": STRENGTH_LABELS.get(it.get("require_significance"), STRENGTH_LABELS[None]),
            # 2026-10 리스크 중심: 비중(자산 대비, 숏은 음수)과 신호 알림 on/off. 비중이 없으면 None(동일비중 처리)
            "비중": it.get("weight"),
            "신호알림": bool(it.get("signal_alerts", False)),
        }
        for it in items
    ]


def rows_to_items(rows: list[dict]) -> list[dict]:
    """items_to_rows()의 역변환. 종목코드가 빈 행(새로 추가했다가 안 채운 줄)은
    건너뛴다. '강도'가 '기본값'이면 require_significance 키 자체를 넣지 않는다
    (watchlist.json에 쓸데없는 null을 남기지 않고, check_signals.py의 시장별
    기본값 로직이 그대로 적용되게 하기 위함)."""
    out = []
    for row in rows:
        symbol = str(row.get("종목코드", "")).strip()
        if not symbol:
            continue
        item = {"market": row["시장"], "symbol": symbol}
        label = str(row.get("이름", "")).strip()
        if label:
            item["label"] = label
        req = _STRENGTH_VALUES.get(row.get("강도"))
        if req is not None:
            item["require_significance"] = req
        w = row.get("비중")
        if w is not None and not (isinstance(w, float) and w != w):      # NaN(빈 칸)은 '미지정'
            item["weight"] = float(w)
        if bool(row.get("신호알림", False)):
            item["signal_alerts"] = True
        out.append(item)
    return out


def fetch_watchlist(token: str, repo: str, path: str = "alerts/watchlist.json",
                     branch: str = "main") -> tuple[list[dict], str]:
    """현재 GitHub에 올라간 watchlist.json 내용과 sha(업데이트 시 필요)를 가져온다."""
    url = f"{_API_BASE}/repos/{repo}/contents/{path}"
    try:
        resp = requests.get(
            url, params={"ref": branch},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        raise GitHubSyncError(f"GitHub에 연결하지 못했습니다: {type(e).__name__}: {e}") from e

    if resp.status_code != 200:
        raise GitHubSyncError(
            f"watchlist.json을 불러오지 못했습니다 (status={resp.status_code}). "
            f"토큰 권한/저장소 경로를 확인하세요. 응답: {resp.text[:300]}"
        )
    payload = resp.json()
    try:
        content = base64.b64decode(payload["content"]).decode("utf-8")
        items = json.loads(content)
    except (KeyError, ValueError) as e:
        raise GitHubSyncError(f"watchlist.json 내용을 해석하지 못했습니다: {type(e).__name__}: {e}") from e
    return items, payload["sha"]


def update_watchlist(token: str, repo: str, items: list[dict], sha: str,
                      path: str = "alerts/watchlist.json", branch: str = "main",
                      message: str = "update: 대시보드에서 감시리스트 수정") -> str:
    """새 items 리스트로 watchlist.json을 덮어쓰고 커밋한다. 새 sha를 반환한다.

    sha가 현재 GitHub의 파일 상태와 다르면(그 사이 다른 곳에서 먼저 고쳤으면)
    GitHub API가 409를 돌려준다 — 조용히 덮어쓰지 않고 예외로 알려서, 호출부가
    "새로고침 후 다시 시도하세요"라고 안내할 수 있게 한다.
    """
    url = f"{_API_BASE}/repos/{repo}/contents/{path}"
    new_content = json.dumps(items, ensure_ascii=False, indent=2) + "\n"
    encoded = base64.b64encode(new_content.encode("utf-8")).decode("ascii")
    try:
        resp = requests.put(
            url,
            json={"message": message, "content": encoded, "sha": sha, "branch": branch},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as e:
        raise GitHubSyncError(f"GitHub에 연결하지 못했습니다: {type(e).__name__}: {e}") from e

    if resp.status_code == 409:
        raise GitHubSyncError(
            "저장 충돌 — 그 사이 watchlist.json이 다른 곳에서 바뀌었습니다. "
            "새로고침한 뒤 다시 시도하세요."
        )
    if resp.status_code not in (200, 201):
        raise GitHubSyncError(
            f"watchlist.json 저장에 실패했습니다 (status={resp.status_code}). "
            f"응답: {resp.text[:300]}"
        )
    return resp.json()["content"]["sha"]
