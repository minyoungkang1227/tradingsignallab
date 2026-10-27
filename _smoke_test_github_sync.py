"""alerts/github_sync.py 스모크테스트. 실제 네트워크 호출 없이 requests.get/put을
몽키패치해서, base64 인코딩/디코딩 왕복과 에러 처리(비정상 status, 409 충돌)가
손으로 계산한 기대값과 일치하는지 확인한다."""
import base64
import json

import requests
import alerts.github_sync as gs


class FakeResp:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


ORIGINAL_ITEMS = [{"market": "krx", "symbol": "005930", "label": "삼성전자"}]
ORIGINAL_CONTENT_B64 = base64.b64encode(
    json.dumps(ORIGINAL_ITEMS, ensure_ascii=False).encode("utf-8")
).decode("ascii")

captured = {}


def fake_get(url, params=None, headers=None, timeout=None):
    captured["get_url"] = url
    captured["get_headers"] = headers
    return FakeResp(200, {"content": ORIGINAL_CONTENT_B64, "sha": "abc123"})


requests.get = fake_get

# 1) fetch_watchlist: base64 디코딩 후 원래 items와 동일해야 하고, sha도 그대로 반환돼야 함
items, sha = gs.fetch_watchlist(token="t", repo="x/y")
assert items == ORIGINAL_ITEMS, items
assert sha == "abc123", sha
assert captured["get_headers"]["Authorization"] == "Bearer t"
print("[PASS] 1) fetch_watchlist: base64 왕복 + sha 반환 확인")


# 2) fetch_watchlist: 404 등 비정상 status면 GitHubSyncError
def fake_get_404(url, params=None, headers=None, timeout=None):
    return FakeResp(404, text="Not Found")


requests.get = fake_get_404
try:
    gs.fetch_watchlist(token="t", repo="x/y")
    assert False, "예외가 발생했어야 함"
except gs.GitHubSyncError as e:
    assert "404" in str(e)
    print("[PASS] 2) fetch_watchlist: 404 -> GitHubSyncError 발생 확인")


# 3) update_watchlist: PUT 바디에 들어가는 content가 새 items를 올바르게 인코딩했는지
NEW_ITEMS = ORIGINAL_ITEMS + [{"market": "upbit", "symbol": "KRW-ETH", "label": "이더리움",
                                "require_significance": False}]


def fake_put(url, json=None, headers=None, timeout=None):
    captured["put_json"] = json
    decoded = base64.b64decode(json["content"]).decode("utf-8")
    captured["decoded_content"] = decoded
    return FakeResp(200, {"content": {"sha": "def456"}})


requests.put = fake_put
new_sha = gs.update_watchlist(token="t", repo="x/y", items=NEW_ITEMS, sha="abc123")
assert new_sha == "def456", new_sha
assert captured["put_json"]["sha"] == "abc123"
decoded_items = __import__("json").loads(captured["decoded_content"])
assert decoded_items == NEW_ITEMS, decoded_items
print("[PASS] 3) update_watchlist: PUT content가 새 items와 정확히 일치, 새 sha 반환 확인")


# 4) update_watchlist: 409면 충돌 전용 메시지의 GitHubSyncError
def fake_put_409(url, json=None, headers=None, timeout=None):
    return FakeResp(409, text="Conflict")


requests.put = fake_put_409
try:
    gs.update_watchlist(token="t", repo="x/y", items=NEW_ITEMS, sha="stale")
    assert False, "예외가 발생했어야 함"
except gs.GitHubSyncError as e:
    assert "충돌" in str(e)
    print("[PASS] 4) update_watchlist: 409 -> 충돌 전용 메시지 확인")

# 5) items_to_rows: require_significance가 없는 항목은 '기본값' 라벨로 표시돼야 함
items_v = [
    {"market": "krx", "symbol": "005930", "label": "삼성전자"},
    {"market": "upbit", "symbol": "KRW-BTC", "label": "비트코인", "require_significance": False},
    {"market": "krx", "symbol": "000660", "label": "SK하이닉스", "require_significance": True},
]
rows_v = gs.items_to_rows(items_v)
assert rows_v == [
    {"시장": "krx", "종목코드": "005930", "이름": "삼성전자", "강도": "기본값 (시장별 자동)", "비중": None, "신호알림": False},
    {"시장": "upbit", "종목코드": "KRW-BTC", "이름": "비트코인", "강도": "즉시 알림", "비중": None, "신호알림": False},
    {"시장": "krx", "종목코드": "000660", "이름": "SK하이닉스", "강도": "BH보정 필요", "비중": None, "신호알림": False},
], rows_v
print("[PASS] 5) items_to_rows: require_significance -> 한글 강도 라벨 매핑 확인")

# 6) rows_to_items: items_to_rows의 역변환 왕복(round-trip)이 원래 items와 일치해야 함
roundtrip = gs.rows_to_items(rows_v)
assert roundtrip == items_v, roundtrip
print("[PASS] 6) rows_to_items: items -> rows -> items 왕복 시 원본과 정확히 일치 확인")

# 7) rows_to_items: 종목코드가 빈 행(새로 추가했다가 안 채운 줄)은 건너뛰어야 함
rows_with_blank = rows_v + [{"시장": "krx", "종목코드": "", "이름": "", "강도": "기본값 (시장별 자동)"}]
assert gs.rows_to_items(rows_with_blank) == items_v
print("[PASS] 7) rows_to_items: 종목코드 빈 행 건너뜀 확인")

# 8) 비중·신호알림 왕복: weight는 숫자로, NaN(편집기 빈 칸)은 미지정으로, signal_alerts는 True일 때만 기록
items_w = [{"market": "krx", "symbol": "005930", "label": "삼성전자", "weight": 0.3, "signal_alerts": True},
           {"market": "krx", "symbol": "000660", "label": "SK하이닉스"}]
rows_w = gs.items_to_rows(items_w)
assert rows_w[0]["비중"] == 0.3 and rows_w[0]["신호알림"] is True and rows_w[1]["비중"] is None and rows_w[1]["신호알림"] is False
assert gs.rows_to_items(rows_w) == items_w, gs.rows_to_items(rows_w)
rows_nan = [dict(rows_w[1], **{"비중": float("nan")})]
assert "weight" not in gs.rows_to_items(rows_nan)[0]
print("[PASS] 8) 비중·신호알림 왕복, NaN 비중은 미지정, signal_alerts는 True일 때만 기록")

# 8) rows_to_items: 이름이 빈 문자열이면 label 키 자체가 없어야 함(쓸데없는 빈 값 저장 방지)
rows_no_label = [{"시장": "krx", "종목코드": "123456", "이름": "", "강도": "기본값 (시장별 자동)"}]
out = gs.rows_to_items(rows_no_label)
assert out == [{"market": "krx", "symbol": "123456"}], out
assert "label" not in out[0]
print("[PASS] 8) rows_to_items: 이름 빈 문자열 -> label 키 생략 확인")

print("\n모든 스모크테스트 통과.")
