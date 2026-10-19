"""live 경로 보조 함수 스모크 테스트 (네트워크 불필요).

손으로 계산한 기대값만 쓴다.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
for k in ("DART_API_KEY", "ECOS_API_KEY", "FRED_API_KEY", "FINAGENT_MODE"):
    os.environ.pop(k, None)

import finagent_mcp_servers as S

# 1) _num: DART 금액 파싱. 결측은 '-' / '' / None 으로 온다.
assert S._num("1,234") == 1234.0
assert S._num("-1,234") == -1234.0          # 부호 있는 금액은 숫자로 살아야 한다
assert S._num("-") == 0.0                    # 결측 placeholder
assert S._num("") == 0.0
assert S._num(None) == 0.0
assert S._num("abc") == 0.0
assert S._num("12") == 12.0
print("[OK] _num: 콤마 제거, 음수 유지, 결측('-'/''/None)과 파싱 실패는 0.0")

# 2) _need_key: live 인데 키가 없으면 예외가 아니라 ok=False 응답
os.environ["FINAGENT_MODE"] = "live"
for impl, label in ((lambda: S.dart_financials_impl("00126380", 2024), "dart_financials"),
                    (lambda: S.dart_disclosures_impl("00126380", 30), "dart_disclosures"),
                    (lambda: S.ecos_series_impl("usdkrw", 3), "ecos_series"),
                    (lambda: S.fred_series_impl("VIXCLS"), "fred_series")):
    out = json.loads(impl())
    assert out["ok"] is False, label
    assert any("환경변수가 없습니다" in w for w in out["warnings"]), (label, out["warnings"])
print("[OK] 키 없는 live 호출: KeyError 대신 ok=False + 안내 경고")

# 3) 허용목록 밖 시리즈는 호출 전에 거절된다
assert json.loads(S.ecos_series_impl("gdp"))["ok"] is False
assert json.loads(S.fred_series_impl("GDP"))["ok"] is False
print("[OK] 허용목록 밖 시리즈 거절 (ecos/fred)")

# 4) _err: 예외 메시지에서 키가 가려진다
os.environ["DART_API_KEY"] = "supersecretkey1234"
msg = S._err(RuntimeError("url=...&crtfc_key=supersecretkey1234"))
assert "supersecretkey1234" not in msg, msg
assert "***" in msg and msg.startswith("RuntimeError:"), msg
os.environ["DART_API_KEY"] = "abc"           # 4자 미만 키는 가리지 않는다(오탐 방지)
assert "abc" in S._err(RuntimeError("abc"))
print("[OK] _err: 4자 이상 키는 ***로 가림, 짧은 값은 그대로")

# 5) fixture 모드는 영향을 받지 않는다
os.environ["FINAGENT_MODE"] = "fixture"
out = json.loads(S.dart_financials_impl("FIX001", 2024))
assert out["ok"] is True and out["mode"] == "fixture"
# baseline 매출 100 -> 112 이므로 증가율 0.12, 영업이익률 13.5/112 = 0.1205...
assert out["data"]["revenue_growth"] == 0.12, out["data"]
assert out["data"]["op_margin"] == round(13.5 / 112, 4), out["data"]
assert out["data"]["debt_ratio"] == round(62 / 110, 4), out["data"]
print("[OK] fixture 회귀: 매출증가율 0.12, 영업이익률 13.5/112, 부채비율 62/110")

print("\nSMOKE TEST (finagent live paths) OK")
