"""상담 세션: 고객 입력 저장·복원과 재현용 계산 ID.

금융기관에서는 "작년에 이 고객에게 보여준 숫자를 다시 뽑아 보라"는 요구가 반드시 들어온다.
그래서 입력과 모델 버전을 한 파일에 담고, 둘을 합쳐 만든 짧은 ID를 리포트에 찍어 둔다.
같은 입력 + 같은 모델 버전 + 같은 시드 = 같은 결과.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date

from . import __version__

FORMAT = "retire-compass/1"


def calc_id(inputs: dict, version: str = __version__) -> str:
    """입력과 모델 버전으로 만든 10자리 재현 ID. 키 순서가 달라도 같은 값이 나온다."""
    payload = json.dumps({"p": inputs, "v": str(version)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10].upper()


def save_blob(inputs: dict, client: str = "", advisor: str = "", today: date = None) -> str:
    """고객 파일(JSON 문자열)."""
    return json.dumps({"format": FORMAT, "model_version": __version__,
                       "calc_id": calc_id(inputs), "saved_at": (today or date.today()).isoformat(),
                       "client": client, "advisor": advisor, "inputs": inputs},
                      ensure_ascii=False, indent=2)


def load_blob(raw) -> tuple:
    """고객 파일을 읽어 (입력, 고객명, 상담자, 경고문)을 돌려준다.

    모델 버전이 다르면 경고문을 함께 준다 — 제도·가정이 바뀌어 그때와 결과가 다를 수 있기 때문이다."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    d = json.loads(raw)
    if not isinstance(d, dict) or d.get("format") != FORMAT or not isinstance(d.get("inputs"), dict):
        raise ValueError("노후나침반 고객 파일이 아닙니다.")
    saved = str(d.get("model_version"))
    warn = ("" if saved == str(__version__) else
            f"이 파일은 모델 v{saved}에서 저장됐고 지금은 v{__version__}입니다. "
            "제도·가정이 바뀌었을 수 있어 결과가 그때와 다를 수 있습니다.")
    return d["inputs"], d.get("client", ""), d.get("advisor", ""), warn
