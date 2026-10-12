"""
문서-코드 동기화 스모크테스트.
(1) OVERVIEW.md의 '옵션·기본값 대조표'가 argparse 실제 정의에서 생성한 표와 글자 단위로 같은가.
    (옵션을 추가/삭제/기본값 변경하고 문서를 안 고치면 여기서 실패한다. 고치는 법:
     python tools/gen_option_table.py 출력으로 마커 사이를 교체)
(2) README/PIPELINE/OVERVIEW가 핵심 주장(할 수 있는 일/입증되지 않은 일, 확장 단계, 검증 현황 분리)을 유지하는가.
(3) 문서가 언급한 코드 파일이 실제로 존재하는가(문서가 없는 모듈을 가리키지 않게).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tools.gen_option_table import build_table, START, END


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def main():
    ov = read("OVERVIEW.md")
    # (1)
    assert ov.count(START) == 1 and ov.count(END) == 1, "OVERVIEW 옵션표 마커가 정확히 1쌍이어야 함"
    doc = ov.split(START, 1)[1].split(END, 1)[0].strip()
    gen = build_table().strip()
    if doc != gen:
        dl, gl = doc.splitlines(), gen.splitlines()
        diff = [l for l in gl if l not in dl][:3] + [l for l in dl if l not in gl][:3]
        raise AssertionError("OVERVIEW 옵션표가 코드와 다름. 예: " + " || ".join(diff))
    print("1) 옵션표 일치:", len(gen.splitlines()) - 2, "행")

    # (2)
    rd, pl = read("README.md"), read("PIPELINE.md")
    for key in ["현재 할 수 있는 일", "현재 입증되지 않은 일", "검증 현황", "성과 리포트 표준 형식"]:
        assert key in rd, f"README에 '{key}' 없음"
    for key in ["[8]", "[9]", "[10]"]:
        assert key in pl and key in ov, f"{key} 단계가 PIPELINE/OVERVIEW에 없음"
    for key in ["15-1. 합성 데이터 정확성 테스트", "15-2. 실제 시장 데이터 검증 현황"]:
        assert key in ov, f"OVERVIEW에 '{key}' 없음"
    print("2) 핵심 주장 유지")

    # (3) 문서가 백틱으로 가리킨 .py 경로 중 backtest/·indicators/·alerts/·tools/ 하위는 실제로 있어야 함
    missing = []
    for text in (rd, pl, ov):
        for m in re.findall(r"`((?:backtest|indicators|alerts|tools|research|advisor|decision|factors|data)/[A-Za-z0-9_]+\.py)`", text):
            if not os.path.exists(os.path.join(ROOT, m)):
                missing.append(m)
    # 일부 모듈은 이 테스트가 도는 사본에 없을 수 있으므로 전부 없는 경우만 실패시키지 않고 목록을 보여 준다
    assert not missing, "문서가 가리키는데 없는 파일: " + ", ".join(sorted(set(missing)))
    print("3) 문서가 가리키는 코드 파일 모두 존재")
    print("ALL OK")


if __name__ == "__main__":
    main()
