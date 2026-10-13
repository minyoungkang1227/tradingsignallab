"""pytest가 저장소 폴더를 파이썬 경로에 넣도록 하는 파일.

이 파일이 없으면 `pytest`(콘솔 스크립트)로 실행할 때 `retire_sim`을 찾지 못한다.
`python -m pytest`는 현재 폴더가 자동으로 경로에 들어가서 문제가 드러나지 않았다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
