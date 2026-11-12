@echo off
REM pytest로 루트 _smoke_test*.py 전체를 실행한다. 이전 버전은 cmd에서
REM 유효하지 않은 /dev/null 리디렉션을 써서 성공/실패 판정이 신뢰할 수 없었다.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

python -c "import pytest" 2>nul
if errorlevel 1 (
  echo pytest not found. Install first:
  echo     pip install -r requirements.txt -r requirements-dev.txt
  exit /b 1
)

python -m pytest %*
set RC=%ERRORLEVEL%
echo.
if %RC%==0 (echo ALL PASSED) else (echo FAILURES - see summary above)
exit /b %RC%
