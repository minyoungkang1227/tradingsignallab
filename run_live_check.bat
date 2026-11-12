@echo off
cd /d "%~dp0finagent"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo === live_check ===
python live_check.py --corp 00126380 --symbol 005930 --year 2024
echo.
echo Copy the output above and paste it to Claude.
pause
