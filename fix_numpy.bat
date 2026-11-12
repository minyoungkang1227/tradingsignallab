@echo off
cd /d "%~dp0"
echo [1/3] current numpy
python -c "import numpy;print(numpy.__version__)"
echo [2/3] pin numpy 1.26.4 (scipy/streamlit compat)
pip install "numpy==1.26.4"
echo [3/3] verify
python -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"
echo.
echo DONE - close this window
pause
