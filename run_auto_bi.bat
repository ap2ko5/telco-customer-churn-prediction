@echo off
REM Start always-on BI auto-update watcher
setlocal

cd /d "%~dp0"

SET "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  pause
  exit /b 1
)

echo ================================================
echo   Auto BI Update Watcher
echo ================================================
echo Any change in data CSV files will:
echo 1) run pipeline
echo 2) update outputs/churn_predictions.csv
echo 3) open Power BI dashboard
echo.

call "%PYTHON_EXE%" bi_auto_update.py --skip-ai --close-lock-holders
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Watcher exited with code %EXIT_CODE%
pause
exit /b %EXIT_CODE%
