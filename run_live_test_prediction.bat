@echo off
REM Run predictions on dedicated live test set without altering main output CSV
setlocal

cd /d "%~dp0"

SET "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
SET "TEST_DATA=%~dp0data\live_test_set.csv"
SET "OUT_CSV=%~dp0outputs\live_test_predictions.csv"
SET "READABLE_OUT=%~dp0outputs\live_test_predictions_readable.csv"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  pause
  exit /b 1
)

if not exist "%TEST_DATA%" (
  echo [ERROR] Test dataset not found: %TEST_DATA%
  pause
  exit /b 1
)

echo Running prediction on %TEST_DATA%
set "POWERBI_CSV_PATH=%OUT_CSV%"
call "%PYTHON_EXE%" run_demo.py --data "%TEST_DATA%" --skip-ai
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Prediction run failed. Exit code: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo Generating readable output view...
call "%PYTHON_EXE%" scripts\make_readable_predictions.py --input "%OUT_CSV%" --output "%READABLE_OUT%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Readable output generation failed. Exit code: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo.
echo [OK] New prediction file ready: %OUT_CSV%
echo [OK] Readable prediction file: %READABLE_OUT%
echo Existing outputs\churn_predictions.csv was not changed by this run.
pause
exit /b 0
