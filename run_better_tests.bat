@echo off
setlocal
cd /d "%~dp0"

SET "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
SET "TEST_SET=%~dp0data\live_test_set.csv"
SET "TEST_OUT=%~dp0outputs\live_test_predictions.csv"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%TEST_SET%" (
  echo [ERROR] Test set not found: %TEST_SET%
  exit /b 1
)

echo ================================================================
echo HIGH-RISK SINGLE CUSTOMER TEST
echo ================================================================
call "%PYTHON_EXE%" src\predict_stacked.py --input-file data\high_risk_test_customer.json
if errorlevel 1 exit /b %errorlevel%

echo.
echo ================================================================
echo LOW-RISK SINGLE CUSTOMER TEST
echo ================================================================
call "%PYTHON_EXE%" src\predict_stacked.py --input-file data\low_risk_test_customer.json
if errorlevel 1 exit /b %errorlevel%

echo.
echo ================================================================
echo LIVE TEST SET RUN (APPEND MODE)
echo ================================================================
set "POWERBI_CSV_PATH=%TEST_OUT%"
call "%PYTHON_EXE%" run_demo.py --data "%TEST_SET%" --skip-ai --append-output
if errorlevel 1 exit /b %errorlevel%

echo.
echo [OK] All tests completed.
echo [OK] Test set predictions appended to: %TEST_OUT%
endlocal
exit /b 0
