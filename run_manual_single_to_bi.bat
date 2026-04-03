@echo off
REM One-click manual single-customer test + BI update
setlocal

title Manual Single Customer to BI
echo ================================================
echo   Manual Single-Customer Prediction -> BI
echo ================================================
echo.
echo Step 1/4: Checking Python and input files...

cd /d "%~dp0"

SET "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
SET "INPUT_JSON=%~dp0data\one_customer_template.json"
SET "OUTPUT_CSV=%~dp0outputs\one_customer_prediction.csv"
SET "PBIX_PATH=C:\Users\ABEL ABRAHAM\Downloads\dash1.pbix"

if not exist "%PYTHON_EXE%" (
  echo [WARN] Venv Python not found: %PYTHON_EXE%
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] No Python interpreter found.
    echo Install Python or create .venv first.
    pause
    exit /b 1
  )
  SET "PYTHON_EXE=python"
  echo [INFO] Falling back to system Python: %PYTHON_EXE%
)

if not exist "%INPUT_JSON%" (
  echo [ERROR] Input JSON not found: %INPUT_JSON%
  pause
  exit /b 1
)

echo [INFO] Running prediction script...
echo Step 2/4: Running churn prediction for one customer...
call "%PYTHON_EXE%" src\predict_stacked.py --input-file "%INPUT_JSON%" --write-csv "%OUTPUT_CSV%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Manual single-customer run failed. Exit code: %EXIT_CODE%
  pause
  exit /b 1
)

echo.
echo Step 3/4: CSV updated and ready for dashboard refresh.
echo [OK] CSV updated for BI: %OUTPUT_CSV%
timeout /t 2 /nobreak >nul

if exist "%PBIX_PATH%" (
  echo Step 4/4: Opening Power BI dashboard...
  start "" "%PBIX_PATH%"
  echo [OK] Power BI opened: %PBIX_PATH%
) else (
  echo [WARN] PBIX file not found: %PBIX_PATH%
  echo Update PBIX_PATH in run_manual_single_to_bi.bat
)

echo.
echo Done. Press any key to close.
pause >nul

exit /b 0
