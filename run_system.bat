@echo off
REM ══════════════════════════════════════════════════════════════════════════════
REM  run_system.bat — ONE-CLICK DEMO ORCHESTRATOR
REM ══════════════════════════════════════════════════════════════════════════════
REM
REM  This batch file orchestrates the complete demo:
REM    1. Runs the Python prediction pipeline
REM    2. Waits for the CSV to be fully written
REM    3. Opens the Power BI dashboard
REM
REM  Setup:
REM    - Update PBIX_PATH below with your actual Power BI file path
REM    - Double-click this file to run the full pipeline
REM
REM ══════════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM ─ CONFIGURATION ────────────────────────────────────────────────────────────

REM  ⚠️  IMPORTANT: Update this path to your Power BI file location
REM      Example for Windows:
REM        C:\Users\YourUsername\Documents\churn_dashboard.pbix
REM        C:\Users\ABEL ABRAHAM\Downloads\dash1.pbix
SET "PBIX_PATH=C:\Users\ABEL ABRAHAM\Downloads\dash1.pbix"

REM  Python executable (uses the project's virtual environment)
SET "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

REM  Demo script location
SET "DEMO_SCRIPT=%~dp0run_demo.py"

REM  Reliable demo default: skip external AI dependency for faster/stable runs.
REM  Set to empty if you want Gemini recommendations in this flow.
SET "DEMO_ARGS=--skip-ai"

REM  Fixed CSV path used by both Python and Power BI
SET "CSV_PATH=%~dp0outputs\churn_predictions.csv"

REM  CSV readiness checks before opening Power BI
SET "CSV_READY_RETRIES=10"
SET "CSV_READY_WAIT_SECONDS=1"
SET "CSV_MIN_BYTES=128"

REM ─ COLOR CODES FOR OUTPUT ──────────────────────────────────────────────────
REM  (Windows console colors)

CLS
echo.
echo ════════════════════════════════════════════════════════════════════════════════
echo   ⚡ CHURN INTELLIGENCE DEMO — ONE-CLICK AUTOMATION
echo ════════════════════════════════════════════════════════════════════════════════
echo.

SET "INPUT_DATA_PATH=%~1"
if not "%INPUT_DATA_PATH%"=="" (
    if not exist "%INPUT_DATA_PATH%" (
        echo ❌ ERROR: Provided dataset path does not exist:
        echo    %INPUT_DATA_PATH%
        echo.
        pause
        exit /b 1
    )
)

REM ─ STEP 1: Verify Python ───────────────────────────────────────────────────

echo 📋 Checking environment...
if not exist "%PYTHON_EXE%" (
    echo.
    echo ❌ ERROR: Python executable not found at:
    echo    %PYTHON_EXE%
    echo.
    echo    Make sure the virtual environment is initialized:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo    ✓ Python found: %PYTHON_EXE%

REM ─ STEP 2: Run the Python Pipeline ────────────────────────────────────────

echo.
echo 🚀 Running Churn Intelligence Pipeline...
echo    (This may take 30-120 seconds depending on data size)
echo    Mode: %DEMO_ARGS%
if "%INPUT_DATA_PATH%"=="" (
    echo    Data: data\customer_churn.csv
) else (
    echo    Data: %INPUT_DATA_PATH%
)
echo.

pushd "%~dp0"
set "POWERBI_CSV_PATH=%CSV_PATH%"
if "%INPUT_DATA_PATH%"=="" (
    call "%PYTHON_EXE%" "%DEMO_SCRIPT%" %DEMO_ARGS%
) else (
    call "%PYTHON_EXE%" "%DEMO_SCRIPT%" %DEMO_ARGS% --data "%INPUT_DATA_PATH%"
)
set "PYTHON_EXIT_CODE=!ERRORLEVEL!"
popd

if not !PYTHON_EXIT_CODE! equ 0 (
    echo.
    echo ❌ Pipeline failed with exit code !PYTHON_EXIT_CODE!
    echo.
    echo    Check the error messages above for details.
    echo.
    pause
    exit /b 1
)

REM ─ STEP 3: Wait for File to Stabilize ─────────────────────────────────────

echo.
echo ⏳ Waiting for CSV file readiness checks...

REM ─ STEP 4: Verify CSV Created ─────────────────────────────────────────────

if not exist "%CSV_PATH%" (
    echo.
    echo ❌ ERROR: Expected CSV file not found at:
    echo    %CSV_PATH%
    echo.
    echo    Power BI launch is blocked to avoid loading stale data.
    echo.
    pause
    exit /b 1
)

set "CSV_READY=0"
for /L %%i in (1,1,%CSV_READY_RETRIES%) do (
    for %%F in ("%CSV_PATH%") do set "CSV_SIZE=%%~zF"
    if !CSV_SIZE! GEQ %CSV_MIN_BYTES% (
        set "CSV_READY=1"
        goto :csv_ready
    )
    timeout /t %CSV_READY_WAIT_SECONDS% /nobreak >nul
)

:csv_ready
if not "%CSV_READY%"=="1" (
    echo.
    echo ❌ ERROR: CSV readiness check failed.
    echo    File exists but appears incomplete or empty:
    echo    %CSV_PATH%
    echo.
    pause
    exit /b 1
)

powershell -NoProfile -Command "$h=(Get-Content -Path '%CSV_PATH%' -TotalCount 1); if($h -notmatch 'churn_probability' -or $h -notmatch 'churn_band'){ exit 1 }"
if not %ERRORLEVEL%==0 (
    echo.
    echo ❌ ERROR: CSV header validation failed.
    echo    Expected columns not found in first row.
    echo    File: %CSV_PATH%
    echo.
    pause
    exit /b 1
)

REM ─ STEP 5: Open Power BI Dashboard ────────────────────────────────────────

echo.
echo 📊 Opening Power BI Dashboard...
echo    File: %PBIX_PATH%
echo    CSV : %CSV_PATH%
echo.

if not exist "%PBIX_PATH%" (
    echo.
    echo ❌ Power BI file not found at:
    echo    %PBIX_PATH%
    echo.
    echo    👉 Update PBIX_PATH at the top of this batch file:
    echo       (Line 19 in run_system.bat)
    echo.
    echo    The CSV has been generated successfully at:
    echo    %CSV_PATH%
    echo.
    pause
    exit /b 1
)

REM Open the Power BI file
start "" "%PBIX_PATH%"

REM ─ SUCCESS MESSAGE ─────────────────────────────────────────────────────────

echo.
echo ════════════════════════════════════════════════════════════════════════════════
echo   ✅ DEMO COMPLETE
echo ════════════════════════════════════════════════════════════════════════════════
echo.
echo   📊 Power BI should open in a moment...
echo.
echo   📁 Files created:
echo      • outputs\churn_predictions.csv       (Updated predictions)
echo      • outputs\summary_report.txt          (Executive summary)
echo      • outputs\prob_distribution.png       (Probability chart)
echo      • outputs\band_distribution.png       (Risk band chart)
echo      • outputs\shap_importance.png         (Feature importance)
echo.
echo   💡 Next steps:
echo      1. In Power BI, refresh the data source if needed
echo      2. Review the dashboard for new predictions
echo      3. Run this script again anytime to update
echo.
echo ════════════════════════════════════════════════════════════════════════════════
echo.

REM Small delay before closing (gives user time to read the output)
timeout /t 3 /nobreak

exit /b 0
