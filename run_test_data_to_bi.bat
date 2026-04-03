@echo off
REM One-click run using test dataset + auto open Power BI
setlocal

cd /d "%~dp0"

SET "TEST_DATA=%~dp0data\test_dataset.csv"

if not exist "%TEST_DATA%" (
  echo [ERROR] Test dataset not found: %TEST_DATA%
  pause
  exit /b 1
)

call "%~dp0run_system.bat" "%TEST_DATA%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Test-data run failed. Exit code: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

exit /b 0
