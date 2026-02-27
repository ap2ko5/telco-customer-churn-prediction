@echo off
echo.
echo =======================================================
echo   CHURN INTELLIGENCE - TEST RUNNER
echo =======================================================
echo.
echo Running all automated unit tests (PyTest)...
echo.
python -m pytest tests/ -v
echo.
echo =======================================================
echo   TEST RUN COMPLETE
echo =======================================================
echo.
pause
