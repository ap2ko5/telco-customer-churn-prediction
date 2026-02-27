@echo off
echo.
echo =======================================================
echo   CHURN INTELLIGENCE - PIPELINE TRAINER
echo =======================================================
echo.
echo Re-training the entire 14-step machine learning pipeline...
echo.
python src/train_pipeline.py
echo.
echo =======================================================
echo   TRAINING COMPLETE
echo =======================================================
echo.
pause
