@echo off
title Penguin Detector
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONNOUSERSITE=1
set PYTHONDONTWRITEBYTECODE=1
"%~dp0python\python.exe" "%~dp0launcher.py"
if errorlevel 1 (
    echo.
    echo Penguin Detector stopped with an error. See the messages above.
    pause
)
