@echo off
cd /d "%~dp0"
echo Starting Data Processor...
python Data_Processor_r0.py
if %errorlevel% neq 0 (
    echo.
    echo Error occurred. Press any key to close...
    pause >nul
) else (
    echo.
    echo Data Processor closed normally.
)