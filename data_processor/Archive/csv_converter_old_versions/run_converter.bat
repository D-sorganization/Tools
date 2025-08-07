@echo off
echo CSV to Parquet Bulk Converter
echo =============================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import PyQt6, pandas, pyarrow" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install required packages
        pause
        exit /b 1
    )
)

echo Starting CSV to Parquet Converter...
python csv_to_parquet_converter.py

if errorlevel 1 (
    echo.
    echo Error: Application failed to start
    pause
    exit /b 1
)

echo.
echo Application closed.
pause
