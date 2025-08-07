@echo off
echo Starting Parquet File Analyzer...
echo.
REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)
REM Check if required packages are installed
echo Checking dependencies...
python -c "import PyQt6, pyarrow" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install PyQt6 pyarrow
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)
REM Launch the parquet analyzer
echo Launching Parquet File Analyzer...
python Parquet_File_Analyzer_r0.py
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Check the console output for details
    pause
) 