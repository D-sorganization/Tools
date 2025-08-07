@echo off
echo Starting File Size Analyzer...
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
python -c "import PyQt6, pandas, pyarrow" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install PyQt6 pandas pyarrow numpy openpyxl h5py scipy joblib
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Launch the file analyzer
echo Launching File Size Analyzer...
python file_size_analyzer.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Check the console output for details
    pause
) 