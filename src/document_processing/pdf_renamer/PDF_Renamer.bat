@echo off
REM PDF Renamer GUI Launcher
REM This batch file launches the PDF Renamer GUI application

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Check if dependencies are installed
echo Checking dependencies...
python verify_installation.py >nul 2>&1
if errorlevel 1 (
    echo.
    echo Some dependencies are missing. Installing...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo.
        echo Please try manually:
        echo   pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo Dependencies installed successfully!
    echo.
)

REM Launch the GUI
echo Starting PDF Renamer...
python launch_gui.py

REM If there was an error, pause to see the message
if errorlevel 1 (
    echo.
    echo An error occurred. Check the error message above.
    pause
)
