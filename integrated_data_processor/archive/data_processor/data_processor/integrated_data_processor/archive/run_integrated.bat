@echo off
echo Starting Integrated Data Processor...
echo.
echo This version includes:
echo - Original CSV processing functionality
echo - Format converter with support for multiple file formats
echo - Parquet file analyzer
echo - All existing plotting and analysis features
echo.

cd /d "%~dp0"
python launch_integrated.py

if errorlevel 1 (
    echo.
    echo Error running the application.
    echo Please ensure Python and all dependencies are installed.
    echo.
    pause
)
