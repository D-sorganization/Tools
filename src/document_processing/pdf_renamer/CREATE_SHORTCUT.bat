@echo off
REM Simple batch file to create PDF Renamer desktop shortcut
REM This will run the PowerShell script to create a professional shortcut

echo.
echo 🚀 PDF Renamer Desktop Shortcut Creator
echo =====================================
echo.

REM Check if PowerShell is available
powershell -Command "Get-Host" >nul 2>&1
if errorlevel 1 (
    echo ❌ PowerShell is not available on this system
    echo Please use the VBScript method instead: double-click create_shortcut.vbs
    pause
    exit /b 1
)

echo Creating desktop shortcut...
echo.

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0create_pdf_renamer_shortcut.ps1"

echo.
echo Done! Check your desktop for the PDF Renamer shortcut.
pause