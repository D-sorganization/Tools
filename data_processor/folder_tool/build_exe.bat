@echo off
echo Building Folder Fix executable...
echo.

REM Clean previous build
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Build executable
echo Building executable with PyInstaller...
pyinstaller FolderFix.spec --clean

REM Check if build succeeded
if exist "dist\FolderFix.exe" (
    echo.
    echo ✓ Build successful! 
    echo ✓ Executable: dist\FolderFix.exe
    echo ✓ Launch script: Launch_FolderFix.bat
    echo.
    pause
) else (
    echo.
    echo ✗ Build failed! Check the output above for errors.
    echo.
    pause
)
