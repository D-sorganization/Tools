@echo off
echo Building Folder Packer Executable...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Run the build script
python build_exe.py

echo.
echo Build process completed.
if exist "dist\FolderPacker.exe" (
    echo.
    echo ✅ SUCCESS: Executable created at dist\FolderPacker.exe
    echo.
    echo Would you like to run the executable now? (y/n)
    set /p choice=
    if /i "%choice%"=="y" (
        start "" "dist\FolderPacker.exe"
    )
) else (
    echo.
    echo ❌ Build failed. Check the output above for errors.
)

pause
