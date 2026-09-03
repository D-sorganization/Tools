@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src;%~dp0src\shared\python;%PYTHONPATH%
rem Prefer the Python launcher (py -3) when present; fall back to python on PATH.
where py >nul 2>&1 && (start "" py -3 "%~dp0src\rate_of_closure\launch_pyqt6.py") || start "" python "%~dp0src\rate_of_closure\launch_pyqt6.py"
