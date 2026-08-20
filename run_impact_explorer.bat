@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src;%~dp0src\shared\python;%PYTHONPATH%
start "" "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" "%~dp0src\rate_of_closure\launch_pyqt6.py"
