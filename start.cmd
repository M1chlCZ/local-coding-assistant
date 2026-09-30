@echo off
cd /d "%~dp0"
set "PYTHON=python"
where py >nul 2>&1
if not errorlevel 1 set "PYTHON=py -3"
%PYTHON% launcher.py setup
if errorlevel 1 goto end
%PYTHON% launcher.py serve --open
:end
pause
