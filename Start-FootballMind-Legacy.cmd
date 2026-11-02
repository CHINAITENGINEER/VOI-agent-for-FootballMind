@echo off
chcp 65001 >nul
python "%~dp0start_footballmind.py" %*
if errorlevel 1 pause
