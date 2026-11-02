@echo off
chcp 65001 >nul
python "%~dp0start_footballmind_defense.py" --page defense %*
if errorlevel 1 pause
