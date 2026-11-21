@echo off
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-model.ps1" -Admin -OpenBrowser
pause
