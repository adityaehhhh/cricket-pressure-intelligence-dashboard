@echo off
title Cricket Pressure Performance Intelligence Dashboard
echo ======================================================================
echo STARTING CRICKET PRESSURE INTELLIGENCE DASHBOARD
echo ======================================================================
echo Loading dataset cache across 7 competitions and 1,635 players...
echo.
python "%~dp0app.py"
pause
