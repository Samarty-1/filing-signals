@echo off
rem Phase 4, end to end: fetch prices within the Tiingo free tier, then run the
rem pre-registered backtest once and refresh the explorer's JSON. Resumable:
rem rerun it if the month's symbol budget runs out. Start it detached with
rem   powershell Start-Process -WindowStyle Hidden cmd.exe -ArgumentList '/c','scripts\run_phase4.cmd'
cd /d %~dp0..
if "%TIINGO_API_KEY%"=="" (echo set TIINGO_API_KEY first & exit /b 1)
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
if not exist data\prices mkdir data\prices
.venv\Scripts\filing-signals tickers >> data\prices\phase4.log 2>&1
.venv\Scripts\filing-signals prices >> data\prices\phase4.log 2>&1 || exit /b 1
findstr /c:"'remaining': 0" data\prices\phase4.log >nul || (echo prices incomplete; rerun next month >> data\prices\phase4.log & exit /b 2)
.venv\Scripts\filing-signals backtest >> data\prices\phase4.log 2>&1 || exit /b 1
.venv\Scripts\filing-signals export >> data\prices\phase4.log 2>&1
echo phase4 done >> data\prices\phase4.log
