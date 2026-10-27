@echo off
echo ============================================================
echo  ECG MI Detection System - Starting Backend
echo ============================================================
cd /d "%~dp0"
echo Starting FastAPI backend on http://localhost:8000 ...
echo Open frontend\index.html in your browser after this starts.
echo Press Ctrl+C to stop.
echo.
python run.py
