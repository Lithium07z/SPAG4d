@echo off
cd /d "%~dp0"

echo ========================================
echo   SPAG-4D Launcher
echo ========================================
echo.

set "PYTHON_EXE=python_embed\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Embedded Python not found.
    echo Please run 'install.bat' first!
    pause
    exit /b 1
)

REM Kill any existing process on port 7860
echo Checking for existing servers on port 7860...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING 2^>nul') do (
    echo Killing existing process PID: %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo Starting web server on http://localhost:7860
echo Press Ctrl+C to stop
echo.

REM Open browser after short delay
start "" /min cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:7860"

"%PYTHON_EXE%" -m spag4d.cli serve --port 7860

pause
