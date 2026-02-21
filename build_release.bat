@echo off
REM ============================================
REM  SPAG-4D Release Builder
REM  Packages the venv + app into a release folder
REM ============================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "RELEASE_DIR=dist\SPAG-4D-v0.2.0-win-x64"
set "VERSION=0.2.0"

echo.
echo ========================================
echo   SPAG-4D Release Builder v%VERSION%
echo ========================================
echo.

REM Check venv exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Run: python -m venv .venv
    echo Then install deps: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Clean previous build
if exist "%RELEASE_DIR%" (
    echo Cleaning previous build...
    rmdir /s /q "%RELEASE_DIR%"
)
mkdir "%RELEASE_DIR%"

echo [1/6] Copying application code...
xcopy /E /I /Q /Y "spag4d" "%RELEASE_DIR%\spag4d" >nul
copy /Y "api.py" "%RELEASE_DIR%\" >nul
copy /Y "pyproject.toml" "%RELEASE_DIR%\" >nul
copy /Y "requirements.txt" "%RELEASE_DIR%\" >nul
copy /Y "README.md" "%RELEASE_DIR%\" >nul

echo [2/6] Copying static web files...
xcopy /E /I /Q /Y "static" "%RELEASE_DIR%\static" >nul

echo [3/6] Copying ml-sharp submodule...
if exist "ml-sharp" (
    xcopy /E /I /Q /Y "ml-sharp" "%RELEASE_DIR%\ml-sharp" >nul
)

echo [4/6] Copying Python virtual environment (this takes a while)...
xcopy /E /I /Q /Y ".venv" "%RELEASE_DIR%\.venv" >nul

echo [5/6] Cleaning up unnecessary files from venv...
REM Remove __pycache__ directories
for /d /r "%RELEASE_DIR%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" 2>nul
)
REM Remove .pyc files
del /s /q "%RELEASE_DIR%\*.pyc" 2>nul >nul
REM Remove pip/setuptools caches
if exist "%RELEASE_DIR%\.venv\Lib\site-packages\pip" rmdir /s /q "%RELEASE_DIR%\.venv\Lib\site-packages\pip" 2>nul
if exist "%RELEASE_DIR%\.venv\Lib\site-packages\setuptools" rmdir /s /q "%RELEASE_DIR%\.venv\Lib\site-packages\setuptools" 2>nul
REM Remove .dist-info except critical ones
REM Remove test directories from site-packages
for /d %%d in ("%RELEASE_DIR%\.venv\Lib\site-packages\*test*") do (
    rmdir /s /q "%%d" 2>nul
)

echo [6/6] Copying launcher and docs...
copy /Y "SPAG4D.bat" "%RELEASE_DIR%\" >nul 2>nul
copy /Y "INSTALL.md" "%RELEASE_DIR%\" >nul 2>nul

echo.
echo ========================================
echo   Build complete!
echo ========================================
echo.
echo Output: %RELEASE_DIR%
echo.

REM Show size
for /f "tokens=3" %%a in ('dir /s "%RELEASE_DIR%" ^| findstr "File(s)"') do (
    set "SIZE=%%a"
)
echo Total size: ~%SIZE% bytes
echo.
echo Next steps:
echo   1. Test: cd %RELEASE_DIR% ^& SPAG4D.bat
echo   2. Zip:  Compress %RELEASE_DIR% for GitHub release
echo.

pause
