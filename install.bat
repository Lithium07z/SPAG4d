@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==================================================
echo   SPAG-4D Automatic Installer (Embedded Python)
echo ==================================================
echo.

set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYTHON_ZIP=python_embed.zip"
set "PYTHON_DIR=python_embed"

:: Check if already installed
if exist "%PYTHON_DIR%\python.exe" (
    echo [INFO] Embedded Python already found in "%PYTHON_DIR%".
    goto :InstallDeps
)

echo [1/4] Downloading Python 3.11 Embedded...
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'"
if not exist "%PYTHON_ZIP%" (
    echo [ERROR] Failed to download Python.
    exit /b 1
)

echo [2/4] Extracting Python...
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force"
del "%PYTHON_ZIP%"

echo [3/4] Configuring Embedded Python...
:: Uncomment the import site line in the ._pth file to allow pip to work
set "PTH_FILE=%PYTHON_DIR%\python311._pth"
powershell -Command "(Get-Content '%PTH_FILE%') -replace '#import site', 'import site' | Set-Content '%PTH_FILE%'"

echo [4/4] Installing pip...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py'"
"%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py"
del "%PYTHON_DIR%\get-pip.py"

:InstallDeps
echo.
echo ==================================================
echo   Installing Dependencies...
echo ==================================================

set "PIP=%PYTHON_DIR%\Scripts\pip.exe"
if not exist "!PIP!" (
    :: Fallback if pip goes to the root folder
    set "PIP=%PYTHON_DIR%\python.exe -m pip"
)

echo.
echo [1/4] Installing PyTorch (CUDA 12.1)...
!PIP! install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/4] Installing SPAG-4D Core Requirements...
!PIP! install -r requirements.txt
!PIP! install -e ".[server,download]"

echo.
echo [3/4] Installing Optional SHARP Model...
!PIP! install git+https://github.com/apple/ml-sharp.git

echo.
echo [4/4] Installing Optional Depth Anything V3 Model...
!PIP! install -e "git+https://github.com/ByteDance-Seed/depth-anything-3.git#egg=depth_anything_3"

echo.
echo ==================================================
echo   Installation Complete!
echo   You can now double-click 'run.bat' to start!
echo ==================================================
echo Done
