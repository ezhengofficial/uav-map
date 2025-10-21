@echo off
setlocal
REM =============================================================
REM  UAV-MVP Setup Script (Windows 10/11)
REM  - Creates a Python 3.11 venv (via py launcher if available)
REM  - Installs requirements in correct order
REM  - Copies AirSim settings.json to Documents\AirSim
REM =============================================================

REM Move to repo root even if started from tools\
pushd "%~dp0\.."

echo =============================================================
echo   UAV-MVP Environment Setup
echo =============================================================

REM 1) Create virtual environment (prefer Python 3.11 via launcher)
echo.
echo [STEP] Creating virtual environment...

REM Try with py -3.11 first
py -3.11 -m venv venv >nul 2>&1
IF NOT EXIST "venv\Scripts\python.exe" (
  REM Fall back to whatever "python" is
  python --version >nul 2>&1
  IF ERRORLEVEL 1 (
    echo [ERROR] Python not found in PATH or not installed.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Then run this script again.
    popd
    pause
    exit /b 1
  )
  python -m venv venv
)

IF NOT EXIST "venv\Scripts\python.exe" (
  echo [ERROR] Failed to create virtual environment.
  popd
  pause
  exit /b 1
)

call "venv\Scripts\activate"

REM 2) Upgrade pip toolchain
echo.
echo [STEP] Upgrading pip, setuptools, wheel...
python -m ensurepip --upgrade >nul 2>&1
python -m pip install --upgrade pip setuptools wheel
IF ERRORLEVEL 1 (
  echo [ERROR] Failed to upgrade pip toolchain.
  popd
  pause
  exit /b 1
)

REM 3) Install dependencies
echo.
echo [STEP] Installing dependencies from requirements.txt...
IF NOT EXIST "requirements.txt" (
  echo [ERROR] requirements.txt not found in %CD%
  popd
  pause
  exit /b 1
)
pip install -r requirements.txt
IF ERRORLEVEL 1 (
  echo [ERROR] Dependency installation failed.
  popd
  pause
  exit /b 1
)

REM 4) Copy AirSim settings.json
echo.
echo [STEP] Copying AirSim settings.json to Documents\AirSim...
set "AIRSIM_DIR=%USERPROFILE%\Documents\AirSim"
IF NOT EXIST "%AIRSIM_DIR%" mkdir "%AIRSIM_DIR%" >nul 2>&1
IF EXIST "config\settings.json" (
  copy /Y "config\settings.json" "%AIRSIM_DIR%\settings.json" >nul
) ELSE (
  echo [WARN] config\settings.json not found. Skipping copy.
)

echo.
echo =============================================================
echo   Setup complete!
echo =============================================================
echo - Virtual environment: .\venv
echo - AirSim settings at:  %AIRSIM_DIR%\settings.json
echo.
echo Next steps:
echo   1. Launch sim:  unreal\Blocks\WindowsNoEditor\Blocks.exe
echo   2. Activate venv:  call venv\Scripts\activate
echo   3. Run tests:
echo        python scripts\connect_test.py
echo        python scripts\collect_lidar.py
echo        python scripts\merge_and_view.py
echo.
echo Tip: If Documents is synced to OneDrive and causes issues,
echo      move settings.json into a non-OneDrive Documents path.
echo =============================================================

popd
endlocal
pause
