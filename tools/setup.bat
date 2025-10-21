@echo off
setlocal
REM =============================================================
REM  UAV-MVP Setup Script
REM  Creates a Python 3.11 venv, installs dependencies, and
REM  copies AirSim settings to the correct Documents folder.
REM =============================================================

echo =============================================================
echo   UAV-MVP Environment Setup
echo =============================================================

REM 1) Ensure Python is installed and accessible
python --version >nul 2>&1
IF ERRORLEVEL 1 (
  echo [ERROR] Python not found in PATH or not installed.
  echo Please install Python 3.11 (64-bit) from https://www.python.org/downloads/
  echo and rerun this script.
  pause
  exit /b 1
)

REM 2) Create virtual environment
echo.
echo [STEP] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
  echo [ERROR] Failed to create virtual environment.
  pause
  exit /b 1
)

call venv\Scripts\activate

REM 3) Upgrade pip and install dependencies
echo.
echo [STEP] Installing dependencies from requirements.txt...
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  pause
  exit /b 1
)

REM 4) Copy AirSim settings.json to proper Documents path
echo.
echo [STEP] Copying AirSim settings.json to Documents...
set "AIRSIM_DIR=%USERPROFILE%\Documents\AirSim"
if not exist "%AIRSIM_DIR%" mkdir "%AIRSIM_DIR%"
copy /Y "config\settings.json" "%AIRSIM_DIR%\settings.json" >nul

echo.
echo =============================================================
echo   ✅ Setup complete!
echo =============================================================
echo - Your virtual environment is ready: .\venv\
echo - AirSim settings copied to: %AIRSIM_DIR%\settings.json
echo.
echo Next steps:
echo   1. Launch Unreal sim: unreal\Blocks\WindowsNoEditor\Blocks.exe
echo   2. Activate venv:     call venv\Scripts\activate
echo   3. Run tests:
echo        python scripts\connect_test.py
echo        python scripts\collect_lidar.py
echo        python scripts\merge_and_view.py
echo.
echo Tip: If you get connection errors, ensure the sim is running
echo and settings.json is in a non-OneDrive Documents folder.
echo =============================================================

endlocal
pause
