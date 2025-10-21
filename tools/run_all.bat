@echo off
setlocal
echo Ensure Blocks.exe is already running before continuing.
pause
call venv\Scripts\activate
python scripts\collect_lidar.py
python scripts\merge_and_view.py
endlocal
pause
