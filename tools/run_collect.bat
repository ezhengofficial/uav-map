@echo off
setlocal
call venv\Scripts\activate
python scripts\collect_lidar.py
endlocal
pause
