@echo off
setlocal
call venv\Scripts\activate
python scripts\merge_and_view.py
endlocal
pause
