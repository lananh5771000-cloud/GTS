@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

echo [1/3] Bien dich toan bo project...
%PY% -m compileall -q . || goto :fail
echo [2/3] Chay toan bo pytest, gom 1550 ca ngau nhien co seed...
call pytest -q || goto :fail
echo [3/3] Kiem tra kha nang chay offline...
%PY% KIEM_TRA_MAY.py || goto :fail
echo.
echo TAT CA KIEM THU DA DAT.
goto :end

:fail
echo.
echo KIEM THU THAT BAI. Xem thong bao loi o phia tren.
pause
exit /b 1

:end
echo.
pause
exit /b 0
