@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

echo [1/4] Bien dich tat ca tep Python...
%PY% -m compileall -q . || goto :fail
echo [2/4] Chay pytest...
call pytest -q || goto :fail
echo [3/4] Kiem tra import va thuat toan mau...
%PY% KIEM_TRA_MAY.py || goto :fail
echo [4/4] Kiem tra rieng bo dinh dang va bo nhap...
call pytest -q tests/test_exam_format_and_edges.py tests/test_input_utils.py || goto :fail
echo.
echo TAT CA KIEM THU DA DAT.
goto :end

:fail
echo.
echo KIEM THU THAT BAI. Xem thong bao loi o phia tren.
:end
pause
