@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo ЗАПУСК ALPR ПРИЛОЖЕНИЯ
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ОШИБКА] Виртуальное окружение не найдено
    echo Запустите 01_INSTALL.cmd сначала
    pause
    exit /b 2
)

echo Запуск Streamlit приложения...
echo Откройте в браузере: http://localhost:8501
echo.
echo Для остановки нажмите Ctrl+C
echo.

set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
".venv\Scripts\python.exe" -m pip install plotly -q
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.address localhost

pause