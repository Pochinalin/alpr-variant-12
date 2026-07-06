@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo УСТАНОВКА ALPR СИСТЕМЫ
echo ============================================================
echo.

set "PYTHON_EXE="
set "PYTHON_ARGS="

REM Поиск Python
where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.12 3.11 3.10) do (
        if not defined PYTHON_EXE (
            py -%%V -c "import struct,sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) and struct.calcsize('P')*8 == 64 else 1)" >nul 2>nul
            if not errorlevel 1 (
                set "PYTHON_EXE=py"
                set "PYTHON_ARGS=-%%V"
            )
        )
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import struct,sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,13) and struct.calcsize('P')*8 == 64 else 1)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo [ОШИБКА] Python 3.10-3.12 x64 не найден.
    echo Установите Python с https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Найден Python: %PYTHON_EXE% %PYTHON_ARGS%
echo.

REM Создание виртуального окружения
if not exist ".venv\Scripts\python.exe" (
    echo Создание виртуального окружения...
    "%PYTHON_EXE%" %PYTHON_ARGS% -m venv ".venv"
    if errorlevel 1 goto failed
    echo [OK] Виртуальное окружение создано
) else (
    echo [OK] Виртуальное окружение уже существует
)

echo.

REM Установка пакетов напрямую без активации
echo Установка Python пакетов...
echo.

echo 1. Обновление pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo.

echo 2. Установка PyTorch...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
echo.

echo 3. Установка основных пакетов...
".venv\Scripts\python.exe" -m pip install numpy opencv-python Pillow
if errorlevel 1 goto failed
echo.

echo 4. Установка Streamlit и pandas...
".venv\Scripts\python.exe" -m pip install streamlit pandas
if errorlevel 1 goto failed
echo.

echo 5. Установка OCR...
".venv\Scripts\python.exe" -m pip install pytesseract rapidocr-onnxruntime onnxruntime
if errorlevel 1 goto failed
echo.

echo 6. Установка дополнительных пакетов...
".venv\Scripts\python.exe" -m pip install scipy matplotlib scikit-learn
if errorlevel 1 goto failed
echo.

echo 7. Проверка установки...
".venv\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__)"
".venv\Scripts\python.exe" -c "import cv2; print('OpenCV:', cv2.__version__)"
".venv\Scripts\python.exe" -c "import streamlit; print('Streamlit:', streamlit.__version__)"
".venv\Scripts\python.exe" -c "import pytesseract; print('Tesseract OK')"

echo.
echo ============================================================
echo УСТАНОВКА ЗАВЕРШЕНА
echo ============================================================
echo.
echo Для запуска приложения выполните: 03_RUN_APP.cmd
echo.
pause
exit /b 0

:failed
echo.
echo [ОШИБКА] Установка не удалась
echo.
pause
exit /b 1