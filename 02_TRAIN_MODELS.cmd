@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo ОБУЧЕНИЕ ВСЕХ 5 МОДЕЛЕЙ
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ОШИБКА] Виртуальное окружение не найдено
    echo Запустите 01_INSTALL.cmd сначала
    pause
    exit /b 2
)

echo Модели для обучения:
echo  1. U-Net
echo  2. ResNet50 + FPN
echo  3. DeepLabV3
echo  4. PSPNet
echo  5. SegNet
echo.
echo Это займет много времени (2-4 часа)
echo.

set /p confirm="Начать обучение всех 5 моделей? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Отмена
    pause
    exit /b 0
)

echo.
echo ============================================================
echo НАЧАЛО ОБУЧЕНИЯ
echo ============================================================
echo.

.venv\Scripts\python.exe train_all.py

if errorlevel 1 (
    echo.
    echo [ОШИБКА] Обучение не удалось
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ОБУЧЕНИЕ ЗАВЕРШЕНО!
echo ============================================================
echo.
echo Результаты сохранены в папке checkpoints/
echo.
echo Запустите 03_RUN_APP.cmd для просмотра результатов
echo.
pause