@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo ОБУЧЕНИЕ 5 РАЗНЫХ АРХИТЕКТУР
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ОШИБКА] Виртуальное окружение не найдено
    echo Запустите 01_INSTALL.cmd сначала
    pause
    exit /b 2
)

echo Выберите режим обучения:
echo.
echo 1. Обучить все 5 моделей (U-Net, ResNet+FPN, DeepLabV3, PSPNet, SegNet)
echo 2. Обучить только PSPNet и SegNet (4 и 5 модели)
echo 3. Обучить только одну модель
echo 4. Проверить статус моделей
echo.

set /p choice="Ваш выбор (1-4): "

if "%choice%"=="1" goto train_all
if "%choice%"=="2" goto train_missing
if "%choice%"=="3" goto train_one
if "%choice%"=="4" goto check_status

:check_status
echo.
echo Проверка статуса моделей...
.venv\Scripts\python.exe -c "from pathlib import Path; import json; models=['unet','resnet_fpn','deeplabv3','pspnet','segnet']; names=['U-Net','ResNet+FPN','DeepLabV3','PSPNet','SegNet']; [print(f'{names[i]}: {\"✅ обучена\" if (Path(f\"checkpoints/{m}/summary.json\").exists() and json.load(open(f\"checkpoints/{m}/summary.json\"))[\"best_val_dice\"]>0.3) else \"❌ не обучена\"}') for i,m in enumerate(models)]"
pause
exit /b 0

:train_all
echo.
echo Обучение всех 5 моделей...
echo Это займет много времени (2-4 часа)
echo.
pause
.venv\Scripts\python.exe train_all.py
pause
exit /b 0

:train_missing
echo.
echo Обучение только PSPNet и SegNet...
echo Это займет ~1-2 часа
echo.
pause
.venv\Scripts\python.exe train_missing.py
pause
exit /b 0

:train_one
echo.
echo Доступные модели:
echo 1. unet - U-Net
echo 2. resnet_fpn - ResNet50 + FPN
echo 3. deeplabv3 - DeepLabV3
echo 4. pspnet - PSPNet
echo 5. segnet - SegNet
echo.
set /p model_choice="Выберите модель (1-5): "
if "%model_choice%"=="1" set model_name=unet
if "%model_choice%"=="2" set model_name=resnet_fpn
if "%model_choice%"=="3" set model_name=deeplabv3
if "%model_choice%"=="4" set model_name=pspnet
if "%model_choice%"=="5" set model_name=segnet

echo.
echo Обучение: %model_name%
.venv\Scripts\python.exe train.py --model %model_name% --epochs 100 --batch-size 32 --size 64 --threads 8
pause
exit /b 0