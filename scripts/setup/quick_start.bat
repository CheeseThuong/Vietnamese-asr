@echo off
REM Quick start script để test quy trình với VIVOS only

echo ==========================================
echo Quick Start - VIVOS Dataset Only
echo ==========================================

REM 0. Check dependencies first
echo.
echo [0/5] Checking dependencies...
python scripts\check_dependencies.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================
    echo Please install dependencies first!
    echo ==========================================
    echo Run: pip install -r requirements.txt
    echo ==========================================
    pause
    exit /b 1
)

echo.
echo Press any key to continue...
pause >nul

REM 1. Check audio files
echo.
echo [1/5] Checking audio files...
python src\data\normalize_audio.py --analyze-only
if %ERRORLEVEL% NEQ 0 (
    echo Error: Audio analysis failed!
    pause
    exit /b 1
)

echo.
echo Press any key to continue with normalization (or Ctrl+C to skip)...
pause >nul

REM 2. Normalize audio (optional - to new directory)
echo.
echo [2/5] Normalizing audio files...
echo This will create normalized copies in Data/vivos_normalized
python src\data\normalize_audio.py --output_dir Data/vivos_normalized
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Audio normalization had issues, continuing...
)

REM 3. Prepare dataset (VIVOS only)
echo.
echo [3/5] Preparing VIVOS dataset...
python prepare_vivos.py --check-audio
if %ERRORLEVEL% NEQ 0 (
    echo Error: Dataset preparation failed!
    pause
    exit /b 1
)

REM 4. Create vocabulary and processor
echo.
echo [4/5] Creating vocabulary and processor...
python src\data\preprocessing.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Data preprocessing failed!
    pause
    exit /b 1
)

REM 5. Ready to train
echo.
echo [5/5] Setup complete!
echo ==========================================
echo Ready to train!
echo ==========================================
echo.
echo Next steps:
echo   1. Review the dataset summary in processed_data_vivos/
echo   2. Start training: python train.py
echo   3. Or run demo on a sample audio: python src\utils\demo.py [audio_file]
echo   4. Start API server: python run_server.py
echo.
echo ==========================================
pause
