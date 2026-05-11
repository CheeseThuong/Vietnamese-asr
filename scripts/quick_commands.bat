@echo off
REM Quick commands for common tasks

echo ==========================================
echo Vietnamese ASR - Quick Commands
echo ==========================================
echo.
echo Choose an option:
echo   1. Check dependencies
echo   2. Prepare VIVOS dataset
echo   3. Prepare full dataset (VIVOS + VinBigData)
echo   4. Train model (LOCAL - CPU)
echo   5. Evaluate model
echo   6. Start API server
echo   7. Run demo
echo   8. Profile code
echo   9. Open Google Colab
echo   10. Full pipeline
echo   0. Exit
echo.

set /p choice="Enter choice (0-10): "

if "%choice%"=="1" (
    python scripts\check_dependencies.py
) else if "%choice%"=="2" (
    python prepare_vivos.py
) else if "%choice%"=="3" (
    python prepare_full_dataset.py
) else if "%choice%"=="4" (
    echo WARNING: Training on CPU will take ~6 days!
    echo Consider using Kaggle (option 9) for GPU training.
    set /p confirm="Continue anyway? (y/n): "
    if /i "%confirm%"=="y" python train.py
) else if "%choice%"=="5" (
    python run_evaluation.py
) else if "%choice%"=="6" (
    python run_server.py
) else if "%choice%"=="7" (
    set /p audio="Enter audio file path: "
    python src\utils\demo.py %audio%
) else if "%choice%"=="8" (
    python scripts\profiling\flamegraph_guide.py
) else if "%choice%"=="9" (
    echo.
    echo Opening Google Colab...
    echo.
    echo 1. Upload colab_train.ipynb to https://colab.research.google.com
    echo 2. Runtime -^> Change runtime type -^> GPU
    echo 3. Read instructions at: colab_setup.md
    echo.
    start https://colab.research.google.com
    start colab_setup.md
    pause
) else if "%choice%"=="10" (
    call scripts\run_pipeline.bat
) else if "%choice%"=="0" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice!
)

pause
