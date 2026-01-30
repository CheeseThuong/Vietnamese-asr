@echo off
REM Script để chạy toàn bộ pipeline từ đầu đến cuối (Windows version)

echo ==========================================
echo Vietnamese ASR - Complete Pipeline
echo ==========================================

REM 1. Prepare dataset
echo.
echo [1/7] Preparing dataset...
python prepare_full_dataset.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Dataset preparation failed!
    exit /b 1
)

REM 2. Data preprocessing
echo.
echo [2/7] Preprocessing data...
python src\data\preprocessing.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Data preprocessing failed!
    exit /b 1
)

REM 3. Train model
echo.
echo [3/7] Training Wav2Vec2 model...
python train.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Training failed!
    exit /b 1
)

REM 4. Build language model
echo.
echo [4/7] Building language model...
python src\training\language_model.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Language model building failed, continuing...
)

REM 5. Evaluate
echo.
echo [5/7] Evaluating model...
python run_evaluation.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Evaluation failed!
    exit /b 1
)

REM 6. Optimize
echo.
echo [6/7] Optimizing model...
python src\utils\optimization.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Optimization failed, continuing...
)

REM 7. Done
echo.
echo [7/7] Pipeline complete!
echo ==========================================
echo All steps completed successfully!
echo To start the web server, run:
echo   python run_server.py
echo ==========================================
pause
