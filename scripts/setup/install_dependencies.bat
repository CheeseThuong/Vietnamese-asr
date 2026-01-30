@echo off
REM Script đơn giản để cài đặt dependencies

echo ==========================================
echo Installing Dependencies
echo ==========================================
echo.
echo This will install all required packages
echo from requirements.txt
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo Installing packages...
echo ==========================================

pip install -r requirements-core.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo Installation complete!
    echo ==========================================
    echo.
    echo Running dependency check...
    python check_dependencies.py
    echo.
    echo ==========================================
    echo You can now run: quick_start.bat
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo Installation failed!
    echo ==========================================
    echo.
    echo Please check your internet connection and try again.
    echo Or install packages manually:
    echo   pip install torch torchaudio transformers datasets
)

pause
