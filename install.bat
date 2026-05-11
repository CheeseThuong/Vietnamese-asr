@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: =====================================================================
:: VietASR Pro - Auto Installer (Windows)
:: Cài đặt tự động: Python check, venv, dependencies, model, Flask
:: =====================================================================

title VietASR Pro - Auto Installer

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║         VietASR Pro - Cai Dat Tu Dong (Windows)         ║
echo  ║         Vietnamese ASR - Automatic Installer             ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: =====================================================================
:: STEP 1: Kiem tra Python
:: =====================================================================
echo [1/7] Kiem tra Python...

set PYTHON_CMD=
set PYTHON_VERSION=

:: Tìm Python trong PATH
for %%P in (python python3 py) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2 delims= " %%V in ('%%P --version 2^>^&1') do (
            set PYTHON_CMD=%%P
            set PYTHON_VERSION=%%V
        )
        if defined PYTHON_CMD goto :python_found
    )
)

:: Python không tìm thấy
echo.
echo  [LOI] Khong tim thay Python!
echo.
echo  Vui long cai dat Python 3.9 tro len tu:
echo    https://www.python.org/downloads/
echo.
echo  QUAN TRONG: Tick vao "Add Python to PATH" khi cai dat!
echo.
pause
exit /b 1

:python_found
echo   Python: %PYTHON_VERSION% (%PYTHON_CMD%)

:: Kiểm tra version >= 3.9
for /f "tokens=1,2 delims=." %%A in ("%PYTHON_VERSION%") do (
    set MAJOR=%%A
    set MINOR=%%B
)

if %MAJOR% lss 3 (
    echo  [LOI] Can Python 3.9+, ban dang dung %PYTHON_VERSION%
    pause
    exit /b 1
)
if %MAJOR% equ 3 if %MINOR% lss 9 (
    echo  [LOI] Can Python 3.9+, ban dang dung %PYTHON_VERSION%
    pause
    exit /b 1
)

echo   [OK] Python %PYTHON_VERSION% phu hop!

:: =====================================================================
:: STEP 2: Tao Virtual Environment
:: =====================================================================
echo.
echo [2/7] Tao virtual environment...

set VENV_DIR=%~dp0.venv

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo   [OK] venv da ton tai, su dung lai.
) else (
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo  [LOI] Khong tao duoc virtual environment!
        pause
        exit /b 1
    )
    echo   [OK] Da tao venv tai: %VENV_DIR%
)

:: =====================================================================
:: STEP 3: Activate venv
:: =====================================================================
echo.
echo [3/7] Kich hoat virtual environment...

call "%VENV_DIR%\Scripts\activate.bat"
echo   [OK] venv da kich hoat

:: Upgrade pip
python -m pip install --upgrade pip --quiet >nul 2>&1
echo   [OK] pip da cap nhat

:: =====================================================================
:: STEP 4: Cai dat dependencies
:: =====================================================================
echo.
echo [4/7] Cai dat thu vien (co the mat vai phut)...

if exist "%~dp0requirements.txt" (
    pip install -r "%~dp0requirements.txt" --quiet
    if errorlevel 1 (
        echo.
        echo  [CANH BAO] Mot so thu vien co the khong cai duoc.
        echo  Dang thu cai tung thu vien...
        for /f "usebackq eol=# tokens=*" %%L in ("%~dp0requirements.txt") do (
            pip install %%L --quiet >nul 2>&1
        )
    )
    echo   [OK] Da cai dat dependencies
) else (
    echo  [CANH BAO] Khong tim thay requirements.txt
    echo  Dang cai dat thu vien co ban...
    pip install torch torchaudio transformers datasets librosa soundfile flask flask-cors pydub jiwer --quiet
    echo   [OK] Da cai dat thu vien co ban
)

:: Cài thêm Flask nếu thiếu (vì requirements.txt gốc có FastAPI)
pip install flask flask-cors pydub --quiet >nul 2>&1

:: =====================================================================
:: STEP 5: Setup ffmpeg
:: =====================================================================
echo.
echo [5/7] Kiem tra ffmpeg...

set FFMPEG_TOOLS_DIR=%~dp0tools\ffmpeg\bin
set FFMPEG_TOOLS_EXE=%FFMPEG_TOOLS_DIR%\ffmpeg.exe

:: Check local tools/ first
if exist "%FFMPEG_TOOLS_EXE%" (
    echo   [OK] ffmpeg da co tai: %FFMPEG_TOOLS_EXE%
    goto :ffmpeg_done
)

:: Check system PATH
where ffmpeg >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] ffmpeg da co trong system PATH
    goto :ffmpeg_done
)

echo   ffmpeg chua cai dat. Dang tai tu dong...
echo.

:: Create tools directory
if not exist "%~dp0tools" mkdir "%~dp0tools"

set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
set FFMPEG_ZIP=%~dp0tools\ffmpeg.zip

powershell -NoProfile -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
    "$ProgressPreference = 'SilentlyContinue'; " ^
    "try { " ^
    "    Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing; " ^
    "    Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%~dp0tools' -Force; " ^
    "    Remove-Item '%FFMPEG_ZIP%' -Force; " ^
    "    Write-Host '  [OK] Tai va giai nen thanh cong!' " ^
    "} catch { " ^
    "    Write-Host '  [CANH BAO] Khong tai duoc ffmpeg: ' $_.Exception.Message; " ^
    "}"

:: Rename extracted folder
if exist "%~dp0tools\ffmpeg" rmdir /s /q "%~dp0tools\ffmpeg" 2>nul
for /d %%D in ("%~dp0tools\ffmpeg-*") do (
    rename "%%D" "ffmpeg"
)

if exist "%FFMPEG_TOOLS_EXE%" (
    echo   [OK] ffmpeg da san sang!
) else (
    echo   [CANH BAO] Khong cai duoc ffmpeg tu dong.
    echo   Upload file MP3/M4A co the khong hoat dong.
    echo   Chay setup_ffmpeg.bat de thu lai.
)

:ffmpeg_done

:: =====================================================================
:: STEP 6: Kiem tra model
:: =====================================================================
echo.
echo [6/7] Kiem tra model ASR...

set MODEL_DIR=%~dp0final_model

if exist "%MODEL_DIR%" (
    echo   [OK] Da tim thay model tai: %MODEL_DIR%
) else (
    echo.
    echo   Khong tim thay model local tai: %MODEL_DIR%
    echo   He thong se tu dong tai model tu Hugging Face khi chay.
    echo.
    echo   Neu ban co model rieng, hay copy vao thu muc:
    echo     %MODEL_DIR%
    echo.
    set /p USER_MODEL="  Nhap duong dan model (Enter de bo qua): "
    if defined USER_MODEL (
        if exist "!USER_MODEL!" (
            xcopy /E /I /Q "!USER_MODEL!" "%MODEL_DIR%" >nul
            echo   [OK] Da copy model tu: !USER_MODEL!
        ) else (
            echo   [CANH BAO] Duong dan khong ton tai, bo qua.
        )
    ) else (
        echo   [OK] Se su dung model tu Hugging Face
    )
)

:: =====================================================================
:: STEP 7: Khoi dong server
:: =====================================================================
echo.
echo [7/7] Khoi dong VietASR Pro...
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                 CAI DAT HOAN TAT!                       ║
echo  ║                                                          ║
echo  ║  Server se khoi dong tai: http://localhost:5000          ║
echo  ║  Nhan Ctrl+C de dung server                              ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: Mở trình duyệt sau 3 giây
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:5000"

:: Chạy server
cd /d "%~dp0"
python -m app.app

:: Nếu server dừng
echo.
echo  Server da dung. Nhan phim bat ky de thoat...
pause >nul
