@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: =====================================================================
:: VietASR Pro - FFmpeg Auto Setup (Windows)
:: Tự động kiểm tra, tải, và cấu hình ffmpeg cho pydub
:: =====================================================================

title VietASR Pro - FFmpeg Setup

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║       VietASR Pro - FFmpeg Auto Setup (Windows)         ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

set PROJECT_DIR=%~dp0
set TOOLS_DIR=%PROJECT_DIR%tools
set FFMPEG_DIR=%TOOLS_DIR%\ffmpeg
set FFMPEG_BIN=%FFMPEG_DIR%\bin
set FFMPEG_EXE=%FFMPEG_BIN%\ffmpeg.exe
set FFPROBE_EXE=%FFMPEG_BIN%\ffprobe.exe
set FFMPEG_ZIP=%TOOLS_DIR%\ffmpeg.zip

:: =====================================================================
:: STEP 1: Kiểm tra ffmpeg đã có chưa
:: =====================================================================
echo [1/3] Kiem tra ffmpeg...

:: Kiểm tra trong thư mục tools/ trước
if exist "%FFMPEG_EXE%" (
    echo   [OK] ffmpeg da co tai: %FFMPEG_EXE%
    goto :verify
)

:: Kiểm tra trong system PATH
where ffmpeg >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] ffmpeg da co trong system PATH
    for /f "delims=" %%i in ('where ffmpeg') do echo   Path: %%i
    echo.
    echo   Tuy nhien, de dam bao on dinh, se tai them ban local...
    echo.
)

:: =====================================================================
:: STEP 2: Tải ffmpeg
:: =====================================================================
echo [2/3] Tai ffmpeg tu GitHub...

:: Tạo thư mục tools/ nếu chưa có
if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"

:: URL tải ffmpeg
set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip

echo   URL: %FFMPEG_URL%
echo   Dang tai... (co the mat vai phut)

:: Dùng PowerShell để tải
powershell -NoProfile -Command ^
    "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
    "try { " ^
    "    $ProgressPreference = 'SilentlyContinue'; " ^
    "    Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing; " ^
    "    Write-Host '  [OK] Tai thanh cong!' " ^
    "} catch { " ^
    "    Write-Host '  [LOI] Khong tai duoc ffmpeg: ' $_.Exception.Message; " ^
    "    exit 1 " ^
    "}"

if errorlevel 1 (
    echo.
    echo  [LOI] Khong tai duoc ffmpeg tu GitHub.
    echo  Ban co the tai thu cong tu:
    echo    %FFMPEG_URL%
    echo  Giai nen vao: %FFMPEG_DIR%
    echo.
    pause
    exit /b 1
)

:: =====================================================================
:: STEP 3: Giải nén
:: =====================================================================
echo.
echo [3/3] Giai nen ffmpeg...

:: Giải nén bằng PowerShell
powershell -NoProfile -Command ^
    "try { " ^
    "    Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%TOOLS_DIR%' -Force; " ^
    "    Write-Host '  [OK] Giai nen thanh cong!' " ^
    "} catch { " ^
    "    Write-Host '  [LOI] Khong giai nen duoc: ' $_.Exception.Message; " ^
    "    exit 1 " ^
    "}"

if errorlevel 1 (
    echo  [LOI] Giai nen that bai!
    pause
    exit /b 1
)

:: Rename extracted folder to simpler name
:: The zip extracts to something like ffmpeg-master-latest-win64-gpl/
if exist "%FFMPEG_DIR%" rmdir /s /q "%FFMPEG_DIR%" 2>nul

for /d %%D in ("%TOOLS_DIR%\ffmpeg-*") do (
    rename "%%D" "ffmpeg"
    echo   Renamed: %%~nxD -^> ffmpeg
)

:: Dọn dẹp file zip
if exist "%FFMPEG_ZIP%" del /f /q "%FFMPEG_ZIP%"
echo   [OK] Da don dep file tam

:: =====================================================================
:: Verify
:: =====================================================================
:verify
echo.
echo  ═══════════════════════════════════════════════════════════
echo   Kiem tra ket qua:
echo  ═══════════════════════════════════════════════════════════

if exist "%FFMPEG_EXE%" (
    echo   [OK] ffmpeg:  %FFMPEG_EXE%
    "%FFMPEG_EXE%" -version 2>nul | findstr /i "ffmpeg version" 
) else (
    echo   [LOI] Khong tim thay ffmpeg.exe tai %FFMPEG_EXE%
    pause
    exit /b 1
)

if exist "%FFPROBE_EXE%" (
    echo   [OK] ffprobe: %FFPROBE_EXE%
) else (
    echo   [CANH BAO] Khong tim thay ffprobe.exe
)

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║              SETUP FFMPEG HOAN TAT!                     ║
echo  ║  ffmpeg da san sang cho VietASR Pro                     ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

pause
