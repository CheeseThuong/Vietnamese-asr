@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM VietASR Pro launcher. Keep this file ASCII-only for Windows cmd.exe.
title VietASR Pro - Launcher

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%venv\Scripts\python.exe"
set "APP_ENTRY=%ROOT%app\app.py"
set "PORT=5000"
set "NEW_VENV=0"

cd /d "%ROOT%"

echo.
echo ==========================================
echo  VietASR Pro v2.1.0
echo ==========================================
echo.

REM Always reuse the project venv. This prevents the Windows Store Python
REM or another global Python installation from running the application.
echo [1/5] Checking project virtual environment...
if exist "%VENV_PY%" goto :venv_ready

echo   Project venv was not found. Creating it now...
set "BASE_PY="

py -3.11 --version >nul 2>&1
if not errorlevel 1 set "BASE_PY=py -3.11"

if not defined BASE_PY (
    py -3.10 --version >nul 2>&1
    if not errorlevel 1 set "BASE_PY=py -3.10"
)

if not defined BASE_PY (
    python --version >nul 2>&1
    if not errorlevel 1 set "BASE_PY=python"
)

if not defined BASE_PY (
    echo   ERROR: Python was not found.
    echo   Install Python 3.10 or 3.11, then run this file again.
    goto :error_exit
)

!BASE_PY! -m venv "%ROOT%venv"
if errorlevel 1 (
    echo   ERROR: Could not create the project venv.
    goto :error_exit
)
set "NEW_VENV=1"

:venv_ready
for /f "tokens=*" %%V in ('"%VENV_PY%" --version 2^>^&1') do set "VENV_VERSION=%%V"
echo   Using: %VENV_PY%
echo   Version: !VENV_VERSION!

echo [2/5] Checking dependencies...
if "%NEW_VENV%"=="1" (
    echo   New venv detected. Installing requirements...
    "%VENV_PY%" -m pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 (
        echo   ERROR: Dependency installation failed.
        goto :error_exit
    )
)

"%VENV_PY%" -c "import flask_socketio" >nul 2>&1
if errorlevel 1 (
    echo   Flask-SocketIO is missing. Installing it into the project venv...
    "%VENV_PY%" -m pip install "flask-socketio>=5.3.0"
    if errorlevel 1 (
        echo   ERROR: Could not install Flask-SocketIO.
        goto :error_exit
    )
)
echo   Flask-SocketIO import: OK

echo [3/5] Validating application imports...
"%VENV_PY%" -c "import app.app; print('  Application imports: OK')"
if errorlevel 1 (
    echo.
    echo   ERROR: The application import check failed.
    echo   Run this command for full dependency repair:
    echo   "%VENV_PY%" -m pip install -r "%ROOT%requirements.txt"
    goto :error_exit
)

if /i "%~1"=="--check" (
    echo.
    echo   Launcher check completed successfully.
    exit /b 0
)

if not exist "%APP_ENTRY%" (
    echo   ERROR: Cannot find %APP_ENTRY%
    goto :error_exit
)

if exist "%ROOT%.env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%.env") do (
        if /i "%%A"=="FLASK_PORT" set "PORT=%%B"
    )
)

echo [4/5] Checking port !PORT!...
powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -State Listen -LocalPort !PORT! -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    curl.exe -s --max-time 5 -o nul -w "%%{http_code}" "http://127.0.0.1:!PORT!/api/health" 2>nul | findstr /c:"200" >nul 2>&1
    if not errorlevel 1 goto :already_running

    echo   ERROR: Port !PORT! is already in use.
    echo   Another program is using this port. Stop it or change FLASK_PORT in .env.
    goto :error_exit
)
echo   Port !PORT!: available

echo [5/5] Starting server with the project venv...
start "VietASR-Server" cmd /k ""%VENV_PY%" "%APP_ENTRY%""

echo   Waiting for http://127.0.0.1:!PORT!/api/health ...
set "READY=0"
for /l %%I in (1,1,40) do (
    timeout /t 3 /nobreak >nul
    curl.exe -s --max-time 3 -o nul -w "%%{http_code}" "http://127.0.0.1:!PORT!/api/health" 2>nul | findstr /c:"200" >nul 2>&1
    if not errorlevel 1 (
        set "READY=1"
        goto :server_ready
    )
)

echo.
echo   ERROR: Server did not become ready within 120 seconds.
echo   Read the error in the VietASR-Server window.
goto :error_exit

:server_ready
echo   Server is ready: http://127.0.0.1:!PORT!
start "" "http://127.0.0.1:!PORT!"
echo.
echo   Close the VietASR-Server window or press Ctrl+C there to stop.
exit /b 0

:already_running
echo   VietASR is already running: http://127.0.0.1:!PORT!
start "" "http://127.0.0.1:!PORT!"
exit /b 0

:error_exit
echo.
echo ==========================================
echo  STARTUP FAILED
echo ==========================================
echo.
pause
exit /b 1
