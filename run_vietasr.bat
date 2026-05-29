@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  run_vietasr.bat - VietASR Pro One-Click Launcher (Windows)
REM  Pure ASCII only - compatible with all Windows codepages
REM  Requirements: Python 3.10 or 3.11 recommended
REM ============================================================

title VietASR Pro - Launcher

echo.
echo ==========================================
echo  VietASR Pro v2.1.0
echo  Vietnamese Speech Recognition Demo
echo  Recommended: Python 3.10 or 3.11
echo ==========================================
echo.

REM -- Set working directory to script location
set "ROOT=%~dp0"
cd /d "%ROOT%"

REM ============================================================
REM  STEP 1: Find the right Python interpreter
REM  Priority: py -3.10 -> py -3.11 -> python3.10 ->
REM            common install paths -> python (with warning)
REM ============================================================
echo [1/7] Finding Python interpreter...
set "PYEXE="

REM Try py launcher with 3.10
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set "PYEXE=py -3.10"
    for /f "tokens=*" %%v in ('py -3.10 --version 2^>^&1') do set "PYVER=%%v"
    echo   Found: !PYVER! via py -3.10
    goto :py_found
)

REM Try py launcher with 3.11
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set "PYEXE=py -3.11"
    for /f "tokens=*" %%v in ('py -3.11 --version 2^>^&1') do set "PYVER=%%v"
    echo   Found: !PYVER! via py -3.11
    goto :py_found
)

REM Try common standalone install paths for 3.10
for %%p in (
    "C:\Python310\python.exe"
    "C:\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
) do (
    if exist "%%~p" (
        set "PYEXE=%%~p"
        for /f "tokens=*" %%v in ('"%%~p" --version 2^>^&1') do set "PYVER=%%v"
        echo   Found: !PYVER! at %%~p
        goto :py_found
    )
)

REM Try python3.10 or python3.11 on PATH
for %%c in (python3.10 python3.11) do (
    %%c --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXE=%%c"
        for /f "tokens=*" %%v in ('%%c --version 2^>^&1') do set "PYVER=%%v"
        echo   Found: !PYVER! via %%c
        goto :py_found
    )
)

REM Fallback: use whatever 'python' is on PATH
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ERROR: No Python found on this system.
    echo   Please install Python 3.10 from: https://www.python.org/downloads/
    echo   During install, check "Add Python to PATH".
    echo.
    goto :error_exit
)
set "PYEXE=python"
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"

REM Warn if Python 3.13 (Windows Store or otherwise)
echo !PYVER! | findstr /c:"3.13" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   WARNING: Detected !PYVER!
    echo   Python 3.13 may have compatibility issues with some ASR libraries.
    echo   RECOMMENDED: Install Python 3.10 from https://www.python.org/downloads/
    echo.
    echo   To proceed anyway with Python 3.13, press any key.
    echo   To exit and install Python 3.10 first, close this window.
    echo.
    pause
)

REM Also warn for Python 3.12
echo !PYVER! | findstr /c:"3.12" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   NOTE: Detected !PYVER!
    echo   Python 3.10 or 3.11 is recommended for best compatibility.
    echo   Continuing with !PYVER!...
    echo.
)

echo   Using: !PYVER! [!PYEXE!]

:py_found

REM ============================================================
REM  STEP 2: Check if port 5000 is already in use
REM ============================================================
echo [2/7] Checking port 5000...
set "PORT=5000"
if exist "%ROOT%.env" (
    for /f "usebackq tokens=1,2 delims==" %%a in ("%ROOT%.env") do (
        set "_K=%%a"
        set "_V=%%b"
        if /i "!_K!"=="FLASK_PORT" set "PORT=!_V!"
        if /i "!_K!"=="PORT" (
            if "!PYEXE!"=="python" set "PORT=!_V!"
        )
    )
)

netstat -ano 2>nul | findstr /r ":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   WARNING: Port %PORT% is already in use.
    for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /r ":%PORT% .*LISTENING"') do (
        set "_PID=%%p"
        goto :got_pid
    )
    :got_pid
    echo   PID using port %PORT%: !_PID!
    for /f "skip=3 tokens=1" %%n in ('tasklist /FI "PID eq !_PID!" 2^>nul') do (
        echo   Process: %%n
    )
    echo.
    echo   Options:
    echo   1. Stop the process using port %PORT%
    echo   2. Edit .env and change FLASK_PORT to a different port (e.g. 5001)
    echo.
    echo   Press any key to try starting anyway (may fail).
    echo   Or close this window to fix the port conflict first.
    echo.
    pause
) else (
    echo   Port %PORT% is available - OK
)

REM ============================================================
REM  STEP 3: Check / recreate venv
REM ============================================================
echo [3/7] Checking virtual environment...

set "NEED_NEW_VENV=0"

REM Check if venv exists at all
if not exist "%ROOT%venv\Scripts\activate.bat" (
    set "NEED_NEW_VENV=1"
    echo   No venv found - will create new one
    goto :create_venv
)

REM Check if existing venv uses the correct Python version
REM Read version from pyvenv.cfg
set "VENV_VER="
if exist "%ROOT%venv\pyvenv.cfg" (
    for /f "usebackq tokens=1,2 delims== " %%a in ("%ROOT%venv\pyvenv.cfg") do (
        if /i "%%a"=="version" set "VENV_VER=%%b"
    )
)

REM If venv uses Python 3.13 but we found a better Python, warn and ask
if "!VENV_VER:~0,4!"=="3.13" (
    REM Check if current PYEXE is NOT 3.13
    echo !PYVER! | findstr /c:"3.13" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   WARNING: Existing venv was built with Python 3.13 (!VENV_VER!)
        echo   but you now have !PYVER! selected.
        echo   The venv must be recreated for the correct Python version.
        echo.
        echo   Press Y to delete old venv and create a new one.
        echo   Press any other key to keep the old venv and continue.
        echo.
        set /p "_CHOICE=Your choice [Y/N]: "
        if /i "!_CHOICE!"=="Y" (
            echo   Deleting old venv...
            rmdir /s /q "%ROOT%venv" 2>nul
            set "NEED_NEW_VENV=1"
        ) else (
            echo   Keeping existing venv. Errors may occur if Python mismatch.
        )
    ) else (
        echo   Existing venv: Python !VENV_VER! - OK
    )
) else if defined VENV_VER (
    echo   Existing venv: Python !VENV_VER! - OK
) else (
    echo   Existing venv found - OK
)

:create_venv
if "!NEED_NEW_VENV!"=="1" (
    echo   Creating new virtual environment with !PYEXE!...
    !PYEXE! -m venv venv
    if errorlevel 1 (
        echo.
        echo   ERROR: Failed to create venv.
        echo   Try manually: !PYEXE! -m venv venv
        goto :error_exit
    )
    echo   New venv created - OK
)

REM ============================================================
REM  STEP 4: Activate venv
REM ============================================================
call "%ROOT%venv\Scripts\activate.bat"
if errorlevel 1 (
    echo   ERROR: Failed to activate venv.
    goto :error_exit
)
echo [4/7] Virtual environment activated - OK

REM ============================================================
REM  STEP 5: Upgrade pip and install dependencies
REM          STOP if this fails - do NOT start the server
REM ============================================================
echo [5/7] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 (
    echo.
    echo   ERROR: Could not upgrade pip/setuptools/wheel.
    echo   ERROR: Khong the cap nhat pip. Kiem tra ket noi mang.
    goto :error_exit
)
echo   pip, setuptools, wheel upgraded - OK

echo   Installing requirements from requirements.txt...
if not exist "%ROOT%requirements.txt" (
    echo   WARNING: requirements.txt not found - skipping
    goto :deps_done
)

python -m pip install -r "%ROOT%requirements.txt" -i https://pypi.org/simple
if errorlevel 1 (
    echo.
    echo ==========================================
    echo  INSTALLATION FAILED
    echo  Khong the cai thu vien. Server chua duoc khoi dong.
    echo.
    echo  Common fixes:
    echo  1. Check internet connection
    echo  2. Try: python -m pip install -r requirements.txt --no-deps
    echo  3. Install PyTorch separately from https://pytorch.org
    echo  4. Make sure you are using Python 3.10 or 3.11
    echo ==========================================
    echo.
    goto :error_exit
)
echo   Dependencies installed - OK

:deps_done

REM ============================================================
REM  STEP 6: Check .env file
REM ============================================================
echo [6/7] Checking .env config...
if not exist "%ROOT%.env" (
    if exist "%ROOT%.env.example" (
        copy "%ROOT%.env.example" "%ROOT%.env" >nul
        echo   WARNING: Created .env from .env.example
        echo   Please edit .env and set GEMINI_API_KEY to use Gemini AI.
    ) else (
        echo   WARNING: No .env found - server will run without API key
    )
) else (
    echo   .env exists - OK
)

REM Re-read port in case .env was just created
set "PORT=5000"
if exist "%ROOT%.env" (
    for /f "usebackq tokens=1,2 delims==" %%a in ("%ROOT%.env") do (
        set "_K=%%a"
        set "_V=%%b"
        if /i "!_K!"=="FLASK_PORT" set "PORT=!_V!"
    )
)

REM ============================================================
REM  STEP 7: Find entry point and start server (VISIBLE window)
REM ============================================================
echo [7/7] Starting Flask server...
set "ENTRY="

if exist "%ROOT%app\app.py" (
    set "ENTRY=app\app.py"
    goto :found_entry
)
if exist "%ROOT%app.py" (
    set "ENTRY=app.py"
    goto :found_entry
)

echo   ERROR: Cannot find app\app.py
goto :error_exit

:found_entry
echo   Entry point: %ENTRY%
echo   Server port: %PORT%
echo.
echo ------------------------------------------
echo  Starting server in a new visible window
echo  so you can see import errors if any occur
echo ------------------------------------------
echo.

REM Start in a VISIBLE window (not minimized) for debugging
start "VietASR-Server" cmd /c "python %ENTRY% & pause"

REM ============================================================
REM  Poll /api/health for up to 120 seconds
REM  (Wav2Vec2 model loading can be slow on CPU)
REM ============================================================
echo   Waiting for server to be ready (up to 120 seconds)...
echo   Wav2Vec2 model loading may take 30-90 seconds on CPU.
echo.

set "RETRY=0"
set "MAX_RETRY=40"
set "SERVER_READY=0"

:poll_loop
REM Wait 3 seconds between each poll
timeout /t 3 /nobreak >nul
set /a "RETRY+=1"
set /a "ELAPSED=RETRY*3"

<nul set /p "=  Checking... !ELAPSED!s / 120s"
echo.

REM Try /api/health
curl -s --max-time 3 -o nul -w "%%{http_code}" "http://127.0.0.1:%PORT%/api/health" 2>nul | findstr /c:"200" >nul 2>&1
if not errorlevel 1 (
    set "SERVER_READY=1"
    goto :poll_done
)

REM Try /health fallback
curl -s --max-time 3 -o nul -w "%%{http_code}" "http://127.0.0.1:%PORT%/health" 2>nul | findstr /c:"200" >nul 2>&1
if not errorlevel 1 (
    set "SERVER_READY=1"
    goto :poll_done
)

REM Try root path fallback
curl -s --max-time 3 -o nul -w "%%{http_code}" "http://127.0.0.1:%PORT%/" 2>nul | findstr /c:"200" >nul 2>&1
if not errorlevel 1 (
    set "SERVER_READY=1"
    goto :poll_done
)

if !RETRY! lss !MAX_RETRY! goto :poll_loop

:poll_done

if "!SERVER_READY!"=="0" (
    echo.
    echo   WARNING: Server did not respond after 120 seconds.
    echo   Check the server window for error messages.
    echo   Common causes:
    echo     - Import error: run  pip install -r requirements.txt
    echo     - Port conflict: change FLASK_PORT in .env
    echo     - Missing ffmpeg: run setup_ffmpeg.bat
    echo.
    echo   Opening browser anyway - you may see ERR_CONNECTION_REFUSED
    echo   Press any key to open browser or close window to cancel.
    pause
) else (
    echo.
    echo   Server is ready!
)

REM Open browser
echo   Opening browser at http://127.0.0.1:%PORT% ...
start "" "http://127.0.0.1:%PORT%"

echo.
echo ==========================================
echo  VietASR Pro is running at:
echo  http://127.0.0.1:%PORT%
echo.
echo  Close the "VietASR-Server" window to stop.
echo  This window can be closed safely.
echo ==========================================
echo.
pause
goto :end

REM ============================================================
REM  ERROR EXIT - always keep terminal open
REM ============================================================
:error_exit
echo.
echo ==========================================
echo  STARTUP FAILED - Server was NOT started.
echo  Browser will NOT be opened.
echo.
echo  Read the error above and fix it first.
echo  Then run this script again.
echo ==========================================
echo.
pause
exit /b 1

:end
endlocal
