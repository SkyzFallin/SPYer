@echo off
:: ============================================================
::  SPYer - Double-Click Installer for Windows
::  Just double-click this file to install everything.
::  It will:
::    1. Check for / install Python
::    2. Create a virtual environment
::    3. Install all dependencies
::    4. Create a "Run SPYer.bat" launcher
::
::  PORTABLE: You can move this folder anywhere after install.
:: ============================================================

title SPYer Installer
color 0A
echo.
echo  ============================================
echo     SPYer - SPY Options Signal Monitor
echo     Windows Installer
echo  ============================================
echo.

:: --- Work from the folder this .bat lives in ---
cd /d "%~dp0"

:: --- Check spyer.py exists ---
if not exist "spyer.py" (
    color 0C
    echo [ERROR] spyer.py not found in this folder.
    echo         Make sure INSTALL.bat is in the same folder as spyer.py
    echo.
    pause
    exit /b 1
)

:: --- Check for Python ---
echo [*] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found on PATH.
    echo.
    echo     Trying to install via winget...
    winget --version >nul 2>&1
    if %errorlevel% neq 0 (
        color 0E
        echo [!] winget not available either.
        echo.
        echo     Please install Python manually:
        echo       1. Go to https://www.python.org/downloads/
        echo       2. Download Python 3.9 or higher
        echo       3. IMPORTANT: Check "Add Python to PATH" during install
        echo       4. Re-run this installer
        echo.
        pause
        exit /b 1
    )
    winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
    echo.
    echo [*] Python install attempted. Verifying...

    :: Refresh PATH for this session
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"

    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        color 0E
        echo [!] Python still not found after install.
        echo     Please close this window, restart your computer, then run this again.
        echo.
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [+] Found: %PYVER%
echo.

:: --- Upgrade pip ---
echo [*] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo [+] pip upgraded
echo.

:: --- Create virtual environment ---
if exist ".venv" (
    echo [+] Virtual environment already exists
) else (
    echo [*] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [+] Virtual environment created
)
echo.

:: --- Activate venv and install deps ---
echo [*] Installing dependencies (this may take a minute)...
call .venv\Scripts\activate.bat

pip install --upgrade yfinance pandas numpy PyQt5 plyer >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Some packages may have failed. Retrying with verbose output...
    pip install --upgrade yfinance pandas numpy PyQt5 plyer
)
echo [+] Dependencies installed
echo.

:: --- Create portable run_app.bat ---
echo [*] Creating launcher...

(
    echo @echo off
    echo cd /d "%%~dp0"
    echo call .venv\Scripts\activate.bat
    echo start "" pythonw spyer.py
) > run_app.bat

echo [+] Created: run_app.bat  (double-click this to launch SPYer^)
echo.

:: --- Done ---
color 0A
echo  ============================================
echo     INSTALL COMPLETE
echo  ============================================
echo.
echo  To run SPYer:
echo    Double-click "run_app.bat" in this folder.
echo.
echo  This folder is portable - you can move it
echo  anywhere and it will still work.
echo.
echo  Starting SPYer now...
echo.

:: Launch the app
start "" pythonw spyer.py

timeout /t 5
