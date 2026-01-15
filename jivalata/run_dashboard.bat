@echo off
setlocal EnableDelayedExpansion

title JIVALATA Dashboard Launcher

echo ==================================================
echo         JIVALATA - Flood Restoration MVP
echo ==================================================
echo.

:: 1. Check Python
echo [1/3] Checking Python environment...
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in your PATH.
    echo         Please install Python and check "Add to PATH" during installation.
    pause
    exit /b 1
)

:: 2. Check/Install Dependencies
echo [2/3] Verifying dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to install dependencies. 
    echo         Check your internet connection or Python permission settings.
    pause
    exit /b 1
)

:: 3. Launch Dashboard via Module
echo.
echo [3/3] Launching Dashboard...
echo       (If the browser does not open, check the URL below)
echo ==================================================

python -m streamlit run src/dashboard.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The Dashboard application crashed.
    echo         See the error message above for details.
    pause
    exit /b 1
)

echo.
echo [INFO] Dashboard session ended.
pause
