@echo off
setlocal
title GPT-SoVITS Inference

REM Keep this file ASCII-only to avoid cmd encoding issues.

cd /d "%~dp0"
set "LOG_FILE=start_inference.log"
echo [%date% %time%] start_inference started > "%LOG_FILE%"

if "%infer_ttswebui%"=="" set infer_ttswebui=9872
if "%language%"=="" set language=Auto
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "http_proxy="
set "https_proxy="
set "all_proxy="
set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"

echo ==========================================
echo GPT-SoVITS Inference Start
echo ==========================================
echo Working directory: %CD%
echo Port: %infer_ttswebui%
echo Language: %language%
echo Log file: %CD%\%LOG_FILE%
echo ==========================================
echo [%date% %time%] working_dir=%CD% port=%infer_ttswebui% language=%language% >> "%LOG_FILE%"

REM Set console to UTF-8 to avoid encoding issues with various languages
chcp 65001 >nul

set VENV_ACTIVATED=0

REM 1) venv in current directory
if exist "venv\Scripts\activate.bat" (
    echo venv found in current directory. Activating...
    call venv\Scripts\activate.bat
    set VENV_ACTIVATED=1
REM 2) venv in sibling project directory
) else if exist "..\GPT-SoVITS-v2-2025\venv\Scripts\activate.bat" (
    echo venv found in project directory. Activating...
    call ..\GPT-SoVITS-v2-2025\venv\Scripts\activate.bat
    set VENV_ACTIVATED=1
REM 3) venv in parent directory
) else if exist "..\venv\Scripts\activate.bat" (
    echo venv found in parent directory. Activating...
    call ..\venv\Scripts\activate.bat
    set VENV_ACTIVATED=1
)

if %VENV_ACTIVATED%==0 (
    echo No venv found. Checking system Python...
    echo [%date% %time%] no venv found, checking system python >> "%LOG_FILE%"
    python -c "import torch" 2>nul
    if errorlevel 1 (
        echo.
        echo WARNING: no venv found and torch is not installed in system Python.
        echo Please run setup_windows.bat first.
        echo [%date% %time%] torch missing in system python >> "%LOG_FILE%"
        echo.
        pause
        exit /b 1
    ) else (
        echo torch detected in system Python. Continue.
        echo [%date% %time%] torch found in system python >> "%LOG_FILE%"
    )
)

# Check if core dependencies are installed. If not, run full requirements sync.
python -c "import onnxruntime, opencc, pytorch_lightning" 2>nul
if errorlevel 1 (
    echo.
    echo Detecting missing dependencies...
    echo Starting full dependency sync from requirements.txt...
    echo This may take a few minutes for the first time.
    echo [%date% %time%] missing deps, running full sync >> "%LOG_FILE%"
    
    python -m pip install -r requirements.txt
    
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies. 
        echo Please check your internet connection and run:
        echo python -m pip install -r requirements.txt
        echo [%date% %time%] requirements sync failed >> "%LOG_FILE%"
        echo.
        pause
        exit /b 1
    ) else (
        echo.
        echo Dependencies synced successfully.
        echo [%date% %time%] requirements sync completed >> "%LOG_FILE%"
    )
)

set PYTHONPATH=%CD%;%CD%\GPT_SoVITS;%PYTHONPATH%

if not exist "output\batch_result" mkdir "output\batch_result"
if not exist "GPT_weights_v2" mkdir "GPT_weights_v2"
if not exist "GPT_weights" mkdir "GPT_weights"
if not exist "SoVITS_weights_v2" mkdir "SoVITS_weights_v2"
if not exist "SoVITS_weights" mkdir "SoVITS_weights"

echo Starting inference service...
echo [%date% %time%] launching inference_webui.py >> "%LOG_FILE%"
REM Use -u for unbuffered output. Explicitly handle encoding for Tee-Object compatibility.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$OutputEncoding = [System.Text.Encoding]::UTF8; $env:PYTHONPATH='%PYTHONPATH%'; python -u 'GPT_SoVITS\inference_webui.py' '%language%' 2>&1 | ForEach-Object { Write-Host $_; $_ | Out-File -FilePath '%LOG_FILE%' -Append -Encoding utf8 }"
echo [%date% %time%] inference process exited with code %errorlevel% >> "%LOG_FILE%"

pause
endlocal

