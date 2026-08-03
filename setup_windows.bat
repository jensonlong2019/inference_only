@echo off
setlocal
title GPT-SoVITS Windows Setup

REM Keep this file ASCII-only to avoid cmd encoding issues.

cd /d "%~dp0"
set "LOG_FILE=setup_windows.log"
set "VC_EXE=%TEMP%\vc_redist.x64.exe"
set "PIP_NET_OPTS=--disable-pip-version-check --retries 20 --timeout 120"
echo [%date% %time%] setup started > "%LOG_FILE%"

echo ==========================================
echo   GPT-SoVITS Inference - Windows Setup
echo ==========================================
echo Working directory: %CD%
echo Log file: %CD%\%LOG_FILE%
echo.

REM ----- 1) Detect Python command -----
set "PY_CMD="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3"

if "%PY_CMD%"=="" (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

if "%PY_CMD%"=="" (
    echo [1/4] Python not found. Trying to install Python 3.11 with winget...
    echo [%date% %time%] python not found, trying winget >> "%LOG_FILE%"
    echo.
    winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo.
        echo [%date% %time%] winget install failed >> "%LOG_FILE%"
        echo Auto install failed. Please install Python manually:
        echo 1. Open https://www.python.org/downloads/
        echo 2. Install Python 3.10 or 3.11 (64-bit)
        echo 3. IMPORTANT: check "Add python.exe to PATH"
        echo 4. Re-run setup_windows.bat after installation
        echo See log: %CD%\%LOG_FILE%
        echo.
        pause
        exit /b 1
    )
    echo [%date% %time%] winget install succeeded, retry python detection >> "%LOG_FILE%"
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -3"
    ) else (
        python --version >nul 2>&1
        if not errorlevel 1 set "PY_CMD=python"
    )
    if "%PY_CMD%"=="" (
        echo.
        echo Python installed, but current shell cannot find it yet.
        echo Please close this window and run setup_windows.bat again.
        echo See log: %CD%\%LOG_FILE%
        echo.
        pause
        exit /b 0
    )
)

echo [1/4] Python detected: %PY_CMD%
%PY_CMD% --version
echo [%date% %time%] python cmd: %PY_CMD% >> "%LOG_FILE%"
%PY_CMD% -c "import struct; print(struct.calcsize('P')*8)" > "%TEMP%\gsv_python_bits.txt" 2>> "%LOG_FILE%"
set /p PY_BITS=<"%TEMP%\gsv_python_bits.txt"
del /q "%TEMP%\gsv_python_bits.txt" >nul 2>&1
if not "%PY_BITS%"=="64" (
    echo.
    echo Detected Python is not 64-bit. Please install 64-bit Python 3.10/3.11.
    echo [%date% %time%] python bitness invalid: %PY_BITS% >> "%LOG_FILE%"
    pause
    exit /b 1
)
echo [%date% %time%] python bitness: %PY_BITS% >> "%LOG_FILE%"
echo.

REM ----- 2) Create venv if missing -----
if not exist "venv\Scripts\activate.bat" (
    echo [2/4] Creating virtual environment: venv
    echo [%date% %time%] creating venv >> "%LOG_FILE%"
    %PY_CMD% -m venv venv >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo [%date% %time%] create venv failed >> "%LOG_FILE%"
        echo Failed to create virtual environment.
        echo See log: %CD%\%LOG_FILE%
        pause
        exit /b 1
    )
    echo venv created.
) else (
    echo [2/4] Existing venv found, skip create.
)
echo.

REM ----- 3) Install dependencies -----
echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat
set "TORCH_VER=2.3.1+cpu"
set "TORCHAUDIO_VER=2.3.1+cpu"
echo Installing Microsoft VC++ Runtime (required by torch DLLs)...
set "VC_OK=0"
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed 2>nul | find "0x1" >nul
if not errorlevel 1 (
    set "VC_OK=1"
    echo VC++ Runtime already installed, skip.
    echo [%date% %time%] vcredist already installed >> "%LOG_FILE%"
)
if "%VC_OK%"=="0" (
    winget install --id Microsoft.VCRedist.2015+.x64 -e --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
    if not errorlevel 1 set "VC_OK=1"
)
if "%VC_OK%"=="0" (
    winget install --id Microsoft.VCRedist.2015+.x86 -e --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
    REM x86 install is optional; no VC_OK switch here.
)
if "%VC_OK%"=="0" (
    echo winget install failed. Trying direct Microsoft installer...
    echo [%date% %time%] vcredist winget failed, trying direct download >> "%LOG_FILE%"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%VC_EXE%'" >> "%LOG_FILE%" 2>&1
    if not exist "%VC_EXE%" (
        echo [%date% %time%] powershell download missing file, trying curl >> "%LOG_FILE%"
        curl.exe -L "https://aka.ms/vs/17/release/vc_redist.x64.exe" -o "%VC_EXE%" >> "%LOG_FILE%" 2>&1
    )
    if exist "%VC_EXE%" (
        "%VC_EXE%" /install /quiet /norestart >> "%LOG_FILE%" 2>&1
        if not errorlevel 1 (
            set "VC_OK=1"
            echo [%date% %time%] vcredist direct install succeeded >> "%LOG_FILE%"
        ) else (
            echo [%date% %time%] vcredist direct install failed >> "%LOG_FILE%"
        )
        del /q "%VC_EXE%" >nul 2>&1
    ) else (
        echo [%date% %time%] vcredist download failed >> "%LOG_FILE%"
    )
)
if "%VC_OK%"=="0" (
    echo.
    echo VC++ Runtime install failed.
    echo Install manually: https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo Then reboot Windows and re-run setup_windows.bat
    echo See log: %CD%\%LOG_FILE%
    pause
    exit /b 1
)
echo [%date% %time%] vcredist install OK >> "%LOG_FILE%"
python -m pip install %PIP_NET_OPTS% --upgrade pip
if errorlevel 1 (
    echo.
    echo [%date% %time%] pip upgrade failed >> "%LOG_FILE%"
    echo pip upgrade failed.
    echo See log: %CD%\%LOG_FILE%
    pause
    exit /b 1
)

echo Enforcing NumPy compatibility for torch (numpy^<2)...
python -m pip install %PIP_NET_OPTS% --no-cache-dir --force-reinstall --progress-bar on "numpy<2"
if errorlevel 1 (
    echo.
    echo [%date% %time%] numpy compatibility install failed >> "%LOG_FILE%"
    echo NumPy compatibility install failed.
    echo See log: %CD%\%LOG_FILE%
    pause
    exit /b 1
)

python -c "import torch, torchaudio; print(torch.__version__)" >nul 2>&1
if errorlevel 1 (
    echo torch/torchaudio missing or broken, reinstalling...
    echo [%date% %time%] torch missing/broken, reinstalling >> "%LOG_FILE%"
    python -m pip uninstall -y torch torchaudio torchvision >> "%LOG_FILE%" 2>&1
    python -m pip install %PIP_NET_OPTS% --no-cache-dir --force-reinstall --no-deps --progress-bar on torch==%TORCH_VER% torchaudio==%TORCHAUDIO_VER% --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo.
        echo [%date% %time%] torch install failed >> "%LOG_FILE%"
        echo torch/torchaudio install failed. Tried versions: %TORCH_VER% / %TORCHAUDIO_VER%
        echo See log: %CD%\%LOG_FILE%
        pause
        exit /b 1
    )
    python -m pip install %PIP_NET_OPTS% --no-cache-dir --upgrade --progress-bar on filelock typing-extensions setuptools sympy networkx fsspec
    if errorlevel 1 (
        echo [%date% %time%] torch prerequisite refresh warning (non-fatal) >> "%LOG_FILE%"
    )
) else (
    echo torch/torchaudio already healthy, skip reinstall.
    echo [%date% %time%] torch/torchaudio already healthy >> "%LOG_FILE%"
)

echo Installing Python requirements. First run may download very large wheels (e.g. mkl ~228MB).
python -m pip install %PIP_NET_OPTS% --progress-bar on -r requirements.txt
if errorlevel 1 (
    echo [%date% %time%] requirements install failed on default index, retrying with Tsinghua mirror >> "%LOG_FILE%"
    python -m pip install %PIP_NET_OPTS% --progress-bar on -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
)
if errorlevel 1 (
    echo.
    echo [%date% %time%] dependency install failed >> "%LOG_FILE%"
    echo Dependency install failed. Check network and try again.
    echo See log: %CD%\%LOG_FILE%
    pause
    exit /b 1
)

echo Installing LangSegment (fix setLangfilters import error)...
python install_langsegment.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo LangSegment install failed.
    echo See log: %CD%\%LOG_FILE%
    pause
    exit /b 1
)

echo Enforcing compatibility pins for gradio ecosystem...
python -m pip install %PIP_NET_OPTS% --progress-bar on --upgrade --force-reinstall "numpy<2" "huggingface_hub<1.0" "markupsafe~=2.1.5" "transformers<4.46.0" "fastapi==0.112.4" "starlette==0.38.2" "pydantic<2.10"
if errorlevel 1 (
    echo [%date% %time%] compatibility pin install failed >> "%LOG_FILE%"
    echo Compatibility dependency install failed.
    echo See log: %CD%\%LOG_FILE%
    pause
    exit /b 1
)

python -c "import torch; print(torch.__version__)" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo.
    echo [%date% %time%] torch import check failed >> "%LOG_FILE%"
    echo torch DLL load failed in venv.
    echo Please reboot Windows once and run setup_windows.bat again.
    echo If still failing, install VC runtime manually:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo See log: %CD%\%LOG_FILE%
    pause
    exit /b 1
)

echo.
echo [4/4] Starting inference service...
echo ==========================================
echo Setup completed. Inference service is starting...
echo ==========================================
echo [%date% %time%] setup completed >> "%LOG_FILE%"
call start_inference.bat
endlocal
