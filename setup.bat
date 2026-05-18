@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: Bootstrap the local Windows development environment.
cd /d "%~dp0"
echo [INFO] Starting RL Portfolio Management environment setup...

set "PYTHON=py -3.13"
%PYTHON% --version >nul 2>nul
if !ERRORLEVEL! neq 0 (
    set "PYTHON=py -3.12"
    %PYTHON% --version >nul 2>nul
)
if !ERRORLEVEL! neq 0 (
    set "PYTHON=py -3.11"
    %PYTHON% --version >nul 2>nul
)
if !ERRORLEVEL! neq 0 (
    set "PYTHON=python"
    %PYTHON% --version >nul 2>nul
)
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo [ERROR] Supported versions are ^>=3.11 and ^<3.14.
    exit /b 1
)

%PYTHON% -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)"
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Unsupported Python version.
    echo [ERROR] Supported versions are ^>=3.11 and ^<3.14, as defined in pyproject.toml.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Virtual environment not found or incompatible. Creating...
    %PYTHON% -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

echo [SETUP] Installing project with development dependencies...
"%VENV_PYTHON%" -m pip install --upgrade pip >nul
"%VENV_PYTHON%" -m pip install -e ".[dev]"
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Failed to install dependencies from pyproject.toml.
    exit /b 1
)

if exist ".git\" (
    echo [SETUP] Installing pre-commit hooks...
    "%VENV_PYTHON%" -m pre_commit install
    if !ERRORLEVEL! neq 0 (
        echo [WARNING] pre-commit hook installation failed. Dependencies are installed, but hooks are not active.
    )
) else (
    echo [WARNING] Skipping pre-commit hook installation because .git was not found.
)

echo.
echo [SUCCESS] Environment setup completed successfully.
echo [INFO] Available commands:
echo   python -m scripts.build_dataset
echo   python -m scripts.train_agent
echo   python -m scripts.evaluate_agent
echo   python -m scripts.optimize_hyperparams

exit /b 0
