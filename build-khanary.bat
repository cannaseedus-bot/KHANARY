@echo off
setlocal

:: build-khanary.bat — rebuild llama.cpp with OpenCL GPU (Intel HD 4600)
:: Wraps build_gpu.ps1 so it works from cmd, Explorer, or Claude Code's ! prefix.
:: Usage:  build-khanary            (full rebuild if cache stale)
::         build-khanary clean      (wipe CMakeCache before rebuilding)

set ROOT=%~dp0
set SCRIPT=%ROOT%khanary-llama-build\build_gpu.ps1
set CACHE=%ROOT%khanary-llama-build\llama.cpp\build\CMakeCache.txt

if /i "%1"=="clean" (
    echo [khanary] Wiping CMakeCache for clean rebuild...
    if exist "%CACHE%" del /f /q "%CACHE%"
)

echo [khanary] Starting GPU build...
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [khanary] BUILD FAILED ^(exit %ERRORLEVEL%^)
    exit /b %ERRORLEVEL%
)

echo.
echo [khanary] Build complete.
echo   llama-cli.exe  -m ^<model.gguf^> -ngl 999 --threads 4
echo   llama-server   -m ^<model.gguf^> -ngl 999 --threads 4 --port 17480
endlocal
