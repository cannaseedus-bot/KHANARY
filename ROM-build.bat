@echo off
setlocal enabledelayedexpansion
:: ============================================================================
:: ROM-build.bat - compile studio-dist/ into a WWA ROM bundle (.wwa)
::
:: The K'UHUL Studio ROM is a single-file PWA (studio-dist/). "Compiling" it
:: means packaging the folder into a deterministic .wwa zip container that
:: WWAHost.exe can launch as a Windows Web App.
::
:: Packaging backend, in preference order:
::   1. kuhul_engine_driver.dll  - ke_package_wwa (native, deterministic STORE)
::      (compile first: drivers\build_all.bat)
::   2. PowerShell Compress-Archive - fallback (no native DLL required)
::
:: Usage:
::   ROM-build              - validate + package studio-dist\studio.wwa
::   ROM-build deploy       - package + copy to dist\khanary-server\
::   ROM-build launch       - package + launch via WWAHost.exe
::   ROM-build clean        - remove output artifacts
:: ============================================================================

set "ROOT=%~dp0"
set "ROM_DIR=%ROOT%studio-dist"
set "OUT_WWA=%ROM_DIR%\studio.wwa"
set "DEPLOY_DIR=%ROOT%dist\khanary-server"

:: required ROM files
set "REQUIRED=index.html user-profile.html kuhul-studio.webmanifest sw.js kuhul-icon.svg studio.manifest.json"
set "MISSING=0"

echo [ROM] K'UHUL Studio ROM build
echo [ROM] validating %ROM_DIR%
echo.

for %%f in (%REQUIRED%) do (
    if not exist "%ROM_DIR%\%%f" (
        echo [ROM]   MISSING: %%f
        set "MISSING=1"
    )
)

if "%MISSING%"=="1" (
    echo.
    echo [ROM] ERROR: required ROM files missing ^(above^). Aborting.
    exit /b 1
)
echo [ROM]   all 6 core ROM files present

if /i "%~1"=="clean" goto :clean

:: package
echo.
echo [ROM] packaging to %OUT_WWA%
if exist "%OUT_WWA%" del /q "%OUT_WWA%"

echo [ROM]   backend: tools\package_wwa.cjs ^(native DLL if available, else PowerShell^)
node "%ROOT%tools\package_wwa.cjs" "%ROM_DIR%" "%OUT_WWA%"
if errorlevel 1 (
    echo [ROM] ERROR: packaging failed
    exit /b 1
)

:: deploy / launch
if /i "%~1"=="deploy" goto :deploy
if /i "%~1"=="launch" goto :launch
echo [ROM] done. Bundle: %OUT_WWA%
echo [ROM]   deploy : ROM-build deploy
echo [ROM]   launch : ROM-build launch
exit /b 0

:deploy
echo [ROM] deploying to %DEPLOY_DIR%
if not exist "%DEPLOY_DIR%" mkdir "%DEPLOY_DIR%"
copy /y "%OUT_WWA%" "%DEPLOY_DIR%\studio.wwa" >nul
copy /y "%ROM_DIR%\index.html" "%DEPLOY_DIR%\studio-index.html" >nul
echo [ROM] done. %DEPLOY_DIR%\studio.wwa + studio-index.html
exit /b 0

:launch
set "WWAHOST=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\WWAHost.exe"
if not exist "%WWAHOST%" (
    echo [ROM] WWAHost.exe not found at %WWAHOST%
    exit /b 1
)
echo [ROM] launching %OUT_WWA% via WWAHost.exe
start "" "%WWAHOST%" "%OUT_WWA%"
echo [ROM] launched.
exit /b 0

:clean
echo [ROM] cleaning artifacts
if exist "%OUT_WWA%" del /q "%OUT_WWA%" & echo [ROM]   removed %OUT_WWA%
if exist "%DEPLOY_DIR%\studio.wwa" del /q "%DEPLOY_DIR%\studio.wwa" & echo [ROM]   removed deploy copy
exit /b 0
