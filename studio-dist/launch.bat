@echo off
setlocal

:: launch.bat — K'UHUL STUDIO ROM launcher
:: Mounts the 4-file PWA as a static sidecar on port 8820
:: via json_runtime.exe

set "ROOT=%~dp0"
set "JSON_RT=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json_runtime.exe"

if not exist "%JSON_RT%" (
    echo [studio] json_runtime.exe not found: %JSON_RT%
    echo          Start it first: START-SERVERS
    exit /b 1
)

netstat -ano | findstr /r /c:":8787 .*LISTENING" >nul 2>&1
if errorlevel 1 (
    echo [studio] json_runtime not running on :8787
    echo          Start it first: START-SERVERS
    exit /b 1
)

echo [studio] Mounting K'UHUL STUDIO ROM on :8820...
"%JSON_RT%" --mount "%ROOT%" --port 8820 --manifest studio.manifest.json
exit /b %ERRORLEVEL%
