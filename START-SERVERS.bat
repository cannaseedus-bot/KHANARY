@echo off
setlocal

:: ============================================================================
:: START-SERVERS.bat — KUHUL APPS stack: one-shot startup with MCP hookup info
::
::   START-SERVERS                           start all services, default model
::   START-SERVERS "path\to\model.gguf"     start with specific model on :9000
::   START-SERVERS "path\to\model.gguf" 3080  start model on custom port
::   START-SERVERS --status                  show which ports are listening
::   START-SERVERS --stop                    stop all managed services
::
:: On startup, writes active-model.json so MicrosoftSDK.ps1 / kuhul-server
:: can find the active inference endpoint without hardcoding port 17480.
::
:: MCP hookup: after startup, paste the MCP Gateway URL shown below into
::   llama-server UI → Settings → MCP Servers
:: ============================================================================

set "ROOT=C:\Users\canna\_khanary_inspect"

:: --- binaries ----------------------------------------------------------------
set "JSON_RT_EXE=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json_runtime.exe"
set "JSON_RT_DIR=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json-runtime"
set "GATEWAY=%ROOT%\dist\khanary-server\kuhul-server.cjs"
set "ENGINE_EXE=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\build-llama\bin\Release\kuhul_engine.exe"
:: Studio server — khanary-server.exe in dist, must run from that dir so DLLs resolve
set "UI_SERVER=%ROOT%\dist\khanary-server\khanary-server.exe"
set "UI_SERVER_DIR=%ROOT%\dist\khanary-server"

:: --- ports -------------------------------------------------------------------
set "JSON_RT_PORT=8787"
set "GATEWAY_PORT=8764"
set "ENGINE_PORT=17480"

:: --- model arg ---------------------------------------------------------------
set "MODEL_PATH=%~1"
set "MODEL_PORT=%~2"
if "%MODEL_PORT%"=="" set "MODEL_PORT=9000"

:: Default model if none provided
if "%MODEL_PATH%"=="" set "MODEL_PATH=%ROOT%\models\from_zero\from_zero_v0.1.f32.gguf"

:: Dispatch
if /i "%MODEL_PATH%"=="--status" goto :status
if /i "%MODEL_PATH%"=="--stop"   goto :stop

:: ============================================================================
:: Start all services
:: ============================================================================

echo.
echo [START-SERVERS] KUHUL APPS stack starting...
echo.

:: --- json_runtime :8787 ------------------------------------------------------
netstat -ano | findstr /r /c:":%JSON_RT_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [json_runtime]  already on :%JSON_RT_PORT%
) else if not exist "%JSON_RT_EXE%" (
    echo [json_runtime]  WARN: not found at %JSON_RT_EXE%
) else (
    echo [json_runtime]  starting on :%JSON_RT_PORT%...
    pushd "%JSON_RT_DIR%"
    start "json_runtime" "..\json_runtime.exe" --manifest manifest.json
    popd
)

:: --- kuhul-server :8764 (MCP gateway) ----------------------------------------
netstat -ano | findstr /r /c:":%GATEWAY_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [kuhul-server]  already on :%GATEWAY_PORT%
) else if not exist "%GATEWAY%" (
    echo [kuhul-server]  WARN: not found at %GATEWAY%
) else (
    echo [kuhul-server]  starting MCP gateway on :%GATEWAY_PORT%...
    start "kuhul-server" "%ROOT%\dist\khanary-server\start-gateway.bat"
)

:: --- kuhul_engine :17480 (waits 3s for gateway to auto-start it) -------------
timeout /t 3 /nobreak >nul
netstat -ano | findstr /r /c:":%ENGINE_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [kuhul_engine]  already on :%ENGINE_PORT%
) else if not exist "%ENGINE_EXE%" (
    echo [kuhul_engine]  WARN: not found at %ENGINE_EXE% (gateway will retry on 60s tick^)
) else (
    echo [kuhul_engine]  starting on :%ENGINE_PORT%...
    start "kuhul_engine" "%ENGINE_EXE%" --serve %ENGINE_PORT%
)

:: --- llama-server :<MODEL_PORT> (active inference model) ---------------------
netstat -ano | findstr /r /c:":%MODEL_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [llama-server]  already on :%MODEL_PORT%
    goto :write_active_model
)
if not exist "%UI_SERVER%" (
    echo [llama-server]  WARN: not found at %UI_SERVER%
    echo                 Rebuild: cd khanary-llama-build\llama.cpp ^&^& cmake --build build --target llama-server
    goto :write_active_model
)
if exist "%MODEL_PATH%" (
    echo [khanary-server] starting with model on :%MODEL_PORT%...
    pushd "%UI_SERVER_DIR%"
    start "khanary-server" ".\khanary-server.exe" ^
        -m "%MODEL_PATH%" ^
        -ngl 0 --threads 4 --port %MODEL_PORT% ^
        --repeat-penalty 1.3 --repeat-last-n 64
    popd
) else (
    echo [khanary-server] starting UI-only on :%MODEL_PORT% (model loads on demand^)...
    pushd "%UI_SERVER_DIR%"
    start "khanary-server" ".\khanary-server.exe" --port %MODEL_PORT%
    popd
)

:: ============================================================================
:: Write active-model.json — MicrosoftSDK.ps1 and kuhul-server read this
:: to know which port the active inference model is on.
:: ============================================================================
:write_active_model
set "ACTIVE_JSON=%ROOT%\active-model.json"
(
    echo {
    echo   "endpoint": "http://127.0.0.1:%MODEL_PORT%/v1/chat/completions",
    echo   "port": %MODEL_PORT%,
    echo   "model_path": "%MODEL_PATH:\=\\%",
    echo   "planner_endpoint": "http://127.0.0.1:%GATEWAY_PORT%/api/plan",
    echo   "mcp_gateway": "http://127.0.0.1:%GATEWAY_PORT%",
    echo   "engine_endpoint": "http://127.0.0.1:%ENGINE_PORT%/v1/chat/completions"
    echo }
) > "%ACTIVE_JSON%"
echo.
echo [active-model.json] written: %ACTIVE_JSON%

goto :summary

:: ============================================================================
:: --status
:: ============================================================================
:status
echo.
echo [status] KUHUL APPS stack:
call :check_port %JSON_RT_PORT%  "json_runtime    "
call :check_port %GATEWAY_PORT%  "kuhul-server    "
call :check_port %ENGINE_PORT%   "kuhul_engine    "
echo.
echo [status] llama-server — check your model port (default 9000 or custom):
call :check_port 9000            "llama-server    "
if exist "%ROOT%\active-model.json" (
    echo.
    echo [active-model.json]:
    type "%ROOT%\active-model.json"
)
goto :eof

:check_port
netstat -ano | findstr /r /c:":%~1 .*LISTENING" >nul 2>&1
if errorlevel 1 (
    echo   %~2 :%~1   DOWN
) else (
    echo   %~2 :%~1   LISTENING
)
goto :eof

:: ============================================================================
:: --stop
:: ============================================================================
:stop
echo [stop] Stopping KUHUL APPS services...
taskkill /f /im json_runtime.exe >nul 2>&1
taskkill /f /im kuhul_engine.exe >nul 2>&1
taskkill /f /im llama-server.exe >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c.":%GATEWAY_PORT% .*LISTENING"') do taskkill /f /pid %%p >nul 2>&1
if exist "%ROOT%\active-model.json" del /f /q "%ROOT%\active-model.json"
echo [stop] Done.
goto :eof

:: ============================================================================
:: Summary — server info display for MCP hookup
:: ============================================================================
:summary
echo.
echo =============================================================
echo   KUHUL APPS Stack — Server Info
echo =============================================================
echo.
echo   KUHUL APPS Studio  :  http://127.0.0.1:%MODEL_PORT%
echo   Canvas (studio)    :  http://127.0.0.1:%MODEL_PORT%/chat/^<id^>/canvas
echo                         ^(create a chat first, then append /canvas to the URL^)
echo   K'UHUL STUDIO ROM  :  http://127.0.0.1:8820
echo                         ^(run studio-dist\launch.bat to mount^)
echo.
echo   MCP Gateway  :  http://127.0.0.1:%GATEWAY_PORT%
echo   Planner API  :  http://127.0.0.1:%GATEWAY_PORT%/api/plan
echo   Engine       :  http://127.0.0.1:%ENGINE_PORT%
echo   Runtime      :  http://127.0.0.1:%JSON_RT_PORT%
echo   Active model :  http://127.0.0.1:%MODEL_PORT%/v1/chat/completions
echo.
echo =============================================================
echo   MCP HOOKUP — paste into KUHUL APPS Settings:
echo     Settings ^> MCP Servers ^> Add Server URL:
echo.
echo       http://127.0.0.1:%GATEWAY_PORT%
echo.
echo   With MCP wired: typing "Create an app for X" in the studio
echo   chat routes through PM-1 ^> TaskEngine.cpp (not raw inference).
echo   The three-panel studio canvas is at /chat/^<id^>/canvas.
echo =============================================================
echo.
echo   NOTE: kuhul_engine auto-detects its port — if %ENGINE_PORT% is Windows-excluded
echo   (HyperV/WSL range), it scans 14800, 14810, 15000... and uses the first
echo   available port. Actual port is written to active-model.json after startup.
echo   To see exclusions: netsh int ipv4 show excludedportrange protocol=tcp
echo.
echo   To plan a task manually (e.g. "Create an app for X"):
echo     powershell -File "%ROOT%\tools\call_planner.ps1" -Prompt "Create an app for X"
echo.
echo   Use --status to re-check, --stop to shut down.
echo.

endlocal
