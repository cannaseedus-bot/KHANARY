@echo off
setlocal

:: ============================================================================
:: SERVER-LAUNCHER.bat — KUHUL APPS stack: start every server, or make sure
:: every server is already running. Idempotent: missing services are started,
:: running services are left alone.
::
::   SERVER-LAUNCHER            start all missing services
::   SERVER-LAUNCHER --status   show which ports are listening
::   SERVER-LAUNCHER --stop     stop the services this script manages
::   SERVER-LAUNCHER <model>    start all + launch llama-server with <model>
::
:: Services:
::   1. json_runtime.exe   :8787  hosting API + port manager (file-manager, sidecars)
::   2. kuhul-server.cjs   :8764  gateway — auto-starts kuhul_engine :17474 (watchdog)
::   3. kuhul_engine.exe   :17474 OpenAI-compatible chat origin (started by gateway;
::                              direct fallback below if the gateway cannot)
::   4. llama-server.exe   :8085  embedded KUHUL APPS UI (router mode — a model is
::                              optional; pass one as arg to enable inference)
::
:: Rebuild the UI server first when UI source changed (see notes.txt):
::   cd khanary-llama-build\llama.cpp\tools\ui && npm run build
::   call "<VS2022 vcvars64.bat>" && cmake -B build-ninja -S ..\.. -G Ninja ...
::   cmake --build build-ninja --target llama-server
:: ============================================================================

set "ROOT=C:\Users\canna\_khanary_inspect"

:: --- service binaries -------------------------------------------------------
set "JSON_RT_EXE=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json_runtime.exe"
set "JSON_RT_DIR=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json-runtime"
set "GATEWAY=%ROOT%\dist\khanary-server\kuhul-server.cjs"
set "ENGINE_EXE=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\build-llama\bin\Release\kuhul_engine.exe"
set "UI_SERVER=%ROOT%\khanary-llama-build\llama.cpp\build-ninja\bin\llama-server.exe"

:: --- ports (edit here to change) ---------------------------------------------
set "JSON_RT_PORT=8787"
set "GATEWAY_PORT=8764"
set "ENGINE_PORT=17474"
set "UI_PORT=8085"

:: --- optional model (router mode loads on demand; pass one as arg) -----------
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%\models\from_zero\from_zero_v0.1.f32.gguf"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=start"

:: ============================================================================
:: dispatch
:: ============================================================================
if /i "%MODE%"=="--status" goto :status
if /i "%MODE%"=="--stop"   goto :stop
if /i "%MODE%"=="start"    goto :start
echo [launcher] unknown argument: %MODE%
echo             use: SERVER-LAUNCHER ^| SERVER-LAUNCHER --status ^| SERVER-LAUNCHER --stop
goto :eof

:: ============================================================================
:: --status
:: ============================================================================
:status
echo [status] json_runtime :%JSON_RT_PORT%
call :is_listening %JSON_RT_PORT%
echo [status] kuhul-server :%GATEWAY_PORT%
call :is_listening %GATEWAY_PORT%
echo [status] kuhul_engine :%ENGINE_PORT%
call :is_listening %ENGINE_PORT%
echo [status] llama-server :%UI_PORT%
call :is_listening %UI_PORT%
goto :eof

:is_listening
netstat -ano | findstr /r /c:":%~1 .*LISTENING" >nul 2>&1
if errorlevel 1 (
    echo             down
) else (
    echo             LISTENING
)
goto :eof

:: ============================================================================
:: --stop
:: ============================================================================
:stop
echo [stop] stopping KUHUL APPS services...
taskkill /f /im json_runtime.exe >nul 2>&1
taskkill /f /im kuhul_engine.exe >nul 2>&1
taskkill /f /im llama-server.exe >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%GATEWAY_PORT% .*LISTENING"') do taskkill /f /pid %%p >nul 2>&1
echo [stop] done.
goto :eof

:: ============================================================================
:: start-all
:: ============================================================================
:start

call :is_listening %JSON_RT_PORT%
if not errorlevel 1 goto json_rt_up
if not exist "%JSON_RT_EXE%" goto json_rt_missing
echo [launcher] starting: json_runtime.exe --manifest manifest.json (:%JSON_RT_PORT%)
pushd "%JSON_RT_DIR%"
start "json_runtime" "..\json_runtime.exe" --manifest manifest.json
popd
goto gateway
:json_rt_up
echo [launcher] json_runtime already on :%JSON_RT_PORT%
goto gateway
:json_rt_missing
echo [launcher] ERROR json_runtime.exe not found: %JSON_RT_EXE%

:gateway
call :is_listening %GATEWAY_PORT%
if not errorlevel 1 goto gw_up
if not exist "%GATEWAY%" goto gw_missing
echo [launcher] starting: kuhul-server.cjs (:%GATEWAY_PORT%) — auto-starts kuhul_engine :%ENGINE_PORT%
pushd "%ROOT%"
start "kuhul-server" cmd /c "node dist\khanary-server\kuhul-server.cjs"
popd
goto engine_check
:gw_up
echo [launcher] kuhul-server already on :%GATEWAY_PORT%
goto engine_check
:gw_missing
echo [launcher] ERROR kuhul-server.cjs not found: %GATEWAY%

:engine_check
timeout /t 3 /nobreak >nul
call :is_listening %ENGINE_PORT%
if not errorlevel 1 goto engine_up
if not exist "%ENGINE_EXE%" goto engine_missing
echo [launcher] starting: kuhul_engine.exe --serve (:%ENGINE_PORT%)
start "kuhul_engine" "%ENGINE_EXE%" --serve %ENGINE_PORT%
goto ui
:engine_up
echo [launcher] kuhul_engine already on :%ENGINE_PORT%
goto ui
:engine_missing
echo [launcher] WARN kuhul_engine.exe not found: %ENGINE_EXE%
echo             (gateway retries it on its 60s tick)

:ui
call :is_listening %UI_PORT%
if not errorlevel 1 goto ui_up
if not exist "%UI_SERVER%" goto ui_missing
if exist "%MODEL%" (
    echo [launcher] starting: llama-server.exe -m %MODEL% (:%UI_PORT%, router mode)
    start "llama-server" "%UI_SERVER%" -m "%MODEL%" -ngl 999 --threads 4 --port %UI_PORT%
) else (
    echo [launcher] starting: llama-server.exe UI-only (:%UI_PORT%, router mode — model loads on demand)
    start "llama-server" "%UI_SERVER%" --port %UI_PORT%
)
goto summary
:ui_up
echo [launcher] llama-server already on :%UI_PORT%
goto summary
:ui_missing
echo [launcher] WARN llama-server.exe not found: %UI_SERVER%
echo             Rebuild first:  cd khanary-llama-build\llama.cpp ^&^& cmake --build build-ninja --target llama-server

:summary
echo.
echo [launcher] KUHUL APPS stack check complete.
echo             Studio UI : http://localhost:%UI_PORT%
echo             Engine    : http://127.0.0.1:%ENGINE_PORT%  ^(OpenAI-compatible^)
echo             Gateway   : http://127.0.0.1:%GATEWAY_PORT% ^(MCP / micronauts^)
echo             Hosting   : http://127.0.0.1:%JSON_RT_PORT% ^(json_runtime^)
echo.
echo             Use SERVER-LAUNCHER --status to re-check, --stop to shut down.
goto :eof
