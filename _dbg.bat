setlocal enabledelayedexpansion

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
::   2. kuhul-server.cjs   :8764  gateway — auto-starts kuhul_engine :17480 (watchdog)
::   3. kuhul_engine.exe   :17480 OpenAI-compatible chat origin (started by gateway;
::                              direct fallback below if the gateway cannot)
::   4. llama-server.exe   :9000  embedded KUHUL APPS UI (router mode — a model is
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

:: --- ports (override any with UI_PORT= etc.) --------------------------------
if "%JSON_RT_PORT%"=="" set "JSON_RT_PORT=8787"
if "%GATEWAY_PORT%"=="" set "GATEWAY_PORT=8764"
if "%ENGINE_PORT%"==""  set "ENGINE_PORT=17480"
if "%UI_PORT%"==""      set "UI_PORT=9000"

:: --- optional model for the UI server (router mode; inference on demand) ----
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%\models\from_zero\from_zero_v0.1.f32.gguf"

:: ============================================================================
:: helpers
:: ============================================================================

:is_port_listening
set "port=%~1"
netstat -ano | findstr /r /c:":%port% .*LISTENING" >nul 2>&1
if %errorlevel%==0 ( exit /b 0 ) else ( exit /b 1 )

:log_start
echo [launcher] starting: %~1
exit /b 0

:: ============================================================================
:: --status
:: ============================================================================
if /i "%~1"=="--status" (
    echo [status] json_runtime :%JSON_RT_PORT%
    call :is_port_listening %JSON_RT_PORT% && echo            LISTENING || echo            down
    echo [status] kuhul-server :%GATEWAY_PORT%
    call :is_port_listening %GATEWAY_PORT% && echo            LISTENING || echo            down
    echo [status] kuhul_engine :%ENGINE_PORT%
    call :is_port_listening %ENGINE_PORT% && echo            LISTENING || echo            down
    echo [status] llama-server :%UI_PORT%
    call :is_port_listening %UI_PORT% && echo            LISTENING || echo            down
    exit /b 0
)

:: ============================================================================
:: --stop
:: ============================================================================
if /i "%~1"=="--stop" (
    echo [stop] stopping KUHUL APPS services...
    taskkill /f /im json_runtime.exe    >nul 2>&1
    taskkill /f /im kuhul_engine.exe    >nul 2>&1
    taskkill /f /im llama-server.exe    >nul 2>&1
    :: stop the gateway node process (kuhul-server.cjs) — match by port first
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%GATEWAY_PORT% .*LISTENING"') do taskkill /f /pid %%p >nul 2>&1
    echo [stop] done.
    exit /b 0
)

:: ============================================================================
:: 1. json_runtime :8787
:: ============================================================================
call :is_port_listening %JSON_RT_PORT%
if %errorlevel%==0 (
    echo [launcher] json_runtime already on :%JSON_RT_PORT%
) else (
    if not exist "%JSON_RT_EXE%" (
        echo [launcher] ERROR json_runtime.exe not found: %JSON_RT_EXE%
    ) else (
        call :log_start "json_runtime.exe --manifest manifest.json (:%JSON_RT_PORT%)"
        pushd "%JSON_RT_DIR%"
        start "json_runtime" "..\json_runtime.exe" --manifest manifest.json
        popd
    )
)

:: ============================================================================
:: 2. kuhul-server gateway :8764  (auto-starts kuhul_engine :17480 + watchdog)
:: ============================================================================
call :is_port_listening %GATEWAY_PORT%
if %errorlevel%==0 (
    echo [launcher] kuhul-server already on :%GATEWAY_PORT%
) else (
    if not exist "%GATEWAY%" (
        echo [launcher] ERROR kuhul-server.cjs not found: %GATEWAY%
    ) else (
        call :log_start "kuhul-server.cjs (:%GATEWAY_PORT%)"
        pushd "%ROOT%"
        start "kuhul-server" cmd /c "node dist\khanary-server\kuhul-server.cjs"
        popd
        echo [launcher] gateway starting — kuhul_engine auto-starts on :%ENGINE_PORT%
    )
)

:: ============================================================================
:: 3. kuhul_engine :17480 — direct fallback if the gateway is not up to do it
:: ============================================================================
timeout /t 3 /nobreak >nul
call :is_port_listening %ENGINE_PORT%
if %errorlevel%==0 (
    echo [launcher] kuhul_engine already on :%ENGINE_PORT%
) else (
    if exist "%ENGINE_EXE%" (
        call :log_start "kuhul_engine.exe --serve (:%ENGINE_PORT%)"
        start "kuhul_engine" "%ENGINE_EXE%" --serve %ENGINE_PORT%
    ) else (
        echo [launcher] WARN kuhul_engine.exe not found: %ENGINE_EXE%
        echo             (gateway will retry it on its 60s tick)
    )
)

:: ============================================================================
:: 4. llama-server :%UI_PORT% — embedded KUHUL APPS UI (router mode, model optional)
:: ============================================================================
call :is_port_listening %UI_PORT%
if %errorlevel%==0 (
    echo [launcher] llama-server already on :%UI_PORT%
) else (
    if not exist "%UI_SERVER%" (
        echo [launcher] WARN llama-server.exe not found: %UI_SERVER%
        echo             Rebuild first:  cd khanary-llama-build\llama.cpp ^&^& cmake --build build-ninja --target llama-server
    ) else (
        if exist "%MODEL%" (
            call :log_start "llama-server.exe -m %MODEL% (:%UI_PORT%, router mode)"
            start "llama-server" "%UI_SERVER%" -m "%MODEL%" -ngl 999 --threads 4 --port %UI_PORT%
        ) else (
            call :log_start "llama-server.exe UI-only (:%UI_PORT%, router mode — model loads on demand)"
            start "llama-server" "%UI_SERVER%" --port %UI_PORT%
        )
    )
)

echo.
echo [launcher] KUHUL APPS stack check complete.
echo             Studio UI : http://localhost:%UI_PORT%
echo             Engine    : http://127.0.0.1:%ENGINE_PORT%  ^(OpenAI-compatible^)
echo             Gateway   : http://127.0.0.1:%GATEWAY_PORT% ^(MCP / micronauts^)
echo             Hosting   : http://127.0.0.1:%JSON_RT_PORT% ^(json_runtime^)
echo.
echo             Use SERVER-LAUNCHER --status to re-check, --stop to shut down.
endlocal
