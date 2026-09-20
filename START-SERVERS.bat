@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: START-SERVERS.bat -- KUHUL APPS stack
::
::   START-SERVERS                           start all services, GPT-OSS default
::   START-SERVERS "path\to\model.gguf"     start with specific model on :9000
::   START-SERVERS "path\to\model.gguf" 3080  start model on custom port
::   START-SERVERS --use-build [model] [port] launch using build llama-server.exe
::   START-SERVERS --use-dist  [model] [port] launch using dist khanary-server.exe
::   START-SERVERS --kson-contract <path>      apply KSON contract to main lane routing
::   START-SERVERS --contract-strict           fail startup if contract is not admitted
::   START-SERVERS --studio-use-build ...   launch studio lane (:25110) on build server
::   START-SERVERS --studio-use-dist  ...   launch studio lane (:25110) on dist server
::   START-SERVERS --status                  show which ports are listening
::   START-SERVERS --stop                    stop all managed services
::
:: All services launch as hidden background processes (no visible windows).
:: Logs written to logs\ inside the project root.
:: Force-kills existing instances before each start for a clean restart.
:: ============================================================================

set "ROOT=C:\Users\canna\_khanary_inspect"
set "LOGDIR=%ROOT%\logs"

:: --- binaries ----------------------------------------------------------------
set "JSON_RT_EXE=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json_runtime.exe"
set "JSON_RT_DIR=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\bin\json-runtime"
set "GATEWAY=%ROOT%\dist\khanary-server\kuhul-server.cjs"
set "ENGINE_EXE=%ROOT%\dist\v3.5.0-WebX\build\bin\Release\kuhul_engine.exe"
if not exist "%ENGINE_EXE%" set "ENGINE_EXE=C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\build-llama\bin\Release\kuhul_engine.exe"
for %%I in ("%ENGINE_EXE%") do set "ENGINE_DIR=%%~dpI"
set "ENGINE_WORKDIR=%ROOT%"
if exist "%ENGINE_DIR%" set "ENGINE_WORKDIR=%ENGINE_DIR%"
if exist "%ENGINE_DIR%" set "PATH=%ENGINE_DIR%;%PATH%"
set "DIST_SERVER=%ROOT%\dist\khanary-server\llama-server.exe"
set "BUILD_SERVER=%ROOT%\khanary-llama-build\llama.cpp\build\bin\Release\llama-server.exe"
set "DIST_RUNTIME_DIR=%ROOT%\dist\khanary-server"
set "DRIVERS_DIR=%ROOT%\dist\khanary-server\drivers"
if exist "%DIST_RUNTIME_DIR%" set "PATH=%DIST_RUNTIME_DIR%;%PATH%"
if exist "%DRIVERS_DIR%" set "PATH=%DRIVERS_DIR%;%PATH%"
set "XCFE_BACKEND_DLL=%DIST_RUNTIME_DIR%\ggml-xcfe.dll"
set "GL_OPS_DLL=%DIST_RUNTIME_DIR%\xcfe_gl_ops.dll"
set "RUNTIME_ENV_PREFIX="
if exist "%XCFE_BACKEND_DLL%" set "RUNTIME_ENV_PREFIX=$env:GGML_BACKEND_PATH='%XCFE_BACKEND_DLL%';"
if exist "%GL_OPS_DLL%" set "RUNTIME_ENV_PREFIX=%RUNTIME_ENV_PREFIX%$env:KHANARY_GL_OPS='%GL_OPS_DLL%';"

:: --- canonical model lanes --------------------------------------------------
set "GPT_OSS_MODEL=C:\Users\canna\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
set "LFM_MODEL=C:\Users\canna\.lmstudio\models\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf"
set "DOLPHIN_MODEL=E:\models\lmstudio\TheBloke\dolphin-2_6-phi-2-GGUF\dolphin-2_6-phi-2.Q5_K_S.gguf"
set "PHI3_MODEL=E:\models\lmstudio\microsoft\Phi-3-mini-4k-instruct-gguf\Phi-3-mini-4k-instruct-q4-micronaut.gguf"
set "CODER_MODEL=E:\models\GPT2\coder_micronaut\ultrachat_coder_skeleton_slerp_0p40.gguf"
set "GEMMA_MODEL=E:\models\GPT2\gemma-3-1b-it\gemma-3-1b-it-q8_0.gguf"
set "BASE_MODEL=%GEMMA_MODEL%"

:: --- ports -------------------------------------------------------------------
set "JSON_RT_PORT=8787"
set "GATEWAY_PORT=8764"
set "ENGINE_PORT=17480"

:: --- model arg ---------------------------------------------------------------
set "SERVER_MODE=dist"
set "SERVER_MODE_EXPLICIT=0"
set "STUDIO_SERVER_MODE=dist"
set "MODEL_CONTRACT_PATH="
set "CONTRACT_STRICT=0"
:parse_modes
if /i "%~1"=="--use-build" (
    set "SERVER_MODE=build"
    set "SERVER_MODE_EXPLICIT=1"
    shift
    goto :parse_modes
)
if /i "%~1"=="--use-dist" (
    set "SERVER_MODE=dist"
    set "SERVER_MODE_EXPLICIT=1"
    shift
    goto :parse_modes
)
if /i "%~1"=="--kson-contract" (
    if "%~2"=="" (
        echo  [contract]       WARN: --kson-contract provided without a path
        shift
        goto :parse_modes
    )
    set "MODEL_CONTRACT_PATH=%~2"
    shift
    shift
    goto :parse_modes
)
if /i "%~1"=="--contract-strict" (
    set "CONTRACT_STRICT=1"
    shift
    goto :parse_modes
)
if /i "%~1"=="--studio-use-build" (
    set "STUDIO_SERVER_MODE=build"
    shift
    goto :parse_modes
)
if /i "%~1"=="--studio-use-dist" (
    set "STUDIO_SERVER_MODE=dist"
    shift
    goto :parse_modes
)

set "MODEL_PATH=%~1"
set "MODEL_PORT=%~2"
if "%MODEL_PORT%"=="" set "MODEL_PORT=9000"
if "%MODEL_PATH%"=="" set "MODEL_PATH=%BASE_MODEL%"

:: --- resolve optional KSON model contract (main lane authority) -------------
set "CONTRACT_STATUS=none"
set "CONTRACT_PATH="
set "CONTRACT_PROVIDER="
set "CONTRACT_MODE=build"
set "CONTRACT_XCFE=0"
set "CONTRACT_CAPS="
set "CONTRACT_ENCODINGS="
set "CONTRACT_REASON=not_resolved"
set "MODEL_HEALTH_STATUS=unknown"
set "MODEL_HEALTH_REASON=not_checked"
set "MODEL_HEALTH_ENDPOINT=http://127.0.0.1:%MODEL_PORT%/v1/chat/completions"
set "MODEL_HEALTH_TEXT_LEN=0"

if /i not "%MODEL_PATH%"=="--status" if /i not "%MODEL_PATH%"=="--stop" (
    call :resolve_contract
    if errorlevel 1 exit /b 1
)

:: --- select main chat serving binary -----------------------------------------
set "UI_SERVER=%BUILD_SERVER%"
set "UI_SERVER_TAG=llama-server"
if /I "%SERVER_MODE%"=="build" (
    if not exist "%BUILD_SERVER%" (
        if exist "%DIST_SERVER%" (
            set "UI_SERVER=%DIST_SERVER%"
            set "UI_SERVER_TAG=khanary-server"
            echo  [startup]        WARN: build server missing; main chat falling back to dist server.
        )
    )
) else (
    set "UI_SERVER=%DIST_SERVER%"
    set "UI_SERVER_TAG=khanary-server"
    if not exist "%DIST_SERVER%" if exist "%BUILD_SERVER%" (
        set "UI_SERVER=%BUILD_SERVER%"
        set "UI_SERVER_TAG=llama-server"
        echo  [startup]        WARN: dist server missing; main chat falling back to build server.
    )
)
for %%I in ("%UI_SERVER%") do set "UI_SERVER_DIR=%%~dpI"

:: --- main lane runtime env gate ----------------------------------------------
set "MAIN_RUNTIME_ENV_PREFIX="
if /I "%CONTRACT_STATUS%"=="admitted" (
    if "%CONTRACT_XCFE%"=="1" set "MAIN_RUNTIME_ENV_PREFIX=%RUNTIME_ENV_PREFIX%"
) else (
    if /I "%SERVER_MODE%"=="dist" set "MAIN_RUNTIME_ENV_PREFIX=%RUNTIME_ENV_PREFIX%"
)

:: --- select studio lane serving binary (:25110) ------------------------------
set "STUDIO_SERVER=%DIST_SERVER%"
set "STUDIO_SERVER_TAG=khanary-server"
if /I "%STUDIO_SERVER_MODE%"=="build" (
    set "STUDIO_SERVER=%BUILD_SERVER%"
    set "STUDIO_SERVER_TAG=llama-server"
    if not exist "%BUILD_SERVER%" if exist "%DIST_SERVER%" (
        set "STUDIO_SERVER=%DIST_SERVER%"
        set "STUDIO_SERVER_TAG=khanary-server"
        echo  [startup]        WARN: studio build server missing; studio falling back to dist server.
    )
) else (
    if not exist "%DIST_SERVER%" if exist "%BUILD_SERVER%" (
        set "STUDIO_SERVER=%BUILD_SERVER%"
        set "STUDIO_SERVER_TAG=llama-server"
        echo  [startup]        WARN: studio dist server missing; studio falling back to build server.
    )
)
for %%I in ("%STUDIO_SERVER%") do set "STUDIO_SERVER_DIR=%%~dpI"

:: Dispatch
if /i "%MODEL_PATH%"=="--status" goto :status
if /i "%MODEL_PATH%"=="--stop"   goto :stop

:: ============================================================================
:: Start all services
:: ============================================================================
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1

echo.
echo  ============================================================
echo   KUHUL APPS stack starting...
echo  ============================================================
echo.

:: Kill named executables we own outright (force-kill for clean restart)
taskkill /f /im json_runtime.exe               >nul 2>&1
taskkill /f /im kuhul_engine.exe               >nul 2>&1
taskkill /f /im supernaut_native.exe           >nul 2>&1
taskkill /f /im Micronaut.Worker.Host.Http.exe >nul 2>&1

:: --- json_runtime :8787 -------------------------------------------------------
call :kill_port %JSON_RT_PORT%
if not exist "%JSON_RT_EXE%" (
    echo  [json_runtime]   WARN: not found
) else (
    echo  [json_runtime]   starting on :%JSON_RT_PORT%...
    powershell -NoProfile -NonInteractive -Command "Start-Process '%JSON_RT_EXE%' -ArgumentList '--manifest','manifest.json' -WorkingDirectory '%JSON_RT_DIR%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\json_runtime.log' -RedirectStandardError '%LOGDIR%\json_runtime.err'" >nul 2>&1
)

:: --- kuhul-server :8764 (MCP gateway) -----------------------------------------
call :kill_port %GATEWAY_PORT%
if not exist "%GATEWAY%" (
    echo  [kuhul-server]   WARN: not found
) else (
    echo  [kuhul-server]   starting MCP gateway on :%GATEWAY_PORT%...
    powershell -NoProfile -NonInteractive -Command "$env:KUHUL_REGISTRY_PORT='8764'; Start-Process node -ArgumentList 'dist\khanary-server\kuhul-server.cjs' -WorkingDirectory '%ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\kuhul-server.log' -RedirectStandardError '%LOGDIR%\kuhul-server.err'" >nul 2>&1
)

:: --- distil_provider :1236 ----------------------------------------------------
call :kill_port 1236
if not exist "%ROOT%\tools\distil_provider.py" (
    echo  [distil_provider] WARN: not found
) else (
    echo  [distil_provider] starting on :1236...
    powershell -NoProfile -NonInteractive -Command "Start-Process python -ArgumentList '%ROOT%\tools\distil_provider.py' -WorkingDirectory '%ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\distil_provider.log' -RedirectStandardError '%LOGDIR%\distil_provider.err'" >nul 2>&1
)

:: --- Supernaut orchestration stack :5774-5776 ---------------------------------
set "SN_DIR=%ROOT%\dist\supernaut-cpp"
call :bg_node   5774 "mn-supernaut" "%SN_DIR%"         "%SN_DIR%\run.mjs"
call :bg_python 5775 "mn-s7api    " "%SN_DIR%\supernaut_api.py"
call :bg_exe    5776 "mn-asxr     " "%SN_DIR%\native\build\bin\Debug" "%SN_DIR%\native\build\bin\Debug\supernaut_native.exe"

:: --- MM-* hive fleet + compiled coordinator/factory/executor -----------------
set "MNAUT_DIR=E:\models\micronaut\micronaut"
set "MN_KUHUL=%ROOT%\models\from_zero\from_zero_v0.6_kuhul.gguf"
set "MN_BASE=%CODER_MODEL%"
call :bg_llama 25500 "mm-coder   " "%CODER_MODEL%"
call :bg_llama 25501 "mm-gemma-env" "%GEMMA_MODEL%" "chatml"
call :bg_llama 25502 "mm-toolcall" "%LFM_MODEL%" "chatml"
call :bg_llama 25503 "mm-kuhul   " "%MN_KUHUL%"
call :bg_llama 25504 "mm-research" "%LFM_MODEL%" "chatml"
call :bg_llama 25505 "mm-agent   " "%DOLPHIN_MODEL%" "chatml"
call :bg_llama 25506 "mm-skill   " "%PHI3_MODEL%" "chatml"
call :bg_llama 25507 "mm-design  " "%GEMMA_MODEL%" "chatml"
call :bg_node   25100 "mn-coord   " "%MNAUT_DIR%" "%MNAUT_DIR%\coordinator.mjs"
call :bg_node   25101 "mn-factory " "%MNAUT_DIR%" "%MNAUT_DIR%\factory-micronaut.mjs"
call :bg_node   25103 "mn-executor" "%MNAUT_DIR%" "%MNAUT_DIR%\executor-micronaut.mjs"
call :bg_exe    5010  "mn-worker  " "%MNAUT_DIR%" "%MNAUT_DIR%\Micronaut.Worker.Host.Http.exe"

:: --- GC-1: Gemma Chat+Toolcall :25110 ----------------------------------------
set "GC1_MODEL=%GEMMA_MODEL%"
call :kill_port 25110
if not exist "%STUDIO_SERVER%" (
    echo  [gc-1]           WARN: %STUDIO_SERVER_TAG%.exe not found
) else if not exist "%GC1_MODEL%" (
    echo  [gc-1]           WARN: model not found at %GC1_MODEL%
) else (
    echo  [gc-1]           starting on :25110 using %STUDIO_SERVER_TAG%...
    powershell -NoProfile -NonInteractive -Command "%RUNTIME_ENV_PREFIX% Start-Process '%STUDIO_SERVER%' -ArgumentList @('-m','%GC1_MODEL%','--host','127.0.0.1','--port','25110','--ctx-size','8192','--parallel','4','--path','%ROOT%\studio-dist') -WorkingDirectory '%STUDIO_SERVER_DIR%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\gc-1.log' -RedirectStandardError '%LOGDIR%\gc-1.err'" >nul 2>&1
    echo {"port":25110,"model":"gemma-3-1b-it-q8_0.gguf","host":"127.0.0.1"} > "%ROOT%\active-model.json"
)

:: --- MCP-CS-1: C# MCP Server :8766 -------------------------------------------
set "MCP_CS_LAUNCH=%ROOT%\dist\micronauts\micronaut-mcp-cs\launch.bat"
call :kill_port 8766
if not exist "%MCP_CS_LAUNCH%" (
    echo  [mcp-cs-1]       WARN: not found
) else (
    echo  [mcp-cs-1]       starting on :8766...
    powershell -NoProfile -NonInteractive -Command "Start-Process cmd -ArgumentList @('/c','%MCP_CS_LAUNCH%') -WorkingDirectory '%ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\mcp-cs-1.log' -RedirectStandardError '%LOGDIR%\mcp-cs-1.err'" >nul 2>&1
)

:: --- kuhul_engine :17480 (give gateway ~3s to auto-start it) -----------------
timeout /t 3 /nobreak >nul
call :kill_port %ENGINE_PORT%
if not exist "%ENGINE_EXE%" (
    echo  [kuhul_engine]   WARN: not found
) else (
    echo  [kuhul_engine]   starting on :%ENGINE_PORT%...
    powershell -NoProfile -NonInteractive -Command "Start-Process '%ENGINE_EXE%' -ArgumentList '--serve','%ENGINE_PORT%' -WorkingDirectory '%ENGINE_WORKDIR%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\kuhul_engine.log' -RedirectStandardError '%LOGDIR%\kuhul_engine.err'" >nul 2>&1
)

:: --- llama-server / khanary-server :<MODEL_PORT> ------------------------------
call :kill_port %MODEL_PORT%
if not exist "%UI_SERVER%" (
    echo  [%UI_SERVER_TAG%] WARN: not found
    set "MODEL_HEALTH_STATUS=degraded"
    set "MODEL_HEALTH_REASON=server_binary_missing"
) else if exist "%MODEL_PATH%" (
    echo  [%UI_SERVER_TAG%] starting with model on :%MODEL_PORT%...
    powershell -NoProfile -NonInteractive -Command "%MAIN_RUNTIME_ENV_PREFIX% Start-Process '%UI_SERVER%' -ArgumentList @('-m','%MODEL_PATH%','-ngl','0','--threads','4','--port','%MODEL_PORT%','--repeat-penalty','1.3','--repeat-last-n','64','--ui-mcp-proxy') -WorkingDirectory '%UI_SERVER_DIR%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\khanary-server.log' -RedirectStandardError '%LOGDIR%\khanary-server.err'" >nul 2>&1
    call :probe_output_health
) else (
    echo  [%UI_SERVER_TAG%] starting UI-only on :%MODEL_PORT%...
    powershell -NoProfile -NonInteractive -Command "%MAIN_RUNTIME_ENV_PREFIX% Start-Process '%UI_SERVER%' -ArgumentList @('--port','%MODEL_PORT%','--ui-mcp-proxy') -WorkingDirectory '%UI_SERVER_DIR%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\khanary-server.log' -RedirectStandardError '%LOGDIR%\khanary-server.err'" >nul 2>&1
    set "MODEL_HEALTH_STATUS=skipped"
    set "MODEL_HEALTH_REASON=ui_only_no_model"
)

:: ============================================================================
:: Write active-model.json
:: ============================================================================
set "ACTIVE_JSON=%ROOT%\active-model.json"
set "CONTRACT_PATH_ESC=%CONTRACT_PATH%"
if defined CONTRACT_PATH_ESC set "CONTRACT_PATH_ESC=%CONTRACT_PATH_ESC:\=\\%"
(
    echo {
    echo   "endpoint": "http://127.0.0.1:%MODEL_PORT%/v1/chat/completions",
    echo   "port": %MODEL_PORT%,
    echo   "model_path": "%MODEL_PATH:\=\\%",
    echo   "contract_status": "%CONTRACT_STATUS%",
    echo   "contract_provider": "%CONTRACT_PROVIDER%",
    echo   "contract_mode": "%CONTRACT_MODE%",
    echo   "contract_path": "%CONTRACT_PATH_ESC%",
    echo   "output_health": "%MODEL_HEALTH_STATUS%",
    echo   "output_health_reason": "%MODEL_HEALTH_REASON%",
    echo   "output_health_endpoint": "%MODEL_HEALTH_ENDPOINT%",
    echo   "output_health_text_len": %MODEL_HEALTH_TEXT_LEN%,
    echo   "planner_endpoint": "http://127.0.0.1:%GATEWAY_PORT%/api/plan",
    echo   "mcp_gateway": "http://127.0.0.1:%GATEWAY_PORT%",
    echo   "engine_endpoint": "http://127.0.0.1:%ENGINE_PORT%/v1/chat/completions"
    echo }
) > "%ACTIVE_JSON%"

:: ============================================================================
:: Write chat.manifest.json — canonical "is chat op?" file read by PRIMEOS
:: ============================================================================
(
    echo {
    echo   "@kind": "primeos.chat.manifest.v1",
    echo   "@status": "op",
    echo   "@written_by": "START-SERVERS.bat",
    echo   "gateway": {
    echo     "url": "http://127.0.0.1:%GATEWAY_PORT%",
    echo     "chat": "http://127.0.0.1:%GATEWAY_PORT%/v1/chat/completions",
    echo     "models": "http://127.0.0.1:%GATEWAY_PORT%/v1/models",
    echo     "health": "http://127.0.0.1:%GATEWAY_PORT%/health",
    echo     "tool_chat": "http://127.0.0.1:%GATEWAY_PORT%/v1/tool-chat",
    echo     "forge": "http://127.0.0.1:%GATEWAY_PORT%/v1/forge",
    echo     "forge_chat": "http://127.0.0.1:%GATEWAY_PORT%/v1/forge/chat",
    echo     "plan": "http://127.0.0.1:%GATEWAY_PORT%/api/plan"
    echo   },
    echo   "engine": {
    echo     "url": "http://127.0.0.1:%ENGINE_PORT%",
    echo     "chat": "http://127.0.0.1:%ENGINE_PORT%/v1/chat/completions",
    echo     "health": "http://127.0.0.1:%ENGINE_PORT%/health"
    echo   },
    echo   "active_model": {
    echo     "port": %MODEL_PORT%,
    echo     "chat": "http://127.0.0.1:%MODEL_PORT%/v1/chat/completions",
    echo     "health": "http://127.0.0.1:%MODEL_PORT%/health",
    echo     "output_health": "%MODEL_HEALTH_STATUS%"
    echo   },
    echo   "fallback_chain": [
    echo     "http://127.0.0.1:%GATEWAY_PORT%/v1/chat/completions",
    echo     "http://127.0.0.1:%MODEL_PORT%/v1/chat/completions",
    echo     "http://127.0.0.1:25110/v1/chat/completions",
    echo     "http://127.0.0.1:%ENGINE_PORT%/v1/chat/completions"
    echo   ],
    echo   "programs_api": "programs/api.manifest.json",
    echo   "model_manifest": "programs/manifest.json",
    echo   "capability_routes": "programs/model-capability-routes.json"
    echo }
) > "%ROOT%\chat.manifest.json"

goto :summary

:: ============================================================================
:: --status
:: ============================================================================
:status
echo.
echo  [status] KUHUL APPS stack:
call :check_port %JSON_RT_PORT%  "json_runtime    "
call :check_port %GATEWAY_PORT%  "kuhul-server    "
call :check_port %ENGINE_PORT%   "kuhul_engine    "
call :check_port 1236            "distil_provider "
echo.
echo  [status] Supernaut:
call :check_port 5774            "mn-supernaut    "
call :check_port 5775            "mn-s7api        "
call :check_port 5776            "mn-asxr         "
echo.
echo  [status] MM-* hive fleet:
call :check_port 25100           "mn-coord        "
call :check_port 25101           "mn-factory      "
call :check_port 25103           "mn-executor     "
call :check_port 5010            "mn-worker       "
call :check_port 25500           "mm-coder        "
call :check_port 25501           "mm-gemma-env    "
call :check_port 25502           "mm-toolcall     "
call :check_port 25503           "mm-kuhul        "
call :check_port 25504           "mm-research     "
call :check_port 25505           "mm-agent        "
call :check_port 25506           "mm-skill        "
call :check_port 25507           "mm-design       "
echo.
echo  [status] Extended:
call :check_port 25110           "gc-1            "
call :check_port 8766            "mcp-cs-1        "
echo.
echo  [status] Inference server:
set "STATUS_MODEL_PORT=9000"
if exist "%ROOT%\active-model.json" (
    for /f %%P in ('powershell -NoProfile -NonInteractive -Command "$raw = [System.IO.File]::ReadAllText('%ROOT%\active-model.json'); $j = ConvertFrom-Json -InputObject $raw; [int]$j.port" 2^>nul') do set "STATUS_MODEL_PORT=%%P"
)
call :check_port %STATUS_MODEL_PORT% "active-model    "
if exist "%ROOT%\active-model.json" (
    echo.
    type "%ROOT%\active-model.json"
)
echo.
goto :eof

:: ============================================================================
:: --stop
:: ============================================================================
:stop
echo.
echo  [stop] Stopping all KUHUL APPS services...
set "ACTIVE_STOP_PORT=9000"
if exist "%ROOT%\active-model.json" (
    for /f %%P in ('powershell -NoProfile -NonInteractive -Command "$raw = [System.IO.File]::ReadAllText('%ROOT%\active-model.json'); $j = ConvertFrom-Json -InputObject $raw; [int]$j.port" 2^>nul') do set "ACTIVE_STOP_PORT=%%P"
)
taskkill /f /im json_runtime.exe               >nul 2>&1
taskkill /f /im kuhul_engine.exe               >nul 2>&1
taskkill /f /im supernaut_native.exe           >nul 2>&1
taskkill /f /im Micronaut.Worker.Host.Http.exe >nul 2>&1
call :kill_port %ACTIVE_STOP_PORT%
for %%H in (%GATEWAY_PORT% 1236 5774 5775 5776 25100 25101 25103 5010 25110 25500 25501 25502 25503 25504 25505 25506 25507 8766 9000) do (
    call :kill_port %%H
)
if exist "%ROOT%\active-model.json" del /f /q "%ROOT%\active-model.json"
echo  [stop] Done.
echo.
goto :eof

:: ============================================================================
:: Summary -- shown after startup
:: ============================================================================
:summary
set "STACK_STATE=ready"
if /I not "%MODEL_HEALTH_STATUS%"=="healthy" set "STACK_STATE=degraded"
echo.
echo  ============================================================
echo   KUHUL APPS Stack -- %STACK_STATE%
echo  ============================================================
echo.
echo   Main chat      http://127.0.0.1:%MODEL_PORT%
echo   Canvas         http://127.0.0.1:%MODEL_PORT%/chat/[id]/canvas
echo   Studio lane    http://127.0.0.1:25110
echo.
echo   MCP Gateway    http://127.0.0.1:%GATEWAY_PORT%
echo   Planner API    http://127.0.0.1:%GATEWAY_PORT%/api/plan
echo   Engine         http://127.0.0.1:%ENGINE_PORT%
echo   Runtime        http://127.0.0.1:%JSON_RT_PORT%
echo   Distil         http://127.0.0.1:1236
echo   Active model   http://127.0.0.1:%MODEL_PORT%/v1/chat/completions
echo   Output health  %MODEL_HEALTH_STATUS% (%MODEL_HEALTH_REASON%)
echo.
echo   Supernaut      mn-supernaut :5774   mn-s7api :5775   mn-asxr :5776
echo   Compiled       mn-coord :25100   mn-factory :25101   mn-executor :25103   mn-worker :5010
echo   Hive           mm-* :25500-25507
echo   Extended       gc-1 :25110   mcp-cs-1 :8766
echo.
echo   Logs           %LOGDIR%
echo.
echo   Claude Desktop -- add to MCP servers:
echo     http://127.0.0.1:%GATEWAY_PORT%
echo.
echo  ============================================================
echo   All services running in the background.
echo   You can close this window now.
echo  ============================================================
echo.

endlocal
goto :eof

:: ============================================================================
:: Subroutines
:: ============================================================================

:bg_llama
:: %1=port %2=label %3=model %4=chatml (pass "chatml" to add --chat-template chatml)
call :kill_port %~1
if not exist "%DIST_SERVER%" (
    echo  [%~2] WARN: khanary-server not found
    goto :eof
)
if not exist "%~3" (
    echo  [%~2] WARN: model not found %~3
    goto :eof
)
set "BGL_CHAT_ARG="
if /i "%~4"=="chatml" set "BGL_CHAT_ARG=,'--chat-template','chatml'"
echo  [%~2] starting on :%~1...
powershell -NoProfile -NonInteractive -Command "%RUNTIME_ENV_PREFIX% Start-Process '%DIST_SERVER%' -ArgumentList @('-m','%~3','--host','127.0.0.1','--port','%~1','-ngl','0','-t','4','--parallel','2','--ctx-size','4096'%BGL_CHAT_ARG%) -WorkingDirectory '%DIST_RUNTIME_DIR%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\%~2.log' -RedirectStandardError '%LOGDIR%\%~2.err'" >nul 2>&1
goto :eof

:kill_port
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /r /c:":%~1 .*LISTENING"') do (
    taskkill /f /pid %%p >nul 2>&1
)
goto :eof

:bg_node
:: %1=port %2=label %3=workdir %4=script
call :kill_port %~1
if not exist "%~4" (
    echo  [%~2] WARN: not found %~4
    goto :eof
)
echo  [%~2] starting on :%~1...
powershell -NoProfile -NonInteractive -Command "Start-Process node -ArgumentList '%~4' -WorkingDirectory '%~3' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\%~2.log' -RedirectStandardError '%LOGDIR%\%~2.err'" >nul 2>&1
goto :eof

:bg_python
:: %1=port %2=label %3=script
call :kill_port %~1
if not exist "%~3" (
    echo  [%~2] WARN: not found %~3
    goto :eof
)
echo  [%~2] starting on :%~1...
powershell -NoProfile -NonInteractive -Command "Start-Process python -ArgumentList '%~3' -WorkingDirectory '%ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\%~2.log' -RedirectStandardError '%LOGDIR%\%~2.err'" >nul 2>&1
goto :eof

:bg_exe
:: %1=port %2=label %3=workdir %4=exe
call :kill_port %~1
if not exist "%~4" (
    echo  [%~2] WARN: not found %~4
    goto :eof
)
echo  [%~2] starting on :%~1...
powershell -NoProfile -NonInteractive -Command "Start-Process '%~4' -WorkingDirectory '%~3' -WindowStyle Hidden -RedirectStandardOutput '%LOGDIR%\%~2.log' -RedirectStandardError '%LOGDIR%\%~2.err'" >nul 2>&1
goto :eof

:resolve_contract
set "CONTRACT_STRICT_ARG="
if "%CONTRACT_STRICT%"=="1" set "CONTRACT_STRICT_ARG=--strict"
set "CONTRACT_TMP=%TEMP%\khanary-contract-%RANDOM%-%RANDOM%.env"

if "%MODEL_CONTRACT_PATH%"=="" (
    python "%ROOT%\tools\resolve_llama_contract.py" --model "%MODEL_PATH%" %CONTRACT_STRICT_ARG% > "%CONTRACT_TMP%"
) else (
    python "%ROOT%\tools\resolve_llama_contract.py" --model "%MODEL_PATH%" --kson "%MODEL_CONTRACT_PATH%" %CONTRACT_STRICT_ARG% > "%CONTRACT_TMP%"
)

set "CONTRACT_RC=%ERRORLEVEL%"
if exist "%CONTRACT_TMP%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%CONTRACT_TMP%") do (
        set "%%A=%%B"
    )
    del /f /q "%CONTRACT_TMP%" >nul 2>&1
)

if not "%CONTRACT_RC%"=="0" (
    if "%CONTRACT_STRICT%"=="1" (
        echo  [contract]       ERROR: contract strict mode rejected model contract: %CONTRACT_REASON%.
        exit /b 1
    ) else (
        echo  [contract]       WARN: contract not admitted: %CONTRACT_REASON%; using safe main lane defaults.
    )
)

if /I "%CONTRACT_STATUS%"=="admitted" (
    echo  [contract]       admitted provider=%CONTRACT_PROVIDER% mode=%CONTRACT_MODE% xcfe=%CONTRACT_XCFE%
    if "%SERVER_MODE_EXPLICIT%"=="0" (
        set "SERVER_MODE=%CONTRACT_MODE%"
    ) else (
        echo  [contract]       note: explicit --use-* preserved main lane mode=%SERVER_MODE%
    )
)
goto :eof

:probe_output_health
set "HEALTH_TMP=%TEMP%\khanary-health-%RANDOM%-%RANDOM%.env"
python "%ROOT%\tools\probe_llama_output_health.py" --base-url "http://127.0.0.1:%MODEL_PORT%" > "%HEALTH_TMP%"
set "HEALTH_RC=%ERRORLEVEL%"
if exist "%HEALTH_TMP%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%HEALTH_TMP%") do (
        set "%%A=%%B"
    )
    del /f /q "%HEALTH_TMP%" >nul 2>&1
)
if /I "%MODEL_HEALTH_STATUS%"=="healthy" (
    echo  [health]         healthy endpoint=%MODEL_HEALTH_ENDPOINT% text_len=%MODEL_HEALTH_TEXT_LEN%
) else (
    echo  [health]         WARN: output health %MODEL_HEALTH_STATUS% reason=%MODEL_HEALTH_REASON%
)
if not "%HEALTH_RC%"=="0" exit /b 0
goto :eof

:check_port
netstat -ano 2>nul | findstr /r /c:":%~1 .*LISTENING" >nul 2>&1
if errorlevel 1 (
    echo    %~2  :%~1   DOWN
) else (
    echo    %~2  :%~1   UP
)
goto :eof
