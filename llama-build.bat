@echo off
setlocal

:: llama-build.bat ? rebuild llama-server with a fresh webUI baked in
::
:: WHY THIS EXISTS:
::   The webUI (tools/ui/ SvelteKit app) is compiled into the llama-server
::   binary at build time via llama-ui-embed. CMake checks priority order:
::     1. tools/ui/dist/index.html present? ? bake it immediately (NO npm run)
::     2. build/tools/ui/.ui-stamp exists?  ? npm build skipped as "up-to-date"
::     3. npm run build                     ? only if both above are absent
::   Result: every previous build leaves dist/ and the stamp behind, so cmake
::   always reuses stale assets. You must delete them to force a real rebuild.
::
:: Usage:
::   llama-build          ? clear stale UI, npm build, cmake llama-server only
::   llama-build full     ? clear stale UI, npm build, full GPU reconfigure + rebuild
::   llama-build clean    ? wipe CMakeCache too, then full rebuild
::   llama-build deploy   ? copy fresh binary + DLLs to dist/khanary-server/
::   llama-build minimal  ? inference-only server (LLAMA_BUILD_UI=OFF): no embedded
::                          web UI, no npm. Studio serves the UI. Faster, smaller.

set ROOT=%~dp0
set SRC=%ROOT%khanary-llama-build\llama.cpp
set BUILD=%SRC%\build
set UI_SRC=%SRC%\tools\ui
set UI_BUILD=%BUILD%\tools\ui
set DML_SRC=%ROOT%scratch\dml
set GGML_SRC=%ROOT%khanary-llama-build\ggml
set GGML_BUILD=%GGML_SRC%\build
set BIN_OUT=%SRC%\build-ninja\bin

:: ?? minimal mode: skip stale-UI clearing + npm entirely ????????????????
set MINIMAL=0
if /i "%1"=="minimal" set MINIMAL=1
if /i "%2"=="minimal" set MINIMAL=1

:: Locate cmake (bundled with VS 2022 BuildTools or standalone install)
set cmake="C:\Program Files\CMake\bin\cmake.exe"
if not exist %cmake% set cmake="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist %cmake% (
    echo [llama-build] cmake.exe not found. Install CMake or add VS BuildTools.
    exit /b 1
)

:: ?? 0. Kill any running server that would lock the output binary ?????????????
::   khanary-server.exe / llama-server.exe hold a Windows file lock on themselves.
::   If they are running when cmake links or when deploy copies, the write fails silently.
echo [llama-build] Stopping any running server (unlock binary for overwrite)...
taskkill /f /im khanary-server.exe >nul 2>&1
taskkill /f /im llama-server.exe   >nul 2>&1
timeout /t 1 /nobreak >nul

:: ?? 1. Clear stale cmake webUI embed artifacts ???????????????????????????????
::   cmake priority-1 bakes tools\ui\dist\ into the binary if present.
::   Deleting the stamp + generated files forces cmake to rebuild the embed.
::   kuhul.dev now serves studio-dist\ directly (no npm build needed here).
if "%MINIMAL%"=="1" goto skip_ui_steps
echo [llama-build] Clearing stale cmake UI embed artifacts...
if exist "%UI_BUILD%\.ui-stamp"     del   /f /q "%UI_BUILD%\.ui-stamp"
if exist "%UI_BUILD%\ui.cpp"        del   /f /q "%UI_BUILD%\ui.cpp"
if exist "%UI_BUILD%\ui.h"          del   /f /q "%UI_BUILD%\ui.h"
echo [llama-build] cmake embed artifacts cleared.
echo.
:skip_ui_steps

:: ?? 2b. KLSL / DirectML forward pass ? rebuild dml_gemm.dll ?????????????????
::   dml_gemm.dll exposes dml_gemm_bt_f32 (C-ABI), loaded at runtime by ggml-xcfe.dll
::   via LoadLibraryA. It carries the full KLSL forward pass: DirectML GEMM with an
::   amortised D3D12+DML device, per-shape resource cache, and GPU-resident weight store.
::   Built with vcvars64 + cl.exe (MSVC); DirectML.lib links the driver-provided DML layer.
::   Source: scratch\dml\dml_gemm_dll.cpp
echo [llama-build] Bootstrapping MSVC toolchain for dml_gemm.dll build...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [llama-build] WARNING: vcvars64.bat not found ^(neither BuildTools nor Community^)
    echo [llama-build]   dml_gemm.dll will NOT be rebuilt ? existing copy will be used if present.
    goto skip_dml_build
)
echo [llama-build] Building dml_gemm.dll ^(KLSL DirectML forward pass^)...
pushd "%DML_SRC%"
cl /nologo /std:c++17 /EHsc /O2 /I include /LD dml_gemm_dll.cpp /link /LIBPATH:lib DirectML.lib /OUT:dml_gemm.dll
set DML_RC=%ERRORLEVEL%
popd
if %DML_RC% neq 0 (
    echo [llama-build] dml_gemm.dll build FAILED ^(exit %DML_RC%^). Check MSVC + DirectML headers.
    exit /b %DML_RC%
)
echo [llama-build] dml_gemm.dll built ^(%DML_SRC%\dml_gemm.dll^)
echo.

:: ?? 2c. Native D3D11 inference DLL ? rebuild d3d11_infer.dll ?????????????????
::   d3d11_infer.dll goes directly to igd10iumd64.dll (the Intel HD 4600 native
::   D3D11 game driver ? same path WoW and other DX11 games use). No DirectML,
::   no D3D12 compat shim. Compiles 7 cs_5_0 shaders at init time via D3DCompiler.
::   Ops: gemm / embed / layernorm / attention / gelu / add_bias / add (residual).
::   Source: scratch\dml\d3d11_infer_dll.cpp
echo [llama-build] Building d3d11_infer.dll ^(native D3D11 cs_5_0 path^)...
pushd "%DML_SRC%"
cl /nologo /std:c++17 /EHsc /O2 /LD d3d11_infer_dll.cpp /link d3d11.lib D3DCompiler.lib /OUT:d3d11_infer.dll
set D3D11_RC=%ERRORLEVEL%
popd
if %D3D11_RC% neq 0 (
    echo [llama-build] WARNING: d3d11_infer.dll build FAILED ^(exit %D3D11_RC%^) ? native D3D11 path unavailable
) else (
    echo [llama-build] d3d11_infer.dll built ^(%DML_SRC%\d3d11_infer.dll^)
)
echo.

:: ?? 2d. Khanary native driver DLLs ?????????????????????????????????????????????
::   khanary_driver.dll       ? TaskEngine + DAG + provider dispatch
::   khanary_glyph_driver.dll ? K'UHUL phase/fold glyph + compute lane registry
::   kuhul_engine_driver.dll  ? Atomic DOM manifest loader + model status
::   gl_infer_driver.dll      ? OpenGL 4.3 compute probe
::   qwen_infer_driver.dll    ? Qwen architecture probe
::   Loaded by kuhul-server.cjs via koffi (ffi-shim). Build only if the C++ sources exist.
echo [llama-build] Building Khanary native driver DLLs...
if exist "%ROOT%drivers\build_drivers.bat" (
    pushd "%ROOT%drivers"
    call build_drivers.bat
    set DRV_RC=%ERRORLEVEL%
    popd
    if %DRV_RC% neq 0 (
        echo [llama-build] WARNING: one or more driver DLLs failed to build ^(exit %DRV_RC%^)
    ) else (
        echo [llama-build] Driver DLLs built
    )
) else (
    echo [llama-build] NOTE: drivers\build_drivers.bat not found
)
echo.

:skip_dml_build

:: ?? 3. cmake rebuild ?????????????????????????????????????????????????????????
if /i "%1"=="clean"  goto clean_rebuild
if /i "%1"=="full"   goto full_rebuild
if /i "%1"=="deploy" goto deploy_only
if "%MINIMAL%"=="1"  goto minimal_rebuild

:fast_rebuild
echo [llama-build] Rebuilding llama-server (incremental, no reconfigure)...
%cmake% --build "%BUILD%" --config Release --target llama-server -j 4
if %ERRORLEVEL% neq 0 (
    echo [llama-build] cmake build FAILED ^(exit %ERRORLEVEL%^)
    exit /b %ERRORLEVEL%
)
goto done

:minimal_rebuild
echo [llama-build] Inference-only rebuild (LLAMA_BUILD_UI=OFF)...
%cmake% -S "%SRC%" -B "%BUILD%" -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo [llama-build] cmake reconfigure FAILED ^(exit %ERRORLEVEL%^)
    exit /b %ERRORLEVEL%
)
%cmake% --build "%BUILD%" --config Release --target llama-server -j 4
if %ERRORLEVEL% neq 0 (
    echo [llama-build] cmake build FAILED ^(exit %ERRORLEVEL%^)
    exit /b %ERRORLEVEL%
)
goto done

:clean_rebuild
echo [llama-build] Wiping CMakeCache for clean rebuild...
if exist "%BUILD%\CMakeCache.txt" del /f /q "%BUILD%\CMakeCache.txt"
:: fall through to full_rebuild

:full_rebuild
if "%MINIMAL%"=="1" (
    echo [llama-build] Full GPU rebuild, inference-only (UiOff)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%khanary-llama-build\build_gpu.ps1" -UiOff
) else (
    echo [llama-build] Full GPU rebuild (OpenCL reconfigure via build_gpu.ps1)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%khanary-llama-build\build_gpu.ps1"
)
if %ERRORLEVEL% neq 0 (
    echo [llama-build] Full rebuild FAILED ^(exit %ERRORLEVEL%^)
    exit /b %ERRORLEVEL%
)
goto done

:done
:: ?? 4a. Auto-deploy: copy fresh binary + DLLs to dist\khanary-server\ every build
::   Only reached on successful build -- safe to delete the old binary here.
::   llama-server.exe is a thin DLL-entry stub; real impl is llama-server-impl.dll.
::   All DLLs from build-ninja\bin must travel with the exe.
set DEPLOY_DIR=%ROOT%dist\khanary-server
if not exist "%DEPLOY_DIR%" mkdir "%DEPLOY_DIR%"
if exist "%BIN_OUT%\llama-server.exe" (
    if exist "%DEPLOY_DIR%\khanary-server.exe" del /f /q "%DEPLOY_DIR%\khanary-server.exe"
    copy /y "%BIN_OUT%\llama-server.exe" "%DEPLOY_DIR%\khanary-server.exe" >nul
    for %%f in ("%BIN_OUT%\*.dll") do copy /y "%%f" "%DEPLOY_DIR%\" >nul
    echo [llama-build] Deployed: llama-server.exe + DLLs -^> dist\khanary-server\
) else (
    echo [llama-build] WARNING: %BIN_OUT%\llama-server.exe not found -- deploy skipped
)

:: ?? 4. Copy KLSL / DirectML runtime DLLs alongside llama-server.exe ?????????
::   ggml-xcfe.dll calls LoadLibraryA("dml_gemm.dll") at first MUL_MAT dispatch.
::   If either DLL is missing the backend silently falls back to CPU.
::   dml_gemm.dll  ? KLSL forward pass (GEMM, attention, MLP, layernorm via DirectML)
::   DirectML.dll  ? Microsoft's DirectML runtime (driver-agnostic GPU ML ops)
echo [llama-build] Copying GPU runtime DLLs to %BIN_OUT%\...
if exist "%DML_SRC%\dml_gemm.dll" (
    copy /y "%DML_SRC%\dml_gemm.dll" "%BIN_OUT%\" >nul
    echo [llama-build]   dml_gemm.dll  ^(KLSL DirectML forward pass^)
) else (
    echo [llama-build]   WARNING: %DML_SRC%\dml_gemm.dll not found ? GPU matmul will use CPU fallback
)
:: Also copy ggml GPU backends from build output
if exist "%BIN_OUT%\ggml-opencl.dll" (
    copy /y "%BIN_OUT%\ggml-opencl.dll" "%DEPLOY_DIR%\" >nul
    echo [llama-build]   ggml-opencl.dll  ^(OpenCL GPU backend^)
) else (
    echo [llama-build]   NOTE: ggml-opencl.dll not found ? OpenCL backend unavailable
)
if exist "%DML_SRC%\DirectML.dll" (
    copy /y "%DML_SRC%\DirectML.dll" "%BIN_OUT%\" >nul
    echo [llama-build]   DirectML.dll  ^(DirectML runtime^)
) else (
    echo [llama-build]   WARNING: %DML_SRC%\DirectML.dll not found ? dml_gemm.dll will fail to load
)

if exist "%DML_SRC%\d3d11_infer.dll" (
    copy /y "%DML_SRC%\d3d11_infer.dll" "%BIN_OUT%\" >nul
    echo [llama-build]   d3d11_infer.dll  ^(native D3D11 cs_5_0 inference^)
) else (
    echo [llama-build]   NOTE: d3d11_infer.dll not found ? native D3D11 path will be unavailable
)

:: Also deploy to bin\ggml\ so json_runtime.exe can find them at relative ..\\ggml\\dml_gemm.dll
set GGML_BIN_OUT=%ROOT%khanary-llama-build\ggml\build\bin\Release
if not exist "%GGML_BIN_OUT%" mkdir "%GGML_BIN_OUT%"
echo [llama-build] Copying GPU runtime DLLs to %GGML_BIN_OUT%\ (json_runtime path)...
if exist "%DML_SRC%\dml_gemm.dll" (
    copy /y "%DML_SRC%\dml_gemm.dll" "%GGML_BIN_OUT%\" >nul
    echo [llama-build]   dml_gemm.dll  ^(json_runtime KLSL path^)
)
if exist "%DML_SRC%\DirectML.dll" (
    copy /y "%DML_SRC%\DirectML.dll" "%GGML_BIN_OUT%\" >nul
    echo [llama-build]   DirectML.dll  ^(json_runtime DirectML path^)
)
echo.
echo [llama-build] Done.
echo   Binary : %BUILD%\bin\Release\llama-server.exe
echo   Port   : 17480
echo   Launch : node dist\khanary-server\kuhul-server.cjs
echo.
echo   If webUI still looks stale after first launch, hard-refresh the browser
echo   ^(Ctrl+Shift+R^) ? the old service worker may have cached the previous build.

:: ?? 4b. Deploy to dist/khanary-server/ (on-demand via `llama-build deploy`) ??
::   START-SERVERS.bat expects the binary at dist/khanary-server/khanary-server.exe.
::   This copies the fresh build + GPU DLLs there so the launcher picks them up.
if /i "%1"=="deploy" goto do_deploy
goto no_deploy

:deploy_only
echo [llama-build] Deploy-only mode ? skipping build, copying existing artifacts...
:: Build may not be fresh; copy whatever is there.
goto do_deploy

:do_deploy
set DEPLOY_DIR=%ROOT%dist\khanary-server
if not exist "%DEPLOY_DIR%" mkdir "%DEPLOY_DIR%"
echo [llama-build] Deploying to %DEPLOY_DIR%...
if exist "%BIN_OUT%\llama-server.exe" (
    copy /y "%BIN_OUT%\llama-server.exe" "%DEPLOY_DIR%\khanary-server.exe" >nul
    echo [llama-build]   llama-server.exe -^> khanary-server.exe
    for %%f in ("%BIN_OUT%\*.dll") do (
        copy /y "%%f" "%DEPLOY_DIR%\" >nul
        echo [llama-build]   %%~nxf
    )
) else (
    echo [llama-build]   WARNING: %BIN_OUT%\llama-server.exe not found -^> build first
)
echo [llama-build] Deploy complete. START-SERVERS.bat will use this binary.
echo.

:no_deploy
endlocal
