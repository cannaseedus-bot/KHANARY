@echo off
setlocal

:: build-khanary.bat — now delegates to llama-build.bat
::
:: WHY: The old build-khanary.bat ran cmake directly, which reuses stale
:: webUI assets (tools/ui/dist/ + .ui-stamp) from previous builds. If you
:: changed the Svelte UI, the binary would still contain old code.
::
:: llama-build.bat properly deletes these stale artifacts before building.
:: All build-khanary modes now forward to the equivalent llama-build command.
::
:: Usage:
::   build-khanary            → llama-build          (fast, clears stale UI)
::   build-khanary full       → llama-build full     (full GPU reconfigure)
::   build-khanary clean      → llama-build clean    (wipe CMakeCache)

set ROOT=%~dp0
set LLAMA_BUILD=%ROOT%llama-build.bat

if not exist "%LLAMA_BUILD%" (
    echo [build-khanary] ERROR: llama-build.bat not found at %LLAMA_BUILD%
    exit /b 1
)

if /i "%1"=="clean" (
    echo [build-khanary] → llama-build clean
    call "%LLAMA_BUILD%" clean
) else if /i "%1"=="full" (
    echo [build-khanary] → llama-build full
    call "%LLAMA_BUILD%" full
) else if /i "%1"=="deploy" (
    echo [build-khanary] → llama-build deploy
    call "%LLAMA_BUILD%" deploy
) else (
    echo [build-khanary] → llama-build
    call "%LLAMA_BUILD%"
)

endlocal
