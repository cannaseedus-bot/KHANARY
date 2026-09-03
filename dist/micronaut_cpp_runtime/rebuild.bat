@echo off
setlocal

set SRC=C:\Users\canna\_khanary_inspect\dist\micronaut_cpp_runtime
set BLD=%SRC%\build-nnck
set EXE=%BLD%\micronaut_xjson.exe
set DIST=C:\Users\canna\_khanary_inspect\dist

REM ── Bootstrap MSVC (VS 2022 BuildTools preferred, VS 2019 fallback) ───────
set VS22=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat
set VS19=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat

if exist "%VS22%" (
    call "%VS22%" -arch=x64 2>nul
    goto :build
)
if exist "%VS19%" (
    call "%VS19%" -arch=x64 2>nul
    goto :build
)
echo [rebuild] ERROR: no MSVC BuildTools found (tried VS 2022 and VS 2019)
exit /b 1

:build
REM ── CMake + Ninja ─────────────────────────────────────────────────────────
echo [rebuild] configuring micronaut_xjson...
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo [rebuild] CMake configure failed
    exit /b %errorlevel%
)

echo [rebuild] building micronaut_xjson.exe...
ninja -C "%BLD%"
if %errorlevel% neq 0 (
    echo [rebuild] ninja build failed
    exit /b %errorlevel%
)

if not exist "%EXE%" (
    echo [rebuild] ERROR: %EXE% not found after build
    exit /b 1
)

REM ── Distribute to every active skin dir ───────────────────────────────────
echo [rebuild] distributing micronaut_xjson.exe to skin directories...

for %%S in (
    adam-micronaut
    alice-micronaut
    eliza-micronaut
    jyggalag-micronaut
    regex-micronaut
    semantic-cube-micronaut
    sheogorath-micronaut
) do (
    if exist "%DIST%\%%S" (
        copy /Y "%EXE%" "%DIST%\%%S\micronaut_xjson.exe" >nul
        echo   [ok] %%S
    ) else (
        echo   [skip] %%S — directory not found
    )
)

echo [rebuild] done. micronaut_xjson.exe deployed to 7 skins.
endlocal
