@echo off
setlocal EnableDelayedExpansion

set "SRC=C:\Users\canna\_khanary_inspect\dist\micronaut_cpp_runtime"
set "BLD=%SRC%\build-nnck"
set "EXE=%BLD%\micronaut_xjson.exe"
set "DIST=C:\Users\canna\_khanary_inspect\dist"

set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
set "VS_CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "VS_NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "SYS_CMAKE=C:\Program Files\CMake\bin"

REM -- Bootstrap MSVC environment --
if not exist "!VSDEVCMD!" (
    echo [rebuild] ERROR: VsDevCmd.bat not found
    exit /b 1
)
echo [rebuild] bootstrapping VS 2022 BuildTools x64...
call "!VSDEVCMD!" -arch=x64 2>nul

REM -- Add cmake to PATH if not present --
where cmake >nul 2>&1
if !errorlevel! neq 0 (
    if exist "!SYS_CMAKE!\cmake.exe" (
        set "PATH=!SYS_CMAKE!;!PATH!"
        echo [rebuild] cmake: !SYS_CMAKE!
    ) else if exist "!VS_CMAKE!\cmake.exe" (
        set "PATH=!VS_CMAKE!;!PATH!"
        echo [rebuild] cmake: VS bundle
    ) else (
        echo [rebuild] ERROR: cmake.exe not found
        exit /b 1
    )
)

REM -- Add ninja to PATH if not present --
where ninja >nul 2>&1
if !errorlevel! neq 0 (
    if exist "!VS_NINJA!\ninja.exe" (
        set "PATH=!VS_NINJA!;!PATH!"
        echo [rebuild] ninja: VS bundle
    ) else (
        echo [rebuild] ERROR: ninja.exe not found
        exit /b 1
    )
)

REM -- CMake configure --
echo [rebuild] configuring micronaut_xjson...
cmake -S "!SRC!" -B "!BLD!" -G Ninja -DCMAKE_BUILD_TYPE=Release
if !errorlevel! neq 0 (
    echo [rebuild] CMake configure failed
    exit /b !errorlevel!
)

REM -- Ninja build --
echo [rebuild] building micronaut_xjson.exe...
ninja -C "!BLD!"
if !errorlevel! neq 0 (
    echo [rebuild] ninja build failed
    exit /b !errorlevel!
)

if not exist "!EXE!" (
    echo [rebuild] ERROR: EXE not found after build
    exit /b 1
)

REM -- Distribute to every active skin dir --
echo [rebuild] distributing micronaut_xjson.exe to skin directories...

for %%S in (adam-micronaut alice-micronaut code-micronaut eliza-micronaut jyggalag-micronaut regex-micronaut semantic-cube-micronaut sheogorath-micronaut) do (
    if exist "!DIST!\%%S" (
        copy /Y "!EXE!" "!DIST!\%%S\micronaut_xjson.exe" >nul
        echo   [ok] %%S
    ) else (
        echo   [skip] %%S -- not found
    )
)

echo [rebuild] done. micronaut_xjson.exe deployed to 8 skins.
endlocal
