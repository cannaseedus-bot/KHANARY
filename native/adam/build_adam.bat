@echo off
setlocal EnableDelayedExpansion

:: Build Adam.dll using VS 2022 BuildTools (x64)
:: Output: ../../versions/khlc-v1.0.0/bin/Adam.dll

set SCRIPT_DIR=%~dp0
set BUILD_DIR=%SCRIPT_DIR%build
set OUT_DLL=%SCRIPT_DIR%..\..\versions\khlc-v1.0.0\bin\Adam.dll

:: Bootstrap VS 2022 dev environment
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist %VCVARS% (
    echo ERROR: VS 2022 BuildTools not found
    exit /b 1
)
call %VCVARS% >nul 2>&1

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

pushd "%BUILD_DIR%"

cmake .. -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl

if errorlevel 1 (
    echo CMake configure failed
    popd
    exit /b 1
)

cmake --build . --config Release

if errorlevel 1 (
    echo Build failed
    popd
    exit /b 1
)

popd

:: Deploy to khlc-v1.0.0/bin/
set SRC_DLL=%BUILD_DIR%\bin\Adam.dll
if exist "%SRC_DLL%" (
    copy /Y "%SRC_DLL%" "%OUT_DLL%"
    echo Deployed Adam.dll to versions\khlc-v1.0.0\bin\
) else (
    echo FAILED: Adam.dll not found at %SRC_DLL%
    exit /b 1
)

endlocal
