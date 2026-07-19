@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++17 /EHsc /O2 skin_run.cpp /Fe:skin_run.exe >build_skin.log 2>&1
if errorlevel 1 (type build_skin.log & exit /b 1)
skin_run.exe
