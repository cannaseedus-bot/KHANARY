@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /std:c++17 /EHsc /O2 /I include dml_gemm_bench.cpp /link /LIBPATH:lib /OUT:dml_gemm_bench.exe
copy /y lib\DirectML.dll DirectML.dll >nul
