# build_khanary_server.ps1 — Part 2 (finish): build the FULL llama-server with the KHANARY ggml-xcfe
# backend baked in, then brand it -> khanary-server.exe + bundled DirectML GEMM.
#
# Copies the full llama.cpp source to a workspace (never edits the read-only tree), overlays
# native/ggml-xcfe, wires ggml_add_backend(XCFE) + static register, builds the llama-server target
# with -DGGML_XCFE=ON. The Web UI is npm-built + embedded by llama's own llama-ui target (Node req'd).
$ErrorActionPreference = "Continue"
$REPO = Split-Path -Parent $PSScriptRoot
$SRC  = "C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp"
$WS   = "C:\Users\canna\khanary-llama-build\llama.cpp"
$NAT  = Join-Path $REPO "native\ggml-xcfe"
$DIST = Join-Path $REPO "dist\khanary-server"
$cmake = @(
  "C:\Program Files\CMake\bin\cmake.exe",
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not (Test-Path $SRC)) { throw "vendored llama.cpp source not found: $SRC" }
if (-not $cmake) { throw "cmake.exe not found" }
Write-Host "using cmake: $cmake"

Write-Host "[1/6] copy FULL llama.cpp source -> workspace (this is large)"
if (Test-Path $WS) { Remove-Item -Recurse -Force $WS }
robocopy $SRC $WS /E /NFL /NDL /NJH /NJS /NP /XD build .git node_modules | Out-Null

Write-Host "[2/6] overlay KHANARY ggml-xcfe backend + stub ggml.pc.in"
$xdir = Join-Path $WS "ggml\src\ggml-xcfe"
if (Test-Path $xdir) { Remove-Item -Recurse -Force $xdir }
New-Item -ItemType Directory -Force $xdir | Out-Null
Copy-Item (Join-Path $NAT "ggml-xcfe.cpp")  (Join-Path $xdir "ggml-xcfe.cpp")
Copy-Item (Join-Path $NAT "CMakeLists.txt") (Join-Path $xdir "CMakeLists.txt")
Copy-Item (Join-Path $NAT "ggml-xcfe.h")    (Join-Path $WS  "ggml\include\ggml-xcfe.h")
$pc = Join-Path $WS "ggml\ggml.pc.in"
if (-not (Test-Path $pc)) { Set-Content -Path $pc -Value "Name: ggml`nDescription: ggml`nVersion: 0.0.0`n" }

Write-Host "[3/6] wire ggml_add_backend(XCFE)"
$srcCmake = Join-Path $WS "ggml\src\CMakeLists.txt"
$c = Get-Content -Raw $srcCmake
if ($c -notmatch "ggml_add_backend\(XCFE\)") {
    $c = $c -replace "(ggml_add_backend\(CPU\))", "`$1`r`nggml_add_backend(XCFE)"
    Set-Content -Path $srcCmake -Value $c -NoNewline
}

Write-Host "[4/6] patch ggml-backend-reg.cpp (static include + register)"
$reg = Join-Path $WS "ggml\src\ggml-backend-reg.cpp"
$c = Get-Content -Raw $reg
if ($c -notmatch "ggml_backend_xcfe_reg") {
    $c = $c -replace "(#ifdef GGML_USE_BLAS\r?\n#include ""ggml-blas.h""\r?\n#endif)", "`$1`r`n#ifdef GGML_USE_XCFE`r`n#include ""ggml-xcfe.h""`r`n#endif"
    $c = $c -replace "(#ifdef GGML_USE_CUDA\r?\n\s*register_backend\(ggml_backend_cuda_reg\(\)\);\r?\n#endif)", "`$1`r`n#ifdef GGML_USE_XCFE`r`n        register_backend(ggml_backend_xcfe_reg());`r`n#endif"
    Set-Content -Path $reg -Value $c -NoNewline
}
if ((Get-Content -Raw $reg) -notmatch "ggml_backend_xcfe_reg") { throw "reg.cpp patch anchors did not match" }

Write-Host "[5/6] configure + build llama-server (-DGGML_XCFE=ON; UI npm-built by llama-ui)"
$build = Join-Path $WS "build"
& $cmake -S $WS -B $build -G "Visual Studio 17 2022" -A x64 `
    -DGGML_XCFE=ON -DGGML_BACKEND_DL=OFF `
    -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON `
    -DLLAMA_USE_PREBUILT_UI=OFF 2>&1 | Tee-Object -Variable cfg | Out-Host
if ($cfg -match "Including XCFE backend") { Write-Host "  [ok] XCFE backend wired" } else { Write-Host "  [warn] XCFE not seen in configure output" }
Write-Host "  configure exit=$LASTEXITCODE"
& $cmake --build $build --config Release --target llama-server 2>&1 | Tee-Object -Variable bld | Out-Host
Write-Host "  build exit=$LASTEXITCODE"

Write-Host "[6/6] brand -> khanary-server + bundle DirectML GEMM"
$exe = Get-ChildItem -Path $build -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($exe) {
    New-Item -ItemType Directory -Force $DIST | Out-Null
    Copy-Item $exe.FullName (Join-Path $DIST "khanary-server.exe") -Force
    # sibling runtime dlls next to the server exe (ggml*, llama, etc.) + the DirectML GEMM
    Get-ChildItem -Path $exe.DirectoryName -Filter "*.dll" | ForEach-Object { Copy-Item $_.FullName $DIST -Force }
    foreach ($d in @("dml_gemm.dll", "DirectML.dll")) {
        $s = Join-Path $REPO "scratch\dml\$d"
        if (Test-Path $s) { Copy-Item $s $DIST -Force }
    }
    Copy-Item (Join-Path $REPO "khanary.svg") $DIST -Force -ErrorAction SilentlyContinue
    Write-Host "  [ok] khanary-server bundled at $DIST"
    Write-Host "`n=== khanary-server --version ==="
    & (Join-Path $DIST "khanary-server.exe") --version 2>&1 | Select-Object -First 5 | Out-Host
} else {
    Write-Host "[warn] llama-server.exe not found -- build failed above"
}
