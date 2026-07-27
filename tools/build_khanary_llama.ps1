# build_khanary_llama.ps1 -- Part 2 milestone: de-orphan ggml-xcfe into a real, registering backend
# and prove the branded fork's ggml BUILDS with it wired in (-DGGML_XCFE=ON).
#
# Never modifies the read-only vendored tree: it COPIES the ggml source into a workspace, overlays
# the KHANARY-authored backend (native/ggml-xcfe/), patches the two wiring points, builds ggml + a
# probe, and asserts the registry contains "XCFE". Scope: this builds ggml-lib only (not llama-server)
# and the backend's supports_op is false (registers, doesn't compute yet).
# Continue (not Stop): cmake writes warnings/status to stderr; with Stop + 2>&1 those abort the
# script. Our own critical checks use explicit `throw` (works regardless), and we check $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$REPO = Split-Path -Parent $PSScriptRoot
$SRC  = "C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp\ggml"
$WS   = "C:\Users\canna\khanary-llama-build\ggml"     # workspace (outside the repo)
$NAT  = Join-Path $REPO "native\ggml-xcfe"

if (-not (Test-Path $SRC)) { throw "vendored ggml source not found: $SRC" }
$cmake = @(
  "C:\Program Files\CMake\bin\cmake.exe",
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $cmake) { throw "cmake.exe not found" }
Write-Host "using cmake: $cmake"
Write-Host "[1/6] copy ggml source -> workspace"
if (Test-Path $WS) { Remove-Item -Recurse -Force $WS }
robocopy $SRC $WS /E /NFL /NDL /NJH /NJS /NP | Out-Null   # robocopy exit codes are not errors

Write-Host "[2/6] overlay KHANARY ggml-xcfe backend (replace the orphan webgpu copy)"
$xdir = Join-Path $WS "src\ggml-xcfe"
if (Test-Path $xdir) { Remove-Item -Recurse -Force $xdir }
New-Item -ItemType Directory -Force $xdir | Out-Null
Copy-Item (Join-Path $NAT "ggml-xcfe.cpp")   (Join-Path $xdir "ggml-xcfe.cpp")
Copy-Item (Join-Path $NAT "CMakeLists.txt")  (Join-Path $xdir "CMakeLists.txt")
Copy-Item (Join-Path $NAT "ggml-xcfe.h")     (Join-Path $WS  "include\ggml-xcfe.h")
Copy-Item (Join-Path $NAT "xcfe_probe.c")       (Join-Path $WS "xcfe_probe.c")
Copy-Item (Join-Path $NAT "xcfe_matmul_test.c") (Join-Path $WS "xcfe_matmul_test.c")
# the binary-distro ggml copy prunes tests/, examples/, ggml.pc.in — stub the pc template so the
# GGML_STANDALONE configure_file() succeeds (tests/examples are disabled via -D below).
$pc = Join-Path $WS "ggml.pc.in"
if (-not (Test-Path $pc)) { Set-Content -Path $pc -Value "Name: ggml`nDescription: ggml`nVersion: 0.0.0`n" }

Write-Host "[3/6] wire ggml_add_backend(XCFE) into src/CMakeLists.txt"
$srcCmake = Join-Path $WS "src\CMakeLists.txt"
$c = Get-Content -Raw $srcCmake
if ($c -notmatch "ggml_add_backend\(XCFE\)") {
    $c = $c -replace "(ggml_add_backend\(CPU\))", "`$1`r`nggml_add_backend(XCFE)"
    Set-Content -Path $srcCmake -Value $c -NoNewline
}

Write-Host "[4/6] patch ggml-backend-reg.cpp (static registration: include + register)"
$reg = Join-Path $WS "src\ggml-backend-reg.cpp"
$c = Get-Content -Raw $reg
if ($c -notmatch "ggml_backend_xcfe_reg") {
    $c = $c -replace "(#ifdef GGML_USE_BLAS\r?\n#include ""ggml-blas.h""\r?\n#endif)", "`$1`r`n#ifdef GGML_USE_XCFE`r`n#include ""ggml-xcfe.h""`r`n#endif"
    $c = $c -replace "(#ifdef GGML_USE_CUDA\r?\n\s*register_backend\(ggml_backend_cuda_reg\(\)\);\r?\n#endif)", "`$1`r`n#ifdef GGML_USE_XCFE`r`n        register_backend(ggml_backend_xcfe_reg());`r`n#endif"
    Set-Content -Path $reg -Value $c -NoNewline
}
if ((Get-Content -Raw $reg) -notmatch "ggml_backend_xcfe_reg") { throw "reg.cpp patch anchors did not match -- inspect $reg" }

Write-Host "[5/6] add the probe target to the top CMakeLists"
$topCmake = Join-Path $WS "CMakeLists.txt"
$c = Get-Content -Raw $topCmake
if ($c -notmatch "xcfe_probe") {
    $c += "`r`n# KHANARY: registry probe + MUL_MAT compute test`r`n"
    $c += "add_executable(xcfe_probe `${CMAKE_CURRENT_SOURCE_DIR}/xcfe_probe.c)`r`ntarget_link_libraries(xcfe_probe PRIVATE ggml)`r`n"
    $c += "add_executable(xcfe_matmul_test `${CMAKE_CURRENT_SOURCE_DIR}/xcfe_matmul_test.c)`r`ntarget_link_libraries(xcfe_matmul_test PRIVATE ggml)`r`n"
    Set-Content -Path $topCmake -Value $c -NoNewline
}

Write-Host "[6/6] configure + build (ggml + xcfe_probe), then run the probe"
$build = Join-Path $WS "build"
& $cmake -S $WS -B $build -G "Visual Studio 17 2022" -A x64 -DGGML_XCFE=ON -DGGML_BACKEND_DL=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF 2>&1 | Tee-Object -Variable cfg | Out-Host
if ($cfg -match "Including XCFE backend") { Write-Host "  [ok] CMake wired the XCFE backend" } else { Write-Host "  [warn] 'Including XCFE backend' not seen in configure output" }
if ($LASTEXITCODE -ne 0) { Write-Host "  [warn] configure exit=$LASTEXITCODE" }
& $cmake --build $build --config Release --target xcfe_probe xcfe_matmul_test 2>&1 | Tee-Object -Variable bld | Out-Host
Write-Host "  build exit=$LASTEXITCODE"

$probe = Get-ChildItem -Path $build -Recurse -Filter "xcfe_probe.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($probe) {
    Write-Host "`n=== probe (registration) ==="
    & $probe.FullName
    Write-Host "exit=$LASTEXITCODE  (0 = XCFE registered)"
} else {
    Write-Host "[warn] xcfe_probe.exe not found -- build likely failed above"
}

$mm = Get-ChildItem -Path $build -Recurse -Filter "xcfe_matmul_test.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($mm) {
    Write-Host "`n=== MUL_MAT compute (xcfe vs ggml CPU) ==="
    & $mm.FullName
    Write-Host "exit=$LASTEXITCODE  (0 = XCFE MUL_MAT matches CPU)"
} else {
    Write-Host "[warn] xcfe_matmul_test.exe not found -- build likely failed above"
}
