# reseal_khanary_server.ps1 — Part 3: embed the branded UI into khanary-server (the "seal").
#
# Drops the branded Web UI (from build_khanary_ui.ps1) into the EXISTING server workspace as
# pre-built assets (ui-assets.cmake priority 1: tools/ui/dist -> embedded, no npm), then rebuilds
# ONLY the llama-server target (incremental) so the branded index.html is compiled into the binary.
# Requires build_khanary_server.ps1 (workspace + build/) and build_khanary_ui.ps1 (branded dist) first.
$ErrorActionPreference = "Continue"
$REPO   = Split-Path -Parent $PSScriptRoot
$WS     = "C:\Users\canna\khanary-llama-build\llama.cpp"
$UIDIST = "C:\Users\canna\khanary-ui-build\dist"
$DIST   = Join-Path $REPO "dist\khanary-server"
$cmake  = @(
  "C:\Program Files\CMake\bin\cmake.exe",
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not (Test-Path "$WS\build")) { throw "server workspace build/ not found -- run build_khanary_server.ps1 first" }
if (-not (Test-Path "$UIDIST\index.html")) { throw "branded UI dist not found -- run build_khanary_ui.ps1 first" }

Write-Host "[1/3] drop branded UI -> $WS\tools\ui\dist (priority-1 pre-built assets, no npm)"
$srcDist = Join-Path $WS "tools\ui\dist"
if (Test-Path $srcDist) { Remove-Item -Recurse -Force $srcDist }
New-Item -ItemType Directory -Force $srcDist | Out-Null
robocopy $UIDIST $srcDist /E /NFL /NDL /NJH /NJS /NP | Out-Null

Write-Host "[2/3] rebuild llama-server (incremental; embeds the branded UI)"
& $cmake --build "$WS\build" --config Release --target llama-server 2>&1 | Tee-Object -Variable bld | Out-Host
Write-Host "  build exit=$LASTEXITCODE"
if ($bld -match "UI: using pre-built assets") { Write-Host "  [ok] embedded the pre-built (branded) UI" }

Write-Host "[3/3] re-bundle khanary-server"
$exe = Get-ChildItem -Path "$WS\build" -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($exe) {
    New-Item -ItemType Directory -Force $DIST | Out-Null
    Copy-Item $exe.FullName (Join-Path $DIST "khanary-server.exe") -Force
    Get-ChildItem -Path $exe.DirectoryName -Filter "*.dll" | ForEach-Object { Copy-Item $_.FullName $DIST -Force }
    foreach ($d in @("dml_gemm.dll", "DirectML.dll")) {
        $s = Join-Path $REPO "scratch\dml\$d"; if (Test-Path $s) { Copy-Item $s $DIST -Force }
    }
    Write-Host "  [ok] resealed khanary-server at $DIST"
} else {
    Write-Host "  [warn] llama-server.exe not found -- build failed above"
}
