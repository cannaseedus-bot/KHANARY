# build_khanary_ui.ps1 — Part 3: build the (KHANARY-branded) llama Web UI to an embeddable dist/.
#
# Copies tools/ui (SvelteKit) into a workspace, npm installs + builds with a LARGER Node heap (the
# stock CMake-driven build OOM'd), producing dist/index.html. That dist/ is then dropped into the
# server workspace so the reseal (build_khanary_server.ps1 with LLAMA_BUILD_UI=ON) embeds it —
# priority-1 "pre-built assets", so the server build itself never runs npm.
$ErrorActionPreference = "Continue"
$REPO = Split-Path -Parent $PSScriptRoot
$SRC  = "C:\Users\canna\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp\tools\ui"
$WS   = "C:\Users\canna\khanary-ui-build"
$env:NODE_OPTIONS = "--max-old-space-size=8192"   # fix: SvelteKit/vite build was OOMing

if (-not (Test-Path $SRC)) { throw "tools/ui source not found: $SRC" }
Write-Host "[1/4] copy tools/ui -> workspace (keep any existing node_modules for speed)"
if (-not (Test-Path $WS)) { New-Item -ItemType Directory -Force $WS | Out-Null }
robocopy $SRC $WS /E /NFL /NDL /NJH /NJS /NP /XD node_modules dist .svelte-kit | Out-Null

Push-Location $WS
Write-Host "[2/4] npm install (NODE_OPTIONS=$env:NODE_OPTIONS)"
& npm install 2>&1 | Select-Object -Last 6 | Out-Host
Write-Host "  npm install exit=$LASTEXITCODE"

Write-Host "[3/4] npm run build -> dist/"
$env:LLAMA_UI_OUT_DIR = "$WS\dist"
& npm run build 2>&1 | Select-Object -Last 12 | Out-Host
Write-Host "  npm build exit=$LASTEXITCODE"
Pop-Location

Write-Host "[4/4] verify dist/index.html"
$idx = Join-Path $WS "dist\index.html"
if (Test-Path $idx) {
    $sz = (Get-Item $idx).Length
    Write-Host "  [ok] $idx ($([math]::Round($sz/1024,1)) KB)"
} else {
    Write-Host "  [warn] dist/index.html not produced -- check the build output above"
    # some adapters emit elsewhere; show what dist has
    if (Test-Path (Join-Path $WS "dist")) { Get-ChildItem (Join-Path $WS "dist") | Select-Object -First 10 Name | Out-Host }
}
