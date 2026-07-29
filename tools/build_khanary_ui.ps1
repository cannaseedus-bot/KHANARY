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

Write-Host "[brand] KHANARY name + ASX Atomic teal accent + logo (the sigil, not the llama mark)"
$env:VITE_PUBLIC_APP_NAME = "KHANARY"      # APP_NAME reads VITE_PUBLIC_APP_NAME || 'llama-ui'
# swap the nav/brand logo mark (Logo.svelte imports $lib/assets/logo.svg?raw)
$brandLogo = Join-Path $REPO "native\khanary-ui\assets\logo.svg"
if (Test-Path $brandLogo) { Copy-Item $brandLogo (Join-Path $WS "src\lib\assets\logo.svg") -Force; Write-Host "  [ok] logo.svg -> KHANARY sigil" }
$appcss = Join-Path $WS "src\app.css"
if (Test-Path $appcss) {
    $css = Get-Content -Raw $appcss
    # recolor the shadcn accent tokens to the ASX Atomic teal-mint (oklch)
    $css = $css -replace '(--primary:\s*)oklch\([^)]*\)',         '${1}oklch(0.72 0.14 172)'
    $css = $css -replace '(--ring:\s*)oklch\([^)]*\)',            '${1}oklch(0.72 0.14 172)'
    $css = $css -replace '(--sidebar-primary:\s*)oklch\([^)]*\)', '${1}oklch(0.72 0.14 172)'
    Set-Content -Path $appcss -Value $css -NoNewline
    Write-Host "  [ok] recolored --primary/--ring/--sidebar-primary in app.css"
}

Write-Host "[routes] add KHANARY Models + Train feature routes + nav items"
# overlay the KHANARY route pages (new files)
$overlay = Join-Path $REPO "native\khanary-ui\routes"
if (Test-Path $overlay) { robocopy $overlay (Join-Path $WS "src\routes") /E /NFL /NDL /NJH /NJS /NP | Out-Null }
# ROUTES: add MODELS + TRAIN (routes.ts is overwritten by the copy each run, so re-patch)
$routes = Join-Path $WS "src\lib\constants\routes.ts"
$c = Get-Content -Raw $routes
if ($c -notmatch "MODELS:") {
    $c = $c -replace "(SEARCH:\s*'#/search')", "`$1,`r`n`tMODELS: '#/models',`r`n`tTRAIN: '#/train'"
    Set-Content -Path $routes -Value $c -NoNewline
}
# ui.ts: import the icons + append two SIDEBAR_ACTIONS_ITEMS after the MCP item
$uits = Join-Path $WS "src\lib\constants\ui.ts"
$c = Get-Content -Raw $uits
if ($c -notmatch "GraduationCap") {
    $c = $c -replace "import \{ Search, Settings, SquarePen \} from '@lucide/svelte';", "import { Boxes, GraduationCap, Search, Settings, SquarePen } from '@lucide/svelte';"
    $c = $c -replace "(activeRouteId: '/mcp-servers'\s*\r?\n\s*\},)", "`$1`r`n`t{ icon: Boxes, tooltip: 'Models', route: ROUTES.MODELS, activeRouteId: '/models' },`r`n`t{ icon: GraduationCap, tooltip: 'Train', route: ROUTES.TRAIN, activeRouteId: '/train' },"
    Set-Content -Path $uits -Value $c -NoNewline
}
if ((Get-Content -Raw $routes) -notmatch "MODELS:") { Write-Host "  [warn] routes.ts patch did not apply" }
if ((Get-Content -Raw $uits)   -notmatch "GraduationCap") { Write-Host "  [warn] ui.ts nav patch did not apply" }

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
