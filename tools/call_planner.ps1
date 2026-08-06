# call_planner.ps1 — Micronaut bridge to MicrosoftSDK.ps1
#
# Reads active-model.json (written by START-SERVERS.bat) to find the active
# inference endpoint, then calls MicrosoftSDK.ps1 tasklist with that endpoint.
# This routes "create app / build X" intents through PM-1 → TaskEngine.cpp
# using whatever model is currently loaded, not the hardcoded 17480 default.
#
# Usage:
#   .\tools\call_planner.ps1 -Prompt "Create an app for fantasy football"
#   .\tools\call_planner.ps1 -Prompt "Build a website for X" -DispatchToBoss
#   .\tools\call_planner.ps1 -Prompt "..." -Endpoint http://127.0.0.1:3080/v1/chat/completions

param(
    [Parameter(Mandatory)]
    [string]$Prompt,

    [string]$Endpoint = '',
    [string]$Model = '',
    [switch]$DispatchToBoss,
    [string]$OutputPath = '',
    [ValidateRange(1, 4096)]
    [int]$Tokens = 512
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$activeModelPath = Join-Path $root 'active-model.json'
$sdkScript = 'C:\Users\canna\.NNC-K\bin\v3.5.0-WebX\native\semantic-kernel\MicrosoftSDK.ps1'

# --- resolve endpoint --------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($Endpoint)) {
    if (Test-Path $activeModelPath) {
        $active = Get-Content $activeModelPath -Raw | ConvertFrom-Json
        $Endpoint = $active.endpoint
        Write-Host "[call_planner] Using active model endpoint: $Endpoint"
    } else {
        $Endpoint = 'http://127.0.0.1:17480/v1/chat/completions'
        Write-Warning "[call_planner] active-model.json not found — falling back to kuhul_engine:17480"
        Write-Warning "              Run START-SERVERS.bat first to register the active model."
    }
}

if ([string]::IsNullOrWhiteSpace($Model)) {
    if (Test-Path $activeModelPath) {
        $active = Get-Content $activeModelPath -Raw | ConvertFrom-Json
        $modelFile = $active.model_path
        $Model = if ($modelFile) { Split-Path $modelFile -Leaf } else { 'active-model' }
    } else {
        $Model = 'gpt-oss-20b-MXFP4.gguf'
    }
}

# --- resolve output path -----------------------------------------------------
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $root 'tasklist.generated.json'
}

# --- verify MicrosoftSDK.ps1 exists ------------------------------------------
if (-not (Test-Path $sdkScript)) {
    throw "MicrosoftSDK.ps1 not found at: $sdkScript`nCheck that the NNC-K stack is installed."
}

# --- call the planner --------------------------------------------------------
Write-Host "[call_planner] Sending to planner (PM-1 / MicrosoftSDK.ps1)..."
Write-Host "               Endpoint : $Endpoint"
Write-Host "               Model    : $Model"
Write-Host "               Prompt   : $Prompt"
Write-Host ""

$sdkArgs = @{
    Command    = 'tasklist'
    Prompt     = $Prompt
    Endpoint   = $Endpoint
    Model      = $Model
    OutputPath = $OutputPath
    Tokens     = $Tokens
}
if ($DispatchToBoss) { $sdkArgs['DispatchToBoss'] = $true }

$result = & $sdkScript @sdkArgs | ConvertFrom-Json

Write-Host ""
Write-Host "[call_planner] TaskList written: $($result.output)"
Write-Host "               Tasks    : $($result.tasks)"
Write-Host "               Dispatch : $($result.dispatch)"

if ($result.dispatch -ne 'completed' -and -not $DispatchToBoss) {
    Write-Host ""
    Write-Host "To execute the plan:"
    Write-Host "  $($result.next)"
}

return $result
