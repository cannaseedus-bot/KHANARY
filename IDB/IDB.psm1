#Requires -Version 7.4

<#
.SYNOPSIS
    IDB.psm1 — Identity Database sidecar layer (IDB).
.DESCRIPTION
    IDB is the final pipeline layer: it stores and retrieves compiled kernel
    identity contracts (.smca.json), emits sidecar events to an append-only
    JSONL log, and serves as the PowerShell-side registry-match verification
    surface for external consumers (MCP routes, kuhul-server, PRIMEOS).
    Pipeline position: SCO/1 → IDB → external consumers
    Event log: .idb_events.jsonl (per output directory, same pattern as JROM).
#>

$script:RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$script:DefaultDir  = Join-Path $script:RepoRoot 'versions\kxc-v1.0.0\examples\outputs'
$script:EventLog    = Join-Path $script:RepoRoot '.idb_events.jsonl'
$script:Bus         = $null

# ── Bus registration ──────────────────────────────────────────────────────────

function Register-IdbWithBus {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Bus)
    $script:Bus = $Bus
    Write-Host "[IDB] registered; event log: $script:EventLog"
}

# ── SMCA I/O ──────────────────────────────────────────────────────────────────

function Read-IdbSmca {
    <#
    .SYNOPSIS Reads a .smca.json from the IDB output directory.
              Resolves by kernel name or by direct path.
    #>
    [CmdletBinding()]
    param(
        [string]$KernelName,
        [string]$Path,
        [string]$OutDir = $script:DefaultDir
    )
    if (-not $Path -and -not $KernelName) {
        throw "IDB: supply either -KernelName or -Path"
    }
    $resolved = if ($Path) { $Path }
                else { Join-Path $OutDir "$KernelName.smca.json" }
    if (-not (Test-Path $resolved)) {
        throw "IDB: smca.json not found: $resolved"
    }
    return Get-Content -Raw $resolved | ConvertFrom-Json -AsHashtable
}

function Write-IdbSmca {
    <#
    .SYNOPSIS Persists a compiled contract as .smca.json and appends an idb.write event.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$KernelName,
        [Parameter(Mandatory)][hashtable]$Smca,
        [string]$OutDir = $script:DefaultDir
    )
    if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
    $dest = Join-Path $OutDir "$KernelName.smca.json"
    $Smca | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $dest -Encoding utf8NoBOM
    Write-Host "[IDB] wrote: $dest"
    Append-IdbEvent -Op 'idb.write' -Kernel $KernelName -Path $dest -Meta @{
        kernelClass     = ($Smca['smca'] ?? $Smca)['kernelClass']
        registryMatched = ($Smca['smca'] ?? $Smca)['registryMatched']
    }
}

# ── Event log (append-only JSONL, same SHA-256 hash-chain pattern as JROM) ───

function Append-IdbEvent {
    <#
    .SYNOPSIS Appends a JSON event line to the IDB event log.
    .PARAMETER Op     Operation type (e.g. idb.write, idb.verify, idb.reject).
    .PARAMETER Kernel Kernel name (optional).
    .PARAMETER Path   Artifact path (optional).
    .PARAMETER Meta   Additional metadata hashtable (optional).
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Op,
        [string]$Kernel = '',
        [string]$Path   = '',
        [hashtable]$Meta = @{}
    )
    $prev = ''
    if (Test-Path $script:EventLog) {
        $lines = Get-Content -LiteralPath $script:EventLog -ErrorAction SilentlyContinue
        if ($lines) { $prev = ($lines | Select-Object -Last 1 | ConvertFrom-Json -AsHashtable)['hash'] ?? '' }
    }

    $payload = [ordered]@{
        op        = $Op
        timestamp = [DateTimeOffset]::UtcNow.ToString('o')
        kernel    = $Kernel
        path      = $Path
        meta      = $Meta
    }
    $payloadJson = $payload | ConvertTo-Json -Compress
    $hashBytes   = [System.Security.Cryptography.SHA256]::Create().ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes("$prev|$payloadJson")
    )
    $hash = [System.BitConverter]::ToString($hashBytes).Replace('-','').ToLower()

    $event = [ordered]@{ op = $Op; timestamp = $payload.timestamp; kernel = $Kernel;
                          path = $Path; meta = $Meta; hash = $hash }
    Add-Content -LiteralPath $script:EventLog -Value ($event | ConvertTo-Json -Compress) -Encoding utf8NoBOM
}

# ── Registry-match verification ───────────────────────────────────────────────

function Test-IdbRegistryMatch {
    <#
    .SYNOPSIS Checks that a smca.json has registryMatched=true and lawful=true.
    .OUTPUTS Hashtable with ok, kernelClass, registryMatched, lawful, path.
    #>
    [CmdletBinding()]
    param(
        [string]$KernelName,
        [string]$Path,
        [string]$OutDir = $script:DefaultDir
    )
    $smca = Read-IdbSmca -KernelName $KernelName -Path $Path -OutDir $OutDir
    $block = if ($smca.ContainsKey('smca')) { $smca['smca'] } else { $smca }
    $ok = ($block['registryMatched'] -eq $true) -and ($block['lawful'] -eq $true)
    if (-not $ok) {
        Append-IdbEvent -Op 'idb.reject' -Kernel $KernelName -Path $Path -Meta @{
            registryMatched = $block['registryMatched']
            lawful          = $block['lawful']
        }
    }
    return @{
        ok              = $ok
        kernelClass     = $block['kernelClass']
        registryMatched = $block['registryMatched']
        lawful          = $block['lawful']
        path            = $Path
    }
}

# ── Index query ───────────────────────────────────────────────────────────────

function Get-IdbIndex {
    <#
    .SYNOPSIS Returns all .smca.json records in the output directory as an indexed hashtable.
              Keys are kernel names; values are the smca blocks.
    #>
    [CmdletBinding()]
    param([string]$OutDir = $script:DefaultDir)
    $index = @{}
    if (-not (Test-Path $OutDir)) { return $index }
    Get-ChildItem -LiteralPath $OutDir -Filter '*.smca.json' -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.BaseName -replace '\.smca$', ''
        $smca = Get-Content -Raw $_.FullName | ConvertFrom-Json -AsHashtable
        $index[$name] = if ($smca.ContainsKey('smca')) { $smca['smca'] } else { $smca }
    }
    return $index
}

function Get-IdbKernelContract {
    <#
    .SYNOPSIS Retrieves the compiled contract for a kernel from the IDB index.
              Unlike SMCA.Get-KernelContract (registry lookup), this reads
              the compiled artifact on disk — the runtime-verified contract.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$KernelName,
        [string]$OutDir = $script:DefaultDir
    )
    $smca = Read-IdbSmca -KernelName $KernelName -OutDir $OutDir
    return if ($smca.ContainsKey('smca')) { $smca['smca'] } else { $smca }
}

Export-ModuleMember -Function Register-IdbWithBus, Read-IdbSmca, Write-IdbSmca,
    Append-IdbEvent, Test-IdbRegistryMatch, Get-IdbIndex, Get-IdbKernelContract
