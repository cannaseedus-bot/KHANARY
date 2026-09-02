#Requires -Version 7.4

<#
.SYNOPSIS
    SMCA.psm1 — Semantic Machine Capability Assertion layer.
.DESCRIPTION
    SMCA is the contract surface for compiled kernels. It owns the registry
    (kernel-classes.json / kernel-aliases.json / kernel-extras.json),
    validates .smca.json artifacts for lawfulness and registryMatched state,
    and exposes the kernelClass / collapseClass dispatch table used by kxc
    lower.cpp::classify() — but from PowerShell, for runtime admission checks.
#>

$script:RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$script:RegistryDir = Join-Path $script:RepoRoot 'versions\kxc-v1.0.0\registry'
$script:KernelClasses = $null
$script:KernelAliases = $null
$script:KernelExtras  = $null

# ── Bus registration ──────────────────────────────────────────────────────────

function Register-SmcaWithBus {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Bus)
    $script:Bus = $Bus
    $script:KernelClasses = $null   # flush cache on re-register
    Write-Host "[SMCA] registered; registry: $script:RegistryDir"
}

# ── Registry loading (lazy, cached) ──────────────────────────────────────────

function Get-SmcaRegistry {
    [CmdletBinding()]
    param()
    if (-not $script:KernelClasses) {
        $p = Join-Path $script:RegistryDir 'kernel-classes.json'
        if (-not (Test-Path $p)) { throw "SMCA: kernel-classes.json not found at $p" }
        $script:KernelClasses = Get-Content -Raw $p | ConvertFrom-Json -AsHashtable
    }
    if (-not $script:KernelAliases) {
        $p = Join-Path $script:RegistryDir 'kernel-aliases.json'
        if (Test-Path $p) { $script:KernelAliases = Get-Content -Raw $p | ConvertFrom-Json -AsHashtable }
    }
    if (-not $script:KernelExtras) {
        $p = Join-Path $script:RegistryDir 'kernel-extras.json'
        if (Test-Path $p) { $script:KernelExtras = Get-Content -Raw $p | ConvertFrom-Json -AsHashtable }
    }
    return @{ classes = $script:KernelClasses; aliases = $script:KernelAliases; extras = $script:KernelExtras }
}

# ── Kernel contract lookup ────────────────────────────────────────────────────

function Get-KernelContract {
    <#
    .SYNOPSIS Returns the registry entry for a given kernelClass.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$KernelClass)
    $reg = Get-SmcaRegistry
    if (-not $reg.classes.ContainsKey($KernelClass)) {
        return $null
    }
    $entry = $reg.classes[$KernelClass]
    $extras = if ($reg.extras -and $reg.extras.ContainsKey('extras') -and $reg.extras['extras'].ContainsKey($KernelClass)) {
        $reg.extras['extras'][$KernelClass]
    } else { @{} }
    return @{
        kernelClass  = $KernelClass
        lawful       = $entry['lawful']
        requires     = $entry['requires']
        forbids      = $entry['forbids']
        backend      = $entry['backend']
        layers       = $entry['layers']
        family       = $entry['family']
        collapseClass = $entry['collapseClass']
        extras       = $extras
    }
}

# ── .smca.json I/O ───────────────────────────────────────────────────────────

function Read-SmcaJson {
    <#
    .SYNOPSIS Loads a compiled .smca.json artifact and returns it as a hashtable.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path $Path)) { throw "SMCA: file not found: $Path" }
    return Get-Content -Raw $Path | ConvertFrom-Json -AsHashtable
}

function Write-SmcaJson {
    <#
    .SYNOPSIS Writes a .smca.json artifact. Validates required fields before writing.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][hashtable]$Smca
    )
    $required = @('kernelClass','collapseClass','lawful','registryMatched','layers')
    $missing = $required | Where-Object { -not $Smca.ContainsKey($_) }
    if ($missing) { throw "SMCA: Write-SmcaJson missing fields: $($missing -join ', ')" }
    $Smca | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Path -Encoding utf8NoBOM
    Write-Host "[SMCA] wrote: $Path"
}

# ── Contract validation ───────────────────────────────────────────────────────

function Test-SmcaContract {
    <#
    .SYNOPSIS Validates a .smca.json artifact against the registry.
    .OUTPUTS Hashtable with ok, errors[], warnings[].
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Smca)
    if ($Smca -is [string]) { $Smca = $Smca | ConvertFrom-Json -AsHashtable }
    $errors   = [System.Collections.ArrayList]::new()
    $warnings = [System.Collections.ArrayList]::new()

    $smcaBlock = if ($Smca.ContainsKey('smca')) { $Smca['smca'] } else { $Smca }

    $kc = $smcaBlock['kernelClass']
    if (-not $kc) { $errors.Add('smca.kernelClass is missing') | Out-Null }

    $contract = if ($kc) { Get-KernelContract -KernelClass $kc } else { $null }
    if ($kc -and -not $contract) {
        $errors.Add("kernelClass '$kc' not found in registry") | Out-Null
    }

    if ($contract) {
        if (-not $smcaBlock['lawful']) {
            $errors.Add("kernelClass '$kc' marked not lawful") | Out-Null
        }
        if (-not $smcaBlock['registryMatched']) {
            $errors.Add("kernelClass '$kc' registryMatched=false") | Out-Null
        }
        $cc = $smcaBlock['collapseClass']
        if ($cc -and $cc -ne $contract.collapseClass) {
            $warnings.Add("collapseClass '$cc' differs from registry '$($contract.collapseClass)'") | Out-Null
        }
    }

    $ok = $errors.Count -eq 0
    return @{ ok = $ok; errors = $errors.ToArray(); warnings = $warnings.ToArray(); contract = $contract }
}

function Assert-SmcaLawful {
    <#
    .SYNOPSIS Throws if the SMCA contract fails validation.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Smca)
    $result = Test-SmcaContract -Smca $Smca
    if (-not $result.ok) {
        throw "SMCA contract violation: $($result.errors -join '; ')"
    }
    return $result
}

# ── [Sek] flag → kernelClass dispatch (mirrors lower.cpp::classify()) ─────────

function Get-KernelClass {
    <#
    .SYNOPSIS Maps a hashtable of [Sek] bool flags to a kernelClass.
    Mirrors the priority-ordered classify() in lower.cpp.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Flags)
    function flag([string]$f) { $Flags[$f] -eq $true -or $Flags[$f] -eq 'true' -or $Flags[$f] -eq '1' }

    if ((flag 'needsMoERoute') -or (flag 'needsMoEExpertFFN') -or (flag 'needsMoECombine')) {
        return 'moe_route_top2'
    }
    if ((flag 'needsSoftmax') -and (flag 'needsMatMul')) { return 'tensor_attention_fused' }
    if (flag 'needsMeshlet')                              { return 'mesh_meshlet_cull' }
    if ((flag 'needsNormalCompute') -or (flag 'needsTangentFrame')) { return 'mesh_normal_compute' }
    if (flag 'needsVertexProcess')                        { return 'mesh_vertex_process' }
    if (flag 'needsAdam')                                 { return 'adam_optimizer' }
    if (flag 'needsGradAccum')                            { return 'gradient_accum' }
    if ((flag 'needsSiluGrad') -or ((flag 'needsGradClip') -and (flag 'needsShmRead'))) {
        return 'backward_pass'
    }
    if ((flag 'needsValueClamp') -or (flag 'needsGradClip')) { return 'fold_clamp' }
    return 'generic-compute'
}

Export-ModuleMember -Function Register-SmcaWithBus, Get-SmcaRegistry, Get-KernelContract,
    Read-SmcaJson, Write-SmcaJson, Test-SmcaContract, Assert-SmcaLawful, Get-KernelClass
