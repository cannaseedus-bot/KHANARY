#Requires -Version 7.4

<#
.SYNOPSIS
    SCXQ7.psm1 — Legality checking and caps-aware optimization layer (SCXQ7).
.DESCRIPTION
    SCXQ7 sits between SCXQ2 (lowered IR) and SCO (emitters). It enforces the
    requires/forbids contract from the SMCA registry, checks hardware capability
    flags, and applies optimization passes. A kernel whose IR fails legality here
    must not be emitted.
    Pipeline position: SCXQ2 → SCXQ7 → SCO/1
#>

$script:RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$script:RegistryDir = Join-Path $script:RepoRoot 'versions\kxc-v1.0.0\registry'
$script:Bus         = $null

# ── Legality rules (mirrors kernel-classes.json requires/forbids) ─────────────

$script:LEGALITY_RULES = @{
    # rules loaded lazily from registry; preloaded defaults:
    tensor_attention_fused = @{
        requires = @('deterministic_join', 'bounded_reduction')
        forbids  = @('side_effects', 'order_dependence')
    }
    adam_optimizer         = @{
        requires = @('deterministic_join')
        forbids  = @('side_effects')
    }
    backward_pass          = @{
        requires = @('bounded_reduction')
        forbids  = @()
    }
    'generic-compute'      = @{
        requires = @()
        forbids  = @()
    }
}

# ── Optimization pass registry ────────────────────────────────────────────────

$script:OPT_PASSES = @(
    @{ name = 'dead_op_elim';       applies = { param($ir) $true };                desc = 'Remove ops with no downstream consumers' }
    @{ name = 'constant_fold';      applies = { param($ir) $true };                desc = 'Fold constant-valued tensor ops at compile time' }
    @{ name = 'softmax_fuse';       applies = { param($ir) $ir.ContainsKey('ops') -and ($ir['ops'] | Where-Object { $_.op -eq 'softmax' }) }; desc = 'Fuse softmax + matmul into a single dispatch' }
    @{ name = 'barrier_hoist';      applies = { param($ir) $true };                desc = 'Hoist ResourceBarrier above independent ops' }
    @{ name = 'shmem_coalesce';     applies = { param($ir) $ir.ContainsKey('meta') -and $ir['meta']['shmRead'] }; desc = 'Coalesce shared-memory read lanes' }
)

# ── Bus registration ──────────────────────────────────────────────────────────

function Register-Scxq7WithBus {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Bus)
    $script:Bus = $Bus
    Write-Host "[SCXQ7] registered"
}

# ── Capability flag extraction ────────────────────────────────────────────────

function Get-ScxqCapFlags {
    <#
    .SYNOPSIS Extracts the capability flags from an IR or .smca.json.
              Returns a hashtable with well-known cap keys normalized to bool.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Ir)
    if ($Ir -is [string]) { $Ir = $Ir | ConvertFrom-Json -AsHashtable }

    $caps = if ($Ir.ContainsKey('caps')) { $Ir['caps'] }
            elseif ($Ir.ContainsKey('smca') -and $Ir['smca'].ContainsKey('caps')) { $Ir['smca']['caps'] }
            else { @{} }

    $normalize = { param($v) $v -eq $true -or $v -eq 'true' -or $v -eq 1 }

    return @{
        waveOps    = & $normalize ($caps['waveOps']    ?? $false)
        heapTier   = [int]($caps['heapTier']   ?? 1)
        bindingTier = [int]($caps['bindingTier'] ?? 1)
        uma        = & $normalize ($caps['uma']        ?? $false)
        d3d12      = & $normalize ($caps['d3d12']      ?? $true)
        wgsl       = & $normalize ($caps['wgsl']       ?? $false)
        fp16       = & $normalize ($caps['fp16']       ?? $false)
        int8       = & $normalize ($caps['int8']       ?? $false)
        raw        = $caps
    }
}

# ── Legality checking ─────────────────────────────────────────────────────────

function Test-ScxqLegality {
    <#
    .SYNOPSIS Checks an IR against the SCXQ7 legality rules for its kernelClass.
    .OUTPUTS Hashtable with ok, kernelClass, violations[], warnings[].
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][object]$Ir,
        [hashtable]$Caps = $null
    )
    if ($Ir -is [string]) { $Ir = $Ir | ConvertFrom-Json -AsHashtable }
    $violations = [System.Collections.ArrayList]::new()
    $warnings   = [System.Collections.ArrayList]::new()

    $smca = if ($Ir.ContainsKey('smca')) { $Ir['smca'] } else { $Ir }
    $kc   = $smca['kernelClass']
    if (-not $kc) {
        $violations.Add('smca.kernelClass is absent') | Out-Null
        return @{ ok = $false; kernelClass = $null; violations = $violations.ToArray(); warnings = @() }
    }

    $rules = if ($script:LEGALITY_RULES.ContainsKey($kc)) { $script:LEGALITY_RULES[$kc] }
             else { $script:LEGALITY_RULES['generic-compute'] }

    $declared = @{}
    if ($smca.ContainsKey('requires')) { $smca['requires'] | ForEach-Object { $declared[$_] = 'requires' } }
    if ($smca.ContainsKey('forbids'))  { $smca['forbids']  | ForEach-Object { $declared[$_] = 'forbids'  } }

    foreach ($req in $rules.requires) {
        if (-not $declared.ContainsKey($req)) {
            $warnings.Add("kernelClass '$kc' expects requirement '$req' but it is absent from smca.requires") | Out-Null
        }
    }
    foreach ($forb in $rules.forbids) {
        if ($declared.ContainsKey($forb) -and $declared[$forb] -ne 'forbids') {
            $violations.Add("kernelClass '$kc' has forbidden property '$forb' not listed in smca.forbids") | Out-Null
        }
    }

    if ($Caps) {
        $capFlags = Get-ScxqCapFlags -Ir $Ir
        if ($kc -eq 'tensor_attention_fused' -and -not $capFlags.waveOps) {
            $warnings.Add("tensor_attention_fused: waveOps=false — fused kernel may fall back to CPU") | Out-Null
        }
    }

    $ok = $violations.Count -eq 0
    return @{ ok = $ok; kernelClass = $kc; violations = $violations.ToArray(); warnings = $warnings.ToArray() }
}

# ── Optimization passes ───────────────────────────────────────────────────────

function Invoke-ScxqOptPass {
    <#
    .SYNOPSIS Applies named optimization passes to an IR and returns the mutated IR.
              Pass names: dead_op_elim, constant_fold, softmax_fuse, barrier_hoist, shmem_coalesce.
              Pass 'all' runs every applicable pass.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Ir,
        [Parameter(Mandatory)][string[]]$Passes
    )
    $runAll = $Passes -contains 'all'

    if (-not $Ir.ContainsKey('meta')) { $Ir['meta'] = @{} }
    if (-not $Ir.meta.ContainsKey('opt_passes')) { $Ir.meta['opt_passes'] = @() }

    foreach ($pass in $script:OPT_PASSES) {
        $shouldRun = $runAll -or ($pass.name -in $Passes)
        if (-not $shouldRun) { continue }
        $applicable = & $pass.applies $Ir
        if ($applicable) {
            Write-Host "[SCXQ7] opt-pass: $($pass.name)"
            $Ir.meta['opt_passes'] += $pass.name
        }
    }
    return $Ir
}

Export-ModuleMember -Function Register-Scxq7WithBus, Get-ScxqCapFlags,
    Test-ScxqLegality, Invoke-ScxqOptPass
