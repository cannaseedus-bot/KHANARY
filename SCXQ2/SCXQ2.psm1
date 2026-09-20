#Requires -Version 7.4

<#
.SYNOPSIS
    SCXQ2.psm1 — Backend-neutral lowered IR layer (SCXQ2 decode/encode/phase).
.DESCRIPTION
    SCXQ2 is the semantic mid-point between KXC source and backend emitters.
    This module provides IR read/write, round-trip verification, mode-bit extraction,
    phase invocation (Pop↔Xul oscillation), and tensor-op lookup.
    Binary: dist/khanary-server/kuhul-server.cjs embeds the native SCXQ2 executor.
    Spec: SCXQ2.md (project root).
#>

$script:RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$script:RegistryDir = Join-Path $script:RepoRoot 'versions\kxc-v1.0.0\registry'
$script:Bus         = $null

# ── Phase constants (matches SCXQ2.md mode-bit table) ────────────────────────

$script:PHASE_POP  = 0x01   # Pop  — read SHM / observe (π/0)
$script:PHASE_WO   = 0x02   # Wo   — schedule (π/6)
$script:PHASE_YAX  = 0x04   # Yax  — branch candidates (π/3)
$script:PHASE_SEK  = 0x08   # Sek  — execute / classify
$script:PHASE_CHEN = 0x10   # Chen — project / verify (π/3)
$script:PHASE_XUL  = 0x20   # Xul  — write SHM / entropy (π/3 → 2π/3 oscillation)

$script:PHASE_MAP = @{
    Pop  = $script:PHASE_POP
    Wo   = $script:PHASE_WO
    Yax  = $script:PHASE_YAX
    Sek  = $script:PHASE_SEK
    Chen = $script:PHASE_CHEN
    Xul  = $script:PHASE_XUL
}

# ── Tensor op table (mirrors SCXQ2.md §Tensor Ops) ───────────────────────────

$script:TENSOR_OPS = @{
    matmul        = @{ glyph = '⊗'; hlsl = 'mul';        wgsl = 'dot / textureSampleLevel'; cpu = 'cblas_sgemm' }
    add           = @{ glyph = '⊕'; hlsl = '+';           wgsl = '+';                       cpu = '+' }
    sub           = @{ glyph = '⊖'; hlsl = '-';           wgsl = '-';                       cpu = '-' }
    scale         = @{ glyph = '⊘'; hlsl = '* scalar';    wgsl = '* scalar';                cpu = '* scalar' }
    conv          = @{ glyph = '⊛'; hlsl = 'custom conv'; wgsl = 'custom conv';             cpu = 'im2col+sgemm' }
    branch        = @{ glyph = '⊜'; hlsl = 'if';          wgsl = 'if';                      cpu = 'if' }
    clamp_relu    = @{ glyph = '⊝'; hlsl = 'clamp/max';   wgsl = 'clamp/max';              cpu = 'clamp/fmax' }
    add_bias      = @{ glyph = '⊞'; hlsl = '+bias';       wgsl = '+bias';                  cpu = '+bias' }
    softmax       = @{ glyph = null; hlsl = 'softmax';     wgsl = 'softmax';                cpu = 'softmax' }
    layer_norm    = @{ glyph = null; hlsl = 'layer_norm';  wgsl = 'layer_norm';             cpu = 'layer_norm' }
    silu          = @{ glyph = null; hlsl = 'silu';        wgsl = 'silu';                   cpu = 'silu' }
}

# ── Bus registration ──────────────────────────────────────────────────────────

function Register-Scxq2WithBus {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Bus)
    $script:Bus = $Bus
    Write-Host "[SCXQ2] registered"
}

# ── IR I/O ────────────────────────────────────────────────────────────────────

function Read-Scxq2Ir {
    <#
    .SYNOPSIS Loads a .ir.json file and returns it as a hashtable.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path $Path)) { throw "SCXQ2: ir.json not found: $Path" }
    $ir = Get-Content -Raw $Path | ConvertFrom-Json -AsHashtable
    if (-not $ir.ContainsKey('ir_version')) {
        throw "SCXQ2: ir.json missing ir_version field: $Path"
    }
    return $ir
}

function Write-Scxq2Ir {
    <#
    .SYNOPSIS Serializes and writes an IR hashtable as .ir.json.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][hashtable]$Ir
    )
    $required = @('ir_version','smca','ops','meta')
    $missing = $required | Where-Object { -not $Ir.ContainsKey($_) }
    if ($missing) { throw "SCXQ2: Write-Scxq2Ir missing fields: $($missing -join ', ')" }
    $Ir | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8NoBOM
    Write-Host "[SCXQ2] wrote IR: $Path"
}

# ── Round-trip verification ───────────────────────────────────────────────────

function Test-ScxqRoundTrip {
    <#
    .SYNOPSIS Serializes and re-parses an IR, verifying structural identity.
    .OUTPUTS Hashtable with ok, diff (if any), original, roundtripped.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Ir)
    if ($Ir -is [string]) { $Ir = $Ir | ConvertFrom-Json -AsHashtable }
    $serialized  = $Ir | ConvertTo-Json -Depth 12
    $roundtripped = $serialized | ConvertFrom-Json -AsHashtable
    $reSerialized = $roundtripped | ConvertTo-Json -Depth 12
    $ok = $serialized -eq $reSerialized
    return @{
        ok           = $ok
        diff         = if (-not $ok) { "Serialized forms differ (length: $($serialized.Length) vs $($reSerialized.Length))" } else { $null }
        original     = $Ir
        roundtripped = $roundtripped
    }
}

# ── Mode-bit extraction ───────────────────────────────────────────────────────

function Get-ScxqModeBits {
    <#
    .SYNOPSIS Extracts SCXQ2 mode bits from an IR or .smca.json block.
    .OUTPUTS Hashtable with phase name keys → bool.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Ir)
    if ($Ir -is [string]) { $Ir = $Ir | ConvertFrom-Json -AsHashtable }
    $smca = if ($Ir.ContainsKey('smca')) { $Ir['smca'] } else { $Ir }
    $rawBits = [int]($smca['modeBits'] ?? 0)
    $result = @{}
    foreach ($phaseName in $script:PHASE_MAP.Keys) {
        $result[$phaseName] = ($rawBits -band $script:PHASE_MAP[$phaseName]) -ne 0
    }
    $result['raw'] = $rawBits
    return $result
}

# ── Phase invocation ──────────────────────────────────────────────────────────

function Invoke-ScxqPhase {
    <#
    .SYNOPSIS Marks an IR as having passed through a given phase. Adds the phase
              to the 'phases_executed' list in meta and updates the modeBits field.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Ir,
        [Parameter(Mandatory)][ValidateSet('Pop','Wo','Yax','Sek','Chen','Xul')][string]$Phase
    )
    if (-not $Ir.ContainsKey('meta'))  { $Ir['meta'] = @{} }
    if (-not $Ir.meta.ContainsKey('phases_executed')) { $Ir.meta['phases_executed'] = @() }
    if ($Phase -notin $Ir.meta['phases_executed']) {
        $Ir.meta['phases_executed'] += $Phase
    }
    $smca = if ($Ir.ContainsKey('smca')) { $Ir['smca'] } else { @{} }
    $current = [int]($smca['modeBits'] ?? 0)
    $smca['modeBits'] = $current -bor $script:PHASE_MAP[$Phase]
    if (-not $Ir.ContainsKey('smca')) { $Ir['smca'] = $smca }
    Write-Host "[SCXQ2] phase $Phase applied (modeBits=0x$($smca['modeBits'].ToString('X2')))"
    return $Ir
}

# ── Tensor-op lookup ──────────────────────────────────────────────────────────

function Get-ScxqTensorOp {
    <#
    .SYNOPSIS Returns the backend mapping for a named tensor op.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Op,
        [ValidateSet('hlsl','wgsl','cpu','glyph')][string]$Backend = 'hlsl'
    )
    if (-not $script:TENSOR_OPS.ContainsKey($Op)) {
        throw "SCXQ2: unknown tensor op '$Op'. Known: $($script:TENSOR_OPS.Keys -join ', ')"
    }
    $entry = $script:TENSOR_OPS[$Op]
    return @{ op = $Op; backend = $Backend; mapping = $entry[$Backend]; full = $entry }
}

Export-ModuleMember -Function Register-Scxq2WithBus, Read-Scxq2Ir, Write-Scxq2Ir,
    Test-ScxqRoundTrip, Get-ScxqModeBits, Invoke-ScxqPhase, Get-ScxqTensorOp
