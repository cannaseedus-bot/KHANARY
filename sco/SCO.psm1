#Requires -Version 7.4

<#
.SYNOPSIS
    SCO.psm1 — Backend emitter selection and artifact management (SCO/1).
.DESCRIPTION
    SCO is the output stage of the KXC pipeline: it maps a (kernelClass, caps) pair
    to the correct backend emitter (HLSL/WGSL/cpu.cpp/cl/dml.json) and reports the
    produced artifact paths. The actual compilation is done by kxc.exe; SCO.psm1
    provides PowerShell-side backend selection and artifact discovery.
    Pipeline position: SCXQ7 → SCO/1 → IDB
#>

$script:RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$script:OutputsDir  = Join-Path $script:RepoRoot 'versions\kxc-v1.0.0\examples\outputs'
$script:Bus         = $null

# ── Backend preference table (kernelClass → preferred emitters, in priority order) ─

$script:BACKEND_PREF = @{
    tensor_attention_fused = @('hlsl','wgsl','cpu')
    adam_optimizer         = @('hlsl','cpu','wgsl')
    gradient_accum         = @('hlsl','cpu','wgsl')
    backward_pass          = @('hlsl','cpu','wgsl')
    fold_clamp             = @('hlsl','wgsl','cpu')
    moe_route_top2         = @('hlsl','cpu')
    mesh_meshlet_cull      = @('hlsl','cpu')
    mesh_normal_compute    = @('hlsl','cpu')
    mesh_vertex_process    = @('hlsl','cpu')
    'generic-compute'      = @('cpu','hlsl','wgsl')
}

# ── Emitter extension map ──────────────────────────────────────────────────────

$script:BACKEND_EXT = @{
    hlsl     = '.hlsl'
    wgsl     = '.wgsl'
    cpu      = '.cpu.cpp'
    cl       = '.cl'
    dml      = '.dml.json'
    ir       = '.ir.json'
    smca     = '.smca.json'
}

# ── Bus registration ──────────────────────────────────────────────────────────

function Register-ScoWithBus {
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Bus)
    $script:Bus = $Bus
    Write-Host "[SCO] registered"
}

# ── Backend selection ─────────────────────────────────────────────────────────

function Select-ScoBackend {
    <#
    .SYNOPSIS Returns the ordered list of emitter targets for a given kernelClass,
              filtered against hardware caps.
    .PARAMETER Caps  Hashtable from Get-ScxqCapFlags (or null for generic).
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$KernelClass,
        [hashtable]$Caps = $null
    )
    $prefs = if ($script:BACKEND_PREF.ContainsKey($KernelClass)) {
        $script:BACKEND_PREF[$KernelClass]
    } else {
        $script:BACKEND_PREF['generic-compute']
    }

    if ($Caps) {
        # Filter wgsl out if no D3D12/WebGPU adapter
        $prefs = $prefs | Where-Object {
            -not ($_ -eq 'wgsl' -and $Caps.ContainsKey('wgsl') -and -not $Caps.wgsl)
        }
        # Filter hlsl out if no D3D12 support
        $prefs = $prefs | Where-Object {
            -not ($_ -eq 'hlsl' -and $Caps.ContainsKey('d3d12') -and -not $Caps.d3d12)
        }
    }

    return @($prefs)
}

# ── Artifact discovery ────────────────────────────────────────────────────────

function Get-ScoArtifacts {
    <#
    .SYNOPSIS Discovers all compiled artifacts for a kernel in an output directory.
    .OUTPUTS Hashtable with backend → absolute path for each found artifact.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$KernelName,
        [string]$OutDir = $script:OutputsDir
    )
    $result = @{}
    foreach ($backend in $script:BACKEND_EXT.Keys) {
        $ext  = $script:BACKEND_EXT[$backend]
        $path = Join-Path $OutDir "$KernelName$ext"
        if (Test-Path $path) {
            $result[$backend] = (Resolve-Path $path).Path
        }
    }
    return $result
}

# ── Emitter invocation (delegates to kxc.exe via kxc_compile.ps1) ────────────

function Invoke-ScoEmit {
    <#
    .SYNOPSIS Runs kxc.exe on a .kuhul source and returns the artifact map.
              Requires the kxc.exe binary at versions/kxc-v1.0.0/bin/kxc.exe.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$SourcePath,
        [string]$OutDir = $script:OutputsDir
    )
    $kxcExe  = Join-Path $script:RepoRoot 'versions\kxc-v1.0.0\bin\kxc.exe'
    $kxcReg  = Join-Path $script:RepoRoot 'versions\kxc-v1.0.0\registry'
    if (-not (Test-Path $kxcExe)) {
        throw "SCO: kxc.exe not found at $kxcExe — build with versions/kxc-v1.0.0/build_kxc.bat"
    }
    if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }

    $argList = @($SourcePath, '--outdir', $OutDir, '--registry', $kxcReg)
    Write-Host "[SCO] invoking: $kxcExe $($argList -join ' ')"
    $proc = Start-Process -FilePath $kxcExe -ArgumentList $argList -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) { throw "SCO: kxc.exe exited with code $($proc.ExitCode)" }

    $base = [IO.Path]::GetFileNameWithoutExtension($SourcePath)
    return Get-ScoArtifacts -KernelName $base -OutDir $OutDir
}

# ── Artifact summary ──────────────────────────────────────────────────────────

function Get-ScoArtifactSummary {
    <#
    .SYNOPSIS Scans the outputs directory for all compiled kernels and returns a summary.
    #>
    [CmdletBinding()]
    param([string]$OutDir = $script:OutputsDir)
    if (-not (Test-Path $OutDir)) { return @() }
    $smcaFiles = Get-ChildItem -LiteralPath $OutDir -Filter '*.smca.json' -Recurse -ErrorAction SilentlyContinue
    $results = foreach ($f in $smcaFiles) {
        $smca = Get-Content -Raw $f.FullName | ConvertFrom-Json -AsHashtable
        $smcaBlock = if ($smca.ContainsKey('smca')) { $smca['smca'] } else { $smca }
        @{
            kernel          = $f.BaseName -replace '\.smca$', ''
            kernelClass     = $smcaBlock['kernelClass']
            collapseClass   = $smcaBlock['collapseClass']
            registryMatched = $smcaBlock['registryMatched']
            smcaPath        = $f.FullName
        }
    }
    return @($results)
}

Export-ModuleMember -Function Register-ScoWithBus, Select-ScoBackend,
    Get-ScoArtifacts, Invoke-ScoEmit, Get-ScoArtifactSummary
