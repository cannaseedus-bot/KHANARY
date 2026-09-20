#requires -Version 7.4
#requires -Modules @{ ModuleName = 'JROM'; ModuleVersion = '1.0.0' }

<#
.SYNOPSIS
    Compile a .kuhul source file with the patched KXC compiler and emit a JROM event.
.DESCRIPTION
    This wrapper calls the KXC executable that lives under micronaut‑v4/DRIVERS/kxc/,
    forces JSON reporting, and forwards the resulting SCXQ2 payload (if any) to the
    JROM replay stream. The tool can be invoked from the OpenAI‑style tool‑call
    interface (Chat module) or directly from PowerShell.
.PARAMETER SourcePath
    Absolute path to the .kuhul source file you want to compile.
.PARAMETER OutDir
    Directory where the compiler will drop its JSON SMCA report.
#>

param(
    [Parameter(Mandatory)][string] $SourcePath,
    [Parameter(Mandatory)][string] $OutDir
)

# ---------------------------------------------------------------------
# Locate the KXC binary (the patched version you built earlier).
# ---------------------------------------------------------------------
$kxcExe      = Join-Path $PSScriptRoot '..\..\versions\kxc-v1.0.0\bin\kxc.exe'
$kxcRegistry = Join-Path $PSScriptRoot '..\..\versions\kxc-v1.0.0\registry'
if (-not (Test-Path $kxcExe)) {
    throw "KXC executable not found at $kxcExe — build with versions/kxc-v1.0.0/build_kxc.bat"
}

# ---------------------------------------------------------------------
# Run the compiler.
# ---------------------------------------------------------------------
$argList = @(
    $SourcePath,
    '--outdir'   $OutDir,
    '--registry' $kxcRegistry
)
Write-Host "[kxc.compile] invoking $kxcExe $($argList -join ' ')"
try {
    $proc = Start-Process -FilePath $kxcExe -ArgumentList $argList -NoNewWindow -PassThru -Wait 
            -RedirectStandardOutput $null -RedirectStandardError $null
} catch {
    throw "KXC failed: $($_.Exception.Message)"
}

# ---------------------------------------------------------------------
# Load the generated report (the compiler always writes <basename>.json).
# ---------------------------------------------------------------------
$reportFile = Join-Path $OutDir ( [IO.Path]::GetFileNameWithoutExtension($SourcePath) + '.smca.json' )
if (-not (Test-Path $reportFile)) {
    throw "KXC finished but no JSON report was produced (expected $reportFile)"
}
$report = Get-Content -Raw -LiteralPath $reportFile | ConvertFrom-Json -Depth 10

# ---------------------------------------------------------------------
# Emit a JROM event so the replay stream contains a deterministic record.
# ---------------------------------------------------------------------
$event = [pscustomobject]@{
    type      = 'kxc.compile'
    timestamp = [DateTimeOffset]::Now.ToString('o')
    source    = $SourcePath
    outDir    = $OutDir
    hash      = $report.sourceHash   # the compiler already hashes the source text
    registryMatched = $report.registryMatched
    collapseClass   = $report.collapseClass
    scxq2    = $report.SCXQ2        # deterministic IR for downstream providers
}
Invoke-JROMEvent -Event $event

Write-Host "[kxc.compile] success – JROM event emitted"
