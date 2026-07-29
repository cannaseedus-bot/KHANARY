# ForwardPass.ps1 -- Experiment C orchestrator (PowerShell glue, NNCK-Runtime style).
#
# The measurement boundary is a bridge: the Trinity FIELD adapts (Python @flux over evidence),
# the frozen model's forward pass + alignment are measured (Python eval), and PowerShell hides
# the seam -- same pattern as Invoke-GptOssLayerForward. Proves Experiment C:
#
#   C0 = frozen model + frozen field        (baseline alignment)
#   C1 = frozen model + @flux-adapted field  (field learned from evidence, MODEL UNCHANGED)
#
# If ALIGNMENT(C1) > ALIGNMENT(C0), semantic learning is an independent, measurable process --
# the external field improves guidance with zero neural-weight updates.
#
# Usage:
#   .\ForwardPass.ps1 -ModelDir <frozenB> -Field <field0.json> -Heldout <held.jsonl> -Evidence <ev.jsonl>

param(
    [Parameter(Mandatory)] [string]$ModelDir,   # frozen model B (never modified)
    [Parameter(Mandatory)] [string]$Field,      # field_0 (C0)
    [Parameter(Mandatory)] [string]$Heldout,    # held-out eval slice (disjoint)
    [Parameter(Mandatory)] [string]$Evidence,   # @flux evidence slice (disjoint from fit + heldout)
    [string]$Tools   = $PSScriptRoot,
    [int]$Limit      = 400,
    [double]$FluxLr  = 0.05,
    [string]$Python  = "python",
    [switch]$Pmi,    # C2 value-proxy: reward=sign(PMI) instead of +1 (only the evidence signal changes)
    [switch]$Graded, # C2b: reward=sign(PMI)*tanh(|PMI|/scale) -- direction + bounded strength
    [double]$PmiScale = 2.0
)

function Get-Alignment {
    param([string]$Model, [string]$FieldPath)
    $out  = & $Python "$Tools\eval_field_consistency.py" $Model $Heldout $FieldPath --limit $Limit 2>&1
    $line = $out | Where-Object { $_ -match '^RESULT ' } | Select-Object -First 1
    if (-not $line) { Write-Host ($out | Select-Object -Last 5); throw "no RESULT line from evaluator" }
    return [double](($line -split '\s+')[-1])   # RESULT <dir> <end_lp> <all_lp> <alignment>
}

Write-Host "== C0: frozen model + frozen field ==" -ForegroundColor Cyan
$c0 = Get-Alignment -Model $ModelDir -FieldPath $Field
Write-Host ("   ALIGNMENT(C0) = {0:F4}" -f $c0)

$lbl = if ($Graded) { "C2b (reward=sign(PMI)*tanh(|PMI|/$PmiScale))" } elseif ($Pmi) { "C2 (reward=sign(PMI), value-proxy)" } else { "C1 (reward=+1, frequency)" }
Write-Host "== @flux: field learns from evidence -- $lbl (model FROZEN) ==" -ForegroundColor Cyan
$suffix = if ($Graded) { "_c2b.json" } elseif ($Pmi) { "_c2.json" } else { "_c1.json" }
$fieldC1 = [System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName($Field),
           [System.IO.Path]::GetFileNameWithoutExtension($Field) + $suffix)
$adaptArgs = @("$Tools\trinity_field.py", "adapt", $Field, $Evidence, $fieldC1, "--lr", $FluxLr)
if ($Pmi) { $adaptArgs += "--pmi" }
if ($Graded) { $adaptArgs += @("--graded", "--pmi-scale", $PmiScale) }
& $Python @adaptArgs | Write-Host

Write-Host "== ${lbl}: frozen model + @flux-adapted field ==" -ForegroundColor Cyan
$c1 = Get-Alignment -Model $ModelDir -FieldPath $fieldC1
Write-Host ("   ALIGNMENT = {0:F4}" -f $c1)

$delta = $c1 - $c0
Write-Host ("== RESULT  C0={0:F4}  C1={1:F4}  dALIGN={2:+0.0000} ==" -f $c0, $c1, $delta) -ForegroundColor Yellow
if ($delta -gt 0.0) {
    Write-Host "   PASS: the field learned useful guidance with the model FROZEN (semantic learning is independent)." -ForegroundColor Green
} else {
    Write-Host "   NO GAIN: @flux adaptation did not improve alignment on held-out." -ForegroundColor Red
}
[PSCustomObject]@{ C0 = $c0; C1 = $c1; Delta = $delta; ModelFrozen = $ModelDir; Field0 = $Field; Field1 = $fieldC1 }
