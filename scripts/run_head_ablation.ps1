# Sentiment-head ablation on Restaurant3, fixed split, one seed each.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_head_ablation.ps1 [-Mode pair|joint] [-Seed 42]
param(
    [ValidateSet("pair", "joint")] [string]$Mode = "pair",
    [int]$Seed = 42,
    [string]$Dataset = "restaurant",
    [string]$Encoder = "microsoft/deberta-v3-base"
)

$ErrorActionPreference = "Stop"
$heads = @("softmax", "ordinal", "gate")
foreach ($head in $heads) {
    $output = "runs/${Mode}_${Dataset}_head_${head}_seed${Seed}"
    if (Test-Path $output) { Remove-Item -Recurse -Force $output }
    Write-Host "=== $Mode / $head -> $output ==="
    python -m absa_enhanced train --mode $Mode --dataset $Dataset --encoder $Encoder `
        --output $output --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 `
        --seed $Seed --split-seed 42 --class-weighting none --selection-metric gold_accuracy `
        --sentiment-head $head --device auto
    if ($LASTEXITCODE -ne 0) { throw "Training failed for head $head" }
    Get-Content "$output/test_metrics.json"
}
