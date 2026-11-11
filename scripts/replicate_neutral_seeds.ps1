# Replicate one configuration at extra training seeds on a fixed split.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1
#        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1 -Config control -Seeds 43,44
param(
    [ValidateSet("control", "nocls")]
    [string]$Config = "nocls",
    # Comma-separated, e.g. "43,44". Parsed by hand because `powershell -File`
    # passes the whole token as one string instead of splitting it.
    [string]$Seeds = "43,44",
    [int]$SplitSeed = 42
)
$seedList = @($Seeds -split "[,\s]+" | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
if ($seedList.Count -eq 0) { throw "No seeds parsed from '$Seeds'" }
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# Native commands write progress to stderr; do not treat that as fatal.
$PSNativeCommandUseErrorActionPreference = $false
$env:HF_HOME = "D:\NLP-cache\huggingface"
$env:PYTHONUNBUFFERED = "1"
$python = ".\.venv\Scripts\python.exe"

$prefix = if ($Config -eq "nocls") { "pair_nocls" } else { "pair_ctrl" }
$label = if ($Config -eq "nocls") { "pair_no_cls" } else { "pair_control" }
# Assign inside the branch: an `if` used as an expression unrolls a one-element
# array to a scalar string, and splatting a string passes it character by character.
[string[]]$extra = @()
if ($Config -eq "nocls") { $extra = @("--no-cls-feature") }

foreach ($seed in $seedList) {
    $run = "runs/${prefix}_s$seed"
    if (Test-Path $run) { Remove-Item -Recurse -Force $run }
    Write-Output "===== train $Config seed $seed ====="
    & $python -m absa_enhanced train --mode pair --dataset restaurant --output $run `
        --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 `
        --seed $seed --split-seed $SplitSeed --class-weighting none `
        --selection-metric gold_accuracy --device auto @extra 2>&1 |
        Tee-Object -FilePath "D:\NLP-cache\log_${prefix}_s$seed.txt" | Out-Null
    Write-Output "===== diagnostics $Config seed $seed ====="
    foreach ($split in @("validation", "test")) {
        $out = "runs/analysis_${prefix}_s${seed}_$split"
        if (Test-Path $out) { Remove-Item -Recurse -Force $out }
        $file = if ($split -eq "validation") { "$run/validation_best_gold_predictions.jsonl" } else { "$run/test_gold_predictions.jsonl" }
        $extraArgs = if ($split -eq "test") { @("--gold-xml", "Database/Restaurant3/test.xml") } else { @() }
        & $python -m neutral_diagnostics analyze --predictions $file @extraArgs --dataset restaurant3 `
            --split $split --run-id "${prefix}_seed$seed" --model-label $label `
            --training-seed $seed --output $out | Out-Null
    }
}
Write-Output "===== offset for $Config ====="
$runs = $seedList | ForEach-Object { "runs/${prefix}_s$_" }
& $python scripts/tune_neutral_offset.py --runs @runs --metric macro_f1 --bootstrap 2000 2>&1 |
    Tee-Object -FilePath "D:\NLP-cache\offset_${prefix}.txt"
Write-Output "===== done ====="
