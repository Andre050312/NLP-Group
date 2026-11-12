# Replicate one configuration at extra training seeds on a fixed split.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1
#        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1 -Config control -Seeds 43,44
#        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1 -Config gate -Seeds 43,44
param(
    [ValidateSet("control", "nocls", "gate", "ordinal", "clause")]
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

# Prefixes match the existing seed-42 directories so five seeds of one config share a stem.
# control seed 42 lives in runs/pair_heads_softmax_s42, the rest in runs/pair_ctrl_s43..46.
$prefix = switch ($Config) {
    "nocls"   { "pair_nocls" }
    "control" { "pair_ctrl" }
    "gate"    { "pair_heads_gate" }
    "ordinal" { "pair_heads_ordinal" }
    "clause"  { "pair_clause_softmax" }
}
$label = switch ($Config) {
    "nocls"   { "pair_no_cls" }
    "control" { "pair_control" }
    "gate"    { "pair_gate_head" }
    "ordinal" { "pair_ordinal_head" }
    "clause"  { "pair_clause_scope" }
}
# Assign inside the branch: an `if` used as an expression unrolls a one-element
# array to a scalar string, and splatting a string passes it character by character.
[string[]]$extra = @()
switch ($Config) {
    "nocls"   { $extra = @("--no-cls-feature") }
    "gate"    { $extra = @("--sentiment-head", "gate") }
    "ordinal" { $extra = @("--sentiment-head", "ordinal") }
    "clause"  { $extra = @("--context-scope", "clause") }
}

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
        [string[]]$splitArgs = @()
        if ($split -eq "test") { $splitArgs = @("--gold-xml", "Database/Restaurant3/test.xml") }
        & $python -m neutral_diagnostics analyze --predictions $file @splitArgs --dataset restaurant3 `
            --split $split --run-id "${prefix}_seed$seed" --model-label $label `
            --training-seed $seed --output $out | Out-Null
    }
}
Write-Output "===== offset for $Config ====="
$runs = $seedList | ForEach-Object { "runs/${prefix}_s$_" }
& $python scripts/tune_neutral_offset.py --runs @runs --metric macro_f1 --bootstrap 2000 2>&1 |
    Tee-Object -FilePath "D:\NLP-cache\offset_${prefix}.txt"
Write-Output "===== done ====="
