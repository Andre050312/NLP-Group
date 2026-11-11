# Replicate the no-CLS configuration at extra training seeds on a fixed split.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# Native commands write progress to stderr; do not treat that as fatal.
$PSNativeCommandUseErrorActionPreference = $false
$env:HF_HOME = "D:\NLP-cache\huggingface"
$env:PYTHONUNBUFFERED = "1"
$python = ".\.venv\Scripts\python.exe"

foreach ($seed in 43, 44) {
    $run = "runs/pair_nocls_s$seed"
    if (Test-Path $run) { Remove-Item -Recurse -Force $run }
    Write-Output "===== train seed $seed ====="
    & $python -m absa_enhanced train --mode pair --dataset restaurant --output $run `
        --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 `
        --seed $seed --split-seed 42 --class-weighting none `
        --selection-metric gold_accuracy --no-cls-feature --device auto 2>&1 |
        Tee-Object -FilePath "D:\NLP-cache\log_nocls_s$seed.txt" | Out-Null
    Write-Output "===== diagnostics seed $seed ====="
    foreach ($split in @("validation", "test")) {
        $out = "runs/analysis_nocls_s${seed}_$split"
        if (Test-Path $out) { Remove-Item -Recurse -Force $out }
        $file = if ($split -eq "validation") { "$run/validation_best_gold_predictions.jsonl" } else { "$run/test_gold_predictions.jsonl" }
        $extra = if ($split -eq "test") { @("--gold-xml", "Database/Restaurant3/test.xml") } else { @() }
        & $python -m neutral_diagnostics analyze --predictions $file @extra --dataset restaurant3 `
            --split $split --run-id "pair_nocls_seed$seed" --model-label pair_no_cls `
            --training-seed $seed --output $out | Out-Null
    }
}
Write-Output "===== offset per seed ====="
& $python scripts/tune_neutral_offset.py --runs runs/pair_nocls_s43 runs/pair_nocls_s44 `
    --metric macro_f1 --bootstrap 2000 2>&1 | Tee-Object -FilePath "D:\NLP-cache\offset_seeds.txt"
Write-Output "===== done ====="
