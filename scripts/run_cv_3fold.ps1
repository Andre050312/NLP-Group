# Nested 3-fold cross-validation on the Restaurant3 training pool (official test never used).
# Fold splits are grouped by exact text, so duplicate reviews cannot leak across folds.
# Runs control (with CLS) first, then the --no-cls-feature ablation, sequentially on one GPU.
# Interrupted runs can be continued: --resume skips folds whose fingerprints still match.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_cv_3fold.ps1
#        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_cv_3fold.ps1 -Folds 3 -Config nocls
param(
    [int]$Folds = 3,
    [int]$SplitSeed = 42,
    [ValidateSet("control", "nocls", "both")]
    [string]$Config = "both"
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSNativeCommandUseErrorActionPreference = $false
$env:HF_HOME = "D:\NLP-cache\huggingface"
$env:PYTHONUNBUFFERED = "1"
$python = ".\.venv\Scripts\python.exe"

$configs = if ($Config -eq "both") { @("control", "nocls") } else { @($Config) }

foreach ($c in $configs) {
    $out = "runs/cv_${c}_${Folds}fold"
    # Assign inside the branch: an `if` used as an expression unrolls a one-element
    # array to a scalar string, and splatting a string passes it character by character.
    [string[]]$extra = @()
    if ($c -eq "nocls") { $extra = @("--no-cls-feature") }
    Write-Output "===== cross-validate $c ($Folds folds) ====="
    & $python -m absa_enhanced cross-validate --dataset restaurant --output $out `
        --mode pair --folds $Folds --split-seed $SplitSeed --seed 42 `
        --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 `
        --class-weighting none --selection-metric gold_accuracy --device auto --resume @extra 2>&1 |
        Tee-Object -FilePath "D:\NLP-cache\cv_$c.txt" | Out-Null
    if (Test-Path "$out/cv_summary.json") {
        $s = Get-Content "$out/cv_summary.json" -Raw | ConvertFrom-Json
        $g = $s.oof_gold_aspect_sentiment
        Write-Output ("{0}: OOF accuracy={1:N4} macro={2:N4} neutralF1={3:N4}" -f `
            $c, $g.accuracy, $g.macro_f1, $g.per_class.neutral.f1)
    } else {
        Write-Output "WARNING: $out/cv_summary.json not found"
    }
}
Write-Output "===== done ====="
