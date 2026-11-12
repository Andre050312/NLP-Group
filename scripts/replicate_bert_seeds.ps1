# Train the BERT-base sentence-pair baseline at several training seeds on one fixed split.
# The baseline historically derived its split from --seed; --split-seed pins it so the
# partition matches the DeBERTa pair runs (1682/295/800 sentences).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_bert_seeds.ps1
#        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_bert_seeds.ps1 -Seeds 43,44
param(
    # Comma-separated, e.g. "42,43,44,45,46". Parsed by hand because `powershell -File`
    # passes the whole token as one string instead of splitting it.
    [string]$Seeds = "42,43,44,45,46",
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

foreach ($seed in $seedList) {
    $run = "runs/bert_split${SplitSeed}_seed$seed"
    if (Test-Path $run) { Remove-Item -Recurse -Force $run }
    Write-Output "===== train bert seed $seed ====="
    & $python -m bert_baseline train `
        --train "Database/Restaurant3/train.xml" `
        --test "Database/Restaurant3/test.xml" `
        --output $run --model google-bert/bert-base-uncased `
        --epochs 3 --batch-size 4 --learning-rate 2e-05 --weight-decay 0.01 `
        --seed $seed --split-seed $SplitSeed --validation-fraction 0.15 --device auto 2>&1 |
        Tee-Object -FilePath "D:\NLP-cache\log_bert_s$seed.txt" | Out-Null
    if (-not (Test-Path "$run/test_metrics.json")) { throw "seed $seed produced no test_metrics.json" }
    $m = Get-Content "$run/test_metrics.json" -Raw | ConvertFrom-Json
    Write-Output ("seed {0}: acc={1:N4} macro={2:N4} neutralF1={3:N4}" -f `
        $seed, $m.accuracy, $m.macro_f1, $m.per_class.neutral.f1)
}
Write-Output "===== done ====="
