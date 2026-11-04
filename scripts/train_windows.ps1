param(
    [ValidateSet("restaurant", "laptop", "mams")]
    [string]$Dataset = "restaurant",
    [string]$Output = "",
    [string]$Encoder = "microsoft/deberta-v3-base",
    [int]$Epochs = 6,
    [int]$BatchSize = 2,
    [int]$GradAccum = 4,
    [int]$MaxLength = 192,
    [int]$Stride = 64,
    [int]$Seed = 42,
    [string]$CacheDir = "D:\NLP-cache\huggingface"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = "runs/deberta_joint_${Dataset}_seed${Seed}"
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    $env:HF_HOME = $CacheDir
    & python -m absa_enhanced train `
        --dataset $Dataset `
        --encoder $Encoder `
        --output $Output `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --grad-accum $GradAccum `
        --max-length $MaxLength `
        --stride $Stride `
        --seed $Seed `
        --device cuda
    $TrainingExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $TrainingExitCode
