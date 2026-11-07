param(
    [ValidateSet("restaurant", "laptop", "mams")]
    [string]$Dataset = "restaurant",
    [string]$Output = "",
    [string]$Encoder = "microsoft/deberta-v3-base",
    [ValidateSet("joint", "pair")]
    [string]$Mode = "joint",
    [int]$Epochs = 6,
    [int]$BatchSize = 2,
    [int]$GradAccum = 4,
    [int]$MaxLength = 192,
    [int]$Stride = 64,
    [int]$Seed = 42,
    [int]$SplitSeed = 42,
    [string]$CacheDir = "D:\NLP-cache\huggingface"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = "runs/deberta_${Mode}_${Dataset}_split${SplitSeed}_seed${Seed}"
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    $env:HF_HOME = $CacheDir
    & python -m absa_enhanced train `
        --dataset $Dataset `
        --encoder $Encoder `
        --mode $Mode `
        --output $Output `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --grad-accum $GradAccum `
        --max-length $MaxLength `
        --stride $Stride `
        --seed $Seed `
        --split-seed $SplitSeed `
        --device cuda
    $TrainingExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $TrainingExitCode
