# Historical local validation record

Date: 2026-11-11. Base commit: 41fd25405c00b7bf00dcd6512da48e6c22dcad84.
Branch: analysis/neutral-scope-diagnostics. This record describes that revision.

## Environment

Windows, Python 3.12.10, CPU PyTorch 2.14.1, Transformers 4.57.1,
tokenizers 0.22.2, scikit-learn 1.7.2 and NumPy 2.5.3. CUDA was unavailable.
The project .venv reused CPU PyTorch through system-site-packages and installed the pinned
Transformers and scikit-learn versions locally. Global packages were unchanged.
This differs from the Windows CUDA requirements, so GPU, mixed precision and memory use
were not validated. Global Transformers 5.22.1 lacked the required legacy tokenizer API.

## Completed checks

- Full UTF-8 regression run: 56 tests passed in 236.851 seconds, exit code 0.
- Seven neutral diagnostics tests covered AP/ties, denominators, validation, confidence,
  grouped bootstrap, exports and cross-result target coverage.
- Four CLS tests covered gradients, legacy/new checkpoints, paired prediction for all
  model types, tiny local CLI training and CV argument forwarding.
- Forty candidate pairs passed structural checks: 30 neutral, 5 positive and 5 negative.
  All started pending.
- Both real Restaurant3 three-fold plans generated identical fold assignments, excluding
  one official-test overlap and retaining 3040 sentences, 3601 aspects and 3034 text groups.
- Assignment SHA-256: 7308ebd613d76535ab6b1b061fb2e9231d2c3e9fca4e629a2bda1ed3395e427f.
- Compilation and git diff --check passed.

The clone lacked MAMS gitlink contents. Six XML files were restored from the bundled offline
ZIP and verified against pinned hashes. These local files were excluded from the contribution.

## Reproduce

```powershell
New-Item -ItemType Directory -Path runs/test_tmp -Force | Out-Null
$env:TEMP = Join-Path $PWD.Path 'runs/test_tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest discover -s tests -v
```

Use the active project environment. The workspace TEMP setting was specific to the restricted
host. PYTHONUTF8 kept parent/child encodings consistent on Unicode paths. An initial baseline
CLI failure came from log decoding before that setting. The full rerun log was stored locally
under ignored runs/full_tests_utf8.log.

## Research limitations at that time

Real group checkpoints, per-target predictions and original bias/gate curves were unavailable.
The earlier 45%, 0.963 and neutral F1 claims could not be recomputed in this validation.
Tiny random BERT fixtures checked software behavior. This validation did not train DeBERTa,
complete six real ablation runs, measure improvements or approve diagnostic labels.
