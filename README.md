# Aspect-Based Sentiment Analysis

COMP4136 Group 9 implements BERT and DeBERTa models for sentiment toward a supplied
aspect in a restaurant review. The default labels are negative, neutral and positive.
An explicit four-class mode preserves SemEval-2014 conflict annotations.

## Reading guide

| Location | Purpose |
|---|---|
| [Report draft](docs/report_draft.md) | Research questions, methods, experiments and limitations |
| [Recorded results](docs/neutral_results.md) | Historical three-class tables and reproduction commands |
| [Diagnostic workflow](docs/neutral_scope_study.md) | Neutral errors, controlled pairs and CLS ablation |
| [Validation record](docs/neutral_scope_validation.md) | Historical offline software checks |
| [Data overview](Database/README.md) | Sources, filtering, counts and overlap |
| bert_baseline/ | Sentence–aspect BERT classifier |
| absa_enhanced/ | DeBERTa pair classifier, joint BIO extractor and grouped CV |
| neutral_diagnostics/ | Three-class prediction and controlled-pair analysis |
| scripts/ | Dataset preparation, experiment launchers and result analysis |
| tests/ | Offline regression tests using tiny local models |
| resources/ | Checksum-pinned offline dataset archives |

The report draft and recorded results describe historical experiments. The four-class
implementation adds software support and a separate dataset. Four-class benchmark scores
require new training and evaluation.

## Environment

From the repository root, create the README's Windows environment.

```powershell
conda create --name nlp python=3.12 pip -y
conda activate nlp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
$env:HF_HOME = Join-Path $PWD.Path ".cache\huggingface"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Windows requirements select CUDA 12.8 PyTorch, which needs a compatible NVIDIA GPU and
driver. Transformers is pinned to 4.57.1. Use a complete local model directory with
--encoder for offline training. Model downloads require network access.
Use --device cuda when GPU execution is required. The auto setting can select CPU.

## Restore and verify data

```powershell
python scripts/download_datasets.py --offline
python scripts/download_datasets.py --check
```

The full bundled archive restores 12 three-class XML files, including MAMS.
The downloader checks pinned hashes, skips matching files and preserves the source split.
Network recovery uses the same source catalog.

```powershell
python scripts/download_datasets.py
```

The downloader also accepts --dataset, --import-from, --timeout and --retries.
Dataset manifests describe each conversion. Original review text and valid offsets are
preserved. Restaurant3 removes conflict annotations. Restaurant2015/2016 retain explicit
aspect spans after filtering implicit, invalid, conflicting and overlapping annotations.
MAMS retains its original three-class files.

Cross-dataset exact-text overlap is substantial. For example, Restaurant2016 training
overlaps Restaurant2015 test by 662 sentences. Train each benchmark separately, or remove
overlap with every held-out set before combining sources. See Database/overlap_report.json.

## Offline tests

```powershell
python -m unittest discover -s tests -v
```

Tests use local random tiny models and exercise data, checkpoints, CLI training and CV.
They do not require downloading pretrained models or running a full GPU experiment.

## DeBERTa pair classification

The model encodes the review and aspect query as two sequences. Its classifier combines
CLS, the target-span mean and the query mean, followed by LayerNorm, Dropout, a 256-unit
GELU layer and the sentiment output.

The following reproduces the historical three-class control settings.

```powershell
python -m absa_enhanced train --mode pair --dataset restaurant --output runs/deberta_pair_split42_seed42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto
```

Keep --split-seed fixed while varying --seed to compare training randomness.
Set --selection-metric gold_macro_f1 to select by validation Macro-F1 in a new protocol.
Use a fresh output directory when changing the protocol.

--no-cls-feature removes the final classifier's CLS input. The encoder and its
contextual target representations still access the sentence.
Pair models support --context-scope sentence or clause. The clause option crops the
target-containing clause and can discard useful context.

--sentiment-head accepts softmax, gate, ordinal and ordered.
Softmax is the default. Gate models opinion presence and conditional polarity.
Legacy ordinal retains its historical objective–inference mismatch for checkpoint
compatibility. Ordered uses consistent monotone cumulative probabilities. Gate,
ordinal and ordered require three labels. Their single-seed historical observations
cannot establish a repeatable ranking.

| Value | Output | Objective | Description |
|---|---|---|---|
| `softmax` (default) | 3-class logits | Cross-entropy | Baseline with compatible legacy checkpoints |
| `ordinal` | 2 legacy outputs | Legacy BCE | Legacy checkpoint compatibility with objective/inference mismatch |
| `ordered` | 2 ordered cumulative logits | Consistent cumulative BCE | Separate three-class head without recorded benchmark scores |
| `gate` | 1 presence + 2 polarity logits | BCE + binary CE | Opinion presence then polarity; neutral means no opinion |

## Four-class SemEval-2014

Pass --labels four to include conflict. The canonical order is
negative, neutral, positive, conflict. Neutral remains index 1. The separate
[Restaurant4 manifest](Database/Restaurant4/manifest.json) records the following counts.

| Split | Sentence records | Aspect targets | negative | neutral | positive | conflict |
|---|---:|---:|---:|---:|---:|---:|
| train | 3041 | 3693 | 805 | 633 | 2164 | 91 |
| test | 800 | 1134 | 196 | 196 | 728 | 14 |

The exported XML and per-aspect JSONL preserve conflict as its own label.
To recreate the export, choose a new directory. The exporter rejects existing output
and Restaurant3 paths. The default three-class downloader does not restore Restaurant4.

```powershell
python scripts/prepare_restaurant4.py --download --output Database/Restaurant4_new
python scripts/prepare_restaurant4.py --train-source PATH/Restaurants_Train_v2.xml --test-source PATH/Restaurants_Test_Gold.xml --output Database/Restaurant4_new
```

Use either download or local original four-label XML sources.

```powershell
python -m absa_enhanced train --mode pair --dataset restaurant --labels four --sentiment-head softmax --output runs/pair_four_s42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_macro_f1 --device cuda
```

Four-class mode permits softmax only. The encoder, pair construction and BIO extractor
retain their architecture. settings.json stores the actual labels and loading validates
output shapes strictly. A three-class checkpoint cannot initialize a four-class model
through the checkpoint loader.

Four-class evaluation includes conflict in per-class precision, recall, F1 and support.
There are 14 official-test conflict targets. Report this support alongside the score.
The label set and test denominator differ from Restaurant3, so four-class scores need
their own result tables.

## BERT baseline

```powershell
python -m bert_baseline train --model google-bert/bert-base-uncased --train Database/Restaurant3/train.xml --test Database/Restaurant3/test.xml --output runs/bert_split42_seed42 --epochs 3 --batch-size 4 --learning-rate 2e-5 --max-length 128 --seed 42 --split-seed 42 --device auto
```

For four-class BERT, use Restaurant4/train.xml and test.xml, a new output directory,
and --include-conflict. BERT selects checkpoints by validation Macro-F1.
Historical DeBERTa pair experiments used validation accuracy. Explain that distinction
when comparing the recorded systems.

## Joint extraction and other benchmarks

```powershell
python -m absa_enhanced train --mode joint --dataset restaurant --output runs/joint_s42 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --stride 64 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto
```

Joint models predict BIO aspect spans and their sentiments. Gold-aspect classification,
extraction F1 and end-to-end F1 use different evaluation units. Report the protocol.

| Training option | Directory | Training / validation / test aspects |
|---|---|---|
| `--dataset restaurant` | Restaurant3 | 3602 / internal split / 1120 |
| `--dataset restaurant2015` | Restaurant2015 | 1188 / internal split / 525 |
| `--dataset restaurant2016` | Restaurant2016 | 1714 / internal split / 601 |
| `--dataset mams` | MAMS-ATSA | 11186 / 1332 / 1336 |

Other dataset choices are restaurant2015, restaurant2016 and mams.
MAMS-ATSA supports the current aspect-term model. MAMS-ACSA lacks aspect character
spans and serves a separate category task. Custom XML is supported by the CLI.

## Evaluation and predictions

```powershell
python -m absa_enhanced evaluate --checkpoint runs/deberta_pair_split42_seed42/best --test Database/Restaurant3/test.xml --output runs/pair_evaluation --device auto
python -m absa_enhanced predict --checkpoint runs/deberta_pair_split42_seed42/best --text "The food was excellent, but the service was slow." --aspect service --device auto
```

Evaluation and prediction obtain labels from the checkpoint. A pair model needs a
provided aspect, or --extractor-checkpoint for a separate joint extractor.

A run saves best/, best_macro_f1/, run_config.json, history.json,
checkpoint_index.json, training_summary.json and test_metrics.json.
test_gold_predictions.jsonl contains per-aspect predictions and probabilities.
Pair runs also save test_provided_aspects.jsonl. Joint runs save test_end_to_end.jsonl.
Preserve run configurations, histories and predictions when reporting experiments.
runs/, caches and virtual environments are ignored by Git.

| Output | Contents |
|---|---|
| best/, best_macro_f1/ | Weights, encoder config, tokenizer, architecture and window settings |
| run_config.json | Versions, arguments, split seed, class weights and alignment statistics |
| history.json | Training and validation results per epoch |
| checkpoint_index.json | Best epoch and saving rules |
| training_summary.json | Runtime, best validation score and peak allocated GPU memory |
| test_metrics.json | Gold-aspect accuracy/Macro-F1, plus extraction and joint F1 in joint mode |
| test_gold_predictions.jsonl | Per-aspect classification predictions and probabilities |
| test_provided_aspects.jsonl | Per-sentence pair predictions for provided aspects |
| test_end_to_end.jsonl | Joint automatic extraction outputs |

## Grouped cross-validation

```powershell
python -m absa_enhanced cross-validate --mode pair --dataset restaurant --output runs/cv_pair_s42 --folds 3 --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_accuracy --device auto --plan-only
```

Replace --plan-only with --resume to run the plan. Add --labels four for Restaurant4.
Exact-text groups stay together. The official test is excluded from the source pool.
Each fold uses an inner validation split for selection and an outer held-out split for
evaluation. This is outer CV with inner holdout selection.
A fold's run/test_metrics.json scores its outer held-out set.

cv_config.json, fold_assignment.json and per-fold manifests record the plan and hashes.
cv_summary.json reports fold means and sample SDs, plus pooled OOF metrics from
oof_predictions.jsonl. Pooled F1 differs from the mean fold F1.

Resume skips only complete folds with matching source, implementation and configuration.
An incomplete run requires attention. Resume does not restore a partial optimizer state.

## Neutral decisions and diagnostics

The neutral offset multiplies probabilities by [1, k, 1], or [1, k, 1, 1] for four
classes, and renormalizes. Choose k on validation Macro-F1 and apply it once to held-out
data. At a fixed positive k, neutral probability ranking and neutral AP are preserved,
while the competing class decision can change.

```powershell
python scripts/tune_neutral_offset.py --runs runs/a runs/b --metric macro_f1
python scripts/cv_offset.py --cv-dir runs/cv_control_3fold --cv-dir runs/cv_nocls_3fold
python scripts/ensemble_runs.py --runs runs/a runs/b runs/c --output runs/ensemble --bootstrap 2000
```

The offset tools infer three/four-class axes from prediction files and reject mismatches.
The historical offset bootstrap resamples aspect rows. The three-class analyzer in
neutral_diagnostics supports sentence-text cluster intervals.

The [diagnostic workflow](docs/neutral_scope_study.md) explains neutral subgroups, PR
curves, confidence and human review of 40 synthetic pairs. Review the semantic labels
before interpreting those pairs as gold data.

```powershell
python scripts/build_pair_review_sheet.py
python -m neutral_diagnostics validate-pairs
python scripts/apply_pair_review.py review_A.csv review_B.csv
```

Add --write to apply agreed reviews. Keep each reviewer's independent CSV.

## Historical three-class tables

The tables below preserve the numerical entries formerly shown in this README.
They describe the group's recorded Restaurant3 experiments. Consult
[Recorded results](docs/neutral_results.md) for seeds, protocol differences, test-set
reuse and limits of the statistical interpretation. Legacy ordinal results have an
objective–inference mismatch. The two tied offset changes need explicit treatment.
No four-class training result is represented here.

| Configuration | test Acc | test Macro-F1 | neutral P / R / F1 | neutral F1 95% CI |
|---|---|---|---|---|
| `softmax` (control) | 0.8786 | 0.8076 | 0.782 / 0.531 / 0.632 | [0.552, 0.704] |
| `--no-cls-feature` | 0.8786 | 0.8053 | 0.790 / 0.500 / 0.613 | [0.529, 0.685] |
| `gate` | 0.8723 | 0.8018 | 0.709 / 0.536 / 0.610 | [0.527, 0.683] |
| `ordinal` | 0.8688 | 0.7923 | 0.697 / 0.515 / 0.592 | [0.508, 0.663] |
| `clause` context | 0.8429 | 0.7438 | 0.727 / 0.408 / 0.523 | [0.442, 0.599] |
| Three-head probability ensemble | 0.8839 | 0.8166 | 0.778 / 0.536 / 0.634 | — |
| `softmax` + neutral offset 2.5 | 0.8804 | 0.8114 | 0.748 / 0.561 / 0.641 | — |
| `--no-cls-feature` + neutral offset 2.0 | 0.8848 | 0.8192 | 0.779 / 0.551 / 0.644 | — |

| Configuration | Metric | Before offset | After validation-selected offset |
|---|---|---|---|
| `softmax` (control) | test Macro-F1 | 0.7952 ± 0.0164 | 0.8048 ± 0.0100 |
| `softmax` (control) | test Neutral F1 | 0.5944 ± 0.0438 | 0.6206 ± 0.0264 |
| `--no-cls-feature` | test Macro-F1 | 0.7982 ± 0.0194 | 0.8053 ± 0.0187 |
| `--no-cls-feature` | test Neutral F1 | 0.6034 ± 0.0401 | 0.6206 ± 0.0367 |

| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± SD | Paired t | Sign test |
|---|---|---|---|---|---|---|---|---|
| Macro-F1, k=1 | −0.0023 | +0.0118 | +0.0046 | +0.0059 | −0.0049 | +0.0030 ± 0.0067 | +1.00 | 3/5, p=1.00 |
| Neutral F1, k=1 | −0.0197 | +0.0284 | −0.0040 | +0.0336 | +0.0070 | +0.0091 ± 0.0222 | +0.91 | 3/5, p=1.00 |
| Neutral AP (ranking) | +0.0199 | −0.0060 | +0.0026 | +0.0254 | +0.0014 | +0.0087 ± 0.0133 | +1.46 | 4/5, p=0.38 |
| Macro-F1, after each run uses its selected offset | +0.0078 | +0.0112 | +0.0124 | −0.0037 | −0.0252 | +0.0005 ± 0.0157 | +0.07 | 3/5, p=1.00 |
| Neutral F1, after each run uses its selected offset | +0.0030 | +0.0238 | +0.0184 | +0.0102 | −0.0553 | +0.0000 ± 0.0319 | +0.00 | 3/5, p=1.00 |

| Scope | Macro-F1 | Neutral F1 | Nonnegative changes |
|---|---|---|---|
| control (n=5) | +0.0096 ± 0.0110 | +0.0263 ± 0.0284 | 5/5 |
| `--no-cls-feature` (n=5) | +0.0071 ± 0.0068 | +0.0172 ± 0.0159 | 5/5 |
| Pooled (n=10) | **+0.0084 ± 0.0087** | **+0.0217 ± 0.0222** | **10/10** |

## Experiment and maintenance scripts

PowerShell launchers train the baseline replications, head ablations and three-fold
control/no-CLS experiments. Analysis scripts read the saved run artifacts.

- replicate_bert_seeds.ps1 and replicate_neutral_seeds.ps1 vary training seeds.
- run_head_ablation.ps1 and run_cv_3fold.ps1 reproduce the recorded configurations.
- compare_baselines.py, compare_seed_spread.py and ranking_quality.py summarize runs.
- aggregate_offset_seeds.py, offset_effect.py, fixed_offset_check.py and
  offset_validation_size.py analyze decision rules.
- prepare_restaurant3.py and prepare_restaurant4.py export documented derivatives.
- clean_non_threeclass_data.py previews known retired dataset paths. --apply removes
  only those paths after replacement-data validation.

## Troubleshooting

Missing datasets can be restored from the bundled full archive with --offline.
For failed downloads, inspect network access, local paths and checksum errors.
For GPU memory failures, reduce batch size and increase gradient accumulation.
Preserve logs and use a new run directory for changed settings.
Use --precision no when investigating nonfinite mixed-precision loss.
Run commands from the repository root with the pinned dependencies installed.

## References

- [BERT sentence-pair ABSA](https://aclanthology.org/N19-1035/)
- [DeBERTaV3 model](https://huggingface.co/microsoft/deberta-v3-base)
- [SemEval-2014](https://alt.qcri.org/semeval2014/task4/)
- [SemEval-2015](https://alt.qcri.org/semeval2015/task12/)
- [SemEval-2016](https://alt.qcri.org/semeval2016/task5/)
- [MAMS dataset](https://github.com/siat-nlp/MAMS-for-ABSA)

Cite each dataset and retain its source manifests and original usage conditions.
