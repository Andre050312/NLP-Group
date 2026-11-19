# Recorded neutral-sentiment experiments

These are the group's historical three-class Restaurant3 results, summarized in English.
Numeric result tables retain their recorded values. Full independent reproduction requires
the original per-run predictions and checkpoints. The report draft discusses evidence limitations.

The control and no-CLS models each use seeds 42–46 with split seed 42.
Gate, legacy ordinal, clause cropping and joint observations have one seed.
Pair settings were 6 epochs, batch size 2, accumulation 4, length 192,
no class weighting and validation gold accuracy for checkpoint selection.
The group reported an RTX 5060 Laptop GPU with 8 GB memory, BF16 and about 18 minutes per run.

Interpretation of the historical tables needs care. Two offset changes are ties, so
nonnegative changes across all runs cannot support a sign test treating every run as a strict
improvement. The configurations share seeds and test targets. Threshold maxima computed with
test labels are retrospective diagnostics for those scores. A multiclass argmax has no fixed
0.5 neutral threshold. Similar means do not establish equivalence. The legacy ordinal head has
an objective–inference mismatch. Controlled pairs remain pending human review. Historical
offset-script bootstrap intervals resample aspect rows, while the diagnostic analyzer supports
sentence-text clusters.

## 1 Single-seed reference results

All entries use seed 42. Neutral support is 196 and prevalence is 0.175. Single-seed differences do not establish a repeatable ranking.

| Configuration | Acc | Macro-F1 | Neutral P | Neutral R | Neutral F1 | Neutral F1 95% CI (sentence bootstrap) | Neutral AP |
|---|---|---|---|---|---|---|---|
| control (with CLS) | 0.8786 | 0.8076 | 0.782 | 0.531 | 0.632 | [0.552, 0.704] | 0.725 |
| no CLS (`--no-cls-feature`) | 0.8786 | 0.8053 | 0.790 | 0.500 | 0.613 | [0.529, 0.685] | **0.745** |
| gate head | 0.8723 | 0.8018 | 0.709 | 0.536 | 0.610 | [0.527, 0.683] | 0.712 |
| ordinal head | 0.8688 | 0.7923 | 0.697 | 0.515 | 0.592 | [0.508, 0.663] | 0.687 |
| clause context | 0.8429 | 0.7438 | 0.727 | 0.408 | 0.523 | [0.442, 0.599] | 0.648 |

## 1a BERT baseline on the same split

BERT uses split seed 42 and the same 1682/295/800 sentence partition as pair models. It selects checkpoints by validation Macro-F1, while pair models use accuracy. BERT seed 42 replicated accuracy 0.8446, Macro-F1 0.7657 and neutral F1 0.5948. Its per-seed neutral F1 was 0.595 / 0.641 / 0.493 / 0.602 / 0.480. The second table records descriptive Welch t statistics.

| Model | n | Acc | Macro-F1 | Neutral P | Neutral R | Neutral F1 |
|---|---|---|---|---|---|---|
| BERT-base (pair) | 5 | 0.8398 ± 0.0099 | 0.7535 ± 0.0292 | 0.737 ± 0.081 | 0.469 ± 0.112 | 0.562 ± 0.072 |
| DeBERTa joint | 1 | 0.8714 | 0.7915 | 0.793 | 0.469 | 0.590 |
| DeBERTa pair `control` | 5 | 0.8727 ± 0.0076 | 0.7952 ± 0.0164 | 0.771 ± 0.028 | 0.486 ± 0.055 | 0.594 ± 0.044 |
| DeBERTa pair `no CLS` | 5 | 0.8738 ± 0.0101 | 0.7982 ± 0.0194 | 0.746 ± 0.047 | 0.508 ± 0.047 | 0.603 ± 0.040 |

| Comparison | Acc | Macro-F1 | Neutral F1 |
|---|---|---|---|
| BERT − pair `control` | −0.0329 (t=−5.90) | −0.0417 (t=−2.79) | −0.0322 (t=−0.86) |
| BERT − pair `no CLS` | −0.0339 (t=−5.38) | −0.0447 (t=−2.85) | −0.0413 (t=−1.12) |

## 2 Threshold diagnostics for seed 42

These maxima use test labels for retrospective analysis. They must not select a deployment threshold. Five-seed averages appear in Section 3c.

| Configuration | Neutral F1 at argmax | Maximum neutral F1 over thresholds |
|---|---|---|
| control | 0.632 | 0.658 |
| no CLS | 0.613 | 0.689 |
| gate | 0.610 | 0.673 |
| ordinal | 0.592 | 0.656 |
| clause | 0.523 | 0.610 |

## 3 Validation-selected neutral offset

Each run chooses k on validation Macro-F1 and reports its test result. Mean ± SD uses sample standard deviation across five seeds. Shared seeds, shared test targets and ties prevent interpreting ten nonnegative changes as ten independent improvements.

| Configuration | seed | Selected k | Acc (k=1 → k) | Macro-F1 | Neutral F1 |
|---|---|---|---|---|---|
| control | 42 | 2.50 | 0.8786 → 0.8804 | 0.8076 → 0.8114 | 0.632 → 0.641 |
| control | 43 | 1.50 | 0.8830 → 0.8821 | 0.8159 → 0.8166 | 0.640 → 0.645 |
| control | 44 | 1.25 | 0.8670 → 0.8661 | 0.7909 → 0.7915 | 0.595 → 0.598 |
| control | 45 | 1.50 | 0.8670 → 0.8759 | 0.7757 → 0.7985 | 0.535 → 0.587 |
| control | 46 | 5.00 | 0.8679 → 0.8759 | 0.7859 → 0.8063 | 0.570 → 0.632 |
| no CLS | 42 | 2.00 | 0.8786 → 0.8848 | 0.8053 → 0.8192 | 0.613 → 0.644 |
| no CLS | 43 | 1.00 | 0.8884 → 0.8884 | 0.8278 → 0.8278 | 0.669 → 0.669 |
| no CLS | 44 | 2.00 | 0.8679 → 0.8705 | 0.7955 → 0.8039 | 0.591 → 0.617 |
| no CLS | 45 | 1.50 | 0.8714 → 0.8759 | 0.7815 → 0.7948 | 0.569 → 0.597 |
| no CLS | 46 | 1.00 | 0.8625 → 0.8625 | 0.7810 → 0.7810 | 0.577 → 0.577 |

| Configuration | Metric | Before offset | After offset |
|---|---|---|---|
| control | Macro-F1 | 0.7952 ± 0.0164 | 0.8048 ± 0.0100 |
| control | Accuracy | 0.8727 ± 0.0076 | 0.8761 ± 0.0062 |
| control | Neutral F1 | 0.5944 ± 0.0438 | 0.6206 ± 0.0264 |
| no CLS | Macro-F1 | 0.7982 ± 0.0194 | 0.8053 ± 0.0187 |
| no CLS | Accuracy | 0.8738 ± 0.0101 | 0.8764 ± 0.0105 |
| no CLS | Neutral F1 | 0.6034 ± 0.0401 | 0.6206 ± 0.0367 |

| Scope | Macro-F1 change | Neutral F1 change | Runs with nonnegative change |
|---|---|---|---|
| control (n=5) | +0.0096 ± 0.0110 | +0.0263 ± 0.0284 | 5/5 |
| no CLS (n=5) | +0.0071 ± 0.0068 | +0.0172 ± 0.0159 | 5/5 |
| Pooled (n=10) | **+0.0084 ± 0.0087** | **+0.0217 ± 0.0222** | **10/10** |

## 3a Paired control and no-CLS comparison

Differences pair the same training seed on the same partition. Small differences and mixed signs limit evidence for a repeatable benefit.

| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± SD | Paired t | Sign test |
|---|---|---|---|---|---|---|---|---|
| Macro-F1, k=1 | −0.0023 | +0.0118 | +0.0046 | +0.0059 | −0.0049 | +0.0030 ± 0.0067 | +1.00 | 3/5, p=1.00 |
| Neutral F1, k=1 | −0.0197 | +0.0284 | −0.0040 | +0.0336 | +0.0070 | +0.0091 ± 0.0222 | +0.91 | 3/5, p=1.00 |
| Accuracy, k=1 | ±0.0000 | +0.0054 | +0.0009 | +0.0045 | −0.0054 | +0.0011 ± 0.0043 | +0.56 | 3/5, p=1.00 |
| Neutral AP (ranking) | +0.0199 | −0.0060 | +0.0026 | +0.0254 | +0.0014 | +0.0087 ± 0.0133 | +1.46 | 4/5, p=0.38 |
| Macro-F1, after each run uses its selected offset | +0.0078 | +0.0112 | +0.0124 | −0.0037 | −0.0252 | **+0.0005 ± 0.0157** | **+0.07** | 3/5, p=1.00 |
| Accuracy, after each run uses its selected offset | +0.0045 | +0.0062 | +0.0045 | ±0.0000 | −0.0134 | +0.0004 ± 0.0080 | +0.10 | 3/5, p=1.00 |
| Neutral F1, after each run uses its selected offset | +0.0030 | +0.0238 | +0.0184 | +0.0102 | −0.0553 | +0.0000 ± 0.0319 | +0.00 | 3/5, p=1.00 |

## 3a-prime Sensitivity to the number of seeds

The first three seeds gave a stronger apparent improvement than the complete five-seed set. Report the complete seed set.

| | n=3 (seed 42/43/44) | n=5 (all) |
|---|---|---|
| Macro-F1 difference | +0.0105 ± 0.0024 | +0.0005 ± 0.0157 |
| Paired t | +7.65 | +0.07 |

## 3b Training-seed variation

Training-seed variation provides context for single-seed head and context experiments. Repeated runs are needed to rank those alternatives.

| Configuration (k=1, 5 seeds) | Macro-F1 by seed | Mean ± SD | Range |
|---|---|---|---|
| control | 0.8076 / 0.8159 / 0.7909 / 0.7757 / 0.7859 | 0.7952 ± 0.0164 | 0.0402 |
| no CLS | 0.8053 / 0.8278 / 0.7955 / 0.7815 / 0.7810 | 0.7982 ± 0.0194 | 0.0468 |

## 3c Ranking quality

Neutral AP exceeds prevalence. The AP difference was +0.0087 ± 0.0133, t=1.46. The threshold-maximum difference was +0.0081 ± 0.0289. These observations do not establish equivalence or a ceiling for the dataset or future models.

| Configuration (5-seed mean) | Neutral F1 at argmax | Neutral AP | Maximum neutral F1 over thresholds |
|---|---|---|---|
| control | 0.5944 | 0.7112 | 0.6585 |
| no CLS | 0.6034 | 0.7198 | 0.6665 |

## 3d Interpretation

The next table summarizes recorded comparisons. Inconclusive comparisons provide limited evidence for improvement.

| Question | Interpretation |
|---|---|
| Does removing CLS improve neutral F1? | Limited evidence for benefit: +0.0091 (t=0.91) before offset, +0.0000 (t=0.00) after offset across 5 seeds. Across 3 folds, no CLS is lower in 3/3, with Macro-F1 −0.0059 and neutral F1 −0.0108 (Section 9). |
| Does removing CLS improve ranking? | AP difference +0.0087 (t=1.46), with limited evidence for improvement. |
| Does the neutral offset help? | Protocol-dependent. On the official split 10/10 changes are nonnegative and fixed k=2 improves 9/10. Gains do not consistently transfer to 3-fold CV (Section 9). |
| Report interpretation | With 5 seeds, effects are small and inconsistent. All 10 fixed-split runs have nonnegative changes with selected offsets. At fixed k=2, 9/10 improve with mean Macro-F1 +0.012. The 3-fold evaluation limits claims of a general improvement. |
| Unsupported generalizations | Reliable gains from removing CLS, equivalent rankings, a neutral F1 dataset ceiling of 0.66, universal conclusions from 3 seeds, or portable offset gains. |

## 4 Neighboring aspects and sentiment scope

These observations use seed 42. Their stability across training seeds has not been checked.

## 4.1 Observed subgroups on test, seed 42

Neighbor agreement was 41/92 (0.446) for control and 44/98 (0.449) for no CLS. Among eligible mixed-sentence errors, agreement was 0.911 / 0.898. Agreement is association. Matching any neighbor is less informative if both positive and negative neighbors exist.

| Neutral subgroup | support | control recall | no CLS recall |
|---|---|---|---|
| All | 196 | 0.531 | 0.500 |
| Sentence has a polar aspect | 83 | 0.458 | 0.410 |
| All annotated aspects in sentence are neutral | 113 | 0.584 | 0.566 |

## 4.2 Controlled edits on 40 pending pairs

Edits swap a neighboring opinion while retaining target wording. All 40 pairs await two-person review. The neutral subset contains 30 pairs. Control and no CLS get both sentences correct for 6 and 13 pairs. Clause cropping removes the distractor and can produce identical inputs. Probability-change reduction from 0.354 to 0.257 is approximately 27%. These pilot results cannot establish an internal causal mechanism.

| Model | Neutral-target invariance | Neutral-target both correct | Follows distractor polarity | Mean neutral probability change |
|---|---|---|---|---|
| control | 0.200 | 0.200 | 0.367 | 0.354 |
| no CLS | 0.433 | 0.433 | 0.333 | 0.257 |
| clause | **1.000** | 0.867 | 0.000 | **0.000** |
| Polar controls (control) | 1.000 | 1.000 | 0.000 | 0.002 |

## 5 Validation and test

These are descriptive gaps for seed 42. General claims about systematic validation bias require more splits.

| Configuration | Macro-F1 validation minus test (seed 42) | Neutral F1 validation minus test (seed 42) | Validation neutral support | Test neutral support |
|---|---|---|---|---|
| control | −0.0121 | +0.0530 | 113 | 196 |
| no CLS | −0.0154 | +0.0476 | 113 | 196 |

## 6 Reproduction commands

Run from the repository root. Paths follow the group's original run names. Preserve the actual training partition and validation predictions.

```powershell
$env:HF_HOME = "D:\NLP-cache\huggingface"

python -m absa_enhanced train --mode pair --dataset restaurant --output runs/pair_heads_softmax_s42 `
  --epochs 6 --batch-size 2 --grad-accum 4 --max-length 192 --seed 42 --split-seed 42 `
  --class-weighting none --selection-metric gold_accuracy --device auto

powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1 -Config control -Seeds 43,44,45,46
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/replicate_neutral_seeds.ps1 -Config nocls -Seeds 43,44,45,46

python scripts/compare_seed_spread.py
python scripts/ranking_quality.py
python scripts/aggregate_offset_seeds.py
python scripts/offset_effect.py

python -m neutral_diagnostics analyze --predictions runs/pair_nocls_s42/test_gold_predictions.jsonl `
  --gold-xml Database/Restaurant3/test.xml --dataset restaurant3 --split test `
  --run-id pair_nocls_seed42 --model-label pair_no_cls --training-seed 42 --output runs/analysis_no_cls

python scripts/tune_neutral_offset.py --runs runs/pair_nocls_s42 runs/pair_nocls_s43 runs/pair_nocls_s44 runs/pair_nocls_s45 runs/pair_nocls_s46 `
  --metric macro_f1 --bootstrap 2000 --output runs/offset_nocls.json
python scripts/tune_neutral_offset.py --runs runs/pair_heads_softmax_s42 runs/pair_ctrl_s43 runs/pair_ctrl_s44 runs/pair_ctrl_s45 runs/pair_ctrl_s46 `
  --metric macro_f1 --bootstrap 2000 --output runs/offset_control.json

python -m neutral_diagnostics predict-pairs --checkpoint runs/pair_nocls_s42/best `
  --output runs/scope_pair_nocls --include-unreviewed --device auto
```

## 7 Follow-up work

Five-seed replications and three-fold comparisons are recorded here. Follow-up work includes independent human review of the 40 pairs, complete run archiving and target-attribution evaluation on fresh held-out examples. No fixed seed count guarantees statistical power.

## 8 Repeated use of the official test

The same 1120 test targets were inspected repeatedly during head comparisons, seed expansion and retrospective PR analysis. These comparisons are exploratory. Outer-fold evaluation excludes official-test data, but the folds share training observations.

## 9 Grouped three-fold evaluation with inner validation

The pool has 3040 sentences and 3601 aspects after removing official-test text overlap. Exact-text groups stay within a fold. Each outer training partition has an inner validation holdout with about 361–366 aspects and 62 neutral targets. Settings match the pair experiments. This is outer CV with inner holdout selection.

## 9.1 Control and no CLS

Pooled OOF covers all 3601 aspects once. Control accuracy / Macro-F1 / neutral F1 is 0.8398 / 0.7652 / 0.5740. No CLS gives 0.8367 / 0.7596 / 0.5639. Training-pool sizes and evaluation composition differ from the official-test protocol.

| Configuration | Fold 1 Macro | Fold 2 | Fold 3 | Mean ± SD | Fold 1 Neutral F1 | Fold 2 | Fold 3 | Mean ± SD |
|---|---|---|---|---|---|---|---|---|
| control | 0.7603 | 0.7671 | 0.7675 | 0.7650 ± 0.0041 | 0.5481 | 0.5908 | 0.5808 | 0.5732 ± 0.0223 |
| no CLS | 0.7453 | 0.7654 | 0.7664 | 0.7591 ± 0.0119 | 0.5251 | 0.5901 | 0.5722 | 0.5624 ± 0.0336 |
| Paired difference (no CLS minus control) | −0.0150 | −0.0017 | −0.0010 | **−0.0059** | −0.0230 | −0.0007 | −0.0086 | **−0.0108** |

## 9.2 Validation-selected offsets on outer held-out predictions

Each k comes from that fold's inner validation set. Historical offset intervals use an aspect-row bootstrap. Fitting k directly on all OOF gives retrospective optima: control k=5.0, Macro-F1 +0.0030 and neutral F1 +0.0244; no CLS k=2.5, +0.0074 and +0.0242. The second table records fixed multipliers.

| Configuration | Fold 1 selected k | Fold 2 | Fold 3 | pooled Macro-F1: argmax → inner-validation offset | pooled Neutral F1: argmax → inner-validation offset |
|---|---|---|---|---|---|
| control | 1.25 | 2.00 | 2.00 | 0.7652 → 0.7613 (−0.0039, 95% CI [−0.0102, +0.0023]) | 0.5740 → 0.5739 (−0.0001, CI [−0.0120, +0.0119]) |
| no CLS | 5.00 | 1.00 | 1.00 | 0.7596 → 0.7628 (+0.0032, CI [−0.0041, +0.0107]) | 0.5639 → 0.5767 (+0.0129, CI [−0.0025, +0.0297]) |

| k | Official-test Macro-F1 change across 10 runs | Official-test neutral F1 change | CV control OOF Macro-F1 | CV no CLS OOF Macro-F1 |
|---|---|---|---|---|
| 1.25 | +0.0051 (7/10) | +0.0127 | −0.0004 | +0.0024 |
| 1.5 | +0.0067 (9/10) | +0.0181 | −0.0005 | +0.0043 |
| 2.0 | +0.0121 (9/10) | +0.0311 | −0.0047 | +0.0050 |
| 3.0 | +0.0119 (8/10) | +0.0351 | −0.0031 | +0.0054 |
| 5.0 | +0.0121 (8/10) | +0.0427 | +0.0030 | +0.0048 |

## 9.3 Operating points across protocols

Reducing validation from 555 to 366 aspects retained a positive mean gain in 8 of 10 runs, including control seed 46 +0.0169 and no CLS seed 45 +0.0209. This does not isolate a single cause. Precision/recall trade-offs and training partitions both matter.

| Evaluation | Neutral P | Neutral R | Neutral F1 |
|---|---|---|---|
| Official test (control, 5-seed mean) | 0.771 | 0.486 | 0.594 |
| CV OOF (control) | 0.669 | 0.502 | 0.574 |

## 9.4 Reproduction

Three folds limit precision and share training observations. Preserve the protocol and report fold and pooled metrics separately.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_cv_3fold.ps1 -Config both
python scripts/cv_offset.py --cv-dir runs/cv_control_3fold --cv-dir runs/cv_nocls_3fold --output runs/cv_offset_report.json
python scripts/offset_validation_size.py --runs runs/pair_heads_softmax_s42 runs/pair_ctrl_s43 --sizes 366,555 --draws 200
python scripts/fixed_offset_check.py --runs runs/pair_ctrl_s43 --cv-dir runs/cv_control_3fold
```
