# Neutral-target diagnostics and CLS ablation

This workflow studies whether opinions about neighboring aspects affect a neutral target.
It evaluates three-class sentiment given gold aspects. The analyzer and controlled-pair
workflow support this label set. Four-class support is documented in the root README.

[Recorded results](neutral_results.md) contains the group's historical experiments.
[Validation](neutral_scope_validation.md) records the earlier offline software checks.
The commands below require the actual saved model or complete predictions.

## Analyze complete saved predictions

From the repository root, activate the project environment and supply the real run directory.
Both enhanced *_gold_predictions.jsonl and baseline predictions.jsonl are supported.
Error-only subsets cannot produce valid whole-test metrics or neighbor statistics.

```powershell
python -m neutral_diagnostics analyze --predictions runs/your_run/test_gold_predictions.jsonl --gold-xml Database/Restaurant3/test.xml --dataset restaurant --split test --run-id pair_seed42 --model-label pair_control --training-seed 42 --output runs/analysis_pair_seed42_test
```

| Output | Contents |
|---|---|
| report.md | Accuracy, Macro-F1, per-class P/R/F1/support and neutral subgroups |
| analysis.json | Full metrics, confusion matrix, source SHA-256, target fingerprint, confidence and intervals |
| neutral_pr_probability.csv | Complete neutral-versus-rest PR points and AP ranked by p(neutral) |
| neutral_pr_logit_margin.csv | PR/AP ranked by log p(neutral) minus the largest competing log probability |
| neutral_pr.svg | Both PR curves and neutral prevalence |
| neutral_errors_for_review.csv | Neutral errors, nearest polar aspect, agreement and reviewer fields |

Logit margin corresponds to competition after adding a neutral logit bias.
The analyzer clips probabilities at 1e-30 before logarithms. Saturated probabilities cannot
fully recover the ranking of original logits. The analyzer does not choose test thresholds.

Neutral subgroups distinguish sentences containing another gold polar aspect from sentences
whose annotated aspects are all neutral. The latter can include one-aspect sentences and
unannotated sentiment. Report neighbor agreement over all neutral errors and over eligible
mixed-sentence errors. Matching any neighbor is less informative when both polarities occur.
Human review should distinguish scope interference, negation, implicit opinion, ambiguity,
annotation questions and other errors using reviewed_reason and reviewer.

The default is 1000 percentile bootstrap resamples grouped by identical sentence text.
Intervals describe sampling uncertainty conditional on the fitted model. Training randomness,
selection uncertainty and dependence between different sentences of a review remain.

For validation, use the same run ID with --split validation and the original validation XML.
Do not create a new random partition to reconstruct a missing training-time validation set.
For OOF, use --split oof and avoid supplying the unfiltered original training XML.

```powershell
python -m neutral_diagnostics compare --analyses runs/analysis_pair_seed42_validation/analysis.json runs/analysis_pair_seed42_test/analysis.json --output runs/analysis_comparison
```

Compare reports individual metrics, grouped means and sample SD, and validation-minus-test
gaps for matching run IDs. Target coverage must match within a dataset/split. Repeated runs
require distinct actual training seeds. One run cannot supply a sample SD. Multiple validation
partitions need distinct dataset identifiers.

## Review the 40 synthetic pairs

data/diagnostics/neutral_scope_pairs.csv contains 30 neutral, 5 positive and 5 negative
target pairs, totaling 80 sentences. A pair changes the neighboring opinion while keeping
the target description fixed. Target position alternates across pairs. These are synthetic
diagnostic examples and do not represent the full distribution of natural reviews.

```powershell
python scripts/build_pair_review_sheet.py
python -m neutral_diagnostics validate-pairs
```

Two reviewers independently check the target label, distractor, naturalness and intended
semantic invariance using the HTML review sheet. Enter actual reviewer names and export each
review separately. Review without model outputs. The mechanical validator cannot establish
semantic correctness.

```powershell
python scripts/apply_pair_review.py review_A.csv review_B.csv
python scripts/apply_pair_review.py review_A.csv review_B.csv --write
```

The first command previews the merge. The second writes approved/rejected decisions only
where two distinct reviewers agree and cover all 40 pairs. Disagreements remain pending and
appear in data/diagnostics/pair_review_audit.json. Keep both independent CSV files.
The browser stores progress separately for each reviewer.

```powershell
python -m neutral_diagnostics predict-pairs --checkpoint runs/your_run/best --output runs/scope_pair_control --device auto
```

Baseline, pair and joint checkpoints classify the provided aspects. Add --include-unreviewed
only for exploratory checks. Results record human_review_complete: false for unreviewed
pairs. Rejected rows are always excluded. Outputs are pair_metrics.json, pair_results.csv
and predictions.jsonl.

Report both_correct, invariant, neutral-probability change and follows_distractor_polarity.
Include polar-target controls. An always-neutral model can achieve high invariance.
Controlled edits alone cannot prove a model's internal causal mechanism. If pairs guide model
choice, final confirmation needs fresh examples.

## Remove the final CLS feature

--no-cls-feature is supported by pair, joint, train and cross-validate. Default models retain
CLS. Legacy checkpoints without use_cls load with True. New settings.json stores the choice.
Pair classifiers retain target and query vectors. Joint classifiers retain target and
aspect-guided context vectors. Encoder CLS tokens and attention remain active. Contextual
aspect vectors can carry global information. The classifier parameter count also changes.

```powershell
python -m absa_enhanced cross-validate --mode pair --dataset restaurant --output runs/scope_cv_control_seed42 --folds 3 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_macro_f1 --plan-only
python -m absa_enhanced cross-validate --mode pair --dataset restaurant --output runs/scope_cv_no_cls_seed42 --folds 3 --seed 42 --split-seed 42 --class-weighting none --selection-metric gold_macro_f1 --no-cls-feature --plan-only
```

Check that fold_assignment.json is identical. Replace --plan-only with --resume for training
with the same parameters. This entails six actual model trainings and requires pretrained
weights and compute. Inner validation selects epochs. Outer held-out data evaluate the
selected model. OOF predictions can be analyzed with --split oof.
Use separate directories for additional training seeds with fixed split seed 42.

Fixed-split seeds and multiple folds address different uncertainty sources. Disclose repeated
inspection of the official test. Restaurant2015/2016 neutral test support is only 27/28,
compared with 196 for Restaurant3, and their conversion rules differ.

## Reporting

Report actual completed work, complete seed sets, per-class support, paired comparisons and
protocol differences. A small threshold gain describes the tested rule and score family.
High confidence and neighbor agreement cannot prove a sentence-level shortcut.
Use the real run artifacts for result tables. Keep unreviewed pairs explicitly exploratory.

## References

- [SemEval-2014](https://aclanthology.org/S14-2004/)
- [Aspect scope](https://aclanthology.org/2023.findings-emnlp.727/)
- [Probability calibration](https://proceedings.mlr.press/v70/guo17a.html)
- [Model selection bias](https://jmlr.org/papers/v11/cawley10a.html)
- [Precision, recall and AP](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
