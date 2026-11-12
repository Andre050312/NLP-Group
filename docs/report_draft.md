# Why neutral sentiment stalls at 0.60: a diagnostic study of aspect-based sentiment classification

> Draft for COMP4136 group project report. Every number below is taken from
> `docs/neutral_results.md` and the run directories it cites. Sections marked
> **TODO** need either a human review pass or figures we have not produced yet.

## Abstract

Three-class aspect-based sentiment analysis (ABSA) on SemEval-2014 Restaurant reports
0.87 accuracy while the neutral class sits near F1 0.59. We ask whether that gap comes
from the classifier architecture, from the extent of context the model reads, or from a
decision boundary placed in the wrong place, and we test each hypothesis with controlled
interventions on a fixed data partition. Three findings organise the results. First,
changing the classifier's inductive bias does not help: the ordinal and gate heads sit
within seed noise (single seed each, preliminary), restricting context to the aspect's
clause is clearly worse, and the `[CLS]` ablation — replicated at five seeds and in
three-fold cross-validation — differs from the control by +0.0030 macro-F1
(paired *t* = 1.00) and is worse on all three folds.
Second, the neutral class carries usable ranking signal (average precision 0.71–0.72
against a 0.175 prevalence baseline), but converting that signal into F1 through a
threshold is bounded near 0.66, close to where the argmax already sits.
Third, and most useful methodologically, a validation-tuned neutral logit offset improves
the official test split in 10 of 10 runs (+0.008 macro-F1, +0.022 neutral F1, sign test
*p* ≈ 0.002) yet does not reproduce under nested cross-validation (control −0.004,
95% CI [−0.010, +0.002]). The mechanism is an operating-point difference, not
overfitting to the tuning set: on held-out data from the training pool the model already
over-predicts neutral (precision 0.67) where the test split under-predicts it (0.77).
We report the gain as split-specific and argue that single-split calibration results in
this literature should not be read as portable improvements.

## 1. Introduction

Aspect-based sentiment analysis asks for the sentiment expressed towards a specific
aspect term inside a sentence ("the *pasta* was bland but the *service* was lovely").
Most work reports binary polarity, or three classes in which neutral is the residual
category: SemEval-2014 defines it as *no opinion expressed about the aspect*, which is
not the same as an opinion of zero intensity.

That definitional mismatch produces a signature we observe consistently on Restaurant:
overall accuracy around 0.87, but neutral recall near 0.47–0.51 and neutral F1 near 0.59,
always with high precision and low recall. The model agrees with itself about neutral
when it commits, and rarely commits.

This report is a diagnostic study rather than a leaderboard entry. We fix one data
partition, vary one factor at a time, and ask which of three families of explanation
carries the effect:

* **Representation** — a head with the wrong inductive bias for an ordered three-class
  problem, or a classifier that reads a sentence-level summary instead of the aspect.
* **Scope** — the model reads opinions belonging to other aspects in the same sentence
  and attributes them to the target.
* **Decision rule** — the representations are fine and only the threshold is wrong.

Section 6 reports the outcome: the first two hypotheses yield nothing measurable, the
third yields a small effect that is specific to one evaluation split, and the surviving
explanation is attribution across aspects inside a sentence.

## 2. Related work

Aspect-conditioned sentence-pair classification (Sun et al., 2019) reframed ABSA as
`[CLS] context [SEP] aspect [SEP]` and remains a strong baseline; we use it as the base
architecture with DeBERTa-v3-base as the encoder. Ordinal formulations treat the three
labels as an ordered scale and predict cumulative logits (CORAL), and gate-style heads
decompose the problem into opinion presence followed by polarity; both are motivated by
the observation that neutral is qualitatively different from a midpoint. Calibration
work in classification typically rescales logits on held-out data; we test the simplest
version of that idea, a multiplicative bias on the neutral probability.

## 3. Data and task

We use the three-class conversion of SemEval-2014 Restaurant (`Database/Restaurant3`),
which drops the 286 conflicting polarity annotations and keeps 3041 training sentences
(3602 aspect terms: 805 negative, 633 neutral, 2164 positive) and the official test set
of 1120 aspect terms (196 negative, 196 neutral, 728 positive, 17.5% neutral).

One partition is used throughout: 1682 training sentences, 295 validation sentences and
800 test sentences, produced by `--split-seed 42`. Training seeds vary; the partition does
not. Sentences whose text appears in the official test file are removed from training and
validation, so no test text is ever seen during training. We report gold-aspect
classification only; the pair model has no extraction head, so extraction and
end-to-end numbers are not applicable to it.

## 4. Method

### 4.1 Base classifier

`[CLS] context [SEP] aspect [SEP]` into DeBERTa-v3-base, then an MLP over the
concatenation of the `[CLS]` state, the mean state over the aspect mention inside the
context, and the mean state over the aspect in the query segment. Six epochs, batch 2 with
gradient accumulation 4, learning rate 1e-5, max length 192, no class weighting, model
selection by validation accuracy.

### 4.2 Sentiment heads

`softmax` is the baseline three-way classifier. `ordinal` predicts two cumulative logits
with binary cross-entropy so that negative < neutral < positive and neutral occupies a
band rather than a point. `gate` first decides whether an opinion is present and then its
polarity, mapping "no opinion" to neutral.

### 4.3 Context scope

`sentence` keeps the whole review sentence. `clause` restricts the context to the clause
containing the aspect, using punctuation and contrastive connectives as boundaries, so a
neighbouring opinion cannot be read at all.

### 4.4 `[CLS]` ablation

`--no-cls-feature` drops the `[CLS]` vector from the classifier input, leaving
`(aspect_context, aspect_query)`. The encoder and its attention are unchanged: the aspect
token still encodes sentence-wide information, so this is an ablation of the classifier's
readout, not of the model's access to context.

### 4.5 Neutral logit offset

For a temperature-free version of prior correction, we multiply the neutral probability by
a factor *k* and renormalise, which is equivalent to adding log *k* to the neutral logit.
*k* is selected on a held-out split by macro-F1 over the grid
{1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6}. We evaluate the same rule in two protocols: tuned once
on the 295-sentence validation split and scored on the official test set, and tuned inside
each cross-validation fold on that fold's inner validation split before scoring its outer
held-out fold.

## 5. Experimental setup

All runs use the same partition, the same hyperparameters and the same selection rule.
`control` and `--no-cls-feature` were trained at seeds 42–46 (ten runs, 18 minutes each on
an RTX 5060 Laptop with BF16). `ordinal`, `gate` and `clause` have **one seed each (42)**
and are reported as preliminary observations, not as configurations that can be ranked.
A BERT-base sentence-pair baseline was trained at the same five seeds on the same partition
after adding a separate `--split-seed` so that its split no longer follows the training
seed.

Cross-validation uses three grouped folds over the training pool (3040 sentences, 3601
aspects), grouped by exact text so duplicate reviews never straddle folds. The official
test set is never used for training or for scoring inside cross-validation. Each fold
trains on two folds, selects its epoch on an inner validation split of 361–366 aspects
(about 62 neutral), and is scored once on the outer held-out fold.

**Protocol disclosure.** The official test set was scored many times during this project
(configuration comparisons, ten offset sweeps, threshold scans, two rounds of seed
expansion). Absolute test numbers are therefore optimistic, and the only near-unbiased
estimate in this report is the grouped cross-validation.

## 6. Results

### 6.1 Model ladder and seed variance

| Model | Seeds | Accuracy | Macro-F1 | Neutral P / R / F1 |
|---|---|---|---|---|
| BERT-base pair | 5 | 0.8398 ± 0.0099 | 0.7535 ± 0.0292 | 0.737 / 0.469 / 0.562 ± 0.072 |
| DeBERTa joint | 1 | 0.8714 | 0.7915 | 0.793 / 0.469 / 0.590 |
| DeBERTa pair, `control` | 5 | 0.8727 ± 0.0076 | 0.7952 ± 0.0164 | 0.771 / 0.486 / 0.594 ± 0.044 |
| DeBERTa pair, `no CLS` | 5 | 0.8738 ± 0.0101 | 0.7982 ± 0.0194 | 0.746 / 0.508 / 0.603 ± 0.040 |

Welch tests against the BERT baseline give *t* = −5.90 on accuracy and *t* = −2.79 on
macro-F1, but only *t* = −0.86 on neutral F1. The neutral comparison is inconclusive
because the baseline is unstable across seeds, not because the two are equal: its
per-seed neutral F1 spans 0.480 to 0.641, a range of 0.161, which is wider than the gap
between any two DeBERTa variants. We note that the baseline selects its epoch by
validation macro-F1 while the pair models select by validation accuracy, a difference that
favours the baseline slightly on macro-F1.

### 6.2 Head and scope ablations (single seed, preliminary)

| Configuration | Accuracy | Macro-F1 | Neutral P / R / F1 | Neutral AP |
|---|---|---|---|---|
| `control` (seed 42) | 0.8786 | 0.8076 | 0.782 / 0.531 / 0.632 | 0.725 |
| `--no-cls-feature` | 0.8786 | 0.8053 | 0.790 / 0.500 / 0.613 | 0.745 |
| `gate` head | 0.8723 | 0.8018 | 0.709 / 0.536 / 0.610 | 0.712 |
| `ordinal` head | 0.8688 | 0.7923 | 0.697 / 0.515 / 0.592 | 0.687 |
| `clause` context | 0.8429 | 0.7438 | 0.727 / 0.408 / 0.523 | 0.648 |

These are single-seed observations and all neutral-F1 confidence intervals overlap. The
one difference that clearly exceeds seed noise is `clause` (−0.064 macro-F1 against the
five-seed control mean), and its sign is informative: cutting the sentence down to the
target's clause makes neutral *worse*, so sentence context is not merely a source of
interference, it is also needed. The `gate` and `ordinal` differences (−0.006 and −0.015)
are within the seed spread of the control (SD 0.016) and we draw no conclusion from them.

### 6.3 The `[CLS]` ablation under both protocols

Five paired seeds give +0.0030 ± 0.0067 macro-F1 (*t* = 1.00) and +0.0091 ± 0.0222
neutral F1 (*t* = 0.91); ranking quality is also unchanged (neutral AP +0.0087,
*t* = 1.46). Three-fold cross-validation agrees and points the other way: `no CLS` is
worse on all three folds, by −0.0059 macro-F1 and −0.0108 neutral F1 on average.

| Fold | `control` macro | `no CLS` macro | `control` neutral F1 | `no CLS` neutral F1 |
|---|---|---|---|---|
| 1 | 0.7603 | 0.7453 | 0.5481 | 0.5251 |
| 2 | 0.7671 | 0.7654 | 0.5908 | 0.5901 |
| 3 | 0.7675 | 0.7664 | 0.5808 | 0.5722 |
| mean ± SD | 0.7650 ± 0.0041 | 0.7591 ± 0.0119 | 0.5732 ± 0.0223 | 0.5624 ± 0.0336 |

Pooled out-of-fold scores are 0.8398 / 0.7652 / 0.5740 for `control` and
0.8367 / 0.7596 / 0.5639 for `no CLS`. We do not compare these to test numbers: the
out-of-fold models train on two thirds of the data and are scored on a different
distribution, so the two quantities estimate different things.

The correct statement is therefore not "the two are equivalent" but "there is no evidence
that the ablation helps, and cross-validation shows a small effect in the opposite
direction".

### 6.4 Neutral offset: a split-specific gain

Tuned on the 295-sentence validation split, the offset is positive in every run.

| Scope | Δ Macro-F1 | Δ Neutral F1 | Runs not worse |
|---|---|---|---|
| `control` (n=5) | +0.0096 ± 0.0110 | +0.0263 ± 0.0284 | 5/5 |
| `no CLS` (n=5) | +0.0071 ± 0.0068 | +0.0172 ± 0.0159 | 5/5 |
| Pooled (n=10) | +0.0084 ± 0.0087 | +0.0217 ± 0.0222 | 10/10 |

With a fixed *k* = 2 chosen a priori, so that no per-run selection is involved, the gain
persists on the test split: +0.0121 ± 0.0111 macro-F1, positive in 9 of 10 runs, and
+0.0311 neutral F1.

Under nested cross-validation it does not reproduce.

| Configuration | argmax → nested offset, macro-F1 | argmax → nested offset, neutral F1 |
|---|---|---|
| `control` | 0.7652 → 0.7613 (−0.0039, 95% CI [−0.0102, +0.0023]) | 0.5740 → 0.5739 (−0.0001, CI [−0.0120, +0.0119]) |
| `no CLS` | 0.7596 → 0.7628 (+0.0032, CI [−0.0041, +0.0107]) | 0.5639 → 0.5767 (+0.0129, CI [−0.0025, +0.0297]) |

Even with *k* fitted on the pooled held-out rows — an oracle, not an achievable score —
the macro-F1 headroom is only +0.0030 for `control` (at *k* = 5) and +0.0074 for
`no CLS` (at *k* = 2.5).

Two explanations are testable and only one holds. Tuning-set size does not explain the
discrepancy: subsampling the 555-aspect validation split down to 366 aspects, the size of
a fold's inner validation set, preserves the gain in 8 of 10 runs (e.g. +0.0169 for
`control` seed 46, +0.0209 for `no CLS` seed 45). The operating point does: on held-out
data from the training pool the model predicts neutral more liberally than on the test
split.

| Evaluation | Neutral P | Neutral R | Neutral F1 |
|---|---|---|---|
| Official test (`control`, 5 seeds) | 0.771 | 0.486 | 0.594 |
| Cross-validation, pooled out-of-fold | 0.669 | 0.502 | 0.574 |

Where precision is already 0.67, multiplying the neutral probability by *k* trades
precision for recall at a loss. Where precision is 0.77, the same trade pays. The offset
is thus a compensation for how one particular split misplaces the boundary, not a
property of the model.

### 6.5 How much is left in the decision rule

Thresholding the existing scores yields an achievable neutral F1 of 0.6585 for `control`
and 0.6665 for `no CLS` (five-seed means), against 0.594 and 0.603 at argmax. Ranking
quality is far above chance: neutral average precision is 0.711 (`control`) and 0.720
(`no CLS`) against a 0.175 prevalence baseline. So the representation does separate
neutral from the polar classes, but the total amount recoverable by any threshold on these
scores is about +0.06 F1, and Section 6.4 shows that most of it cannot be located reliably
from a validation split.

### 6.6 Controlled input intervention

**TODO — requires the human review pass to be completed before this section can be used.**

We constructed 40 pairs of sentences in which the target aspect is neutral in both members
and the only change is the opinion word about a *different* aspect in the same sentence
("excellent" → "awful"), with 30 neutral targets and 5 positive / 5 negative controls.
Flipping that one word moves the target's neutral probability by 0.354 on average and
flips the predicted label in 80% of neutral pairs, while polar control aspects move by
0.002. The `no CLS` model halves the sensitivity (0.257, 43% invariant). The `clause`
model is invariant by construction, since the distractor clause is removed from its input
entirely, so its invariance is not evidence of a learned scope.

All 40 pairs are currently marked `pending`. They were drafted for this project and have
not yet been checked by two independent human reviewers, so in the final report they must
be labelled as a development diagnostic rather than as a human-annotated gold standard, or
removed if the review is not completed.

## 7. Discussion

The three hypotheses fail in an informative order. Representation changes do not move
neutral F1: ordinal and gate heads are within seed noise, and the `[CLS]` readout turns out
to be irrelevant (Section 6.3). Scope changes have the wrong sign: removing context makes
things worse, so the model is not simply reading a neighbouring opinion because it cannot
tell clauses apart (Section 6.2). The decision rule has a real but small amount of headroom
that cannot be reliably located (Section 6.5).

What remains is attribution: the sentiment attached to one aspect leaks into the judgment
of another. The observational evidence is that 44.6% of neutral errors (41/92) carry the
polarity of a human-annotated neighbour aspect, rising to 91.1% among errors whose sentence
contains a polar aspect at all, and neutral recall is lower in mixed sentences (0.458) than
in sentences where every aspect is neutral (0.584). The controlled pairs, once reviewed,
would supply the causal version of that statement. Both statistics are measured on the
seed-42 checkpoint only.

This reframes the problem. A threshold correction answers "how confident must the model be
before we call it neutral", but the failure mode is "the model has the wrong evidence for
this aspect". The former is a one-parameter fix with a ceiling of about 0.06 F1; the latter
needs an architectural change, for example an attention prior over the aspect's own clause
that is learned rather than imposed by truncation, or an explicit contrastive comparison
between the target aspect and the other aspects of the same sentence.

## 8. Limitations

* **Single partition for all main results.** Every test-number comparison uses one
  1682/295/800 split. Cross-validation mitigates this for `control` and `no CLS` only.
* **Three configurations have one seed.** `ordinal`, `gate` and `clause` cannot be ranked
  against each other or against the five-seed configurations.
* **Reused test set.** See the disclosure in Section 5. Absolute test scores are
  optimistic, and the offset result in particular does not survive a cleaner protocol.
* **Small cross-validation.** Three folds, two thirds of the data per fold, and about 62
  neutral aspects in each inner validation split. Fold-to-fold SD is 0.004–0.012 on
  macro-F1, so the design has limited power to detect effects below roughly 0.01.
* **Diagnostic pairs unreviewed.** Section 6.6 is exploratory until the human pass is
  done.
* **Pair mode only for the ablations.** Extraction and end-to-end scores are not reported
  for the pair model because it does not perform extraction.

## 9. What we would tell the next group

Three protocol lessons cost us real time and are worth stating explicitly.

1. **Three seeds are not enough.** At seeds 42–44 the `no CLS` configuration with an offset
   looked like a stable win: +0.0105 ± 0.0024 macro-F1, paired *t* = 7.65. Two more seeds
   turned that into +0.0005 ± 0.0157, *t* = 0.07. A very small SD at *n* = 3 is a symptom
   of sampling luck, not of a robust effect.
2. **A positive sign test on one split is not a method.** The offset was non-negative in
   10 of 10 runs on the test split and still failed to transfer to held-out data from the
   training pool. Report the protocol, not just the *p*-value.
3. **Single-seed baselines can point either way.** The BERT baseline's neutral F1 ranges
   over 0.161 across seeds; comparing it at one seed would have produced whatever
   conclusion the seed happened to support.

## 10. Conclusion and future work

On Restaurant, the neutral class is not limited by the classifier head, by the readout
feature, or by the decision threshold to any useful degree. The representation ranks
neutral well above chance (AP 0.71), the recoverable threshold gain is about 0.06 F1, and
a validation-tuned offset that appears to capture part of it on the official test split
fails to do so out of fold. The evidence points to cross-aspect attribution inside the
sentence as the mechanism, which is what the next iteration should attack: a learned
attention prior over the target's clause, and an explicit contrast between the target
aspect and its sentence-mates, evaluated under grouped cross-validation from the start.

## Appendix A. Reproduction

```powershell
$env:HF_HOME = "D:\NLP-cache\huggingface"

# five training seeds for the control and no-CLS configurations
powershell -NoProfile -File scripts/replicate_neutral_seeds.ps1 -Config control -Seeds 43,44,45,46
powershell -NoProfile -File scripts/replicate_neutral_seeds.ps1 -Config nocls -Seeds 43,44,45,46

# BERT baseline on the same partition
powershell -NoProfile -File scripts/replicate_bert_seeds.ps1 -Seeds 42,43,44,45,46

# grouped 3-fold cross-validation plus nested offset evaluation
powershell -NoProfile -File scripts/run_cv_3fold.ps1 -Config both
python scripts/cv_offset.py --cv-dir runs/cv_control_3fold --cv-dir runs/cv_nocls_3fold

# analysis that needs no GPU
python scripts/compare_seed_spread.py
python scripts/ranking_quality.py
python scripts/aggregate_offset_seeds.py
python scripts/offset_effect.py
python scripts/fixed_offset_check.py --runs runs/pair_ctrl_s43 --cv-dir runs/cv_control_3fold
python scripts/offset_validation_size.py --runs runs/pair_ctrl_s43 --sizes 366,555
python scripts/compare_baselines.py --group "bert=runs/bert_split42_seed42" --group "pair_control=runs/pair_ctrl_s43"
```

## References

* Sun, Huang, Qiu. *Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing
  Auxiliary Sentence.* NAACL 2019.
* Pontiki et al. *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.*
* He, Gao, Chen. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with
  Gradient-Disentangled Embedding Sharing.* ICLR 2023.
* Cao, Mirjalili, Raschka. *Rank Consistent Ordinal Regression for Neural Networks with
  Application to Age Estimation.* Pattern Recognition Letters, 2020 (CORAL).
* Saerens, Latinne, Decaestecker. *Adjusting the Outputs of a Classifier to New a Priori
  Probabilities: A Simple Procedure.* Neural Computation, 2002 (the prior-correction view
  of the neutral offset).
* Pleiss et al. *On Fairness and Calibration.* NeurIPS 2017.

**TODO:** verify every citation against the published record and add the page numbers /
DOIs required by the course template. Nothing in this list was read for this draft; they
are the standard references for the methods used.
