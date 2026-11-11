## Problem and behavior

Neutral-target errors need a reproducible evaluation before attributing them to cross-aspect interference. Add a saved-prediction analyzer with per-class metrics, complete neutral PR/AP, confidence/calibration, sentence-cluster bootstrap intervals, and explicit denominators for neighbor-label agreement.

Add 40 synthetic candidate pairs (30 neutral targets and 10 polar controls) that change only the distractor opinion. These require human semantic review; exploratory outputs identify unreviewed cases. Pair inference supports locally trained baseline, joint and pair checkpoints.

Add `--no-cls-feature` for joint/pair training and grouped CV. It removes CLS only from the sentiment head, preserves contextual encoder information, records the architecture in checkpoints, and defaults legacy checkpoints to the original architecture.

## Validation and limits

All 56 tests passed locally with Python 3.12.10, Transformers 4.57.1, scikit-learn 1.7.2 and CPU PyTorch 2.14.1. Control and no-CLS three-fold plans have identical fold assignments on Restaurant3. See `docs/neutral_scope_validation.md` for commands and limitations.

Model tests use small randomly initialized BERT models; they do not establish DeBERTa accuracy or improvement. CUDA has not been verified. The 40 candidates are not an official benchmark and await human review. No empirical performance ceiling, causal mechanism or improvement is claimed.

## Contribution

Evaluation tooling, controlled diagnostic design, CLS-head ablation, regression checks and the experimental/report protocol. Human review and real-model experiments remain explicitly tracked as pending in `docs/neutral_scope_study.md`.
