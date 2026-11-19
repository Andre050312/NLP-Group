# SemEval-2016 Restaurant three-class derivative

Aspect-level labels are negative, neutral and positive. Original training/test membership
is retained. manifest.json records source revisions, hashes and filtering statistics.

Only explicit character spans are retained. Implicit NULL targets, invalid offsets, conflicting same-span labels and overlapping spans are excluded. Identical same-span/same-label opinions count once. Original review text and retained offsets are preserved.
Validation comes from training data. Use validation for tuning.
Cross-year combinations need the overlap filtering described in ../overlap_report.json.

[Official task](https://alt.qcri.org/semeval2016/task5/)
