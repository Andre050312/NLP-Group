# SemEval 2014 Restaurant four class export

Separate from Restaurant3. Original aspect polarities are preserved, including
conflict; no conflict is relabeled as neutral. Label order is negative, neutral,
positive, conflict. The source version and checksums are recorded in manifest.json.

| Split | Sentence records | Aspect terms | Negative | Neutral | Positive | Conflict |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3041 | 3693 | 805 | 633 | 2164 | 91 |
| Test | 800 | 1134 | 196 | 196 | 728 | 14 |

XML preserves category annotations as well, including 195/52 conflict categories;
category counts are not aspect-term counts. JSONL contains one record per aspect
term, with the original text, term, character span, sentence ID and polarity.

Recreate from the pinned original XML with scripts/prepare_restaurant4.py using
an explicit new output directory. Existing directories are never overwritten.
The exporter validates source checksums on download and verifies that the full
aspect-label counts survive conversion before writing either split.

Data provenance follows the existing repository's SemEval-2014 source references;
the original dataset's usage conditions continue to apply. These are data counts,
not new model evaluation results. Three-class reports and Restaurant3 are unchanged.

XML checksums describe LF bytes before Git's Windows newline conversion. Normalize
CRLF to LF when verifying a Windows checkout, as the regression test does.
