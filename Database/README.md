# Restaurant aspect-sentiment datasets

Default datasets use negative, neutral and positive labels for aspect targets.
Explicit four-class SemEval-2014 uses [Restaurant4](Restaurant4/README.md), retaining conflict.

| Directory | Task/source | Training aspects | Validation aspects | Test aspects |
|---|---|---:|---:|---:|
| Restaurant3 | SemEval-2014 explicit-aspect derivative | 3602 | split from training | 1120 |
| Restaurant2015 | SemEval-2015 explicit-aspect derivative | 1188 | split from training | 525 |
| Restaurant2016 | SemEval-2016 explicit-aspect derivative | 1714 | split from training | 601 |
| MAMS/data/MAMS-ATSA/raw | Original multi-aspect term sentiment | 11186 | 1332 | 1336 |
| MAMS/data/MAMS-ACSA/raw | Original aspect-category sentiment | 7090 | 888 | 901 |

Counts refer to targets. MAMS-ACSA has no character spans and cannot use the current
aspect-term model entry points.

## Provenance

Restaurant3 removes conflict annotations from SemEval-2014 Restaurant.
Restaurant2015/2016 convert mirrored official restaurant annotations to aspectTerms XML,
excluding NULL implicit targets, OutOfScope sentences, invalid positions, conflicting
same-span sentiments and overlapping spans. Identical same-span/same-label opinions merge.
Review text and retained offsets remain original. MAMS keeps its original three-class splits
and XML bytes. Manifests and scripts/restaurant_sources.json record revisions, URLs,
source/derived hashes and filtering counts.

Restaurant3 has 1120 classification and extraction test targets after removal of conflict.
Older unfiltered joint runs used 1134 targets and follow a different evaluation protocol.
Use the original providers' conditions and cite their papers.

## Recovery

```powershell
python scripts/download_datasets.py --offline
python scripts/download_datasets.py --check
python scripts/download_datasets.py
```

The first two commands restore/check 12 XML files from resources/restaurant_datasets_3class.zip.
The third enables network recovery. Matching files are skipped.
--dataset selects restaurant, restaurant2015, restaurant2016 or mams.
--import-from accepts a source directory/archive. --timeout and --retries control requests.
Raw mixed-label sources are converted in memory.

Restaurant4 uses its separate exporter and original four-label XML sources.

## Cross-dataset overlap

overlap_report.json records exact-text overlap. Restaurant2016 training overlaps
Restaurant2015 test by 662 sentences. Restaurant3 and MAMS also have cross-split overlaps.
Train independently by default. Combining datasets requires removing training text that
occurs in any validation/test set and recording excluded counts.

## Retired datasets

After all 12 replacement XML files validate, scripts/clean_non_threeclass_data.py can preview
retired ASAP, SemEval2014, IMDB and the former mixed-label archive. --apply removes only those
listed paths and records cleanup_report.json.
