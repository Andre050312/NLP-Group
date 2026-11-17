# Label support and split audit

No trained model was evaluated. Counts are aspect targets, not sentence labels.

| Dataset | Split | Negative | Neutral | Positive | Neutral with polar neighbor | Neutral without polar neighbor |
|---|---|---:|---:|---:|---:|---:|
| restaurant | train | 677 | 520 | 1849 | 236 | 284 |
| restaurant | validation | 128 | 113 | 314 | 52 | 61 |
| restaurant | test | 196 | 196 | 728 | 83 | 113 |
| restaurant2015 | train | 216 | 31 | 777 | 10 | 21 |
| restaurant2015 | validation | 36 | 3 | 125 | 0 | 3 |
| restaurant2015 | test | 179 | 27 | 319 | 11 | 16 |
| restaurant2016 | train | 366 | 52 | 1028 | 20 | 32 |
| restaurant2016 | validation | 67 | 9 | 191 | 1 | 8 |
| restaurant2016 | test | 113 | 28 | 460 | 5 | 23 |

## Reader and split findings

- restaurant2015: train/validation share 102 review IDs. Existing review-grouped CV avoids this holdout dependency.
- restaurant2016: train/validation share 136 review IDs. Existing review-grouped CV avoids this holdout dependency.
- mams: BLOCKED by current reader/data: BIO head cannot represent overlapping annotated aspects

Complete source hashes, overlap examples and recall step sizes are in data_audit.json.
A successful audit command does not imply every dataset is trainable: inspect each dataset status.
These counts neither verify historical predictions nor establish a dataset performance ceiling.
