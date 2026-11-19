# Offline three-class restaurant datasets

- restaurant_datasets_3class.zip contains Restaurant3, Restaurant2015, Restaurant2016,
  MAMS-ATSA and MAMS-ACSA.
- restaurant_3class.zip contains the SemEval-2014 Restaurant three-class derivative.

Run python scripts/download_datasets.py --offline from the repository root to restore the
full collection's 12 XML files with pinned hashes. Include the scripts, source catalog,
converter and full archive when copying the project.

Manifests record source splits and filtering. SemEval-2014 removes conflict.
SemEval-2015/2016 filter implicit, invalid and conflicting aspects. MAMS XML bytes remain
original. Dataset providers' conditions and citations still apply.
Cross-dataset overlap requires independent training or explicit held-out overlap removal.

Packaged project README files use the same English descriptions as Database/. Original MAMS
notes retain their attribution. XML and manifest bytes retain their pinned hashes.
