# Verification record

Checked on 2026-11-17. No new SemEval training result is claimed.

## Software

`python -m unittest discover -s tests -v`: **90 tests passed**, 216.099 seconds,
offline with HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE set. Includes four sentiment
heads in tiny-model train/save/reload/evaluate/predict round trips. Checkpoints
are test fixtures, not trained benchmark artifacts. The raw local test log SHA256
is `bd711af2676ff8ce2489f44dd59ffc2b90fff9f9abf0e7d9a86e4976dfabe755`.

After adding the offset/CV integration regression,
`python -m unittest tests.test_evaluation_stats -v`: **5 tests passed**,
0.307 seconds. Four repeat checks cover ties, shared seeds, cluster preservation
and invalid differences. The additional integration check exercises both real
script entry points, validation offset choice and explicit sentence-cluster
metadata in their JSON reports. An initial test-only encoding failure on a GBK
Windows locale was corrected by specifying UTF-8 for JSON reads.

The final CLI help and explanatory docstring changes do not change model math.
`git diff --check` passes. The repository's existing CRLF conversion notice is
informational; raw working-byte hashes are distinguished from LF source hashes.

## Data and arithmetic

`scripts/audit_report_evidence.py` independently verifies the fixed-split label
counts (3046/555/1120 targets), 1680 unique training texts, two real XML gold
cases, paired rounded-score arithmetic, and 8 positive/2 tied offset changes.
The arithmetic sign-test value 0.0078125 is deliberately not a pooled inference
claim because independent experimental units are not established. No corrected
model interval is reported without raw predictions.

## Word document

Final DOCX SHA256:
`0acb803c547aaa56123c1e0d08189f60585a88355c384c8e601aebf436e4c737`.

- 11 pages, 5 native tables, 3 original figures and 1 native editable equation.
- Names and IDs checked against user-provided text; individual roles pending.
- Titles and headings black; inherited theme font overrides and title border removed.
- Tables have explicit light-gray borders, alternating rows, repeated headers,
  deliberate column widths and no fixed-height clipping.
- Actual PDF export through Microsoft Word 16.0, hidden COM instance; only the
  instance/document opened for this task was closed.
- Canonical render_docx.py attempted first and failed because soffice.exe is
  absent. Its rasterize implementation was then used with the genuine Word PDF
  through scripts/render_report_qa.py, at 140 DPI.
- Visually inspected all eleven final-layout pages for text, tables, equations,
  chart labels, breaks, captions, headers, footers and reference wrapping.
- After the final IEEE first-citation-order correction, only page 2 changed;
  PNG SHA256 comparison confirmed the other ten pages were identical, and the
  revised page 2 was inspected again. No blank or overflow pages remain.

Temporary PDFs and PNGs remain in ignored runs/report_qa* for local QA. The Word
file is the deliverable; these QA files are not a course submission package.
