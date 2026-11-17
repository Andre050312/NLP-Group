The earlier report overstates what the neutral-class experiments establish and
contains a pooled sign-test claim that counts two ties as improvements. The
historical ordinal head also trains cumulative targets but applies a conditional
product at inference, so its score cannot support conclusions about CORAL.

This change rewrites the report around matched-seed ablations, ranking and scope
diagnostics, and protocol-dependent decision tuning. It adds the 11-page Group 9
Word review copy, three original figures, source data, primary-paper reading
records, a reproducible builder and an evidence audit. Model scores remain
explicitly attributed to the committed group records; no new benchmark training
or invented prediction cases are claimed.

Software changes exclude sign-test ties, verify seed sets, use paired exact-text
sentence-cluster resampling for future offset intervals, and add a separately
named consistent `ordered` head. Legacy `ordinal` checkpoint behavior stays
available and is documented as unsuitable for comparative conclusions.

Validation: 90 offline tests passed, including tiny-model head round trips; the
updated statistical module's 5 focused tests also pass, including offset/CV CLI
integration. XML counts and arithmetic are independently checked. All 11 actual
Word-rendered pages were visually reviewed. See docs/report_review/verification.md.

Review before submission: individual contributions are intentionally pending
member confirmation. Original run artifacts and human review of diagnostic
pairs are still needed for complete per-class metrics and real model error cases.
This is a draft for group review and does not merge or replace anyone's work on main.
