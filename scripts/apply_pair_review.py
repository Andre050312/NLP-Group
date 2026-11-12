"""Merge independently exported reviewer CSVs into the canonical pairs file.

Reviewers use data/diagnostics/neutral_scope_pairs_review.html, which exports one CSV
per reviewer (columns identical to the canonical file). This script enforces the rules
the study document states:

* two reviewers, each covering every pair, with a real name on every decision;
* a pair becomes approved/rejected only when both reviewers agree;
* disagreements stay ``pending`` and are listed for discussion;
* nothing is written unless ``--write`` is passed.

Usage:
    python scripts/apply_pair_review.py review_A.csv review_B.csv            # report only
    python scripts/apply_pair_review.py review_A.csv review_B.csv --write    # update canonical
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_PAIRS = Path("data/diagnostics/neutral_scope_pairs.csv")
STATUSES = ("pending", "approved", "rejected")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", nargs="+", type=Path, help="One CSV per reviewer")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--write", action="store_true", help="Rewrite the canonical pairs CSV")
    parser.add_argument("--audit", type=Path, default=Path("data/diagnostics/pair_review_audit.json"))
    args = parser.parse_args()

    if len(args.reviews) != 2:
        raise SystemExit("Exactly two independent reviewer CSVs are required by the study protocol.")

    canonical = read_csv(args.pairs)
    ids = [row["pair_id"] for row in canonical]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{args.pairs} has duplicate pair IDs")

    decisions: dict[str, dict[str, dict[str, str]]] = {}
    used_names: set[str] = set()
    for path in args.reviews:
        rows = read_csv(path)
        seen: dict[str, dict[str, str]] = {}
        for row in rows:
            pid = row["pair_id"]
            if pid in seen:
                raise SystemExit(f"{path}: duplicate pair_id {pid}")
            status = row["review_status"].strip()
            if status not in STATUSES:
                raise SystemExit(f"{path}: invalid review_status {status!r} for {pid}")
            if status == "pending":
                raise SystemExit(f"{path}: {pid} was exported without a verdict")
            if not row["reviewer"].strip():
                raise SystemExit(f"{path}: {pid} has no reviewer name")
            seen[pid] = row
        missing = [pid for pid in ids if pid not in seen]
        if missing:
            raise SystemExit(f"{path}: no decision for {len(missing)} pair(s), e.g. {missing[:5]}")
        names = {row["reviewer"].strip() for row in seen.values()}
        if len(names) != 1:
            raise SystemExit(f"{path}: mixed reviewer names {sorted(names)}")
        name = names.pop()
        if name in used_names:
            raise SystemExit(f"{path}: reviewer {name!r} already used by another file; "
                             "the two reviews must be independent")
        used_names.add(name)
        decisions[path.name] = seen

    (label_a, rows_a), (label_b, rows_b) = decisions.items()
    agree, disagree = [], []
    for pid in ids:
        a, b = rows_a[pid], rows_b[pid]
        if a["review_status"] == b["review_status"]:
            agree.append(pid)
        else:
            disagree.append({"pair_id": pid,
                             label_a: a["review_status"], label_b: b["review_status"],
                             "note_a": a.get("review_notes", ""), "note_b": b.get("review_notes", "")})

    counts = Counter()
    merged = []
    for row in canonical:
        pid = row["pair_id"]
        a, b = rows_a[pid], rows_b[pid]
        out = dict(row)
        if a["review_status"] == b["review_status"] and pid not in {d["pair_id"] for d in disagree}:
            out["review_status"] = a["review_status"]
            reviewers = sorted({a["reviewer"].strip(), b["reviewer"].strip()})
            out["reviewer"] = " + ".join(reviewers)
            notes = [f"{a['reviewer'].strip()}: {a.get('review_notes', '').strip()}".rstrip(": "),
                     f"{b['reviewer'].strip()}: {b.get('review_notes', '').strip()}".rstrip(": ")]
            out["review_notes"] = " | ".join(n for n in notes if not n.endswith(":"))
        else:
            out["review_status"] = "pending"
            out["reviewer"] = ""
            out["review_notes"] = "reviewers disagreed; discuss and re-review"
        counts[out["review_status"]] += 1
        merged.append(out)

    total = len(ids)
    rate = len(agree) / total if total else 0.0
    summary = {
        "pairs": total,
        "reviewers": sorted({rows_a[pid]["reviewer"].strip() for pid in ids} |
                            {rows_b[pid]["reviewer"].strip() for pid in ids}),
        "agreement_count": len(agree),
        "agreement_rate": round(rate, 4),
        "status_counts_after_merge": dict(counts),
        "disagreements": disagree,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"},
                     ensure_ascii=False, indent=2))
    if disagree:
        print(f"\n{len(disagree)} disagreement(s) kept pending:")
        for item in disagree:
            print(f"  {item['pair_id']}: {item[label_a]} vs {item[label_b]}")
    else:
        print("\nNo disagreements.")

    if args.write:
        with args.pairs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(canonical[0]))
            writer.writeheader()
            writer.writerows(merged)
        print(f"\nWrote {args.pairs} and {args.audit}")
    else:
        print(f"\nReport only; wrote {args.audit}. Re-run with --write to update {args.pairs}.")
    return 0 if not disagree else 1


if __name__ == "__main__":
    raise SystemExit(main())
