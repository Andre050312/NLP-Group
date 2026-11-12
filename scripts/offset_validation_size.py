"""Does the neutral offset only look good because the validation split is large?

The single-split experiment tunes k on 555 validation aspects (113 neutral) and reports a
gain on the 1120-aspect official test set. Inside 3-fold cross-validation the inner
validation set is only 366 aspects (62 neutral), and there the same procedure does not
transfer. This script isolates validation-set size as an explanation.

For each run it repeatedly draws a stratified subsample of the validation set at a smaller
size, tunes k by validation macro-F1, and applies it to the same fixed test predictions.
The test set is never subsampled, so only the amount of tuning data changes.

If the gain collapses as the tuning set shrinks, the earlier positive result is best read
as a small effect that needs a large validation set. If the gain survives at the small
size, validation size does not explain the cross-validation result.

Usage:
    python scripts/offset_validation_size.py --runs runs/pair_heads_softmax_s42 runs/pair_ctrl_s43 \
        --sizes 366,555 --draws 200
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_neutral_offset as tno  # noqa: E402

LABELS = tno.LABELS


def stratified_subsample(rows: list[dict], size: int, rng: random.Random) -> list[dict]:
    """Draw `size` rows while keeping the class mix as close to the full set as possible."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    total = len(rows)
    chosen: list[dict] = []
    for label, group in by_label.items():
        quota = max(1, round(len(group) * size / total))
        chosen.extend(rng.sample(group, min(quota, len(group))))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--sizes", default="366,555", help="Candidate tuning-set sizes in aspects")
    parser.add_argument("--draws", type=int, default=200, help="Subsamples per run and size")
    parser.add_argument("--grid", default="1,1.25,1.5,2,2.5,3,4,5,6")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/offset_validation_size.json"))
    args = parser.parse_args()

    grid = [float(value) for value in args.grid.split(",") if value.strip()]
    rng = random.Random(args.seed)
    report: dict[str, dict] = {}

    for run in args.runs:
        directory = Path(run)
        validation = tno.read_rows(directory / tno.VALIDATION)
        test = tno.read_rows(directory / tno.TEST)
        base_metrics = tno.metrics(test, 1.0)
        entry = {"validation_aspects": len(validation),
                 "validation_neutral": sum(row["label"] == "neutral" for row in validation),
                 "test_aspects": len(test),
                 "test_at_k1": {name: base_metrics[name] for name in ("accuracy", "macro_f1")},
                 "test_at_k1_neutral_f1": base_metrics["per_class"]["neutral"]["f1"],
                 "sizes": {}}

        for size in [int(value) for value in args.sizes.split(",") if value.strip()]:
            if size > len(validation):
                continue
            deltas = {"macro_f1": [], "neutral_f1": [], "accuracy": []}
            chosen = []
            for _ in range(args.draws):
                sample = stratified_subsample(validation, size, rng)
                table = [(k, tno.metrics(sample, k)) for k in grid]
                k, _ = max(table, key=lambda item: tno.score_key(item[1], args.metric))
                chosen.append(k)
                tuned = tno.metrics(test, k)
                for name in deltas:
                    value = (tuned["per_class"]["neutral"]["f1"] if name == "neutral_f1"
                             else tuned[name])
                    base = (base_metrics["per_class"]["neutral"]["f1"] if name == "neutral_f1"
                            else base_metrics[name])
                    deltas[name].append(value - base)
            summary = {name: {"mean": statistics.mean(values),
                              "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                              "positive_rate": sum(v > 0 for v in values) / len(values)}
                       for name, values in deltas.items()}
            summary["offset_counts"] = {str(k): chosen.count(k) for k in sorted(set(chosen))}
            entry["sizes"][str(size)] = summary

        report[directory.name] = entry
        print(f"\n=== {directory.name} (validation {len(validation)} aspects, "
              f"{entry['validation_neutral']} neutral) ===")
        print(f"  test at k=1: macro {base_metrics['macro_f1']:.4f} "
              f"neutral F1 {base_metrics['per_class']['neutral']['f1']:.4f}")
        for size, summary in entry["sizes"].items():
            macro = summary["macro_f1"]
            neutral = summary["neutral_f1"]
            print(f"  tuning size {size:>4}: macro {macro['mean']:+.4f} +/- {macro['sd']:.4f} "
                  f"(positive {macro['positive_rate']:.0%}) | "
                  f"neutral F1 {neutral['mean']:+.4f} +/- {neutral['sd']:.4f} "
                  f"(positive {neutral['positive_rate']:.0%}) | k {summary['offset_counts']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
