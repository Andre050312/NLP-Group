"""Compare test metrics across groups of runs (baseline vs model families).

Each group is a comma-separated list of run directories. Prints per-group mean and
sample SD for accuracy / macro-F1 / neutral F1, plus Welch t statistics between every
pair of groups. Welch (unequal variance) is used because the seed-to-seed spread differs
a lot between model families; it is not a paired test because different families do not
share the same random initialisation.

Usage:
    python scripts/compare_baselines.py \
        --group "bert=runs/bert_split42_seed42,...,runs/bert_split42_seed46" \
        --group "pair_control=runs/pair_heads_softmax_s42,runs/pair_ctrl_s43" \
        --output runs/baseline_comparison.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

METRICS = {
    "accuracy": lambda g: g["accuracy"],
    "macro_f1": lambda g: g["macro_f1"],
    "neutral_f1": lambda g: g["per_class"]["neutral"]["f1"],
    "neutral_recall": lambda g: g["per_class"]["neutral"]["recall"],
    "neutral_precision": lambda g: g["per_class"]["neutral"]["precision"],
}


def load(run: str) -> dict:
    path = Path(run) / "test_metrics.json"
    if not path.is_file():
        raise SystemExit(f"{path} is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("gold_aspect_sentiment") or payload


def welch(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return (statistics.mean(a) - statistics.mean(b)) / se if se else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True, metavar="NAME=RUN[,RUN...]")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = {}
    for spec in args.group:
        name, _, runs = spec.partition("=")
        if not runs:
            raise SystemExit(f"Bad group spec: {spec!r}")
        groups[name.strip()] = [load(r.strip()) for r in runs.split(",") if r.strip()]

    report: dict[str, dict] = {}
    print(f"{'group':16s} {'n':>2s} " + " ".join(f"{m:>22s}" for m in METRICS))
    for name, rows in groups.items():
        cells = []
        entry = {"n": len(rows), "metrics": {}}
        for label, getter in METRICS.items():
            values = [getter(row) for row in rows]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            entry["metrics"][label] = {"mean": mean, "sample_sd": sd, "values": values}
            cells.append(f"{mean:.4f} +/- {sd:.4f}")
        report[name] = entry
        print(f"{name:16s} {len(rows):2d} " + " ".join(f"{c:>22s}" for c in cells))

    names = list(groups)
    pairs = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            row = {"comparison": f"{left} vs {right}"}
            for label in METRICS:
                a = report[left]["metrics"][label]["values"]
                b = report[right]["metrics"][label]["values"]
                row[label] = {"mean_difference": statistics.mean(a) - statistics.mean(b),
                              "welch_t": welch(a, b)}
            pairs.append(row)
            print(f"\n{left} vs {right}")
            for label in METRICS:
                d = row[label]
                print(f"  {label:18s} diff {d['mean_difference']:+.4f}   welch t {d['welch_t']:+.2f}")
    report["pairwise"] = pairs

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
