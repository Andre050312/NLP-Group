"""Tune a neutral logit offset on validation, then report the untouched test score.

Multiplying P(neutral) by k and renormalising is the same as adding log(k) to the neutral
logit, so this calibrates the systematic under-prediction of neutral without changing the
model. The offset is chosen on the internal validation split only; the official test set
is scored once with that value. Use the printed value as `--neutral-offset` when scoring.

    python scripts/tune_neutral_offset.py --runs runs/a runs/b --metric macro_f1
"""
import argparse
import json
from pathlib import Path
import random
import sys

if __package__:
    from ..bert_baseline.data import classification_metrics
else:  # `python scripts/tune_neutral_offset.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bert_baseline.data import classification_metrics

LABELS = ("negative", "neutral", "positive")
NEUTRAL = 1
VALIDATION = "validation_best_gold_predictions.jsonl"
TEST = "test_gold_predictions.jsonl"


def read_rows(path):
    if not Path(path).exists():
        raise SystemExit(f"{path} is missing; train the run first")
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def predict(rows, offset):
    gold, predictions = [], []
    for row in rows:
        values = [row["probabilities"][name] for name in LABELS]
        values[NEUTRAL] *= offset
        total = sum(values)
        values = [value / total for value in values]
        gold.append(LABELS.index(row["label"]))
        predictions.append(max(range(3), key=values.__getitem__))
    return gold, predictions


def metrics(rows, offset):
    gold, predictions = predict(rows, offset)
    return classification_metrics(gold, predictions, LABELS)


def score_key(result, metric):
    return result[metric], result["per_class"]["neutral"]["f1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories with validation and test predictions")
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="macro_f1",
                        help="Validation criterion used to pick the offset")
    parser.add_argument("--grid", default="1,1.25,1.5,2,2.5,3,4,5,6", help="Candidate offsets")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Paired resamples for the test delta; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Where to write the JSON report; defaults beside the first run")
    args = parser.parse_args()
    grid = [float(value) for value in args.grid.split(",") if value.strip()]

    report = {}
    for directory in args.runs:
        validation = read_rows(Path(directory) / VALIDATION)
        test = read_rows(Path(directory) / TEST)
        table = []
        for offset in grid:
            inside = metrics(validation, offset)
            table.append({"offset": offset, "validation": inside, "test": metrics(test, offset)})
        chosen = max(table, key=lambda entry: score_key(entry["validation"], args.metric))
        baseline = next(entry for entry in table if entry["offset"] == 1.0)
        entry = {"metric": args.metric, "chosen_offset": chosen["offset"], "table": table,
                 "chosen": chosen, "baseline": baseline}
        if args.bootstrap:
            rng = random.Random(args.seed)
            gold, base_predictions = predict(test, 1.0)
            _, tuned_predictions = predict(test, chosen["offset"])
            indices = list(range(len(gold)))
            deltas = {"accuracy": [], "macro_f1": [], "neutral_f1": []}
            for _ in range(args.bootstrap):
                sample = [rng.choice(indices) for _ in indices]
                a = classification_metrics([gold[i] for i in sample], [base_predictions[i] for i in sample], LABELS)
                b = classification_metrics([gold[i] for i in sample], [tuned_predictions[i] for i in sample], LABELS)
                deltas["accuracy"].append(b["accuracy"] - a["accuracy"])
                deltas["macro_f1"].append(b["macro_f1"] - a["macro_f1"])
                deltas["neutral_f1"].append(b["per_class"]["neutral"]["f1"] - a["per_class"]["neutral"]["f1"])
            entry["bootstrap_vs_offset_1"] = {}
            for name, values in deltas.items():
                values.sort()
                point = (chosen["test"]["per_class"]["neutral"]["f1"] - baseline["test"]["per_class"]["neutral"]["f1"]
                         if name == "neutral_f1" else chosen["test"][name] - baseline["test"][name])
                entry["bootstrap_vs_offset_1"][name] = {
                    "delta": point, "ci95_low": values[int(0.025 * len(values))],
                    "ci95_high": values[int(0.975 * len(values))],
                    "conclusive": values[int(0.025 * len(values))] > 0 or values[int(0.975 * len(values))] < 0}
        report[Path(directory).name] = entry

        print(f"\n=== {Path(directory).name} (offset picked by validation {args.metric}) ===")
        print("  offset   val_macro  val_neutralF1 | test_acc  test_macro  test_neutralF1")
        for row in table:
            inside, outside = row["validation"], row["test"]
            mark = "  <- chosen" if row["offset"] == chosen["offset"] else ""
            print(f"  x{row['offset']:<6} {inside['macro_f1']:.4f}     {inside['per_class']['neutral']['f1']:.4f}    "
                  f"| {outside['accuracy']:.4f}    {outside['macro_f1']:.4f}      "
                  f"{outside['per_class']['neutral']['f1']:.4f}{mark}")
        outside, base = chosen["test"], baseline["test"]
        print(f"  chosen offset {chosen['offset']}: test macro_f1 {base['macro_f1']:.4f} -> {outside['macro_f1']:.4f}, "
              f"neutral F1 {base['per_class']['neutral']['f1']:.4f} -> {outside['per_class']['neutral']['f1']:.4f}, "
              f"accuracy {base['accuracy']:.4f} -> {outside['accuracy']:.4f}")
        for name, value in entry.get("bootstrap_vs_offset_1", {}).items():
            print(f"    delta {name:12s} {value['delta']:+.4f}  95% CI [{value['ci95_low']:+.4f}, "
                  f"{value['ci95_high']:+.4f}]  {'excludes 0' if value['conclusive'] else 'includes 0'}")

    output = Path(args.output) if args.output else Path(args.runs[0]).resolve().parent / "neutral_offset.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
