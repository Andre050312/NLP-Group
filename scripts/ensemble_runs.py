"""Average per-aspect probabilities from several runs into one ensemble score.

Only saved run artefacts are read, so no encoder is loaded and no GPU is needed.
Every run must have predicted exactly the same gold targets; otherwise the rows are
not comparable and the script refuses to average them.

    python scripts/ensemble_runs.py --runs runs/a runs/b runs/c --output runs/ensemble
    python scripts/ensemble_runs.py --runs runs/a runs/b --output runs/ensemble --bootstrap 2000
"""
import argparse
import json
from pathlib import Path
import random
import sys

if __package__:
    from ..bert_baseline.data import classification_metrics
else:  # `python scripts/ensemble_runs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bert_baseline.data import classification_metrics

LABELS = ("negative", "neutral", "positive")
PREDICTIONS = "test_gold_predictions.jsonl"


def read_run(directory):
    path = Path(directory) / PREDICTIONS
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m absa_enhanced train` first")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (row["sentence_id"], row["start"], row["end"])
        if key in rows:
            raise SystemExit(f"Duplicate target {key} in {path}")
        rows[key] = {"label": row["label"], "probabilities": [row["probabilities"][name] for name in LABELS],
                     "text": row["text"], "aspect": row["aspect"]}
    return rows


def scores(gold, predictions):
    return classification_metrics([LABELS.index(label) for label in gold],
                                  [LABELS.index(label) for label in predictions], LABELS)


def predict(probabilities):
    return [LABELS[max(range(3), key=row.__getitem__)] for row in probabilities]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories holding " + PREDICTIONS)
    parser.add_argument("--output", required=True, help="Directory for the ensemble result files")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Paired bootstrap resamples for the difference against the first run; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    runs = [read_run(directory) for directory in args.runs]
    keys = sorted(runs[0])
    for directory, rows in zip(args.runs, runs):
        if set(rows) != set(keys):
            raise SystemExit(f"{directory} was evaluated on different targets; results are not comparable")
        for key in keys:
            if rows[key]["label"] != runs[0][key]["label"]:
                raise SystemExit(f"{directory} disagrees on the gold label of {key}")
    gold = [runs[0][key]["label"] for key in keys]
    averaged = [[sum(row[key]["probabilities"][i] for row in runs) / len(runs) for i in range(3)] for key in keys]

    ensemble = scores(gold, predict(averaged))
    summary = {"runs": [str(Path(d)) for d in args.runs], "examples": len(keys), "ensemble": ensemble,
               "members": [scores(gold, predict([row[key]["probabilities"] for key in keys])) for row in runs]}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, probabilities in zip(keys, averaged):
        values = dict(zip(LABELS, probabilities))
        rows.append({"sentence_id": key[0], "text": runs[0][key]["text"], "aspect": runs[0][key]["aspect"],
                     "start": key[1], "end": key[2], "label": runs[0][key]["label"],
                     "prediction": LABELS[max(range(3), key=probabilities.__getitem__)],
                     "probabilities": values})
    with (output / PREDICTIONS).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.bootstrap:
        rng = random.Random(args.seed)
        indices = list(range(len(keys)))
        base = [runs[0][key]["probabilities"] for key in keys]
        deltas = {"neutral_f1": [], "macro_f1": [], "accuracy": []}
        for _ in range(args.bootstrap):
            sample = [rng.choice(indices) for _ in indices]
            sample_gold = [gold[i] for i in sample]
            a = scores(sample_gold, predict([base[i] for i in sample]))
            b = scores(sample_gold, predict([averaged[i] for i in sample]))
            deltas["accuracy"].append(b["accuracy"] - a["accuracy"])
            deltas["macro_f1"].append(b["macro_f1"] - a["macro_f1"])
            deltas["neutral_f1"].append(b["per_class"]["neutral"]["f1"] - a["per_class"]["neutral"]["f1"])
        summary["bootstrap_vs_first_run"] = {}
        for name, values in deltas.items():
            values.sort()
            point = (ensemble["per_class"]["neutral"]["f1"] - summary["members"][0]["per_class"]["neutral"]["f1"]
                     if name == "neutral_f1" else
                     (ensemble["macro_f1"] - summary["members"][0]["macro_f1"] if name == "macro_f1"
                      else ensemble["accuracy"] - summary["members"][0]["accuracy"]))
            low, high = values[int(0.025 * len(values))], values[int(0.975 * len(values))]
            summary["bootstrap_vs_first_run"][name] = {
                "delta": point, "ci95_low": low, "ci95_high": high,
                "conclusive": low > 0 or high < 0}

    (output / "ensemble_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(f"{len(runs)} runs averaged over {len(keys)} matched aspects")
    for name, member in zip(args.runs, summary["members"]):
        print(f"  member {Path(name).name:28s} acc={member['accuracy']:.4f} macro_f1={member['macro_f1']:.4f} "
              f"neutral_f1={member['per_class']['neutral']['f1']:.4f}")
    print(f"  ensemble {'':21s} acc={ensemble['accuracy']:.4f} macro_f1={ensemble['macro_f1']:.4f} "
          f"neutral_f1={ensemble['per_class']['neutral']['f1']:.4f}")
    for name, value in summary.get("bootstrap_vs_first_run", {}).items():
        print(f"  delta {name:12s} {value['delta']:+.4f}  95% CI [{value['ci95_low']:+.4f}, "
              f"{value['ci95_high']:+.4f}]  {'excludes 0' if value['conclusive'] else 'includes 0'}")
    print(f"wrote {output / 'ensemble_metrics.json'}")


if __name__ == "__main__":
    main()
