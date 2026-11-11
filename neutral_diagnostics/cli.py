"""Analyze saved predictions, compare runs, and evaluate controlled context changes."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .metrics import (LABELS, calibration, classification, cluster_bootstrap,
                      neutral_diagnostics, precision_recall, read_predictions, target_fingerprint)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fresh_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_pr_svg(path, curves, prevalence):
    """Standalone vector plot with fixed axes and no extra plotting dependency."""
    width, height, left, top, size = 650, 570, 70, 35, 440
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#263238">',
             '<text x="70" y="22">Neutral vs other classes: precision-recall</text>']
    for tick in range(6):
        value = tick / 5
        x, y = left + value * size, top + (1 - value) * size
        parts += [f'<path d="M {x} {top} V {top + size} M {left} {y} H {left + size}" stroke="#e0e0e0" fill="none"/>',
                  f'<text x="{x - 8}" y="{top + size + 22}">{value:.1f}</text>',
                  f'<text x="35" y="{y + 4}">{value:.1f}</text>']
    base = top + (1 - prevalence) * size
    parts.append(f'<path d="M {left} {base} H {left + size}" stroke="#777" stroke-dasharray="5 4"/>')
    for index, (name, curve) in enumerate(curves):
        color = ("#0072b2", "#d55e00")[index]
        commands = [f"M {left} {top}"]
        for point in curve["points"]:
            commands += [f"V {top + (1 - point['precision']) * size:.3f}",
                         f"H {left + point['recall'] * size:.3f}"]
        parts.append(f'<path d="{" ".join(commands)}" fill="none" stroke="{color}" stroke-width="2"/>')
        ap = curve["average_precision"]
        label = f"{name} (AP={ap:.3f})" if ap is not None else f"{name} (AP undefined)"
        parts.append(f'<text x="{left}" y="{525 + 19 * index}" fill="{color}">{label}</text>')
    parts += ['<text x="260" y="510">Recall</text>',
              '<text x="18" y="280" transform="rotate(-90 18 280)">Precision</text>',
              f'<text x="70" y="565">Dashed: neutral prevalence = {prevalence:.3f}; no threshold selected.</text>',
              '</g></svg>']
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def analyze(args):
    import math
    rows = read_predictions(args.predictions)
    if args.gold_xml:
        from bert_baseline.data import read_xml
        examples, _ = read_xml(args.gold_xml)
        actual = Counter((r["text"], r["start"], r["end"], r["label"]) for r in rows)
        expected = Counter((r.text, r.start, r.end, r.label) for r in examples)
        if actual != expected:
            raise ValueError("Prediction targets do not exactly cover the provided gold XML")
    metrics = classification(rows)
    metadata_path = Path(args.predictions).parent / "run_config.json"
    cv_path = Path(args.predictions).parent / "cv_config.json"
    config = {}
    if metadata_path.is_file():
        config = json.loads(metadata_path.read_text(encoding="utf-8"))
    elif cv_path.is_file():
        config = json.loads(cv_path.read_text(encoding="utf-8"))["parameters"]
    if config.get("seed") is not None and args.training_seed is not None and config["seed"] != args.training_seed:
        raise ValueError("training-seed disagrees with the adjacent saved run configuration")
    protocol = {key: config[key] for key in ("mode", "encoder", "model", "max_length", "stride",
                "seed", "resolved_split_seed", "split_seed", "selection_metric", "no_cls_feature",
                "class_weighting") if key in config}
    if "model" in config and "encoder" not in config:
        protocol.setdefault("selection_metric", "gold_macro_f1")  # Original baseline selection rule.
    neutral, errors = neutral_diagnostics(rows)
    gold = [r["label"] == "neutral" for r in rows]
    probability_pr = precision_recall(gold, [r["probabilities"]["neutral"] for r in rows])
    # Under neutral-logit bias, argmax selects neutral according to this margin.
    margin_pr = precision_recall(gold, [math.log(max(r["probabilities"]["neutral"], 1e-30))
                               - math.log(max(1e-30, max(r["probabilities"][label] for label in ("negative", "positive"))))
                               for r in rows])
    result = {"schema_version": 1, "task": "provided_aspect_three_class_sentiment",
              "dataset": args.dataset, "split": args.split, "run_id": args.run_id,
              "model_label": args.model_label, "training_seed": args.training_seed,
              "saved_training_protocol": protocol,
              "input_file": str(Path(args.predictions).resolve()),
              "input_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
              "gold_xml_verified": bool(args.gold_xml), "target_fingerprint": target_fingerprint(rows),
              "classification": metrics, "neutral_diagnostics": neutral,
              "calibration": calibration(rows),
              "neutral_ranking": {name: {k: v for k, v in value.items() if k != "points"}
                                  for name, value in (("probability", probability_pr), ("logit_margin", margin_pr))},
              "uncertainty": cluster_bootstrap(rows, args.bootstrap, args.bootstrap_seed),
              "limits": ["No optimal threshold is selected, including on test data.",
                         "Neighbor agreement does not prove a copying mechanism.",
                         "Subgroups contain the supplied annotated targets, not every possible opinion."]}
    output = fresh_output(args.output)
    write_json(output / "analysis.json", result)
    fields = ["threshold", "precision", "recall", "true_positive", "predicted_positive"]
    for name, value in (("probability", probability_pr), ("logit_margin", margin_pr)):
        write_csv(output / f"neutral_pr_{name}.csv", value["points"], fields)
    write_pr_svg(output / "neutral_pr.svg", [("Probability", probability_pr), ("Logit margin", margin_pr)],
                 probability_pr["prevalence"])
    error_fields = ["sentence_id", "text", "aspect", "start", "end", "prediction", "p_neutral", "confidence",
                    "polar_neighbor_labels", "polar_neighbor_count", "matches_any_polar_neighbor",
                    "closest_polar_aspect", "matches_closest_polar_neighbor", "reviewed_reason", "reviewer"]
    write_csv(output / "neutral_errors_for_review.csv", errors, error_fields)
    lines = [f"# Prediction analysis: {args.run_id}", "",
             f"Dataset: {args.dataset}; split: {args.split}; targets: {len(rows)}.", "",
             "Evaluation: sentiment classification with provided gold aspects, not end-to-end extraction.", "",
             f"Macro-F1: {metrics['macro_f1']:.4f}; Accuracy: {metrics['accuracy']:.4f}.", "",
             "| Class | P | R | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        item = metrics["per_class"][label]
        lines.append(f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
    lines += ["", "## Neutral subgroups", "", "| Group | Correct | Support | Recall |", "|---|---:|---:|---:|"]
    for name, item in neutral["groups"].items():
        recall = f"{item['recall']:.4f}" if item["recall"] is not None else "undefined"
        lines.append(f"| {name} | {item['correct']} | {item['support']} | {recall} |")
    lines += ["", "See analysis.json for AP, confidence, calibration, and sentence-cluster bootstrap intervals.", "",
              "Do not interpret a flat PR segment as proof of weak ranking or neighbor-label agreement as causation.",
              "Do not select a threshold/model on this report if it describes the test set.",
              "Subgroup gaps are descriptive; differences in sentence difficulty can confound them."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "macro_f1": metrics["macro_f1"],
                      "neutral": metrics["per_class"]["neutral"]}, indent=2))


def compare(args):
    analyses = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.analyses]
    coverage, identities, tables, by_run, grouped = {}, set(), [], defaultdict(dict), defaultdict(list)
    criteria = defaultdict(set)
    for item in analyses:
        domain = item["dataset"], item["split"]
        if domain in coverage and coverage[domain] != item["target_fingerprint"]:
            raise ValueError(f"Different evaluation targets within dataset/split {domain}")
        coverage[domain] = item["target_fingerprint"]
        criterion = item.get("saved_training_protocol", {}).get("selection_metric")
        if criterion:
            criteria[domain].add(criterion)
            if len(criteria[domain]) > 1:
                raise ValueError(f"Different checkpoint selection rules within dataset/split {domain}")
        identity = domain + (item["run_id"],)
        if identity in identities:
            raise ValueError("Duplicate run ID within dataset/split")
        identities.add(identity)
        metric = item["classification"]
        flat = {name: item[name] for name in ("dataset", "split", "run_id", "model_label", "training_seed")}
        flat.update(accuracy=metric["accuracy"], macro_f1=metric["macro_f1"], examples=metric["examples"])
        for label, values in metric["per_class"].items():
            for key in ("precision", "recall", "f1", "support"):
                flat[f"{label}_{key}"] = values[key]
        tables.append(flat)
        grouped[domain + (item["model_label"],)].append(flat)
        by_run[(item["dataset"], item["run_id"])][item["split"]] = flat
    statistics_rows = []
    for (dataset, split, model), values in grouped.items():
        seeds = [row["training_seed"] for row in values]
        if len(values) > 1 and (None in seeds or len(set(seeds)) != len(seeds)):
            raise ValueError("Repeated-run mean/std requires distinct, explicit training seeds per model/dataset/split")
        for metric in ("macro_f1", "neutral_precision", "neutral_recall", "neutral_f1"):
            scores = [r[metric] for r in values]
            statistics_rows.append({"dataset": dataset, "split": split, "model_label": model,
                                    "metric": metric, "runs": len(scores), "mean": statistics.mean(scores),
                                    "sample_std": statistics.stdev(scores) if len(scores) > 1 else ""})
    gaps = []
    for (dataset, run_id), splits in by_run.items():
        if "validation" not in splits or "test" not in splits:
            continue
        val, test = splits["validation"], splits["test"]
        if (val["model_label"], val["training_seed"]) != (test["model_label"], test["training_seed"]):
            raise ValueError("Validation/test metadata disagree for the same run ID")
        gaps.append({"dataset": dataset, "run_id": run_id,
                     "macro_f1_validation_minus_test": val["macro_f1"] - test["macro_f1"],
                     "neutral_f1_validation_minus_test": val["neutral_f1"] - test["neutral_f1"],
                     "validation_neutral_support": val["neutral_support"], "test_neutral_support": test["neutral_support"]})
    output = fresh_output(args.output)
    write_csv(output / "runs.csv", tables, list(tables[0]))
    write_csv(output / "mean_std.csv", statistics_rows, list(statistics_rows[0]))
    write_csv(output / "validation_test_gaps.csv", gaps, list(gaps[0]) if gaps else
              ["dataset", "run_id", "macro_f1_validation_minus_test", "neutral_f1_validation_minus_test",
               "validation_neutral_support", "test_neutral_support"])
    print(f"Compared {len(tables)} prediction analyses. Gaps are descriptive, not proof of systematic bias.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze", help="Recompute metrics from complete provided-aspect JSONL")
    analysis.add_argument("--predictions", required=True)
    analysis.add_argument("--gold-xml", help="Optional exact coverage check against evaluation XML")
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--dataset", required=True)
    analysis.add_argument("--split", choices=["validation", "test", "oof", "diagnostic"], required=True)
    analysis.add_argument("--run-id", required=True)
    analysis.add_argument("--model-label", required=True)
    analysis.add_argument("--training-seed", type=int)
    analysis.add_argument("--bootstrap", type=int, default=1000)
    analysis.add_argument("--bootstrap-seed", type=int, default=42)
    comparison = commands.add_parser("compare", help="Export comparable runs, mean/std and validation-test gaps")
    comparison.add_argument("--analyses", nargs="+", required=True)
    comparison.add_argument("--output", required=True)
    validation = commands.add_parser("validate-pairs", help="Validate offsets, minimal changes and review status")
    validation.add_argument("--pairs", default="data/diagnostics/neutral_scope_pairs.csv")
    prediction = commands.add_parser("predict-pairs", help="Evaluate a local trained baseline/pair/joint checkpoint")
    prediction.add_argument("--pairs", default="data/diagnostics/neutral_scope_pairs.csv")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--include-unreviewed", action="store_true",
                            help="Exploratory only: also evaluate pending, never rejected, pairs")
    prediction.add_argument("--batch-size", type=int, default=2)
    prediction.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    prediction.add_argument("--precision", choices=["auto", "no", "bf16", "fp16"], default="auto")
    args = parser.parse_args()
    try:
        if args.command == "analyze":
            analyze(args)
        elif args.command == "compare":
            compare(args)
        else:
            from .pairs import read_pairs, predict_pairs
            if args.command == "validate-pairs":
                pairs = read_pairs(args.pairs)
                print(json.dumps({"pairs": len(pairs), "review_status": dict(Counter(p["review_status"] for p in pairs)),
                                  "target_labels": dict(Counter(p["expected_label"] for p in pairs))}, indent=2))
            else:
                predict_pairs(args)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
