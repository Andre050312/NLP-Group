"""Controlled synthetic pairs; candidate labels require independent human review."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from .metrics import LABELS


def read_pairs(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        if not row["pair_id"] or row["pair_id"] in seen:
            raise ValueError("Pair IDs must be nonempty and unique")
        seen.add(row["pair_id"])
        if row["expected_label"] not in LABELS or row["review_status"] not in ("pending", "approved", "rejected"):
            raise ValueError("Invalid target label/review status")
        if row["review_status"] == "approved" and not row["reviewer"].strip():
            raise ValueError("Approved pairs must name their human reviewer")
        normalized = []
        for variant in ("positive", "negative"):
            text, expression = row[f"text_{variant}_neighbor"], row[f"{variant}_expression"]
            if not expression or text.count(expression) != 1:
                raise ValueError(f"Sentiment expression must occur once in {row['pair_id']}")
            spans = []
            for field in ("target", "distractor"):
                term = row[field]
                if not term or text.count(term) != 1:
                    raise ValueError(f"{field} must occur exactly once in {row['pair_id']}")
                spans.append((text.index(term), text.index(term) + len(term)))
            target, distractor = spans
            opinion = text.index(expression), text.index(expression) + len(expression)
            if target[0] < distractor[1] and distractor[0] < target[1]:
                raise ValueError("Target and distractor overlap")
            if target[0] < opinion[1] and opinion[0] < target[1]:
                raise ValueError("Changed opinion overlaps target")
            normalized.append(text.replace(expression, "{CHANGED_OPINION}", 1))
        if normalized[0] != normalized[1] or row["positive_expression"] == row["negative_expression"]:
            raise ValueError("Each pair must change only its stated distractor opinion")
    if not rows:
        raise ValueError("No pairs found")
    return rows


def summarize_pairs(pairs, predictions):
    """Invariance alone rewards constant classifiers, so also report correctness."""
    details = []
    for pair in pairs:
        positive = predictions[(pair["pair_id"], "positive")]
        negative = predictions[(pair["pair_id"], "negative")]
        expected = pair["expected_label"]
        details.append({"pair_id": pair["pair_id"], "target": pair["target"], "expected_label": expected,
                        "positive_neighbor_prediction": positive["prediction"],
                        "negative_neighbor_prediction": negative["prediction"],
                        "invariant": positive["prediction"] == negative["prediction"],
                        "both_correct": positive["prediction"] == negative["prediction"] == expected,
                        "follows_distractor_polarity": positive["prediction"] == "positive" and negative["prediction"] == "negative",
                        "absolute_neutral_probability_change": abs(positive["probabilities"]["neutral"]
                                                                  - negative["probabilities"]["neutral"])})
    groups = {}
    for name, subset in (("all", details), ("neutral_targets", [p for p in details if p["expected_label"] == "neutral"]),
                         ("polar_controls", [p for p in details if p["expected_label"] != "neutral"])):
        groups[name] = {"pairs": len(subset)}
        for field in ("invariant", "both_correct", "follows_distractor_polarity", "absolute_neutral_probability_change"):
            groups[name][field + "_mean"] = sum(p[field] for p in subset) / len(subset) if subset else None
    return {"groups": groups, "note": "Synthetic controlled diagnostic, not benchmark accuracy or a proof of internal mechanism. "
            "Report both_correct with invariance; polar controls detect always-neutral behavior."}, details


def predict_pairs(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    all_pairs = read_pairs(args.pairs)
    pairs = [p for p in all_pairs if p["review_status"] == "approved"
             or (args.include_unreviewed and p["review_status"] == "pending")]
    if not pairs:
        raise ValueError("No approved pairs. Review the CSV, or use --include-unreviewed for exploratory checks only.")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise ValueError("checkpoint must be a local trained model directory")
    from .cli import fresh_output, write_csv, write_json
    # Check the destination before allocating a full model; only create on success.
    destination = Path(args.output)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output must be empty")
    targets, keys = [], []
    for pair in pairs:
        for variant in ("positive", "negative"):
            text = pair[f"text_{variant}_neighbor"]
            start = text.index(pair["target"])
            keys.append((pair["pair_id"], variant))
            targets.append({"sentence_id": pair["pair_id"] + ":" + variant, "text": text, "aspect": pair["target"],
                            "start": start, "end": start + len(pair["target"]), "label": pair["expected_label"]})
    if (checkpoint / "settings.json").is_file():
        from absa_enhanced.cli import resolve_device, resolve_precision
        from absa_enhanced.data import Aspect, Document, WindowDataset
        from absa_enhanced.evaluation import evaluate
        from absa_enhanced.model import JointABSA
        from absa_enhanced.pair import PairDataset
        device = resolve_device(args.device)
        precision = resolve_precision(args.precision, device)
        model, tokenizer, settings = JointABSA.load_checkpoint(checkpoint)
        tokenizer.padding_side = "right"
        model.to(device)
        documents = [Document(t["sentence_id"], t["text"], (Aspect(t["start"], t["end"], t["label"]),)) for t in targets]
        if model.model_kind == "pair":
            # Honour the scope the checkpoint was trained with, so a clause-scoped
            # model is not silently evaluated on whole-sentence context.
            data = PairDataset(documents, tokenizer, settings["max_length"], settings["stride"],
                              scope=settings.get("context_scope", "sentence"))
        else:
            data = WindowDataset(documents, tokenizer, settings["max_length"], settings["stride"])
        _, predicted_rows, _ = evaluate(model, tokenizer, data, device, args.batch_size, precision, compute_metrics=False)
        by_id = {r["sentence_id"]: r for r in predicted_rows}
        predictions = {key: by_id[target["sentence_id"]] for key, target in zip(keys, targets)}
    else:
        from argparse import Namespace
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding
        from bert_baseline.cli import evaluate, load_checkpoint
        from bert_baseline.data import Example
        from bert_baseline.model import AspectDataset
        if args.precision not in ("auto", "no"):
            raise ValueError("Baseline diagnostics use full precision; choose --precision no or auto")
        model, tokenizer, labels, device = load_checkpoint(Namespace(checkpoint=str(checkpoint), device=args.device))
        if set(labels) != set(LABELS):
            raise ValueError("Expected a three-class baseline checkpoint")
        config_path = checkpoint.parent / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        data = AspectDataset([Example(**r) for r in targets], tokenizer, labels, config.get("max_length", 128))
        loader = DataLoader(data, batch_size=args.batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
        _, predicted, probabilities = evaluate(model, loader, device, labels)
        predictions = {key: {**target, "prediction": labels[p], "probabilities": dict(zip(labels, values))}
                       for key, target, p, values in zip(keys, targets, predicted, probabilities)}
    result, details = summarize_pairs(pairs, predictions)
    result.update(checkpoint=str(checkpoint.resolve()),
                  pairs_sha256=hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
                  review_status=dict(Counter(p["review_status"] for p in pairs)),
                  human_review_complete=all(p["review_status"] == "approved" for p in pairs))
    output = fresh_output(args.output)
    write_json(output / "pair_metrics.json", result)
    write_csv(output / "pair_results.csv", details, list(details[0]))
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for key in keys:
            stream.write(json.dumps({**predictions[key], "pair_id": key[0], "neighbor_polarity": key[1]}, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
