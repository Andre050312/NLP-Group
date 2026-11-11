"""Dependency-free, auditable metrics; no threshold/model selection on test data."""
from collections import defaultdict
import hashlib
import json
import math
import random
import statistics

from bert_baseline.data import classification_metrics

LABELS = ("negative", "neutral", "positive")


def read_predictions(path):
    rows, seen = [], set()
    with open(path, encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text, start, end = row["text"], row["start"], row["end"]
            if not isinstance(text, str) or type(start) is not int or type(end) is not int:
                raise ValueError(f"Invalid text/span types at line {line_number}")
            if not 0 <= start < end <= len(text) or text[start:end] != row["aspect"]:
                raise ValueError(f"Invalid aspect offset at line {line_number}")
            if row["label"] not in LABELS or row["prediction"] not in LABELS:
                raise ValueError("Only three-class provided-aspect predictions are supported")
            probabilities = row["probabilities"]
            if set(probabilities) != set(LABELS):
                raise ValueError("Expected probabilities keyed by negative/neutral/positive")
            values = [probabilities[label] for label in LABELS]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-4:
                raise ValueError(f"Invalid probabilities at line {line_number}")
            maximum = max(values)
            if maximum - probabilities[row["prediction"]] > 1e-5:
                raise ValueError("prediction does not match probabilities; export adjusted probabilities if using bias")
            # IDs distinguish separate sentence records with the same text.
            key = (str(row.get("sentence_id", "")), text, start, end)
            if key in seen:
                raise ValueError(f"Duplicate target at line {line_number}: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("Prediction file is empty")
    return rows


def classification(rows):
    return classification_metrics([LABELS.index(r["label"]) for r in rows],
                                  [LABELS.index(r["prediction"]) for r in rows], LABELS)


def target_fingerprint(rows):
    # Preserve multiplicity; omit model-specific sentence ID formatting.
    targets = sorted((r["text"], r["start"], r["end"], r["label"]) for r in rows)
    return hashlib.sha256(json.dumps(targets, ensure_ascii=False).encode("utf-8")).hexdigest()


def precision_recall(gold, scores):
    """Group equal scores before computing AP, avoiding arbitrary tie ordering."""
    positives = sum(gold)
    if not positives:
        return {"average_precision": None, "prevalence": 0.0, "points": []}
    groups = defaultdict(lambda: [0, 0])
    for actual, score in zip(gold, scores):
        groups[score][0] += int(actual)
        groups[score][1] += 1
    tp = predicted = 0
    previous_recall = ap = 0.0
    points = []
    for threshold in sorted(groups, reverse=True):
        hit, count = groups[threshold]
        tp += hit
        predicted += count
        precision, recall = tp / predicted, tp / positives
        ap += (recall - previous_recall) * precision
        points.append({"threshold": threshold, "precision": precision, "recall": recall,
                       "true_positive": tp, "predicted_positive": predicted})
        previous_recall = recall
    return {"average_precision": ap, "prevalence": positives / len(gold), "points": points}


def calibration(rows, bins=10):
    buckets = [[] for _ in range(bins)]
    brier = 0.0
    for row in rows:
        confidence = row["probabilities"][row["prediction"]]
        correct = int(row["label"] == row["prediction"])
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
        brier += sum((row["probabilities"][label] - int(row["label"] == label)) ** 2 for label in LABELS)
    details, ece = [], 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        confidence = statistics.mean(x[0] for x in bucket)
        accuracy = statistics.mean(x[1] for x in bucket)
        ece += len(bucket) / len(rows) * abs(confidence - accuracy)
        details.append({"lower": index / bins, "upper": (index + 1) / bins,
                        "count": len(bucket), "mean_confidence": confidence, "accuracy": accuracy})
    return {"ece": ece, "bins": details, "multiclass_brier": brier / len(rows),
            "note": "ECE depends on binning; softmax confidence does not identify a causal mechanism."}


def neutral_diagnostics(rows):
    sentences = defaultdict(list)
    for row in rows:
        sentences[(str(row.get("sentence_id", "")), row["text"])].append(row)
    buckets = {name: [] for name in ("all", "with_polar_neighbor", "all_annotated_aspects_neutral")}
    errors, eligible_errors = [], 0
    for row in rows:
        if row["label"] != "neutral":
            continue
        others = [other for other in sentences[(str(row.get("sentence_id", "")), row["text"]) ]
                  if (other["start"], other["end"]) != (row["start"], row["end"])]
        polar = [other for other in others if other["label"] != "neutral"]
        buckets["all"].append(row)
        buckets["with_polar_neighbor" if polar else "all_annotated_aspects_neutral"].append(row)
        if row["prediction"] == "neutral":
            continue
        eligible_errors += bool(polar)
        closest = min(polar, key=lambda other: (abs((other["start"] + other["end"])
                     - (row["start"] + row["end"])), other["start"])) if polar else None
        errors.append({"sentence_id": row.get("sentence_id", ""), "text": row["text"],
                       "aspect": row["aspect"], "start": row["start"], "end": row["end"],
                       "prediction": row["prediction"], "p_neutral": row["probabilities"]["neutral"],
                       "confidence": row["probabilities"][row["prediction"]],
                       "polar_neighbor_labels": ";".join(sorted({p["label"] for p in polar})),
                       "polar_neighbor_count": len(polar),
                       "matches_any_polar_neighbor": any(row["prediction"] == p["label"] for p in polar),
                       "closest_polar_aspect": closest["aspect"] if closest else "",
                       "matches_closest_polar_neighbor": row["prediction"] == closest["label"] if closest else False,
                       "reviewed_reason": "", "reviewer": ""})
    groups = {}
    for name, subset in buckets.items():
        hit = sum(r["prediction"] == "neutral" for r in subset)
        groups[name] = {"support": len(subset), "correct": hit,
                        "recall": hit / len(subset) if subset else None}
    confidence = {}
    for correct, name in ((True, "correct"), (False, "incorrect")):
        values = [r["probabilities"][r["prediction"]] for r in buckets["all"]
                  if (r["prediction"] == "neutral") == correct]
        confidence[name] = {"count": len(values), "median": statistics.median(values) if values else None}
    matching = sum(r["matches_any_polar_neighbor"] for r in errors)
    return {"groups": groups, "confidence_on_gold_neutral": confidence,
            "neutral_errors": len(errors), "errors_with_polar_neighbor": eligible_errors,
            "matching_any_neighbor": matching,
            "matching_fraction_of_all_neutral_errors": matching / len(errors) if errors else None,
            "matching_fraction_of_eligible_errors": matching / eligible_errors if eligible_errors else None,
            "note": "Agreement is association, not proof of copying. Uses annotated neighbors only; "
                    "matching either polarity is automatic when both occur. Pure-neutral includes single-aspect sentences."}, errors


def cluster_bootstrap(rows, repeats=1000, seed=42):
    if repeats < 0:
        raise ValueError("bootstrap repetitions must be nonnegative")
    groups = defaultdict(list)
    for row in rows:
        groups[row["text"]].append(row)
    clusters = list(groups.values())
    values = {"macro_f1": [], "neutral_f1": [], "neutral_recall": []}
    rng = random.Random(seed)
    for _ in range(repeats):
        sample = [row for group in rng.choices(clusters, k=len(clusters)) for row in group]
        # An absent neutral class does not define its recall. Record dropped draws.
        if not any(r["label"] == "neutral" for r in sample):
            continue
        metrics = classification(sample)
        values["macro_f1"].append(metrics["macro_f1"])
        values["neutral_f1"].append(metrics["per_class"]["neutral"]["f1"])
        values["neutral_recall"].append(metrics["per_class"]["neutral"]["recall"])
    def interval(samples):
        if not samples:
            return None
        ordered = sorted(samples)
        def quantile(p):
            position = p * (len(ordered) - 1)
            low = int(position)
            high = min(low + 1, len(ordered) - 1)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        return {"lower": quantile(.025), "upper": quantile(.975)}
    return {"method": "percentile bootstrap grouped by exact sentence text", "seed": seed,
            "requested_draws": repeats, "valid_draws": len(values["macro_f1"]),
            "clusters": len(clusters), "confidence_level": .95,
            "intervals": {name: interval(sample) for name, sample in values.items()},
            "limitation": "Sampling uncertainty for this fixed model, not training-seed or split uncertainty; "
                          "different sentences from the same review are not clustered together."}
