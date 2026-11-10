"""Grouped outer CV with a separate inner holdout for checkpoint selection."""
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from bert_baseline.data import classification_metrics
from .data import SENTIMENTS, read_documents

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def review_key(document):
    # read_documents appends ':row_index' to the original sentence ID.
    identifier = document.uid.split("::", 1)[-1].rsplit(":", 1)[0]
    if ":" not in identifier:
        raise ValueError("Review grouping requires IDs of the form review_id:sentence_id; use --group-by sentence")
    return identifier.rsplit(":", 1)[0]


def group_documents(documents, by_review=False):
    """Union equal texts and, optionally, all sentences from the same review."""
    parents = list(range(len(documents)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    texts, reviews = {}, {}
    for i, document in enumerate(documents):
        if document.text in texts:
            union(i, texts[document.text])
        texts[document.text] = i
        if by_review:
            key = review_key(document)
            if key in reviews:
                union(i, reviews[key])
            reviews[key] = i
    return [find(i) for i in range(len(documents))]


def label_counts(documents):
    return Counter(a.label for d in documents for a in d.aspects if a.label in SENTIMENTS)


def grouped_outer_folds(documents, folds=5, seed=42, by_review=False):
    if folds < 2:
        raise ValueError("folds must be at least 2")
    groups = group_documents(documents, by_review)
    if len(set(groups)) < folds:
        raise ValueError("Not enough independent groups for this number of folds")
    # Expand to aspect rows so the splitter sees the full class distribution,
    # rather than forcing a mixed-sentiment sentence into a single class.
    row_document, row_labels, row_groups = [], [], []
    for i, document in enumerate(documents):
        labels = [SENTIMENTS.index(a.label) for a in document.aspects if a.label in SENTIMENTS]
        for label in labels or [3]:  # stratification-only bucket for background sentences
            row_document.append(i)
            row_labels.append(label)
            row_groups.append(groups[i])
    for label in range(3):
        if len({g for g, y in zip(row_groups, row_labels) if y == label}) < folds:
            raise ValueError(f"Too few independent groups containing {SENTIMENTS[label]} for {folds} folds")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for train_rows, held_rows in splitter.split(np.zeros((len(row_labels), 1), dtype=np.uint8), row_labels, row_groups):
        training = sorted({row_document[i] for i in train_rows})
        held = sorted({row_document[i] for i in held_rows})
        if {groups[i] for i in training} & {groups[i] for i in held}:
            raise ValueError("Outer group leakage")
        for indices in (training, held):
            if set(label_counts([documents[i] for i in indices])) != set(SENTIMENTS):
                raise ValueError("A fold is missing a sentiment class; reduce --folds or change --split-seed")
        result.append((training, held))
    return result, groups


def inner_holdout(documents, indices, groups, fraction=0.15, seed=42):
    """Choose a grouped, approximately balanced inner holdout without outer data."""
    if not 0 < fraction < 1:
        raise ValueError("validation-fraction must be in (0, 1)")
    local_groups = [groups[i] for i in indices]
    if len(set(local_groups)) < 2:
        raise ValueError("Not enough groups for an independent inner validation set")
    totals = label_counts([documents[i] for i in indices])
    candidates = GroupShuffleSplit(n_splits=32, test_size=fraction, random_state=seed)
    best = None
    for local_train, local_val in candidates.split(np.zeros((len(indices), 1), dtype=np.uint8), groups=local_groups):
        training = [indices[i] for i in local_train]
        validation = [indices[i] for i in local_val]
        a, b = label_counts([documents[i] for i in training]), label_counts([documents[i] for i in validation])
        if set(a) != set(SENTIMENTS) or set(b) != set(SENTIMENTS):
            continue
        deviation = sum(abs(b[label] / totals[label] - fraction) for label in SENTIMENTS)
        key = deviation, tuple(validation)
        if best is None or key < best[0]:
            best = key, sorted(training), sorted(validation)
    if best is None:
        raise ValueError("Cannot create an inner validation set with all classes; change fold count or validation fraction")
    return best[1], best[2]


def write_documents(path, documents):
    root = ET.Element("sentences")
    for document in documents:
        sentence = ET.SubElement(root, "sentence", {"id": document.uid})
        ET.SubElement(sentence, "text").text = document.text
        aspects = ET.SubElement(sentence, "aspectTerms")
        for aspect in document.aspects:
            ET.SubElement(aspects, "aspectTerm", {"term": document.text[aspect.start:aspect.end],
                                                  "from": str(aspect.start), "to": str(aspect.end),
                                                  "polarity": aspect.label})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_plan(args, train_path, validation_path, test_path):
    by_review = args.group_by == "review" or (args.group_by == "auto" and args.dataset in ("restaurant2015", "restaurant2016"))
    sources = [("train", Path(train_path))]
    if args.include_validation:
        if validation_path is None:
            raise ValueError("--include-validation requires an existing validation source")
        sources.append(("validation", Path(validation_path)))
    documents = []
    source_hashes = {}
    for name, path in sources:
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        documents.extend(replace(d, uid=f"{name}::{d.uid}") for d in read_documents(path))
    official = read_documents(test_path) if test_path else []
    official_texts = {d.text for d in official}
    official_reviews = {review_key(d) for d in official} if by_review else set()
    original_count = len(documents)
    documents = [d for d in documents if d.text not in official_texts and
                 (not by_review or review_key(d) not in official_reviews)]
    if test_path:
        source_hashes["reserved_official_test"] = hashlib.sha256(Path(test_path).read_bytes()).hexdigest()
    seed = args.seed if args.split_seed is None else args.split_seed
    outer, groups = grouped_outer_folds(documents, args.folds, seed, by_review)
    plan = []
    for number, (outer_training, held) in enumerate(outer, 1):
        training, validation = inner_holdout(documents, outer_training, groups, args.validation_fraction, seed + number)
        group_sets = [{groups[i] for i in indices} for indices in (training, validation, held)]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Inner/outer group leakage")
        plan.append({"fold": number, "training": training, "validation": validation, "held_out": held})
    audit = {"group_by": "review_and_exact_text" if by_review else "exact_text",
             "split_seed": seed, "training_seed": args.seed,
             "inner_validation_seed_policy": "split_seed + one_based_fold_number",
             "pool_sentences": len(documents), "pool_aspect_targets": sum(label_counts(documents).values()),
             "independent_groups": len(set(groups)), "excluded_reserved_overlap_sentences": original_count - len(documents),
             "official_test_evaluated": False, "include_original_validation": args.include_validation,
             "ignored_validation_source": str(validation_path) if validation_path and not args.include_validation else None}
    return documents, groups, plan, audit, source_hashes


def training_command(args, split_dir, run_dir):
    command = [sys.executable, "-m", "absa_enhanced", "train", "--dataset", "custom",
               "--train", str(split_dir / "train.xml"), "--validation", str(split_dir / "validation.xml"),
               "--test", str(split_dir / "held_out.xml"), "--output", str(run_dir)]
    parameters = ("encoder", "mode", "sentiment_head", "context_scope", "epochs", "batch_size", "grad_accum",
                  "learning_rate", "head_learning_rate", "weight_decay", "warmup_ratio", "max_length", "stride",
                  "local_radius", "extraction_weight", "class_weighting", "smoothing", "dropout",
                  "selection_metric", "patience", "seed", "device", "precision")
    for name in parameters:
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    for name in ("no_fusion", "no_gradient_checkpointing", "no_cls_feature"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def aggregate(output, documents, plan):
    all_rows, metrics_by_fold = [], []
    expected = {(d.uid, a.start, a.end, a.label) for d in documents for a in d.aspects if a.label in SENTIMENTS}
    observed = set()
    for fold in plan:
        directory = output / f"fold_{fold['fold']:02d}" / "run"
        metrics = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
        metrics_by_fold.append({"fold": fold["fold"], "metrics": metrics})
        held = [documents[i] for i in fold["held_out"]]
        # XML reader adds its local row index to the stable ID written above.
        identifiers = {f"{d.uid}:{i}": d for i, d in enumerate(held)}
        rows = [json.loads(line) for line in (directory / "test_gold_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            document = identifiers[row["sentence_id"]]
            key = document.uid, row["start"], row["end"], row["label"]
            if (key not in expected or key in observed or row["text"] != document.text
                    or row["aspect"] != document.text[row["start"]:row["end"]]):
                raise ValueError("OOF coverage mismatch, duplicate target or changed source text")
            observed.add(key)
            all_rows.append({**row, "source_document_id": document.uid, "fold": fold["fold"]})
        if len(rows) != metrics["gold_aspect_sentiment"]["examples"]:
            raise ValueError("Fold prediction count disagrees with fold metrics")
    if observed != expected:
        raise ValueError(f"Incomplete OOF predictions: {len(observed)}/{len(expected)}")
    stats = {}
    for section, names in (("gold_aspect_sentiment", ("accuracy", "macro_f1")),
                           ("aspect_extraction", ("precision", "recall", "f1")),
                           ("end_to_end", ("precision", "recall", "f1"))):
        if any(item["metrics"].get(section) is None for item in metrics_by_fold):
            continue
        stats[section] = {}
        for name in names:
            values = [item["metrics"][section][name] for item in metrics_by_fold]
            stats[section][name] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "fold_values": values}
    gold = [SENTIMENTS.index(row["label"]) for row in all_rows]
    predictions = [SENTIMENTS.index(row["prediction"]) for row in all_rows]
    summary = {"folds": len(plan), "official_test_evaluated": False,
               "protocol": "Grouped outer evaluation; independent inner validation selects checkpoints. Not full nested hyperparameter search.",
               "fold_statistics": stats, "oof_gold_aspect_sentiment": classification_metrics(gold, predictions, SENTIMENTS),
               "per_fold": metrics_by_fold}
    write_json(output / "cv_summary.json", summary)
    with (output / "oof_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("fold", "accuracy", "macro_f1", "aspect_extraction_f1", "end_to_end_f1", "examples"))
        for item in metrics_by_fold:
            metric = item["metrics"]
            writer.writerow((item["fold"], metric["gold_aspect_sentiment"]["accuracy"],
                             metric["gold_aspect_sentiment"]["macro_f1"],
                             (metric.get("aspect_extraction") or {}).get("f1"),
                             (metric.get("end_to_end") or {}).get("f1"), metric["gold_aspect_sentiment"]["examples"]))
    return summary


def cross_validate(args):
    from .cli import dataset_paths, validate_training_args
    validate_training_args(args)
    train_path, validation_path, test_path = dataset_paths(args)
    documents, groups, plan, audit, hashes = build_plan(args, train_path, validation_path, test_path)
    output = Path(args.output).resolve()
    ignore = {"command", "output", "train", "validation", "test", "data_root", "resume", "plan_only"}
    implementation = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                      for name in ("cli.py", "cross_validation.py", "model.py", "pair.py", "data.py", "evaluation.py", "selection.py")}
    partitions = [{key: [documents[i].uid for i in fold[key]] for key in ("training", "validation", "held_out")}
                  for fold in plan]
    identity = {"parameters": {key: value for key, value in vars(args).items() if key not in ignore},
                "source_hashes": hashes, "implementation_hashes": implementation, "protocol_version": 1,
                "partition_hash": hashlib.sha256(json.dumps(partitions, sort_keys=True).encode()).hexdigest(),
                "versions": {name: version(name) for name in ("torch", "transformers", "scikit-learn", "numpy")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    configuration = {**identity, "fingerprint": fingerprint, "audit": audit}
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError("CV output exists; use a new directory or --resume for a matching run")
        previous = json.loads((output / "cv_config.json").read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration/source mismatch; choose a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "cv_config.json", configuration)
    write_json(output / "fold_assignment.json", [{"document_id": d.uid, "text": d.text, "group": groups[i],
                                                  "outer_fold": next(f["fold"] for f in plan if i in f["held_out"])}
                                                 for i, d in enumerate(documents)])
    for fold in plan:
        directory = output / f"fold_{fold['fold']:02d}"
        split_dir = directory / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for filename, key in (("train", "training"), ("validation", "validation"), ("held_out", "held_out")):
            write_documents(split_dir / f"{filename}.xml", [documents[i] for i in fold[key]])
        manifest = {key: {"documents": [documents[i].uid for i in fold[key]],
                          "label_counts": dict(label_counts([documents[i] for i in fold[key]])),
                          "groups": sorted({groups[i] for i in fold[key]})}
                    for key in ("training", "validation", "held_out")}
        write_json(directory / "fold_manifest.json", manifest)
    print(json.dumps({"output": str(output), "audit": audit, "folds": args.folds}, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        print("Plan written. No tokenizer download or model training was started. Use the same command with --resume and without --plan-only to train.")
        return
    for fold in plan:
        directory = output / f"fold_{fold['fold']:02d}"
        run_dir = directory / "run"
        marker = directory / "completed.json"
        if args.resume and marker.exists():
            completed = json.loads(marker.read_text(encoding="utf-8"))
            if completed.get("fingerprint") != fingerprint:
                raise ValueError("Fold completion fingerprint mismatch")
            for filename, digest in completed["result_hashes"].items():
                if hashlib.sha256((run_dir / filename).read_bytes()).hexdigest() != digest:
                    raise ValueError("Completed fold results were changed or are missing")
            print(f"Skipping completed fold {fold['fold']}", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"Incomplete fold exists at {run_dir}. Step-level resume is unsupported; use a new CV directory or explicitly remove this incomplete run only.")
        print(f"Training outer fold {fold['fold']}/{args.folds}", flush=True)
        subprocess.run(training_command(args, directory / "splits", run_dir), cwd=PROJECT, check=True)
        filenames = ("test_metrics.json", "test_gold_predictions.jsonl")
        write_json(marker, {"fingerprint": fingerprint,
                            "result_hashes": {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest() for name in filenames}})
    summary = aggregate(output, documents, plan)
    print(json.dumps({"fold_statistics": summary["fold_statistics"],
                      "oof_gold_aspect_sentiment": summary["oof_gold_aspect_sentiment"]}, ensure_ascii=False, indent=2))
