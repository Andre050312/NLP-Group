"""Audit actual label support and current reader/split behavior, without a model."""
from argparse import Namespace
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .metrics import LABELS


def raw_summary(path):
    documents, overlaps, counts = 0, [], Counter()
    overlapping_sentences = 0
    for sentence in ET.parse(path).getroot().iter("sentence"):
        documents += 1
        text = sentence.findtext("text")
        if text is None:
            raise ValueError("Sentence has no text")
        spans = {}
        for term in sentence.findall("./aspectTerms/aspectTerm"):
            start, end = int(term.attrib["from"]), int(term.attrib["to"])
            label = term.attrib["polarity"]
            if label not in (*LABELS, "conflict") or not 0 <= start < end <= len(text) or text[start:end] != term.attrib["term"]:
                raise ValueError("Invalid label or source character offset")
            if (start, end) in spans and spans[start, end] != label:
                raise ValueError("One source span has conflicting labels")
            spans[start, end] = label
        counts.update(spans.values())
        ordered = sorted(spans)
        hits = [(a, b) for a, b in zip(ordered, ordered[1:]) if b[0] < a[1]]
        overlapping_sentences += bool(hits)
        for a, b in hits:
            if len(overlaps) < 5:
                overlaps.append({"sentence_id": sentence.attrib.get("id", ""), "text": text,
                                 "left": {"start": a[0], "end": a[1], "aspect": text[a[0]:a[1]], "label": spans[a]},
                                 "right": {"start": b[0], "end": b[1], "aspect": text[b[0]:b[1]], "label": spans[b]}})
    return {"sentences": documents, "unique_span_label_counts": dict(counts),
            "sentences_with_overlapping_spans": overlapping_sentences, "overlap_examples": overlaps,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def split_summary(documents):
    counts = Counter(a.label for doc in documents for a in doc.aspects if a.label in LABELS)
    mixed = both = single = 0
    for doc in documents:
        neutral = sum(a.label == "neutral" for a in doc.aspects)
        polar = {a.label for a in doc.aspects if a.label in ("negative", "positive")}
        mixed += neutral * bool(polar)
        both += neutral * (len(polar) == 2)
        single += neutral * (len(doc.aspects) == 1)
    return {"sentences": len(documents), "unique_texts": len({d.text for d in documents}),
            "targets": sum(counts.values()), "label_counts": {label: counts[label] for label in LABELS},
            "neutral_with_polar_neighbor": mixed, "neutral_without_polar_neighbor": counts["neutral"] - mixed,
            "neutral_with_both_polarities": both, "neutral_single_aspect": single,
            "neutral_recall_step": 1 / counts["neutral"] if counts["neutral"] else None}


def audit_dataset(dataset, data_root="Database", split_seed=42, validation_fraction=.15):
    from absa_enhanced.cli import dataset_paths
    from absa_enhanced.data import prepare_splits
    from absa_enhanced.cross_validation import review_key
    args = Namespace(dataset=dataset, data_root=str(data_root), train=None, validation=None, test=None)
    paths = dataset_paths(args)
    result = {"dataset": dataset, "split_seed": split_seed, "validation_fraction": validation_fraction,
              "source_paths": {name: str(path) if path is not None else None
                               for name, path in zip(("train", "validation", "test"), paths)}, "sources": {}}
    try:
        for name, path in zip(("train", "validation", "test"), paths):
            if path is not None:
                result["sources"][name] = raw_summary(path)
        train, validation, test, audit = prepare_splits(*paths, fraction=validation_fraction, seed=split_seed)
    except (ValueError, OSError, ET.ParseError) as error:
        result.update(status="blocked_by_current_reader_or_data", error=str(error))
        return result
    sets = {"train": train, "validation": validation, "test": test}
    result.update(status="current_split_reproduced", splits={name: split_summary(docs) for name, docs in sets.items()},
                  training_split_audit=audit)
    overlaps = {}
    names = tuple(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlaps[left + "_" + right] = len({d.text for d in sets[left]} & {d.text for d in sets[right]})
    result["exact_text_overlap"] = overlaps
    if dataset in ("restaurant2015", "restaurant2016"):
        groups = {name: {review_key(d) for d in docs} for name, docs in sets.items()}
        result["shared_review_ids"] = {left + "_" + right: len(groups[left] & groups[right])
                                       for i, left in enumerate(names) for right in names[i + 1:]}
    else:
        result["shared_review_ids"] = None
    return result


def audit_data(args):
    from .cli import fresh_output, write_csv, write_json
    results = [audit_dataset(name, args.data_root, args.split_seed, args.validation_fraction) for name in args.datasets]
    report = {"schema_version": 1, "purpose": "Label support and split/reader audit only; no model performance measured",
              "datasets": results,
              "limits": ["Uses current prepare_splits behavior; historical runs may use another seed or protocol.",
                         "Shared review IDs describe review dependence, not measured metric inflation.",
                         "No gold labels are changed and no hyperparameter is selected using test labels."]}
    table = []
    for item in results:
        if "splits" not in item:
            continue
        for split, values in item["splits"].items():
            table.append({"dataset": item["dataset"], "split": split, "sentences": values["sentences"],
                          "targets": values["targets"], **values["label_counts"],
                          "neutral_with_polar_neighbor": values["neutral_with_polar_neighbor"],
                          "neutral_without_polar_neighbor": values["neutral_without_polar_neighbor"],
                          "neutral_recall_step": values["neutral_recall_step"]})
    output = fresh_output(args.output)
    write_json(output / "data_audit.json", report)
    if table:
        write_csv(output / "split_support.csv", table, list(table[0]))
    lines = ["# Label support and split audit", "", "No trained model was evaluated. Counts are aspect targets, not sentence labels.", "",
             "| Dataset | Split | Negative | Neutral | Positive | Neutral with polar neighbor | Neutral without polar neighbor |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['dataset']} | {row['split']} | {row['negative']} | {row['neutral']} | {row['positive']} | "
                     f"{row['neutral_with_polar_neighbor']} | {row['neutral_without_polar_neighbor']} |")
    lines += ["", "## Reader and split findings", ""]
    for item in results:
        if "error" in item:
            lines.append(f"- {item['dataset']}: BLOCKED by current reader/data: {item['error']}")
        elif item["shared_review_ids"] is not None:
            shared = item["shared_review_ids"]["train_validation"]
            lines.append(f"- {item['dataset']}: train/validation share {shared} review IDs. Existing review-grouped CV avoids this holdout dependency.")
    lines += ["", "Complete source hashes, overlap examples and recall step sizes are in data_audit.json.",
              "A successful audit command does not imply every dataset is trainable: inspect each dataset status.",
              "These counts neither verify historical predictions nor establish a dataset performance ceiling."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "datasets": [{"dataset": r["dataset"], "status": r["status"]} for r in results]}, indent=2))
