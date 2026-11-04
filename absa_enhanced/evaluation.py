"""Exact character-span extraction and joint metrics, separate from gold ATSA."""
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bert_baseline.data import classification_metrics
from .data import Collator, SENTIMENTS, trim_span


def autocast_context(device, precision):
    if precision == "no" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def decode_spans(tags, valid, offsets, text):
    """BIO repair: an orphan I starts a span; O/special tokens close it."""
    spans, start = [], None
    for index in range(len(tags) + 1):
        tag = tags[index] if index < len(tags) and valid[index] else 0
        if start is not None and tag != 2:
            end = index
            char_start, char_end = trim_span(text, offsets[start][0], offsets[end - 1][1])
            if char_start < char_end:
                spans.append((start, end, char_start, char_end))
            start = None
        if tag == 1 or (tag == 2 and start is None):
            start = index
    return spans


def set_metrics(gold, predicted):
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": true_positive, "gold": len(gold), "predicted": len(predicted)}


def move_batch(batch, device):
    return ({k: v.to(device) for k, v in batch["inputs"].items()}, batch["valid"].to(device),
            batch["spans"].to(device), batch["labels"].to(device), batch["bio"].to(device))


def evaluate(model, tokenizer, dataset, device, batch_size=2, precision="no", compute_metrics=True):
    model.eval()
    gold_scores, predicted_scores = defaultdict(list), defaultdict(list)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(tokenizer), num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            inputs, valid, gold_spans, _, _ = move_batch(batch, device)
            with autocast_context(device, precision):
                hidden, bio_logits = model.encode(inputs)
                gold_probabilities = model.classify(hidden, valid, gold_spans).float().softmax(-1).cpu().tolist()
                candidate_spans, candidate_keys = [], []
                tags = bio_logits.argmax(-1).cpu().tolist()
                for b, row in enumerate(batch["rows"]):
                    document = dataset.documents[row["doc_index"]]
                    candidates = decode_spans(tags[b][:len(row["valid"])], row["valid"], row["offsets"], document.text)
                    valid_positions = [i for i, value in enumerate(row["valid"]) if value]
                    for start, end, char_start, char_end in candidates:
                        # Avoid accepting partial spans cut at an internal window boundary.
                        if (start == valid_positions[0] and char_start > 0
                                and document.text[:char_start].strip()):
                            continue
                        if (end - 1 == valid_positions[-1] and document.text[char_end:].strip()):
                            continue
                        candidate_spans.append((b, start, end))
                        candidate_keys.append((row["doc_index"], char_start, char_end))
                predicted_tensor = torch.tensor(candidate_spans, dtype=torch.long, device=device).reshape(-1, 3)
                candidate_probabilities = model.classify(hidden, valid, predicted_tensor).float().softmax(-1).cpu().tolist()
            gold_keys = [(row["doc_index"], start, end) for row in batch["rows"]
                         for _, _, _, start, end in row["targets"]]
            for key, probabilities in zip(gold_keys, gold_probabilities):
                gold_scores[key].append(probabilities)
            for key, probabilities in zip(candidate_keys, candidate_probabilities):
                predicted_scores[key].append(probabilities)

    def average(values):
        return [sum(row[i] for row in values) / len(values) for i in range(3)]

    predicted = {key: average(values) for key, values in predicted_scores.items()}
    gold_rows, sentence_rows, gold_labels, gold_predictions = [], [], [], []
    extraction_gold, pair_gold, conflict_keys = set(), set(), set()
    for doc_index, document in enumerate(dataset.documents):
        for aspect in document.aspects:
            key = (doc_index, aspect.start, aspect.end)
            extraction_gold.add(key)
            if aspect.label == "conflict":
                conflict_keys.add(key)
                continue
            probabilities = average(gold_scores[key])
            label = max(range(3), key=probabilities.__getitem__)
            gold_labels.append(SENTIMENTS.index(aspect.label))
            gold_predictions.append(label)
            pair_gold.add((*key, aspect.label))
            gold_rows.append({"sentence_id": document.uid, "text": document.text,
                              "aspect": document.text[aspect.start:aspect.end], "start": aspect.start,
                              "end": aspect.end, "label": aspect.label, "prediction": SENTIMENTS[label],
                              "probabilities": dict(zip(SENTIMENTS, probabilities))})
        predictions = []
        for (index, start, end), probabilities in sorted(predicted.items()):
            if index == doc_index:
                label = max(range(3), key=probabilities.__getitem__)
                predictions.append({"aspect": document.text[start:end], "start": start, "end": end,
                                    "sentiment": SENTIMENTS[label], "confidence": probabilities[label]})
        sentence_rows.append({"sentence_id": document.uid, "text": document.text,
                              "gold": [a.__dict__ for a in document.aspects], "predictions": predictions})
    metrics = {}
    if compute_metrics:
        if not gold_labels:
            raise ValueError("No three-class sentiment targets to evaluate")
        extraction_predicted = set(predicted)
        pairs = {(*key, SENTIMENTS[max(range(3), key=probabilities.__getitem__)])
                 for key, probabilities in predicted.items() if key not in conflict_keys}
        metrics = {"gold_aspect_sentiment": classification_metrics(gold_labels, gold_predictions, SENTIMENTS),
                   "aspect_extraction": set_metrics(extraction_gold, extraction_predicted),
                   "end_to_end": set_metrics(pair_gold, pairs), "sentences": len(dataset.documents),
                   "windows": len(dataset), "excluded_conflict_targets": len(conflict_keys)}
    return metrics, gold_rows, sentence_rows


def save_results(directory, metrics, gold_rows, sentence_rows, prefix="test"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{prefix}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, rows in ((f"{prefix}_gold_predictions.jsonl", gold_rows),
                           (f"{prefix}_end_to_end.jsonl", sentence_rows)):
        with (directory / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
