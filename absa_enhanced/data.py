"""Sentence-level annotations, overflow windows and character-span alignment."""
from dataclasses import asdict, dataclass
from collections import Counter
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

from bert_baseline.data import LABELS, read_xml, split_train_validation

SENTIMENTS = LABELS[:3]
BIO = ("O", "B-ASP", "I-ASP")


@dataclass(frozen=True)
class Aspect:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Document:
    uid: str
    text: str
    aspects: tuple

    def to_dict(self):
        return asdict(self)


def read_documents(path):
    """Retain no-aspect sentences and conflict spans for extraction supervision."""
    documents = []
    for index, sentence in enumerate(ET.parse(path).getroot().iter("sentence")):
        text = sentence.findtext("text")
        if text is None:
            raise ValueError(f"Missing text: {path}, sentence {index}")
        annotations = {}
        for term in sentence.findall("./aspectTerms/aspectTerm"):
            start, end = int(term.attrib["from"]), int(term.attrib["to"])
            label = term.attrib["polarity"]
            if label not in LABELS or not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid annotation in {path}, sentence {index}")
            if text[start:end] != term.attrib["term"]:
                raise ValueError(f"Character offset mismatch: {path}, sentence {index}")
            if (start, end) in annotations and annotations[start, end] != label:
                raise ValueError("One span has inconsistent labels")
            annotations[start, end] = label
        aspects = tuple(Aspect(a, b, label) for (a, b), label in sorted(annotations.items()))
        for a, b in zip(aspects, aspects[1:]):
            if b.start < a.end:
                raise ValueError("BIO head cannot represent overlapping annotated aspects")
        documents.append(Document(f"{sentence.attrib.get('id', index)}:{index}", text, aspects))
    if not documents:
        raise ValueError(f"No sentences in {path}")
    if not any(d.aspects for d in documents):
        raise ValueError(f"No aspect-term annotations in {path}; use ATSA XML, not ACSA")
    return documents


def prepare_splits(train_path, validation_path=None, test_path=None, fraction=0.15, seed=42):
    train = read_documents(train_path)
    if validation_path:
        validation = read_documents(validation_path)
    else:
        # Match the original baseline's validation text selection exactly.
        examples, _ = read_xml(train_path)
        _, val_examples = split_train_validation(examples, fraction, seed)
        validation_texts = {x.text for x in val_examples}
        validation = [d for d in train if d.text in validation_texts]
        train = [d for d in train if d.text not in validation_texts]
        # Sentences with no three-class targets remain in training as extraction data.
    test = read_documents(test_path) if test_path else None
    removed = {}
    if test is not None:
        test_texts = {d.text for d in test}
        for name, documents in (("train", train), ("validation", validation)):
            removed[name] = sum(d.text in test_texts for d in documents)
        train = [d for d in train if d.text not in test_texts]
        validation = [d for d in validation if d.text not in test_texts]
    if not train or not validation:
        raise ValueError("Empty training/validation set")
    if {d.text for d in train} & {d.text for d in validation}:
        raise ValueError("Training/validation contain identical texts")
    counts = Counter(a.label for d in train for a in d.aspects if a.label in SENTIMENTS)
    if set(counts) != set(SENTIMENTS):
        raise ValueError(f"Missing training sentiment class: {dict(counts)}")
    return train, validation, test, {"removed_test_duplicate_sentences": removed,
                                   "training_sentiment_counts": dict(counts)}


def trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def align_span(offsets, start, end):
    positions = [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]
    if not positions:
        return None
    # Reject partial aspects at the boundary of an overflow window.
    if offsets[positions[0]][0] > start or offsets[positions[-1]][1] < end:
        return None
    return positions


class WindowDataset(Dataset):
    def __init__(self, documents, tokenizer, max_length=192, stride=64):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer with character offsets is required")
        if max_length < 8 or not 0 <= stride < max_length - tokenizer.num_special_tokens_to_add():
            raise ValueError("Invalid max_length/stride")
        self.documents, self.rows = documents, []
        covered = set()
        non_exact = {}
        for doc_index, document in enumerate(documents):
            encoded = tokenizer(document.text, truncation=True, max_length=max_length,
                                stride=stride, return_overflowing_tokens=True,
                                return_offsets_mapping=True, return_special_tokens_mask=True)
            for window_index, input_ids in enumerate(encoded["input_ids"]):
                offsets = [tuple(x) for x in encoded["offset_mapping"][window_index]]
                special = encoded["special_tokens_mask"][window_index]
                valid = [not s and b > a for s, (a, b) in zip(special, offsets)]
                bio = [0 if v else -100 for v in valid]
                ignored_bio_positions = set()
                targets = []
                for aspect in document.aspects:
                    positions = align_span(offsets, aspect.start, aspect.end)
                    if positions is None:
                        # A partially visible gold span must not become a false O target.
                        for i, (a, b) in enumerate(offsets):
                            if b > a and b > aspect.start and a < aspect.end:
                                ignored_bio_positions.add(i)
                        continue
                    predicted_span = trim_span(document.text, offsets[positions[0]][0], offsets[positions[-1]][1])
                    covered.add((doc_index, aspect.start, aspect.end))
                    if predicted_span == (aspect.start, aspect.end):
                        for j, pos in enumerate(positions):
                            bio[pos] = 1 if j == 0 else 2
                    else:
                        # Some source annotations end inside a subword (e.g. "dress cod"
                        # inside "dress code"). Pool the covering tokens for sentiment,
                        # but do not teach an incorrect expanded extraction boundary.
                        # Keep the ORIGINAL character span for all gold metrics.
                        ignored_bio_positions.update(positions)
                        non_exact[(doc_index, aspect.start, aspect.end)] = {
                            "sentence_id": document.uid, "start": aspect.start, "end": aspect.end,
                            "aspect": document.text[aspect.start:aspect.end],
                            "covering_token_text": document.text[predicted_span[0]:predicted_span[1]]}
                    if aspect.label in SENTIMENTS:
                        targets.append((positions[0], positions[-1] + 1, SENTIMENTS.index(aspect.label),
                                        aspect.start, aspect.end))
                for pos in ignored_bio_positions:
                    bio[pos] = -100
                model_inputs = {key: encoded[key][window_index]
                                for key in tokenizer.model_input_names if key in encoded}
                self.rows.append({"inputs": model_inputs, "valid": valid, "bio": bio,
                                  "targets": targets, "offsets": offsets, "doc_index": doc_index})
        expected = {(i, a.start, a.end) for i, d in enumerate(documents) for a in d.aspects}
        if expected - covered:
            raise ValueError("Some aspects do not fit any window. Increase --max-length or --stride")
        self.alignment_stats = {"gold_aspects": len(expected),
                                "non_exact_token_spans": len(non_exact),
                                "non_exact_examples": list(non_exact.values())[:10]}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        inputs = self.tokenizer.pad([r["inputs"] for r in rows], padding=True, return_tensors="pt")
        length = inputs["input_ids"].shape[1]
        # All supported encoders use right padding in this project.
        valid = torch.tensor([r["valid"] + [False]*(length-len(r["valid"])) for r in rows])
        bio = torch.tensor([r["bio"] + [-100]*(length-len(r["bio"])) for r in rows])
        spans, labels = [], []
        for batch_index, row in enumerate(rows):
            for start, end, label, _, _ in row["targets"]:
                spans.append((batch_index, start, end))
                labels.append(label)
        return {"inputs": inputs, "valid": valid, "bio": bio,
                "spans": torch.tensor(spans, dtype=torch.long).reshape(-1, 3),
                "labels": torch.tensor(labels, dtype=torch.long), "rows": rows}
