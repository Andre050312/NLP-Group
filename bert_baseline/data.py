"""Raw XML ingestion and sentence-grouped splits; no ML dependencies required."""
from dataclasses import asdict, dataclass
from collections import Counter
import random
import xml.etree.ElementTree as ET

LABELS = ("negative", "neutral", "positive", "conflict")

@dataclass(frozen=True)
class Example:
    sentence_id: str
    text: str
    aspect: str
    start: int
    end: int
    label: str

    def to_dict(self):
        return asdict(self)


def read_xml(path, include_conflict=False):
    """Read SemEval/MAMS ATSA. Fail on inconsistent offsets rather than hide errors."""
    examples = []
    stats = Counter()
    for sentence in ET.parse(path).getroot().iter("sentence"):
        stats["sentences"] += 1
        text = sentence.findtext("text")
        if text is None:
            raise ValueError(f"Missing text in {path}")
        for term in sentence.findall("./aspectTerms/aspectTerm"):
            stats["aspects_total"] += 1
            label = term.attrib["polarity"]
            if label not in LABELS:
                raise ValueError(f"Unknown sentiment: {label}")
            start, end = int(term.attrib["from"]), int(term.attrib["to"])
            aspect = term.attrib["term"]
            if not (0 <= start < end <= len(text)) or text[start:end] != aspect:
                raise ValueError(f"Invalid aspect span in sentence {sentence.attrib.get('id')}: {aspect!r}")
            if label == "conflict" and not include_conflict:
                stats["excluded_conflict"] += 1
                continue
            examples.append(Example(sentence.attrib.get("id", ""), text, aspect, start, end, label))
            stats[f"label_{label}"] += 1
    stats["examples"] = len(examples)
    if not examples:
        raise ValueError(f"No aspect-term examples in {path}; use ATSA, not ACSA XML.")
    return examples, dict(stats)


def split_train_validation(examples, fraction=0.15, seed=42):
    """Group by exact text, including duplicated sentences with different IDs."""
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    texts = sorted({x.text for x in examples})
    if len(texts) < 2:
        raise ValueError("Need at least two distinct sentences")
    random.Random(seed).shuffle(texts)
    count = max(1, min(len(texts)-1, round(len(texts)*fraction)))
    validation_texts = set(texts[:count])
    train = [x for x in examples if x.text not in validation_texts]
    validation = [x for x in examples if x.text in validation_texts]
    return train, validation


def ensure_disjoint(**splits):
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            overlap = {x.text for x in splits[left]} & {x.text for x in splits[right]}
            if overlap:
                raise ValueError(f"Sentence overlap between {left} and {right}: {len(overlap)}. "
                                 "Resolve the dataset overlap before training.")


def classification_metrics(gold, predictions, labels):
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Expected equally sized, nonempty label sequences")
    n = len(labels)
    matrix = [[0]*n for _ in range(n)]
    for y, p in zip(gold, predictions):
        if not (0 <= y < n and 0 <= p < n):
            raise ValueError("Label ID outside configured label set")
        matrix[y][p] += 1
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp/predicted if predicted else 0.0
        recall = tp/support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {"accuracy": sum(matrix[i][i] for i in range(n))/len(gold),
            "macro_f1": sum(x["f1"] for x in per_class.values())/n,
            "per_class": per_class, "confusion_matrix": matrix,
            "label_order": list(labels), "examples": len(gold)}
