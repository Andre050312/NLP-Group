"""Canonical sentiment axes; BIO labels are independent and remain unchanged."""
from bert_baseline.data import LABELS

THREE_LABELS = LABELS[:3]
FOUR_LABELS = LABELS


def sentiment_labels(value=THREE_LABELS):
    if isinstance(value, str):
        if value not in ("three", "four"):
            raise ValueError("labels must be 'three' or 'four'")
        return FOUR_LABELS if value == "four" else THREE_LABELS
    labels = tuple(value)
    if labels not in (THREE_LABELS, FOUR_LABELS):
        raise ValueError(f"Expected canonical labels {THREE_LABELS} or {FOUR_LABELS}, got {labels}")
    return labels
