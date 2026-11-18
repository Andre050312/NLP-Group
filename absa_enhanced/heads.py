"""Three/four-class softmax and optional heads restricted to three sentiment labels.

The corpus defines ``neutral`` as "no opinion expressed about this aspect", which is a
different question from "which polarity". A plain three-way softmax forces the two
questions to compete inside one decision boundary; the alternatives below separate them.

``softmax``  three categorical logits by default, or four with explicit configuration.
``ordinal``  legacy cumulative training / conditional inference, retained for compatibility.
``ordered``  consistent ordered cumulative head, not a reproduction of CORAL.
``gate``     one opinion-presence logit plus two polarity logits. Neutral is predicted
             when the aspect carries no opinion; polarity is only asked of aspects that
             do. This mirrors the annotation rule instead of flattening it.
"""
import torch
from torch.nn import functional as F

HEADS = ("softmax", "ordinal", "ordered", "gate")
NEUTRAL = 1  # Same index on both canonical sentiment axes.


def output_dim(kind, num_classes=3):
    if kind not in HEADS:
        raise ValueError(f"Unknown sentiment head: {kind}")
    if num_classes not in (3, 4):
        raise ValueError("Expected three or four sentiment classes")
    if num_classes == 4 and kind != "softmax":
        raise ValueError("Four-class sentiment requires softmax; gate, ordinal and ordered "
                         "assume three classes and cannot represent conflict")
    return 2 if kind in ("ordinal", "ordered") else num_classes


def ordered_logits(logits):
    return torch.stack((logits[:, 0], logits[:, 0] - F.softplus(logits[:, 1])), -1)


def validate_logits(kind, logits, num_classes=None):
    if num_classes is None:
        num_classes = logits.shape[-1] if kind == "softmax" or logits.shape[-1] == 4 else 3
    if logits.shape[-1] != output_dim(kind, num_classes):
        raise ValueError("Sentiment logits do not match the configured head and labels")
    return num_classes


def probabilities(kind, logits, offset=1.0, num_classes=None):
    """Map raw outputs to probabilities on the configured sentiment axis.

    ``offset`` multiplies P(neutral) before renormalising, which is the same as adding
    log(offset) to the neutral logit. This changes an operating point; tune on
    validation only. It does not establish improved probability calibration.
    """
    if kind not in HEADS:
        raise ValueError(f"Unknown sentiment head: {kind}")
    validate_logits(kind, logits, num_classes)
    logits = logits.float()
    if kind == "softmax":
        values = logits.softmax(-1)
    elif kind == "ordinal":
        # Preserve legacy checkpoint behavior; not a consistent cumulative head.
        at_least_neutral = logits[:, 0].sigmoid()
        at_least_positive = (logits[:, 0] + logits[:, 1]).sigmoid()
        values = torch.stack((1 - at_least_neutral,
                              at_least_neutral * (1 - at_least_positive),
                              at_least_neutral * at_least_positive), -1)
    elif kind == "ordered":
        q = ordered_logits(logits).sigmoid()
        values = torch.stack((1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1]), -1)
    else:
        presence = logits[:, 0].sigmoid()
        polarity = logits[:, 1:].softmax(-1)
        # Label order is (negative, neutral, positive): neutral is "no opinion expressed".
        values = torch.cat((presence.unsqueeze(-1) * polarity[:, :1],
                            (1 - presence).unsqueeze(-1),
                            presence.unsqueeze(-1) * polarity[:, 1:]), -1)
    if offset != 1.0:
        weights = values.new_ones(values.shape[-1])
        weights[NEUTRAL] = offset
        values = values * weights
        values = values / values.sum(-1, keepdim=True)
    return values


def loss(kind, logits, labels, class_weights=None, smoothing=0.05, num_classes=None):
    """Training objective; ordinal preserves the legacy inference mismatch."""
    if kind not in HEADS:
        raise ValueError(f"Unknown sentiment head: {kind}")
    num_classes = validate_logits(kind, logits, num_classes)
    if labels.numel() and ((labels < 0).any() or (labels >= num_classes).any()):
        raise ValueError("Target label is outside the configured sentiment classes")
    logits, labels = logits.float(), labels
    if kind == "softmax":
        return F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=smoothing)
    weights = None if class_weights is None else class_weights[labels]
    if kind in ("ordinal", "ordered"):
        cumulative = (ordered_logits(logits) if kind == "ordered" else
                      torch.stack((logits[:, 0], logits[:, 0] + logits[:, 1]), -1))
        thresholds = torch.stack(((labels >= 1).float(), (labels >= 2).float()), -1)
        thresholds = thresholds * (1 - smoothing) + smoothing / 2
        elementwise = F.binary_cross_entropy_with_logits(cumulative, thresholds, reduction="none")
        if weights is not None and kind == "ordered":
            return (elementwise.mean(-1) * weights).sum() / weights.sum().clamp_min(1e-6)
        if weights is not None:
            return (elementwise * weights.unsqueeze(-1)).sum() / weights.sum().clamp_min(1e-6)
        return elementwise.mean()
    presence_target = (labels != NEUTRAL).float() * (1 - smoothing) + smoothing / 2
    elementwise = F.binary_cross_entropy_with_logits(logits[:, 0], presence_target, reduction="none")
    polar = labels != NEUTRAL
    if polar.any():
        polarity_target = (labels[polar] == 2).long()
        polar_weights = None
        if class_weights is not None:
            # Class weights for the two polar classes only.
            polar_weights = class_weights[[0, 2]]
        polarity_element = F.cross_entropy(logits[polar, 1:], polarity_target, weight=polar_weights,
                                           label_smoothing=smoothing, reduction="none")
        elementwise = elementwise.index_add(0, polar.nonzero(as_tuple=True)[0], polarity_element)
    if weights is not None:
        return (elementwise * weights).sum() / weights.sum().clamp_min(1e-6)
    return elementwise.mean()
