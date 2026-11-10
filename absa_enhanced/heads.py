"""Sentiment heads that differ only in how the three labels are parameterised.

The corpus defines ``neutral`` as "no opinion expressed about this aspect", which is a
different question from "which polarity". A plain three-way softmax forces the two
questions to compete inside one decision boundary; the alternatives below separate them.

``softmax``  three categorical logits, the original baseline.
``ordinal``  two cumulative logits, encoding negative < neutral < positive (CORAL).
``gate``     one opinion-presence logit plus two polarity logits. Neutral is predicted
             when the aspect carries no opinion; polarity is only asked of aspects that
             do. This mirrors the annotation rule instead of flattening it.
"""
import torch
from torch.nn import functional as F

HEADS = ("softmax", "ordinal", "gate")
NEUTRAL = 1  # SENTIMENTS == ("negative", "neutral", "positive")


def output_dim(kind):
    if kind not in HEADS:
        raise ValueError(f"Unknown sentiment head: {kind}")
    return 2 if kind == "ordinal" else 3


def probabilities(kind, logits, offset=1.0):
    """Map raw head outputs to three normalised class probabilities.

    ``offset`` multiplies P(neutral) before renormalising, which is the same as adding
    log(offset) to the neutral logit. Neutral is systematically under-predicted on this
    corpus, so a value tuned on validation is the cheapest calibration; 1.0 is off.
    """
    if kind not in HEADS:
        raise ValueError(f"Unknown sentiment head: {kind}")
    logits = logits.float()
    if kind == "softmax":
        values = logits.softmax(-1)
    elif kind == "ordinal":
        # Multiplicative threshold form (CORAL): always normalised and ordered, even
        # when the two binary decisions disagree.
        at_least_neutral = logits[:, 0].sigmoid()
        at_least_positive = (logits[:, 0] + logits[:, 1]).sigmoid()
        values = torch.stack((1 - at_least_neutral,
                              at_least_neutral * (1 - at_least_positive),
                              at_least_neutral * at_least_positive), -1)
    else:
        presence = logits[:, 0].sigmoid()
        polarity = logits[:, 1:].softmax(-1)
        # Label order is (negative, neutral, positive): neutral is "no opinion expressed".
        values = torch.cat((presence.unsqueeze(-1) * polarity[:, :1],
                            (1 - presence).unsqueeze(-1),
                            presence.unsqueeze(-1) * polarity[:, 1:]), -1)
    if offset != 1.0:
        values = values * values.new_tensor([1.0, offset, 1.0])
        values = values / values.sum(-1, keepdim=True)
    return values


def loss(kind, logits, labels, class_weights=None, smoothing=0.05):
    """Training objective that matches the head's parameterisation."""
    if kind not in HEADS:
        raise ValueError(f"Unknown sentiment head: {kind}")
    logits, labels = logits.float(), labels
    if kind == "softmax":
        return F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=smoothing)
    weights = None if class_weights is None else class_weights[labels]
    if kind == "ordinal":
        cumulative = torch.stack((logits[:, 0], logits[:, 0] + logits[:, 1]), -1)
        thresholds = torch.stack(((labels >= 1).float(), (labels >= 2).float()), -1)
        thresholds = thresholds * (1 - smoothing) + smoothing / 2
        elementwise = F.binary_cross_entropy_with_logits(cumulative, thresholds, reduction="none")
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
