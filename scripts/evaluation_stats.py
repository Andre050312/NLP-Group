"""Small dependency-free helpers for paired evaluation, not independent-run inference."""
from collections import defaultdict
from math import comb, isfinite


def sign_test(deltas, tolerance=1e-12):
    """Exact two-sided binomial sign test, excluding ties (requires independent units)."""
    values = list(deltas)
    if tolerance < 0 or not all(isfinite(v) for v in values):
        raise ValueError("Use finite differences and a nonnegative tie tolerance")
    positive = sum(v > tolerance for v in values)
    negative = sum(v < -tolerance for v in values)
    n = positive + negative
    tail = sum(comb(n, k) for k in range(max(positive, negative), n + 1))
    return {"positive": positive, "negative": negative, "ties": len(values) - n,
            "n_nonzero": n, "p_two_sided": min(1.0, 2 * tail / 2 ** n)}


def common_seeds(config_seeds):
    """Return the intersection explicitly; do not silently discard a whole configuration."""
    sets = [set(s) for s in config_seeds]
    shared = set.intersection(*sets) if sets else set()
    if not shared:
        raise ValueError("No shared training seeds across configurations")
    return sorted(shared)


def sentence_clusters(rows):
    """Conservatively group exact sentence texts; refuse implicit aspect-row fallback."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Sentence-cluster bootstrap requires a nonempty text field")
        groups[text].append(i)
    if not groups:
        raise ValueError("Cannot bootstrap empty predictions")
    return list(groups.values())


def sample_clusters(clusters, rng):
    """Sample G sentence clusters with replacement; retain all aspects in each draw."""
    return [i for _ in clusters for i in rng.choice(clusters)]
