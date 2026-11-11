"""Ranking quality (neutral AP and best-threshold F1) per configuration and seed.

AP and the best achievable F1 depend only on the score ordering, so they separate
"better ranking" from "better calibrated decision boundary".
"""
import json
from pathlib import Path
import statistics

RUNS = {
    ("control", 42): "analysis_control", ("control", 43): "analysis_pair_ctrl_s43_test",
    ("control", 44): "analysis_pair_ctrl_s44_test",
    ("no_cls", 42): "analysis_no_cls", ("no_cls", 43): "analysis_nocls_s43_test",
    ("no_cls", 44): "analysis_nocls_s44_test",
}


def curve_metrics(analysis_dir):
    payload = json.loads((Path("runs") / analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    average_precision = payload["neutral_ranking"]["probability"]["average_precision"]
    best = 0.0
    for row in (Path("runs") / analysis_dir / "neutral_pr_probability.csv").read_text(
            encoding="utf-8-sig").splitlines()[1:]:
        threshold, precision, recall = (float(x) for x in row.split(",")[:3])
        if precision + recall:
            best = max(best, 2 * precision * recall / (precision + recall))
    return average_precision, best, payload["classification"]["per_class"]["neutral"]["f1"]


print(f"{'config':8s} {'seed':>4s} {'neutralF1':>10s} {'AP':>7s} {'bestF1@thr':>11s}")
rows = {}
for key, directory in sorted(RUNS.items()):
    ap, best, at_argmax = curve_metrics(directory)
    rows[key] = (at_argmax, ap, best)
    print(f"{key[0]:8s} {key[1]:>4d} {at_argmax:.4f} {ap:.4f} {best:.4f}")

print()
for config in ("control", "no_cls"):
    subset = [v for k, v in rows.items() if k[0] == config]
    print(f"{config:8s} neutralF1 {statistics.mean(v[0] for v in subset):.4f} | "
          f"AP {statistics.mean(v[1] for v in subset):.4f} | "
          f"bestF1 {statistics.mean(v[2] for v in subset):.4f}")

print("\npaired AP difference (no_cls - control):")
deltas = []
for seed in (42, 43, 44):
    delta = rows[("no_cls", seed)][1] - rows[("control", seed)][1]
    deltas.append(delta)
    print(f"  seed {seed}: {delta:+.4f}")
print(f"  mean {statistics.mean(deltas):+.4f} ± {statistics.stdev(deltas):.4f}")
