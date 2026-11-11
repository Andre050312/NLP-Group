"""Compare control vs no-CLS across training seeds on a fixed split.

Reads saved run directories only; no GPU and no retraining.
"""
import json
import statistics
from pathlib import Path

CONFIGS = {"control": "pair_ctrl", "no_cls": "pair_nocls"}
SEEDS = (42, 43, 44)
# The original control run predates this naming scheme.
RUN_ALIASES = {("pair_ctrl", 42): "pair_heads_softmax_s42"}
ANALYSIS_ALIASES = {("pair_ctrl", 42): "control"}


def run_name(prefix, seed):
    return RUN_ALIASES.get((prefix, seed), f"{prefix}_s{seed}")


def analysis_name(prefix, seed):
    return ANALYSIS_ALIASES.get((prefix, seed), f"{prefix}_s{seed}")


def test_metrics(prefix, seed):
    """Prefer the neutral_diagnostics analysis; fall back to the run's own metrics."""
    name = run_name(prefix, seed)
    analysed = Path(f"runs/analysis_{analysis_name(prefix, seed)}_test/analysis.json")
    if analysed.is_file():
        payload = json.loads(analysed.read_text(encoding="utf-8"))
        result = {"accuracy": payload["classification"]["accuracy"],
                  "macro_f1": payload["classification"]["macro_f1"]}
        for label, values in payload["classification"]["per_class"].items():
            result[f"{label}_f1"] = values["f1"]
            result[f"{label}_recall"] = values["recall"]
        return result
    raw = Path(f"runs/{name}/test_metrics.json")
    if raw.is_file():
        payload = json.loads(raw.read_text(encoding="utf-8"))
        # Run artifacts nest the provided-aspect scores under gold_aspect_sentiment.
        payload = payload.get("gold_aspect_sentiment", payload)
        result = {"accuracy": payload["accuracy"], "macro_f1": payload["macro_f1"]}
        for label, values in payload["per_class"].items():
            result[f"{label}_f1"] = values["f1"]
            result[f"{label}_recall"] = values["recall"]
        return result
    return None


def summary(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


rows = {}
for name, prefix in CONFIGS.items():
    for seed in SEEDS:
        metrics = test_metrics(prefix, seed)
        if metrics:
            rows[(name, seed)] = metrics

print(f"{'config':9s} {'seed':>4s} {'acc':>7s} {'macro':>7s} {'negF1':>7s} {'neuF1':>7s} {'neuR':>7s} {'posF1':>7s}")
for (name, seed), metric in sorted(rows.items()):
    print(f"{name:9s} {seed:>4d} {metric['accuracy']:.4f} {metric['macro_f1']:.4f} "
          f"{metric['negative_f1']:.4f} {metric['neutral_f1']:.4f} {metric['neutral_recall']:.4f} "
          f"{metric['positive_f1']:.4f}")

print()
for name in CONFIGS:
    subset = [m for (n, _), m in rows.items() if n == name]
    if len(subset) < 2:
        print(f"{name}: only {len(subset)} seed(s), cannot estimate spread")
        continue
    parts = []
    for key in ("accuracy", "macro_f1", "neutral_f1", "neutral_recall"):
        mean, std = summary([m[key] for m in subset])
        parts.append(f"{key} {mean:.4f}±{std:.4f}")
    print(f"{name:9s} n={len(subset)} | " + " | ".join(parts))

both = [name for name in CONFIGS if len({s for (n, s) in rows if n == name}) >= 2]
if len(both) == 2 and len({seed for (n, seed) in rows if n == both[0]}) == len({seed for (n, seed) in rows if n == both[1]}):
    seeds = sorted({seed for (_, seed) in rows})
    print("\nper-seed difference (no_cls - control):")
    for key in ("macro_f1", "neutral_f1", "accuracy"):
        deltas = [rows[("no_cls", seed)][key] - rows[("control", seed)][key] for seed in seeds]
        mean, std = statistics.mean(deltas), statistics.stdev(deltas)
        # Paired t with n-1 degrees of freedom; only a rough screen at n=3.
        t = mean / (std / len(deltas) ** 0.5) if std else float("inf")
        print(f"  {key:11s} " + " ".join(f"{d:+.4f}" for d in deltas) +
              f" | mean {mean:+.4f} ± {std:.4f} | paired t={t:+.2f} (n={len(deltas)}, |t|>4.30 needed at p<.05)")
    print("\n  A paired t with 3 seeds needs |t| > 4.30; anything below that stays unresolved.")
else:
    print("\nPaired per-seed comparison needs the same seed set for both configurations.")
