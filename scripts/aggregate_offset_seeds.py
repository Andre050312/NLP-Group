"""Aggregate baseline and validation-tuned offset results across training seeds.

Reads only the saved reports from `tune_neutral_offset.py`; no model runs.
"""
import json
import statistics

CONFIGS = {"control": "runs/offset_control.json", "no_cls": "runs/offset_nocls.json"}
SHORT = {"control": "control", "no_cls": "no-cls"}

table = {}
for config, path in CONFIGS.items():
    payload = json.load(open(path, encoding="utf-8"))
    for run, entry in payload.items():
        offsets = {item["offset"]: item for item in entry["table"]}
        base, tuned = offsets[1.0], offsets[entry["chosen_offset"]]
        seed = int(run.rsplit("_s", 1)[-1])
        table[(config, seed)] = {
            "k": entry["chosen_offset"],
            "base": base["test"], "tuned": tuned["test"],
            "val_base": base["validation"], "val_tuned": tuned["validation"],
            "bootstrap": entry.get("bootstrap_vs_offset_1", {}),
        }

seeds = sorted({seed for (_, seed) in table})
configs = [c for c in CONFIGS if all((c, s) in table for s in seeds)]

print(f"{'config':8s} {'seed':>4s} {'k':>4s} {'test macro':>17s} {'test neutralF1':>17s}")
for config in configs:
    for seed in seeds:
        row = table[(config, seed)]
        print(f"{SHORT[config]:8s} {seed:>4d} {row['k']:>4.2f} "
              f"{row['base']['macro_f1']:.4f}->{row['tuned']['macro_f1']:.4f}".rjust(28)
              + f" {row['base']['per_class']['neutral']['f1']:.4f}->"
                f"{row['tuned']['per_class']['neutral']['f1']:.4f}".rjust(18))

print()
for config in configs:
    for mode, metric in (("base", "macro_f1"), ("tuned", "macro_f1"),
                         ("base", "accuracy"), ("tuned", "accuracy")):
        values = [table[(config, s)][mode][metric] for s in seeds]
        print(f"{SHORT[config]:8s} {mode:5s} {metric:11s} "
              f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}")
    neutral = [table[(config, s)][mode]["per_class"]["neutral"]["f1"]
               for s in seeds for mode in ("tuned",)]
    print(f"{SHORT[config]:8s} tuned neutral_f1  {statistics.mean(neutral):.4f} ± {statistics.stdev(neutral):.4f}")

if len(configs) == 2:
    print("\nPaired per-seed difference (no-cls - control), after each config's own tuned offset:")
    for metric in ("macro_f1", "accuracy"):
        deltas = [table[("no_cls", s)]["tuned"][metric] - table[("control", s)]["tuned"][metric] for s in seeds]
        mean, std = statistics.mean(deltas), statistics.stdev(deltas)
        t = mean / (std / len(deltas) ** 0.5) if std else float("inf")
        print(f"  {metric:11s} " + " ".join(f"{d:+.4f}" for d in deltas) +
              f" | mean {mean:+.4f} ± {std:.4f} | t={t:+.2f}")
    deltas = [table[("no_cls", s)]["tuned"]["per_class"]["neutral"]["f1"]
              - table[("control", s)]["tuned"]["per_class"]["neutral"]["f1"] for s in seeds]
    mean, std = statistics.mean(deltas), statistics.stdev(deltas)
    t = mean / (std / len(deltas) ** 0.5) if std else float("inf")
    print(f"  neutral_f1  " + " ".join(f"{d:+.4f}" for d in deltas) +
          f" | mean {mean:+.4f} ± {std:.4f} | t={t:+.2f}")
    print("\n  |t| must exceed 4.30 for p<.05 with 3 paired seeds; smaller values are unresolved.")
