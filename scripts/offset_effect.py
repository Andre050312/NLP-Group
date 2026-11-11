"""Effect of the validation-tuned neutral offset on each run, per configuration."""
import json
import statistics

for name, path in (("control", "runs/offset_control.json"), ("no_cls", "runs/offset_nocls.json")):
    payload = json.load(open(path, encoding="utf-8"))
    deltas_macro, deltas_neutral, deltas_acc = [], [], []
    for run, entry in sorted(payload.items()):
        table = {item["offset"]: item for item in entry["table"]}
        base, tuned = table[1.0]["test"], table[entry["chosen_offset"]]["test"]
        deltas_macro.append(tuned["macro_f1"] - base["macro_f1"])
        deltas_acc.append(tuned["accuracy"] - base["accuracy"])
        deltas_neutral.append(tuned["per_class"]["neutral"]["f1"] - base["per_class"]["neutral"]["f1"])
    for metric, deltas in (("macro_f1", deltas_macro), ("neutral_f1", deltas_neutral), ("accuracy", deltas_acc)):
        mean, std = statistics.mean(deltas), statistics.stdev(deltas)
        t = mean / (std / len(deltas) ** 0.5) if std else float("inf")
        print(f"{name:8s} {metric:11s} " + " ".join(f"{d:+.4f}" for d in deltas) +
              f" | mean {mean:+.4f} ± {std:.4f} | t={t:+.2f} | non-negative {sum(d >= 0 for d in deltas)}/{len(deltas)}")
    print()

control = json.load(open("runs/offset_control.json", encoding="utf-8"))
nocls = json.load(open("runs/offset_nocls.json", encoding="utf-8"))
all_macro, all_neutral = [], []
for payload in (control, nocls):
    for run, entry in payload.items():
        table = {item["offset"]: item for item in entry["table"]}
        base, tuned = table[1.0]["test"], table[entry["chosen_offset"]]["test"]
        all_macro.append(tuned["macro_f1"] - base["macro_f1"])
        all_neutral.append(tuned["per_class"]["neutral"]["f1"] - base["per_class"]["neutral"]["f1"])
print(f"pooled (n={len(all_macro)}) macro_f1  " + " ".join(f"{d:+.4f}" for d in all_macro) +
      f" | mean {statistics.mean(all_macro):+.4f} ± {statistics.stdev(all_macro):.4f}"
      f" | non-negative {sum(d >= 0 for d in all_macro)}/{len(all_macro)}")
print(f"pooled (n={len(all_neutral)}) neutral_f1 " + " ".join(f"{d:+.4f}" for d in all_neutral) +
      f" | mean {statistics.mean(all_neutral):+.4f} ± {statistics.stdev(all_neutral):.4f}"
      f" | non-negative {sum(d >= 0 for d in all_neutral)}/{len(all_neutral)}")
