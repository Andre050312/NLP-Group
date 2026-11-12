"""Check a fixed neutral offset instead of one tuned per run.

Tuning k on a validation split adds selection noise that differs between protocols. A fixed
k (chosen a priori, identical for every run) removes that noise and answers a simpler
question: does boosting neutral by a constant factor help these checkpoints on average?

Reports, for a fixed grid, the mean change in macro-F1 and neutral F1 over a set of runs
(each with its own test predictions) and, when --cv-dir is given, over pooled out-of-fold
predictions from cross-validation.

Usage:
    python scripts/fixed_offset_check.py --runs runs/a runs/b --cv-dir runs/cv_control_3fold
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_neutral_offset as tno  # noqa: E402

LABELS = tno.LABELS


def deltas(rows: list[dict], grid: list[float]) -> dict[str, dict[float, float]]:
    base = tno.metrics(rows, 1.0)
    out = {"macro_f1": {}, "neutral_f1": {}, "accuracy": {}}
    for k in grid:
        tuned = tno.metrics(rows, k)
        out["macro_f1"][k] = tuned["macro_f1"] - base["macro_f1"]
        out["accuracy"][k] = tuned["accuracy"] - base["accuracy"]
        out["neutral_f1"][k] = (tuned["per_class"]["neutral"]["f1"]
                               - base["per_class"]["neutral"]["f1"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument("--grid", default="1.25,1.5,2,2.5,3,4,5")
    parser.add_argument("--output", type=Path, default=Path("runs/fixed_offset_check.json"))
    args = parser.parse_args()

    grid = [float(value) for value in args.grid.split(",") if value.strip()]
    report: dict[str, object] = {"grid": grid}

    if args.runs:
        per_run = {}
        for run in args.runs:
            directory = Path(run)
            per_run[directory.name] = deltas(tno.read_rows(directory / tno.TEST), grid)
        summary = {}
        print(f"\n=== fixed offset on {len(per_run)} single-split runs (test) ===")
        print("  k      macro delta (mean +/- sd, positive)   neutral F1 delta (mean +/- sd, positive)")
        for k in grid:
            macro = [entry["macro_f1"][k] for entry in per_run.values()]
            neutral = [entry["neutral_f1"][k] for entry in per_run.values()]
            summary[str(k)] = {
                "macro_f1": {"mean": statistics.mean(macro),
                             "sd": statistics.stdev(macro) if len(macro) > 1 else 0.0,
                             "positive": sum(v > 0 for v in macro)},
                "neutral_f1": {"mean": statistics.mean(neutral),
                               "sd": statistics.stdev(neutral) if len(neutral) > 1 else 0.0,
                               "positive": sum(v > 0 for v in neutral)},
            }
            print(f"  {k:<6} {statistics.mean(macro):+.4f} +/- "
                  f"{(statistics.stdev(macro) if len(macro) > 1 else 0.0):.4f} "
                  f"({sum(v > 0 for v in macro)}/{len(macro)})        "
                  f"{statistics.mean(neutral):+.4f} +/- "
                  f"{(statistics.stdev(neutral) if len(neutral) > 1 else 0.0):.4f} "
                  f"({sum(v > 0 for v in neutral)}/{len(neutral)})")
        report["single_split"] = {"per_run": per_run, "summary": summary}

    if args.cv_dir:
        pooled = []
        for fold in sorted(args.cv_dir.glob("fold_*")):
            run = fold / "run"
            if (run / tno.TEST).is_file():
                pooled.extend(tno.read_rows(run / tno.TEST))
        if not pooled:
            raise SystemExit(f"No held-out predictions under {args.cv_dir}")
        result = deltas(pooled, grid)
        print(f"\n=== fixed offset on pooled OOF from {args.cv_dir.name} "
              f"({len(pooled)} aspects) ===")
        for k in grid:
            print(f"  {k:<6} macro {result['macro_f1'][k]:+.4f}   "
                  f"neutral F1 {result['neutral_f1'][k]:+.4f}   acc {result['accuracy'][k]:+.4f}")
        report["cross_validation"] = {"cv_dir": str(args.cv_dir), "aspects": len(pooled),
                                      "deltas": {name: {str(k): v for k, v in values.items()}
                                                 for name, values in result.items()}}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
