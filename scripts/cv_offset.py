"""Evaluate the neutral offset inside cross-validation, tuned on each fold's inner validation set.

For every outer fold the offset k is chosen on that fold's inner validation predictions and
then applied once to that fold's held-out aspects. Held-out rows from all folds are pooled,
so the reported score is an out-of-fold estimate of the whole procedure (train, pick the
epoch, pick k, predict), not of one model on one split.

Also prints an optimistic reference where k is fitted on the pooled held-out rows itself.
That is an upper bound for this post-processing family, not an achieved score.

Usage:
    python scripts/cv_offset.py --cv-dir runs/cv_control_3fold --cv-dir runs/cv_nocls_3fold
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_neutral_offset as tno  # noqa: E402

LABELS = tno.LABELS
SCALARS = ("accuracy", "macro_f1")


def key_metrics(result: dict) -> dict:
    return {**{name: result[name] for name in SCALARS},
            "neutral_f1": result["per_class"]["neutral"]["f1"]}


def choose(rows: list[dict], grid: list[float], metric: str) -> tuple[float, dict]:
    table = [(k, tno.metrics(rows, k)) for k in grid]
    return max(table, key=lambda item: tno.score_key(item[1], metric))


def fold_dirs(cv_dir: Path) -> list[Path]:
    return sorted(p for p in cv_dir.glob("fold_*") if (p / "run").is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", action="append", required=True, type=Path)
    parser.add_argument("--metric", choices=["accuracy", "macro_f1"], default="macro_f1")
    parser.add_argument("--grid", default="1,1.25,1.5,2,2.5,3,4,5,6")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/cv_offset_report.json"))
    args = parser.parse_args()

    grid = [float(value) for value in args.grid.split(",") if value.strip()]
    report = {}

    for cv in args.cv_dir:
        folds = fold_dirs(cv)
        if not folds:
            raise SystemExit(f"No fold_*/run directories under {cv}")

        per_fold, pooled_rows, pooled_base, pooled_nested = [], [], [], []
        for fold in folds:
            run = fold / "run"
            validation = tno.read_rows(run / tno.VALIDATION)
            held = tno.read_rows(run / tno.TEST)
            k, validation_metrics = choose(validation, grid, args.metric)
            _, base = tno.predict(held, 1.0)
            _, tuned = tno.predict(held, k)
            per_fold.append({
                "fold": fold.name,
                "chosen_offset": k,
                "validation": key_metrics(validation_metrics),
                "held_out_at_k1": key_metrics(tno.classification_metrics(
                    [LABELS.index(r["label"]) for r in held], base, LABELS)),
                "held_out_at_k": key_metrics(tno.classification_metrics(
                    [LABELS.index(r["label"]) for r in held], tuned, LABELS)),
                "held_out_rows": len(held),
            })
            pooled_rows.extend(held)
            pooled_base.extend(base)
            pooled_nested.extend(tuned)

        gold = [LABELS.index(row["label"]) for row in pooled_rows]
        argmax = tno.classification_metrics(gold, pooled_base, LABELS)
        nested = tno.classification_metrics(gold, pooled_nested, LABELS)
        oracle_k, oracle = choose(pooled_rows, grid, args.metric)

        summary = {
            "cv_dir": str(cv),
            "folds": len(folds),
            "held_out_aspects": len(pooled_rows),
            "per_fold": per_fold,
            "oof_argmax": key_metrics(argmax),
            "oof_nested_offset": key_metrics(nested),
            "oracle_reference": {"offset": oracle_k,
                                 "note": "offset fitted on the pooled held-out rows; upper bound only",
                                 **key_metrics(oracle)},
        }

        if args.bootstrap:
            rng = random.Random(args.seed)
            indices = list(range(len(gold)))
            deltas = {name: [] for name in (*SCALARS, "neutral_f1")}
            for _ in range(args.bootstrap):
                sample = [rng.choice(indices) for _ in indices]
                a = tno.classification_metrics([gold[i] for i in sample],
                                               [pooled_base[i] for i in sample], LABELS)
                b = tno.classification_metrics([gold[i] for i in sample],
                                               [pooled_nested[i] for i in sample], LABELS)
                for name in SCALARS:
                    deltas[name].append(b[name] - a[name])
                deltas["neutral_f1"].append(b["per_class"]["neutral"]["f1"]
                                            - a["per_class"]["neutral"]["f1"])
            summary["bootstrap_nested_vs_argmax"] = {}
            for name, values in deltas.items():
                values.sort()
                summary["bootstrap_nested_vs_argmax"][name] = {
                    "delta": summary["oof_nested_offset"][name] - summary["oof_argmax"][name],
                    "ci95_low": values[int(0.025 * len(values))],
                    "ci95_high": values[int(0.975 * len(values))],
                }

        report[cv.name] = summary
        print(f"\n=== {cv.name}: {len(folds)} folds, {len(pooled_rows)} held-out aspects ===")
        for entry in per_fold:
            print(f"  {entry['fold']}: k={entry['chosen_offset']:<5} "
                  f"macro {entry['held_out_at_k1']['macro_f1']:.4f} -> {entry['held_out_at_k']['macro_f1']:.4f} | "
                  f"neutral F1 {entry['held_out_at_k1']['neutral_f1']:.4f} -> "
                  f"{entry['held_out_at_k']['neutral_f1']:.4f}")
        for label, values in (("argmax", summary["oof_argmax"]),
                              ("nested offset", summary["oof_nested_offset"])):
            print(f"  pooled {label:14s} acc {values['accuracy']:.4f} macro {values['macro_f1']:.4f} "
                  f"neutral F1 {values['neutral_f1']:.4f}")
        print(f"  oracle k={oracle_k} (upper bound) acc {oracle['accuracy']:.4f} "
              f"macro {oracle['macro_f1']:.4f} neutral F1 {oracle['per_class']['neutral']['f1']:.4f}")
        for name, value in summary.get("bootstrap_nested_vs_argmax", {}).items():
            print(f"    delta {name:11s} {value['delta']:+.4f}  95% CI "
                  f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
