"""Known-answer metrics, malformed inputs, controlled pairs and CLI integration."""
from argparse import Namespace
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from neutral_diagnostics.cli import analyze, compare
from neutral_diagnostics.metrics import (calibration, cluster_bootstrap, neutral_diagnostics,
                                        precision_recall, read_predictions, target_fingerprint)
from neutral_diagnostics.pairs import read_pairs, summarize_pairs

ROOT = Path(__file__).resolve().parents[1]


def row(text, aspect, label, prediction, uid="s"):
    probabilities = {name: .1 for name in ("negative", "neutral", "positive")}
    probabilities[prediction] = .8
    start = text.index(aspect)
    return dict(sentence_id=uid, text=text, aspect=aspect, start=start, end=start + len(aspect),
                label=label, prediction=prediction, probabilities=probabilities)


class DiagnosticTests(unittest.TestCase):
    def test_ap_groups_ties_and_full_recall(self):
        tied = precision_recall([True, False], [.5, .5])
        self.assertEqual(tied["average_precision"], .5)
        self.assertEqual(tied["points"][-1]["recall"], 1)
        self.assertEqual(precision_recall([False, True], [.1, .9])["average_precision"], 1)
        self.assertIsNone(precision_recall([False], [.4])["average_precision"])

    def test_neighbor_denominators_are_explicit(self):
        text = "soup service decor"
        rows = [row(text, "soup", "neutral", "positive"),
                row(text, "service", "positive", "positive"), row(text, "decor", "negative", "negative"),
                row("bread", "bread", "neutral", "negative", "other")]
        summary, errors = neutral_diagnostics(rows)
        self.assertEqual(summary["matching_fraction_of_all_neutral_errors"], .5)
        self.assertEqual(summary["matching_fraction_of_eligible_errors"], 1)
        self.assertEqual(summary["groups"]["with_polar_neighbor"]["support"], 1)
        self.assertEqual(errors[0]["polar_neighbor_count"], 2)

    def test_bootstrap_is_reproducible_and_groups_exact_text(self):
        rows = [row("food soup", "food", "positive", "positive"),
                row("food soup", "soup", "neutral", "neutral"),
                row("bread", "bread", "negative", "positive", "b")]
        result = cluster_bootstrap(rows, 50, 42)
        self.assertEqual(result, cluster_bootstrap(rows, 50, 42))
        self.assertEqual(result["clusters"], 2)
        self.assertLess(result["valid_draws"], 50)
        self.assertIsNone(cluster_bootstrap(rows, 0)["intervals"]["neutral_f1"])
        self.assertAlmostEqual(calibration(rows)["ece"], abs(.8 - 2 / 3))

    def test_reject_invalid_or_duplicate_predictions(self):
        original = row("food", "food", "positive", "positive")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            for mutation in ("duplicate", "span", "nan", "argmax"):
                changed = json.loads(json.dumps(original))
                if mutation == "span":
                    changed["end"] = 100
                elif mutation == "nan":
                    changed["probabilities"]["neutral"] = float("nan")
                elif mutation == "argmax":
                    changed["prediction"] = "neutral"
                rows = [changed, changed] if mutation == "duplicate" else [changed]
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    read_predictions(path)

    def test_end_to_end_reports_and_coverage_checks(self):
        rows = [row("food", "food", "positive", "positive"),
                row("soup", "soup", "neutral", "neutral", "n"),
                row("bread", "bread", "negative", "positive", "b")]
        # Include exact zero probabilities to exercise finite logit-margin handling.
        rows[1]["probabilities"] = {"negative": 0., "neutral": 1., "positive": 0.}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "predictions.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            base = dict(predictions=str(source), gold_xml=None, dataset="fixture", run_id="seed42",
                        model_label="control", training_seed=42, bootstrap=20, bootstrap_seed=1)
            analyze(Namespace(**base, split="validation", output=str(root / "validation")))
            analyze(Namespace(**base, split="test", output=str(root / "test")))
            sources = [str(root / part / "analysis.json") for part in ("validation", "test")]
            compare(Namespace(analyses=sources, output=str(root / "comparison")))
            metrics = json.loads((root / "test/analysis.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(metrics["classification"]["accuracy"], 2 / 3)
            self.assertEqual(metrics["target_fingerprint"], target_fingerprint(rows))
            self.assertTrue((root / "comparison/validation_test_gaps.csv").exists())
            self.assertTrue((root / "test/report.md").exists())
            self.assertTrue((root / "test/neutral_pr.svg").exists())
            with self.assertRaises(ValueError):
                analyze(Namespace(**base, split="test", output=str(root / "test")))
            altered = json.loads((root / "test/analysis.json").read_text(encoding="utf-8"))
            altered["target_fingerprint"], altered["run_id"] = "different", "other"
            path = root / "altered.json"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Different evaluation targets"):
                compare(Namespace(analyses=[sources[1], str(path)], output=str(root / "bad")))
            gold = root / "gold.xml"
            gold.write_text('<sentences><sentence id="n"><text>soup</text><aspectTerms>'
                            '<aspectTerm term="soup" from="0" to="4" polarity="neutral"/>'
                            '</aspectTerms></sentence></sentences>', encoding="utf-8")
            base["gold_xml"] = str(gold)
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                analyze(Namespace(**base, split="test", output=str(root / "bad_gold")))

    def test_candidate_pairs_and_always_neutral_control(self):
        pairs = read_pairs(ROOT / "data/diagnostics/neutral_scope_pairs.csv")
        self.assertEqual(len(pairs), 40)
        self.assertEqual(sum(p["expected_label"] == "neutral" for p in pairs), 30)
        self.assertTrue(all(p["review_status"] in ("pending", "approved", "rejected") for p in pairs))
        predictions = {(p["pair_id"], condition): {"prediction": "neutral", "probabilities":
                       {"negative": .1, "neutral": .8, "positive": .1}}
                       for p in pairs for condition in ("positive", "negative")}
        summary, _ = summarize_pairs(pairs, predictions)
        self.assertEqual(summary["groups"]["all"]["invariant_mean"], 1)
        self.assertEqual(summary["groups"]["polar_controls"]["both_correct_mean"], 0)
        self.assertEqual(summary["groups"]["neutral_targets"]["both_correct_mean"], 1)

    def test_pair_review_and_minimal_change_validation(self):
        pair = read_pairs(ROOT / "data/diagnostics/neutral_scope_pairs.csv")[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.csv"
            for mutation in ("unreviewed_approval", "extra_change", "duplicate_target"):
                changed = dict(pair)
                if mutation == "unreviewed_approval":
                    changed["review_status"] = "approved"
                elif mutation == "extra_change":
                    changed["text_negative_neighbor"] += " Extra context."
                else:
                    changed["text_negative_neighbor"] += " soup"
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(changed))
                    writer.writeheader()
                    writer.writerow(changed)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    read_pairs(path)


if __name__ == "__main__":
    unittest.main()
