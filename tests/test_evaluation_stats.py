import random
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts.evaluation_stats import sign_test, common_seeds, sentence_clusters, sample_clusters


class EvaluationStatsTests(unittest.TestCase):
    def test_ties_are_excluded(self):
        result = sign_test([1] * 8 + [0, 0])
        self.assertEqual(result["n_nonzero"], 8)
        self.assertEqual(result["ties"], 2)
        self.assertEqual(result["p_two_sided"], 0.0078125)
        self.assertEqual(sign_test([0, 0])["p_two_sided"], 1)

    def test_shared_seeds_intersect(self):
        self.assertEqual(common_seeds([{42, 43}, {43, 44}]), [43])
        with self.assertRaises(ValueError):
            common_seeds([{42}, {43}])

    def test_cluster_draw_keeps_aspects_together(self):
        clusters = sentence_clusters([{"text": "same"}, {"text": "same"}, {"text": "other"}])
        sample = sample_clusters(clusters, random.Random(5))
        self.assertEqual(sample.count(0), sample.count(1))
        with self.assertRaises(ValueError):
            sentence_clusters([{"label": "neutral"}])

    def test_nonfinite_differences_rejected(self):
        with self.assertRaises(ValueError):
            sign_test([float("nan")])


class OffsetIntegrationTests(unittest.TestCase):
    def test_offset_and_cv_outputs_declare_cluster_units(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            cv = work / "cv"
            for fold in range(2):
                run = cv / f"fold_{fold}" / "run"
                run.mkdir(parents=True)
                rows = [{"text": f"sentence {fold} {i // 2}", "label": label,
                         "probabilities": dict(zip(("negative", "neutral", "positive"), probs))}
                        for i, (label, probs) in enumerate([
                            ("negative", [.8, .1, .1]), ("neutral", [.45, .4, .15]),
                            ("positive", [.1, .2, .7])])]
                for name in ("validation_best_gold_predictions.jsonl", "test_gold_predictions.jsonl"):
                    (run / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            single, pooled = work / "single.json", work / "pooled.json"
            commands = [
                ["scripts/tune_neutral_offset.py", "--runs", str(cv / "fold_0/run"), "--output", str(single)],
                ["scripts/cv_offset.py", "--cv-dir", str(cv), "--output", str(pooled)]]
            for command in commands:
                result = subprocess.run([sys.executable, *command, "--grid", "1,2", "--bootstrap", "20"],
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            entry = json.loads(single.read_text(encoding="utf-8"))["run"]
            self.assertEqual(entry["chosen_offset"], 2)
            self.assertEqual(entry["bootstrap_design"]["clusters"], 2)
            self.assertEqual(entry["bootstrap_design"]["unit"], "exact_sentence_text")
            summary = json.loads(pooled.read_text(encoding="utf-8"))["cv"]
            self.assertEqual(summary["bootstrap_design"]["clusters"], 4)
            self.assertTrue(summary["bootstrap_design"]["paired"])
            self.assertEqual([f["chosen_offset"] for f in summary["per_fold"]], [2, 2])
