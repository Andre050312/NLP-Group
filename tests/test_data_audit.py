"""Audit source overlap and scarce neutral support without inventing predictions."""
from argparse import Namespace
from collections import namedtuple
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from neutral_diagnostics.data_audit import audit_data, raw_summary, split_summary

Aspect = namedtuple("Aspect", "start end label")
Document = namedtuple("Document", "uid text aspects")


class DataAuditTests(unittest.TestCase):
    def test_subgroup_support_and_recall_resolution(self):
        documents = [Document("a", "soup service decor", (Aspect(0, 4, "neutral"), Aspect(5, 12, "positive"), Aspect(13, 18, "negative"))),
                     Document("b", "bread", (Aspect(0, 5, "neutral"),))]
        summary = split_summary(documents)
        self.assertEqual(summary["label_counts"]["neutral"], 2)
        self.assertEqual(summary["neutral_with_polar_neighbor"], 1)
        self.assertEqual(summary["neutral_without_polar_neighbor"], 1)
        self.assertEqual(summary["neutral_with_both_polarities"], 1)
        self.assertEqual(summary["neutral_single_aspect"], 1)
        self.assertEqual(summary["neutral_recall_step"], .5)

    def test_raw_overlap_is_reported_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.xml"
            content = ('<sentences><sentence id="s"><text>spicy tuna</text><aspectTerms>'
                       '<aspectTerm term="spicy tuna" from="0" to="10" polarity="positive"/>'
                       '<aspectTerm term="tuna" from="6" to="10" polarity="neutral"/>'
                       '</aspectTerms></sentence></sentences>')
            path.write_text(content, encoding="utf-8")
            result = raw_summary(path)
            self.assertEqual(result["sentences_with_overlapping_spans"], 1)
            self.assertEqual(result["overlap_examples"][0]["right"]["aspect"], "tuna")
            self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_real_split_function_and_blocked_dataset_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / "Restaurant3"
            data.mkdir()
            def write(name, indices):
                sentences = []
                for i in indices:
                    sentences.append(f'<sentence id="{i}"><text>food soup service {i}</text><aspectTerms>'
                                     '<aspectTerm term="food" from="0" to="4" polarity="positive"/>'
                                     '<aspectTerm term="soup" from="5" to="9" polarity="neutral"/>'
                                     '<aspectTerm term="service" from="10" to="17" polarity="negative"/>'
                                     '</aspectTerms></sentence>')
                (data / name).write_text('<sentences>' + ''.join(sentences) + '</sentences>', encoding="utf-8")
            write("train.xml", range(8))
            write("test.xml", [8])
            audit_data(Namespace(datasets=["restaurant", "mams"], data_root=str(root), split_seed=42,
                                 validation_fraction=.25, output=str(root / "output")))
            report = json.loads((root / "output/data_audit.json").read_text(encoding="utf-8"))
            ready, blocked = report["datasets"]
            self.assertEqual(ready["status"], "current_split_reproduced")
            self.assertEqual(ready["splits"]["validation"]["label_counts"]["neutral"], 2)
            self.assertTrue(all(count == 0 for count in ready["exact_text_overlap"].values()))
            self.assertEqual(blocked["status"], "blocked_by_current_reader_or_data")
            self.assertIn("BLOCKED", (root / "output/report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
