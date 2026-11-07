"""Three-class export tests with synthetic XML, no training or downloads."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from scripts.prepare_restaurant3 import export_split


class RestaurantExportTests(unittest.TestCase):
    def test_only_conflict_annotations_are_removed(self):
        content = ('<sentences><sentence id="x"><text>food good but service slow</text><aspectTerms>'
                   '<aspectTerm term="food" from="0" to="4" polarity="positive"/>'
                   '<aspectTerm term="service" from="14" to="21" polarity="conflict"/>'
                   '</aspectTerms><aspectCategories>'
                   '<aspectCategory category="food" polarity="positive"/>'
                   '<aspectCategory category="service" polarity="conflict"/>'
                   '</aspectCategories></sentence></sentences>')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, xml, jsonl = root / "source.xml", root / "train.xml", root / "train.jsonl"
            source.write_text(content, encoding="utf-8")
            stats = export_split(source, xml, jsonl)
            self.assertEqual(source.read_text(encoding="utf-8"), content)
            self.assertEqual(ET.parse(xml).findtext(".//text"), "food good but service slow")
            self.assertEqual(stats["aspect_examples"], 1)
            self.assertEqual(stats["excluded_conflict_aspects"], 1)
            self.assertEqual(stats["excluded_conflict_categories"], 1)
            row = json.loads(jsonl.read_text(encoding="utf-8"))
            self.assertEqual((row["start"], row["end"], row["label"]), (0, 4, "positive"))


if __name__ == "__main__":
    unittest.main()
