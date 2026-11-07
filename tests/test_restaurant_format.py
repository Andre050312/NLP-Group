"""Conversion regressions using in-memory fixtures; no downloads or models."""
import unittest
import xml.etree.ElementTree as ET

from scripts.restaurant_format import convert_source, xml_statistics


class RestaurantFormatTests(unittest.TestCase):
    def test_opinions_conversion_preserves_three_classes_and_offsets(self):
        source = ('<Reviews><Review><sentences><sentence id="x"><text>food good service slow price fair</text><Opinions>'
                  '<Opinion target="food" from="0" to="4" polarity="positive"/>'
                  '<Opinion target="food" from="0" to="4" polarity="positive"/>'
                  '<Opinion target="service" from="10" to="17" polarity="negative"/>'
                  '<Opinion target="price" from="23" to="28" polarity="neutral"/>'
                  '<Opinion target="NULL" from="0" to="0" polarity="positive"/>'
                  '</Opinions></sentence></sentences></Review></Reviews>').encode()
        output, audit = convert_source(source, "opinions")
        stats = xml_statistics(output)
        self.assertEqual(stats["label_counts"], {"positive": 1, "negative": 1, "neutral": 1})
        self.assertEqual(audit["excluded_implicit_targets"], 1)
        self.assertEqual(audit["collapsed_duplicate_opinions"], 1)
        self.assertEqual(ET.fromstring(output).findtext("sentence/text"), "food good service slow price fair")

    def test_native_conflict_is_rejected(self):
        source = b'<sentences><sentence><text>food</text><aspectTerms><aspectTerm term="food" polarity="conflict" from="0" to="4"/></aspectTerms></sentence></sentences>'
        with self.assertRaises(ValueError):
            convert_source(source, "native")
        with self.assertRaises(ValueError):
            xml_statistics(source)

    def test_ambiguous_span_is_excluded_without_inventing_a_label(self):
        source = ('<Reviews><sentence id="x"><text>food service</text><Opinions>'
                  '<Opinion target="food" polarity="positive" from="0" to="4"/>'
                  '<Opinion target="food" polarity="negative" from="0" to="4"/>'
                  '<Opinion target="service" polarity="neutral" from="5" to="12"/>'
                  '</Opinions></sentence></Reviews>').encode()
        output, audit = convert_source(source, "opinions")
        self.assertEqual(audit["excluded_ambiguous_spans"], 1)
        self.assertEqual(xml_statistics(output)["label_counts"], {"neutral": 1})


if __name__ == "__main__":
    unittest.main()
