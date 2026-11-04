"""Downloader tests with mocked HTTP; no real network requests."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.download_datasets import fetch_xml, resources, validate_xml


XML = (b'<sentences><sentence id="x"><text>dress code</text><aspectTerms>'
       b'<aspectTerm term="dress cod" from="0" to="9" polarity="negative"/>'
       b'</aspectTerms></sentence></sentences>')


class DownloadTests(unittest.TestCase):
    def test_pinned_resources_cover_expected_xml_paths(self):
        items = resources("all")
        self.assertEqual(len(items), 10)
        self.assertEqual(len({path for path, _, _ in items}), 10)
        self.assertEqual(len(resources("semeval")), 4)
        self.assertEqual(len(resources("mams")), 6)
        self.assertTrue(all(revision in url for _, url, revision in items))

    def test_download_preserves_original_partial_word_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "train.xml"
            with patch("scripts.download_datasets.urlopen", return_value=io.BytesIO(XML)):
                stats = fetch_xml("https://example.org/train.xml", destination, retries=1)
            self.assertEqual(destination.read_bytes(), XML)
            self.assertEqual(stats["aspect_terms"], 1)
            self.assertEqual(validate_xml(destination)["sentences"], 1)

    def test_failed_download_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "train.xml"
            destination.write_bytes(b"old data")
            with patch("scripts.download_datasets.urlopen", return_value=io.BytesIO(b"<html>error</html>")):
                with self.assertRaises(RuntimeError):
                    fetch_xml("https://example.org/train.xml", destination, retries=1)
            self.assertEqual(destination.read_bytes(), b"old data")
            self.assertFalse(list(Path(directory).glob("*.download")))

    def test_invalid_old_file_is_backed_up_before_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "train.xml"
            destination.write_bytes(b"broken XML")
            with patch("scripts.download_datasets.urlopen", return_value=io.BytesIO(XML)):
                fetch_xml("https://example.org/train.xml", destination, retries=1)
            backups = list(Path(directory).glob("train.xml.invalid-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"broken XML")
            self.assertEqual(destination.read_bytes(), XML)


if __name__ == "__main__":
    unittest.main()
