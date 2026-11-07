"""Downloader tests with mocked HTTP; no real network requests."""
import io
from contextlib import redirect_stdout
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.download_datasets import fetch_from_sources, fetch_xml, import_local, main, resources, validate_xml


XML = (b'<sentences><sentence id="x"><text>dress code</text><aspectTerms>'
       b'<aspectTerm term="dress cod" from="0" to="9" polarity="negative"/>'
       b'</aspectTerms></sentence></sentences>')


class DownloadTests(unittest.TestCase):
    def test_checksum_rejects_parseable_wrong_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.xml"
            path.write_bytes(XML)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_xml(path, "0" * 64)

    def test_git_crlf_is_accepted_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            original = XML.replace(b"><", b">\n<")
            path = Path(directory) / "train.xml"
            path.write_bytes(original.replace(b"\n", b"\r\n"))
            stats = validate_xml(path, hashlib.sha256(original).hexdigest())
            self.assertTrue(stats["line_endings_normalized_for_check"])
            self.assertIn(b"\r\n", path.read_bytes())

    def test_network_source_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            urls = ["https://example.org/unavailable", "https://example.org/working"]
            with patch("scripts.download_datasets.urlopen", side_effect=[OSError("offline"), io.BytesIO(XML)]):
                with redirect_stdout(io.StringIO()):
                    stats, source = fetch_from_sources(urls, Path(directory) / "train.xml",
                                                       hashlib.sha256(XML).hexdigest(), retries=1)
            self.assertEqual(source, urls[1])
            self.assertEqual(stats["integrity"], "verified")

    def test_local_zip_handles_duplicate_train_filenames_by_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "data.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("wrong/train.xml", XML.replace(b"negative", b"positive"))
                stream.writestr("nested/MAMS/data/MAMS-ATSA/raw/train.xml", XML)
            target = Path(directory) / "Database/MAMS/data/MAMS-ATSA/raw/train.xml"
            stats, source = import_local(archive, "MAMS/data/MAMS-ATSA/raw/train.xml", target,
                                         hashlib.sha256(XML).hexdigest())
            self.assertEqual(target.read_bytes(), XML)
            self.assertEqual(stats["integrity"], "verified")
            self.assertIn("nested/MAMS", source)

    def test_nested_database_directory_can_be_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Database/Database/SemEval2014"
            source.mkdir(parents=True)
            (source / "Restaurants_Test_Gold.xml").write_bytes(XML)
            target = Path(directory) / "Database/SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml"
            import_local(Path(directory) / "Database", "SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml",
                         target, hashlib.sha256(XML).hexdigest())
            self.assertEqual(target.read_bytes(), XML)

    def test_offline_cli_restores_bundle_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bundle.zip"
            targets = resources("semeval")
            expected = {relative: hashlib.sha256(XML).hexdigest() for relative, _, _ in targets}
            with zipfile.ZipFile(archive, "w") as stream:
                for relative, _, _ in targets:
                    stream.writestr(relative, XML)
            data_root = Path(directory) / "Database"
            with patch("scripts.download_datasets.BUNDLE", archive), \
                 patch("scripts.download_datasets.EXPECTED_SHA256", expected), \
                 patch("scripts.download_datasets.urlopen") as network, \
                 patch("sys.argv", ["download_datasets.py", "--offline", "--dataset", "semeval", "--data-root", str(data_root)]):
                with redirect_stdout(io.StringIO()):
                    main()
                network.assert_not_called()
            for relative, _, _ in targets:
                self.assertTrue((data_root / relative).exists())

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
