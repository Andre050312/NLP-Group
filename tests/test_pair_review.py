"""Pair-review merge tests: only independently agreed verdicts may become canonical."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apply_pair_review.py"
BUILDER = ROOT / "scripts" / "build_pair_review_sheet.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_review(path: Path, rows: list[dict], reviewer: str, flips: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            out = dict(row, review_status=flips.get(row["pair_id"], "approved"), reviewer=reviewer,
                       review_notes="checked")
            writer.writerow(out)


class PairReviewMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pairs = self.dir / "pairs.csv"
        self.rows = read_rows(ROOT / "data" / "diagnostics" / "neutral_scope_pairs.csv")
        with self.pairs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def run_merge(self, first: Path, second: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(first), str(second),
             "--pairs", str(self.pairs), "--audit", str(self.dir / "audit.json"), *extra],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")

    def test_agreement_is_written_and_disagreement_stays_pending(self):
        a, b = self.dir / "a.csv", self.dir / "b.csv"
        write_review(a, self.rows, "Alice", {})
        write_review(b, self.rows, "Bob", {"scope_03": "rejected"})
        result = self.run_merge(a, b, "--write")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

        merged = {row["pair_id"]: row for row in read_rows(self.pairs)}
        self.assertEqual(merged["scope_01"]["review_status"], "approved")
        self.assertEqual(merged["scope_01"]["reviewer"], "Alice + Bob")
        self.assertEqual(merged["scope_03"]["review_status"], "pending")
        self.assertEqual(merged["scope_03"]["reviewer"], "")

        audit = json.loads((self.dir / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["agreement_count"], len(self.rows) - 1)
        self.assertEqual(len(audit["disagreements"]), 1)
        self.assertEqual(audit["disagreements"][0]["pair_id"], "scope_03")

    def test_identical_reviewer_names_are_rejected(self):
        a, b = self.dir / "a.csv", self.dir / "b.csv"
        write_review(a, self.rows, "Alice", {})
        write_review(b, self.rows, "Alice", {})
        result = self.run_merge(a, b)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("independent", result.stderr + result.stdout)

    def test_incomplete_review_is_rejected(self):
        a, b = self.dir / "a.csv", self.dir / "b.csv"
        write_review(a, self.rows, "Alice", {})
        write_review(b, self.rows[:-1], "Bob", {})
        result = self.run_merge(a, b)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no decision", result.stderr + result.stdout)

    def test_pending_verdict_in_export_is_rejected(self):
        a, b = self.dir / "a.csv", self.dir / "b.csv"
        write_review(a, self.rows, "Alice", {})
        write_review(b, self.rows, "Bob", {self.rows[0]["pair_id"]: "pending"})
        result = self.run_merge(a, b)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without a verdict", result.stderr + result.stdout)

    def test_report_only_leaves_canonical_file_untouched(self):
        a, b = self.dir / "a.csv", self.dir / "b.csv"
        write_review(a, self.rows, "Alice", {})
        write_review(b, self.rows, "Bob", {})
        before = self.pairs.read_bytes()
        result = self.run_merge(a, b)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.pairs.read_bytes(), before)


class ReviewSheetTests(unittest.TestCase):
    def test_sheet_embeds_every_pair_without_inventing_a_reviewer(self):
        module = load_module(BUILDER, "build_pair_review_sheet")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sheet.html"
            sys.argv = ["build_pair_review_sheet.py", "--pairs",
                        str(ROOT / "data" / "diagnostics" / "neutral_scope_pairs.csv"),
                        "--output", str(output)]
            self.assertEqual(module.main(), 0)
            html = output.read_text(encoding="utf-8")
        rows = read_rows(ROOT / "data" / "diagnostics" / "neutral_scope_pairs.csv")
        for row in rows:
            self.assertIn(row["pair_id"], html)
        self.assertIn('id="reviewer"', html)
        self.assertNotIn("value=\"pending\"", html)


if __name__ == "__main__":
    unittest.main()
