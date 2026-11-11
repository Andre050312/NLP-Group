"""Clause-scoped context tests: scope isolation must never break target alignment."""
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import BertConfig, BertTokenizerFast

from absa_enhanced.data import Aspect, Document
from absa_enhanced.pair import PairDataset, clause_bounds


class ClauseBoundsTests(unittest.TestCase):
    def test_contrast_sentence_splits_at_but(self):
        text = "the food was great but the service was slow"
        food = text.index("food")
        service = text.index("service")
        left, right = clause_bounds(text, food, food + 4)
        self.assertTrue(left <= food and food + 4 <= right)
        self.assertLess(right, service)
        second_left, second_right = clause_bounds(text, service, service + 7)
        self.assertGreater(second_left, right - 1)
        self.assertTrue(second_left <= service and service + 7 <= second_right)

    def test_punctuation_splits_clauses(self):
        text = "nice atmosphere, friendly staff, and they have a terrace"
        terrace = text.index("terrace")
        left, right = clause_bounds(text, terrace, terrace + 7)
        self.assertNotIn("atmosphere", text[left:right])
        self.assertIn("terrace", text[left:right])

    def test_short_clause_is_widened(self):
        text = "food was great, service, and the room was fine"
        service = text.index("service")
        left, right = clause_bounds(text, service, service + 7)
        # A one-word clause borrows a neighbour so the aspect keeps some context.
        self.assertTrue(left <= service and service + 7 <= right)
        self.assertGreater(len(text[left:right].split()), 1)

    def test_single_clause_returns_whole_text(self):
        text = "the food was very good"
        self.assertEqual(clause_bounds(text, 4, 8), (0, len(text)))


class ClauseScopeDatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "food", "was", "great",
                 "but", "service", "slow", "nice", "staff", "and"]
        path = Path(self.temp.name) / "vocab.txt"
        path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=True)
        self.document = Document("x", "the food was great but service slow",
                                 (Aspect(4, 8, "positive"), Aspect(23, 30, "negative")))

    def test_clause_scope_drops_the_other_opinion(self):
        sentence = PairDataset([self.document], self.tokenizer, 32, scope="sentence")
        clause = PairDataset([self.document], self.tokenizer, 32, scope="clause")
        self.assertEqual(len(sentence), 2)
        self.assertEqual(len(clause), 2)
        first_row = clause[0]
        tokens = self.tokenizer.convert_ids_to_tokens(first_row["inputs"]["input_ids"])
        self.assertIn("food", tokens)
        self.assertNotIn("slow", tokens)
        second_row = clause[1]
        tokens = self.tokenizer.convert_ids_to_tokens(second_row["inputs"]["input_ids"])
        self.assertIn("slow", tokens)
        self.assertNotIn("great", tokens)

    def test_clause_scope_keeps_target_and_query_aligned(self):
        clause = PairDataset([self.document], self.tokenizer, 32, scope="clause")
        for row in clause.rows:
            start, end, _, char_start, char_end = row["targets"][0]
            token_ids = [token for token, query in zip(row["inputs"]["input_ids"], row["valid"]) if query]
            text = self.tokenizer.decode(token_ids).strip()
            self.assertEqual(text, self.document.text[char_start:char_end])
            self.assertEqual(row["inputs"]["input_ids"][start:end], token_ids)
        self.assertEqual(clause.alignment_stats["context_scope"], "clause")
        self.assertEqual(clause.alignment_stats["clause_narrowed_contexts"], 2)

    def test_sentence_scope_is_the_previous_behaviour(self):
        sentence = PairDataset([self.document], self.tokenizer, 32, scope="sentence")
        self.assertEqual(sentence.alignment_stats["clause_narrowed_contexts"], 0)
        tokens = self.tokenizer.convert_ids_to_tokens(sentence[0]["inputs"]["input_ids"])
        self.assertIn("slow", tokens)

    def test_unknown_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            PairDataset([self.document], self.tokenizer, 32, scope="document")


class PairDiagnosticsScopeTests(unittest.TestCase):
    """neutral_diagnostics must honour the scope stored in the checkpoint."""

    def setUp(self):
        import csv
        import subprocess
        import sys
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.subprocess, self.sys, self.csv = subprocess, sys, csv
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "ordered", "the", "soup", ",",
                 "and", "service", "was", "great", "awful", "terrible", "decor", "nice"]
        path = self.directory / "vocab.txt"
        path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=True)
        self.config = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)

    def pairs(self):
        path = self.directory / "pairs.csv"
        rows = []
        for index, (positive, negative) in enumerate((("great", "awful"), ("nice", "terrible"))):
            rows.append({"pair_id": f"scope_{index}", "target": "soup", "expected_label": "neutral",
                         "distractor": "service", "positive_expression": positive,
                         "negative_expression": negative,
                         "text_positive_neighbor": f"i ordered the soup , and service was {positive}",
                         "text_negative_neighbor": f"i ordered the soup , and service was {negative}",
                         "review_status": "pending", "reviewer": "", "review_notes": "", "source": "test"})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = self.csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def predict(self, scope):
        from absa_enhanced.pair import PairABSA
        from transformers import BertModel
        checkpoint = self.directory / f"cpk_{scope}"
        PairABSA(BertModel(self.config), dropout=0.0, use_cls=True).save_checkpoint(
            checkpoint, self.tokenizer,
            {"max_length": 32, "stride": 8, "encoder": "tiny-test", "context_scope": scope})
        output = self.directory / f"pairs_{scope}"
        result = self.subprocess.run([self.sys.executable, "-m", "neutral_diagnostics", "predict-pairs",
                                      "--pairs", str(self.pairs()), "--checkpoint", str(checkpoint),
                                      "--output", str(output), "--include-unreviewed", "--device", "cpu",
                                      "--precision", "no"],
                                     cwd=Path(__file__).resolve().parents[1],
                                     capture_output=True, text=True, timeout=180,
                                     encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        import json
        return json.loads((output / "pair_metrics.json").read_text(encoding="utf-8"))

    def test_clause_checkpoint_is_evaluated_with_clause_context(self):
        clause = self.predict("clause")
        # The target clause is identical in both conditions, so a clause-scoped
        # model cannot change its answer; sentence scope still sees the distractor.
        self.assertEqual(clause["groups"]["neutral_targets"]["invariant_mean"], 1.0)
        self.assertEqual(clause["groups"]["neutral_targets"]["absolute_neutral_probability_change_mean"], 0.0)

    def test_sentence_checkpoint_is_unaffected(self):
        sentence = self.predict("sentence")
        self.assertGreater(sentence["groups"]["neutral_targets"]["absolute_neutral_probability_change_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
