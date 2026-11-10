"""Clause-scoped context tests: scope isolation must never break target alignment."""
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import BertTokenizerFast

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


if __name__ == "__main__":
    unittest.main()
