"""Offline tests to be run by the user on Windows; no pretrained downloads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from absa_enhanced.data import Aspect, Collator, Document, WindowDataset, prepare_splits
from absa_enhanced.evaluation import decode_spans, evaluate, set_metrics
from absa_enhanced.model import JointABSA, joint_loss

ROOT = Path(__file__).resolve().parents[1]


class EnhancedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "food", "service",
                 "good", "bad", "okay", "but", "slow", "filler", "target", "."]
        path = self.directory / "vocab.txt"
        path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=True)
        self.config = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)

    def model(self):
        return JointABSA(BertModel(self.config), local_radius=2, dropout=0.0)

    def document(self):
        text = "food good but service slow"
        return Document("x", text, (Aspect(0, 4, "positive"), Aspect(14, 21, "negative")))

    def test_overflow_retains_late_target(self):
        text = "filler " * 40 + "target"
        doc = Document("long", text, (Aspect(text.index("target"), len(text), "neutral"),))
        data = WindowDataset([doc], self.tokenizer, max_length=12, stride=4)
        self.assertGreater(len(data), 1)
        targets = [t for row in data.rows for t in row["targets"]]
        self.assertTrue(targets)
        self.assertTrue(all(t[3:] == (text.index("target"), len(text)) for t in targets))

    def test_repeated_aspect_and_conflict_labels(self):
        text = "food good but food bad"
        doc = Document("repeat", text, (Aspect(0, 4, "positive"), Aspect(14, 18, "conflict")))
        data = WindowDataset([doc], self.tokenizer, 32, 4)
        self.assertEqual(len(data.rows[0]["targets"]), 1)
        self.assertEqual(data.rows[0]["bio"].count(1), 2)

    def test_partial_subword_keeps_gold_span_and_masks_bio(self):
        # Mirrors source annotations such as "dress cod" inside "dress code".
        document = Document("partial", "food good", (Aspect(0, 3, "negative"),))
        data = WindowDataset([document], self.tokenizer, 32, 4)
        row = data.rows[0]
        start, end, label, char_start, char_end = row["targets"][0]
        self.assertEqual((char_start, char_end), (0, 3))
        self.assertEqual(label, 0)
        self.assertTrue(all(row["bio"][i] == -100 for i in range(start, end)))
        self.assertEqual(data.alignment_stats["non_exact_token_spans"], 1)
        # Gold classification still includes this target; source text was not fixed.
        metrics, gold_rows, _ = evaluate(self.model(), self.tokenizer, data, torch.device("cpu"))
        self.assertEqual(metrics["gold_aspect_sentiment"]["examples"], 1)
        self.assertEqual(gold_rows[0]["aspect"], "foo")
        self.assertEqual(metrics["aspect_extraction"]["gold"], 1)

    def test_joint_gradients_and_local_global_gate(self):
        data = WindowDataset([self.document()], self.tokenizer, 32, 4)
        batch = Collator(self.tokenizer)([data[0]])
        model = self.model()
        bio, sentiment = model(batch["inputs"], batch["valid"], batch["spans"])
        loss, _, _ = joint_loss(bio, sentiment, batch["bio"], batch["labels"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.encoder.embeddings.word_embeddings.weight.grad)
        self.assertGreater(model.extractor.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.gate.weight.grad.abs().sum().item(), 0)
        self.assertEqual(tuple(sentiment.shape), (2, 3))

    def test_no_aspect_sentence_has_finite_training_loss(self):
        data = WindowDataset([Document("empty", "food okay", ())], self.tokenizer, 32, 4)
        batch = Collator(self.tokenizer)([data[0]])
        model = self.model()
        bio, sentiment = model(batch["inputs"], batch["valid"], batch["spans"])
        loss, _, _ = joint_loss(bio, sentiment, batch["bio"], batch["labels"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_decode_and_strict_joint_metrics(self):
        offsets = [(0, 0), (0, 4), (5, 9), (10, 13), (14, 21), (0, 0)]
        spans = decode_spans([0, 2, 0, 0, 1, 0], [False, True, True, True, True, False],
                             offsets, "food good but service")
        self.assertEqual([(a, b) for _, _, a, b in spans], [(0, 4), (14, 21)])
        scores = set_metrics({(0, 0, 4, "positive")}, {(0, 0, 4, "negative")})
        self.assertEqual(scores["f1"], 0)

    def test_save_load_and_evaluate(self):
        data = WindowDataset([self.document()], self.tokenizer, 32, 4)
        model = self.model().eval()
        before, gold_rows, _ = evaluate(model, self.tokenizer, data, torch.device("cpu"))
        self.assertEqual(before["gold_aspect_sentiment"]["examples"], 2)
        self.assertEqual(len(gold_rows), 2)
        model.save_checkpoint(self.directory / "best", self.tokenizer,
                              {"max_length": 32, "stride": 4, "encoder": "tiny-test"})
        restored, tokenizer, _ = JointABSA.load_checkpoint(self.directory / "best")
        after, _, _ = evaluate(restored, tokenizer, data, torch.device("cpu"))
        self.assertEqual(before, after)

    def fixture(self, name, suffix):
        path = self.directory / f"{name}.xml"
        rows = []
        for i, (word, label) in enumerate((("good", "positive"), ("bad", "negative"), ("okay", "neutral"))):
            rows.append(f'<sentence id="{name}{i}"><text>food {word} {suffix}</text><aspectTerms>'
                        f'<aspectTerm term="food" from="0" to="4" polarity="{label}"/>'
                        '</aspectTerms></sentence>')
        path.write_text("<sentences>" + "".join(rows) + "</sentences>", encoding="utf-8")
        return path

    def test_cli_training_evaluation_and_automatic_prediction(self):
        initial = self.directory / "initial"
        BertModel(self.config).save_pretrained(initial)
        self.tokenizer.save_pretrained(initial)
        train, validation, test = (self.fixture("train", "."), self.fixture("val", "but"), self.fixture("test", "slow"))
        output = self.directory / "run"

        def run(*arguments):
            result = subprocess.run([sys.executable, "-m", "absa_enhanced", *map(str, arguments)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=180,
                                    encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            return result.stdout

        run("train", "--dataset", "custom", "--encoder", initial, "--train", train,
            "--validation", validation, "--test", test, "--output", output, "--epochs", 1,
            "--batch-size", 2, "--grad-accum", 2, "--max-length", 32, "--stride", 4,
            "--device", "cpu", "--precision", "no")
        metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["gold_aspect_sentiment"]["examples"], 3)
        index = json.loads((output / "checkpoint_index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["gold_accuracy"]["directory"], "best")
        self.assertEqual(index["gold_macro_f1"]["directory"], "best_macro_f1")
        self.assertTrue((output / "best_macro_f1/model.pt").is_file())
        run("evaluate", "--checkpoint", output / "best", "--test", test,
            "--output", self.directory / "evaluation", "--device", "cpu")
        result = json.loads(run("predict", "--checkpoint", output / "best", "--text", "food good", "--device", "cpu"))
        self.assertIn("predictions", result)
        result = json.loads(run("predict", "--checkpoint", output / "best", "--text", "food good",
                                "--aspect", "food", "--device", "cpu"))
        self.assertIn(result["prediction"], ("negative", "neutral", "positive"))
        self.assertNotIn("label", result)

    def test_explicit_validation_overlap_is_rejected(self):
        train = self.fixture("train", ".")
        validation = self.fixture("val", ".")
        with self.assertRaisesRegex(ValueError, "identical texts"):
            prepare_splits(train, validation)


if __name__ == "__main__":
    unittest.main()
