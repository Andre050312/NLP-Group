"""Offline checks for the sentiment-head parameterisations (no encoder download)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from absa_enhanced import heads
from absa_enhanced.heads import HEADS, output_dim, probabilities, loss

ROOT = Path(__file__).resolve().parents[1]


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_output_dimension_per_head(self):
        self.assertEqual(output_dim("softmax"), 3)
        self.assertEqual(output_dim("ordinal"), 2)
        self.assertEqual(output_dim("gate"), 3)
        with self.assertRaises(ValueError):
            output_dim("unknown")

    def test_probabilities_are_normalised(self):
        for kind in HEADS:
            logits = torch.randn(7, output_dim(kind)) * 3
            values = probabilities(kind, logits)
            self.assertEqual(tuple(values.shape), (7, 3))
            self.assertTrue(torch.all(values >= -1e-6))
            self.assertTrue(torch.allclose(values.sum(-1), torch.ones(7), atol=1e-5), kind)

    def test_ordinal_respects_label_order(self):
        # P(y >= neutral) must never fall below P(y >= positive).
        logits = torch.randn(64, 2) * 5
        values = probabilities("ordinal", logits)
        self.assertTrue(torch.all(values[:, 1] >= -1e-6))
        self.assertTrue(torch.all(values[:, 0] >= -1e-6))
        self.assertTrue(torch.allclose(values.sum(-1), torch.ones(64), atol=1e-5))

    def test_gate_makes_neutral_the_absence_of_opinion(self):
        # A strongly negative opinion logit can still predict neutral when presence is low.
        logits = torch.tensor([[-8.0, 9.0, -9.0]])
        values = probabilities("gate", logits)
        self.assertGreater(values[0, 1].item(), 0.99)
        strong = torch.tensor([[8.0, 9.0, -9.0]])
        self.assertGreater(probabilities("gate", strong)[0, 0].item(), 0.99)

    def test_loss_is_finite_differentiable_and_gradiented(self):
        labels = torch.tensor([0, 1, 2, 1, 0, 2])
        for kind in HEADS:
            logits = torch.randn(6, output_dim(kind), requires_grad=True)
            value = loss(kind, logits, labels)
            self.assertTrue(torch.isfinite(value), kind)
            self.assertGreater(value.item(), 0.0, kind)
            value.backward()
            self.assertIsNotNone(logits.grad)
            self.assertGreater(logits.grad.abs().sum().item(), 0.0, kind)

    def test_class_weights_and_smoothing_run_on_every_head(self):
        labels = torch.tensor([0, 1, 2, 1])
        weights = torch.tensor([1.0, 2.0, 1.0])
        for kind in HEADS:
            logits = torch.randn(4, output_dim(kind))
            value = loss(kind, logits, labels, class_weights=weights, smoothing=0.1)
            self.assertTrue(torch.isfinite(value), kind)

    def test_neutral_only_batch_does_not_break_the_gate(self):
        logits = torch.randn(3, output_dim("gate"), requires_grad=True)
        value = loss("gate", logits, torch.tensor([1, 1, 1]))
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_softmax_matches_plain_cross_entropy(self):
        labels = torch.tensor([0, 1, 2])
        logits = torch.randn(3, 3)
        reference = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
        self.assertTrue(torch.allclose(loss("softmax", logits, labels, smoothing=0.05), reference))

    def test_unknown_head_is_rejected(self):
        with self.assertRaises(ValueError):
            loss("unknown", torch.randn(2, 3), torch.tensor([0, 1]))

    def test_neutral_offset_raises_neutral_probability(self):
        for kind in HEADS:
            logits = torch.randn(16, output_dim(kind))
            base = probabilities(kind, logits)
            self.assertTrue(torch.equal(base, probabilities(kind, logits, 1.0)), kind)
            tuned = probabilities(kind, logits, 3.0)
            self.assertTrue(torch.all(tuned[:, 1] >= base[:, 1] - 1e-6), kind)
            self.assertTrue(torch.allclose(tuned.sum(-1), torch.ones(16), atol=1e-5), kind)

    def test_neutral_offset_can_flip_a_borderline_prediction(self):
        # Neutral second-best by a small margin must win once it is multiplied.
        logits = torch.tensor([[0.4, 0.3, -0.6]])
        self.assertEqual(int(probabilities("softmax", logits).argmax(-1)), 0)
        self.assertEqual(int(probabilities("softmax", logits, 3.0).argmax(-1)), 1)


class HeadCliTests(unittest.TestCase):
    """Each head must survive a real train -> save -> reload -> evaluate round trip."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "food", "good", "service",
                 "slow", "but", "okay", "filler", "."]
        path = self.directory / "vocab.txt"
        path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=True)
        self.config = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
        self.initial = self.directory / "initial"
        BertModel(self.config).save_pretrained(self.initial)
        self.tokenizer.save_pretrained(self.initial)
        self.train = self.fixture("train", ".")
        self.validation = self.fixture("val", "but")
        self.test = self.fixture("test", "slow")

    def fixture(self, split, suffix, sentences=3):
        path = self.directory / f"{split}.xml"
        rows = []
        for i in range(sentences):
            text = f"food good service slow food okay {suffix} {i}"
            rows.append(f'<sentence id="{split}{i}"><text>{text}</text><aspectTerms>'
                        '<aspectTerm term="food" from="0" to="4" polarity="positive"/>'
                        '<aspectTerm term="service" from="10" to="17" polarity="negative"/>'
                        '<aspectTerm term="food" from="23" to="27" polarity="neutral"/>'
                        '</aspectTerms></sentence>')
        path.write_text("<sentences>" + "".join(rows) + "</sentences>", encoding="utf-8")
        return path

    def run_cli(self, *arguments):
        result = subprocess.run([sys.executable, "-m", "absa_enhanced", *map(str, arguments)],
                                cwd=ROOT, capture_output=True, text=True, timeout=180,
                                encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        return result.stdout

    def test_every_head_trains_saves_and_reloads(self):
        for kind in HEADS:
            with self.subTest(head=kind):
                output = self.directory / f"run_{kind}"
                self.run_cli("train", "--mode", "pair", "--dataset", "custom", "--encoder", self.initial,
                             "--train", self.train, "--validation", self.validation, "--test", self.test,
                             "--output", output, "--epochs", 1, "--max-length", 48, "--stride", 8,
                             "--batch-size", 2, "--grad-accum", 2, "--seed", 43, "--split-seed", 42,
                             "--sentiment-head", kind, "--device", "cpu")
                settings = json.loads((output / "best/settings.json").read_text(encoding="utf-8"))
                self.assertEqual(settings["sentiment_head"], kind)
                metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["gold_aspect_sentiment"]["examples"], 9)
                self.assertIsNone(metrics["aspect_extraction"])
                rows = [json.loads(line) for line in
                        (output / "test_gold_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
                self.assertEqual(len(rows), 9)
                for row in rows:
                    values = row["probabilities"]
                    self.assertEqual(set(values), {"negative", "neutral", "positive"})
                    self.assertAlmostEqual(sum(values.values()), 1.0, places=5)
                    self.assertGreaterEqual(min(values.values()), 0.0)
                # Reload through the CLI so the settings dispatch is covered.
                self.run_cli("evaluate", "--checkpoint", output / "best", "--test", self.test,
                             "--output", self.directory / f"evaluated_{kind}", "--device", "cpu")
                prediction = json.loads(self.run_cli("predict", "--checkpoint", output / "best",
                                                     "--text", "food good", "--aspect", "food",
                                                     "--device", "cpu"))
                self.assertIn(prediction["prediction"], ("negative", "neutral", "positive"))

    def test_checkpoint_without_head_setting_defaults_to_softmax(self):
        output = self.directory / "legacy"
        self.run_cli("train", "--mode", "joint", "--dataset", "custom", "--encoder", self.initial,
                     "--train", self.train, "--validation", self.validation, "--test", self.test,
                     "--output", output, "--epochs", 1, "--max-length", 48, "--stride", 16,
                     "--batch-size", 2, "--grad-accum", 2, "--seed", 43, "--split-seed", 42,
                     "--device", "cpu")
        path = output / "best/settings.json"
        settings = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(settings["sentiment_head"], "softmax")
        settings.pop("sentiment_head")
        path.write_text(json.dumps(settings), encoding="utf-8")
        self.run_cli("evaluate", "--checkpoint", output / "best", "--test", self.test,
                     "--output", self.directory / "legacy_evaluated", "--device", "cpu")


if __name__ == "__main__":
    unittest.main()
