"""Offline pair-model and fixed-split tests, to be run on the target machine."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from absa_enhanced.data import Aspect, Collator, Document, prepare_splits
from absa_enhanced.evaluation import evaluate
from absa_enhanced.model import JointABSA, joint_loss
from absa_enhanced.pair import PairABSA, PairDataset

ROOT = Path(__file__).resolve().parents[1]


class PairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "food", "good", "service",
                 "slow", "but", "filler", "okay", "bad", "."]
        path = self.directory / "vocab.txt"
        path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=True)
        self.config = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)

    def document(self):
        return Document("x", "food good but service slow", (Aspect(0, 4, "positive"), Aspect(14, 21, "negative")))

    def test_pair_query_and_target_spans(self):
        dataset = PairDataset([self.document()], self.tokenizer, 32)
        self.assertEqual(len(dataset), 2)
        second = dataset[1]
        query_ids = [token for token, query in zip(second["inputs"]["input_ids"], second["valid"]) if query]
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(query_ids), ["service"])
        start, end, _, a, b = second["targets"][0]
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(second["inputs"]["input_ids"][start:end]), ["service"])
        self.assertEqual((a, b), (14, 21))
        self.assertTrue(all(label == -100 for label in second["bio"]))

    def test_late_repeated_target_survives_cropping(self):
        text = "food " + "filler " * 40 + "food bad"
        start = text.rindex("food")
        dataset = PairDataset([Document("late", text, (Aspect(start, start + 4, "negative"),))], self.tokenizer, 12)
        row = dataset[0]
        self.assertLessEqual(len(row["inputs"]["input_ids"]), 12)
        self.assertEqual(row["targets"][0][3:], (start, start + 4))
        self.assertEqual(dataset.alignment_stats["cropped_contexts"], 1)

    def test_subword_boundary_and_conflict_policy(self):
        dataset = PairDataset([Document("partial", "food good", (Aspect(0, 3, "negative"), Aspect(5, 9, "conflict")))], self.tokenizer, 32)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["targets"][0][3:], (0, 3))
        self.assertEqual(dataset.alignment_stats["non_exact_token_spans"], 1)

    def test_gradients_and_checkpoint_dispatch(self):
        dataset = PairDataset([self.document()], self.tokenizer, 32)
        batch = Collator(self.tokenizer)(dataset.rows)
        model = PairABSA(BertModel(self.config), dropout=0.0)
        bio, sentiment = model(batch["inputs"], batch["valid"], batch["spans"])
        loss, _, extraction = joint_loss(bio, sentiment, batch["bio"], batch["labels"], extraction_weight=0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(extraction.item(), 0)
        loss.backward()
        self.assertGreater(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            before = model(batch["inputs"], batch["valid"], batch["spans"])[1]
        model.save_checkpoint(self.directory / "best", self.tokenizer,
                              {"max_length": 32, "stride": 4, "encoder": "tiny-test"})
        restored, tokenizer, settings = JointABSA.load_checkpoint(self.directory / "best")
        self.assertIsInstance(restored, PairABSA)
        self.assertEqual(settings["model_kind"], "pair")
        restored.eval()
        with torch.no_grad():
            after = restored(batch["inputs"], batch["valid"], batch["spans"])[1]
        torch.testing.assert_close(before, after)
        metrics, gold_rows, _ = evaluate(restored, tokenizer, dataset, torch.device("cpu"))
        self.assertEqual(metrics["gold_aspect_sentiment"]["examples"], 2)
        self.assertIsNone(metrics["end_to_end"])
        self.assertEqual(len(gold_rows), 2)

    def fixture(self, split, suffix, sentences=3):
        path = self.directory / f"{split}.xml"
        rows = []
        for i in range(sentences):
            # Each sentence has every class, keeping any sentence-level split valid.
            text = f"food good service slow food okay {suffix} {i}"
            rows.append(f'<sentence id="{split}{i}"><text>{text}</text><aspectTerms>'
                        '<aspectTerm term="food" from="0" to="4" polarity="positive"/>'
                        '<aspectTerm term="service" from="10" to="17" polarity="negative"/>'
                        '<aspectTerm term="food" from="23" to="27" polarity="neutral"/>'
                        '</aspectTerms></sentence>')
        path.write_text("<sentences>" + "".join(rows) + "</sentences>", encoding="utf-8")
        return path

    def test_fixed_split_does_not_depend_on_training_rng(self):
        source = self.fixture("source", ".", sentences=20)
        torch.manual_seed(42)
        a, b, _, _ = prepare_splits(source, seed=42)
        torch.manual_seed(43)
        c, d, _, _ = prepare_splits(source, seed=42)
        self.assertEqual([x.uid for x in a], [x.uid for x in c])
        self.assertEqual([x.uid for x in b], [x.uid for x in d])

    def test_cli_pair_train_evaluate_and_predict(self):
        initial = self.directory / "initial"
        BertModel(self.config).save_pretrained(initial)
        self.tokenizer.save_pretrained(initial)
        train = self.fixture("train", ".")
        validation = self.fixture("val", "but")
        test = self.fixture("test", "slow")
        output = self.directory / "run"

        def run(*arguments, success=True):
            result = subprocess.run([sys.executable, "-m", "absa_enhanced", *map(str, arguments)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=180,
                                    encoding="utf-8", errors="replace")
            if success:
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
            return result.stdout

        run("train", "--mode", "pair", "--dataset", "custom", "--encoder", initial,
            "--train", train, "--validation", validation, "--test", test, "--output", output,
            "--epochs", 1, "--max-length", 32, "--stride", 4, "--batch-size", 2,
            "--grad-accum", 2, "--seed", 43, "--split-seed", 42, "--device", "cpu")
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["seed"], 43)
        self.assertEqual(config["resolved_split_seed"], 42)
        self.assertEqual(config["effective_extraction_weight"], 0)
        self.assertTrue((output / "best_macro_f1/model.pt").exists())
        metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["gold_aspect_sentiment"]["examples"], 9)
        self.assertIsNone(metrics["aspect_extraction"])
        run("evaluate", "--checkpoint", output / "best", "--test", test,
            "--output", self.directory / "evaluated", "--device", "cpu")
        prediction = json.loads(run("predict", "--checkpoint", output / "best", "--text", "food good",
                                    "--aspect", "food", "--device", "cpu"))
        self.assertIn(prediction["prediction"], ("negative", "neutral", "positive"))
        run("predict", "--checkpoint", output / "best", "--text", "food good", "--device", "cpu", success=False)
        # A deterministic no-aspect extractor checks two-stage dispatch without downloads.
        extractor = JointABSA(BertModel(self.config), dropout=0.0)
        with torch.no_grad():
            extractor.extractor.weight.zero_()
            extractor.extractor.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
        extractor.save_checkpoint(self.directory / "extractor", self.tokenizer,
                                  {"max_length": 32, "stride": 4, "encoder": "tiny-test"})
        prediction = json.loads(run("predict", "--checkpoint", output / "best", "--text", "food good",
                                    "--extractor-checkpoint", self.directory / "extractor", "--device", "cpu"))
        self.assertEqual(prediction["predictions"], [])
        with torch.no_grad():
            extractor.extractor.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
        extractor.save_checkpoint(self.directory / "extractor_nonempty", self.tokenizer,
                                  {"max_length": 32, "stride": 4, "encoder": "tiny-test"})
        prediction = json.loads(run("predict", "--checkpoint", output / "best", "--text", "food good",
                                    "--extractor-checkpoint", self.directory / "extractor_nonempty", "--device", "cpu"))
        self.assertEqual(len(prediction["predictions"]), 2)
        self.assertTrue(all(p["aspect_source"] == "predicted_by_joint_model" for p in prediction["predictions"]))


if __name__ == "__main__":
    unittest.main()
