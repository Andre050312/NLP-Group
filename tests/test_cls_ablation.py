"""Ablation gradients, legacy/new checkpoint loading, inference and CLI training."""
from argparse import Namespace
import contextlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from absa_enhanced.data import Aspect, Collator, Document, WindowDataset
from absa_enhanced.model import JointABSA, joint_loss
from absa_enhanced.pair import PairABSA, PairDataset
from neutral_diagnostics.pairs import predict_pairs

ROOT = Path(__file__).resolve().parents[1]


class CLSAblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "food", "good", "bad", "okay", "."]
        path = self.root / "vocab.txt"
        path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        key = "vocab" if "vocab" in inspect.signature(BertTokenizerFast).parameters else "vocab_file"
        self.tokenizer = BertTokenizerFast(**{key: str(path)}, do_lower_case=True)
        self.config = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)

    def test_joint_pair_gradients_roundtrip_and_legacy_defaults(self):
        document = Document("s", "food good", (Aspect(0, 4, "positive"),))
        for kind, factory in ((JointABSA, WindowDataset), (PairABSA, PairDataset)):
            data = factory([document], self.tokenizer, 32, 4)
            batch = Collator(self.tokenizer)([data[0]])
            for use_cls in (True, False):
                with self.subTest(kind=kind.__name__, use_cls=use_cls):
                    model = kind(BertModel(self.config), dropout=0., use_cls=use_cls)
                    bio, sentiment = model(batch["inputs"], batch["valid"], batch["spans"])
                    loss, _, _ = joint_loss(bio, sentiment, batch["bio"], batch["labels"])
                    loss.backward()
                    self.assertGreater(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum().item(), 0)
                    self.assertEqual(model.sentiment[0].normalized_shape, (24 * (2 + int(use_cls)),))
                    destination = self.root / f"{model.model_kind}_{use_cls}"
                    model.save_checkpoint(destination, self.tokenizer, {"max_length": 32, "stride": 4})
                    if use_cls:
                        settings = json.loads((destination / "settings.json").read_text(encoding="utf-8"))
                        del settings["use_cls"]
                        (destination / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
                    restored, _, _ = JointABSA.load_checkpoint(destination)
                    self.assertEqual(restored.use_cls, use_cls)
                    model.eval()
                    restored.eval()
                    with torch.no_grad():
                        expected = model(batch["inputs"], batch["valid"], batch["spans"])[1]
                        actual = restored(batch["inputs"], batch["valid"], batch["spans"])[1]
                    torch.testing.assert_close(expected, actual)

    def test_counterfactual_inference_for_both_models(self):
        for kind in (JointABSA, PairABSA):
            with self.subTest(kind=kind.__name__), contextlib.redirect_stdout(io.StringIO()):
                model = kind(BertModel(self.config), dropout=0., use_cls=False)
                checkpoint = self.root / kind.__name__
                model.save_checkpoint(checkpoint, self.tokenizer, {"max_length": 32, "stride": 4})
                output = self.root / (kind.__name__ + "_predictions")
                args = Namespace(pairs=str(ROOT / "data/diagnostics/neutral_scope_pairs.csv"),
                                 checkpoint=str(checkpoint), output=str(output), include_unreviewed=True,
                                 batch_size=16, device="cpu", precision="no")
                predict_pairs(args)
                result = json.loads((output / "pair_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(result["groups"]["all"]["pairs"], 40)
                self.assertFalse(result["human_review_complete"])
                self.assertEqual(len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()), 80)
                args.include_unreviewed = False
                with self.assertRaisesRegex(ValueError, "No approved pairs"):
                    predict_pairs(args)

    def test_cli_training_and_cross_validation_forward_ablation_flag(self):
        import subprocess
        import sys
        import os
        from absa_enhanced.cross_validation import training_command
        initial = self.root / "initial"
        BertModel(self.config).save_pretrained(initial)
        self.tokenizer.save_pretrained(initial)
        def fixture(name, suffix):
            rows = []
            for word, label in (("good", "positive"), ("bad", "negative"), ("okay", "neutral")):
                rows.append(f'<sentence id="{word}"><text>food {word} {suffix}</text><aspectTerms>'
                            f'<aspectTerm term="food" from="0" to="4" polarity="{label}"/>'
                            '</aspectTerms></sentence>')
            path = self.root / (name + ".xml")
            path.write_text("<sentences>" + "".join(rows) + "</sentences>", encoding="utf-8")
            return path
        train, val, test = fixture("train", "."), fixture("val", "good"), fixture("test", "bad")
        output = self.root / "run"
        command = [sys.executable, "-m", "absa_enhanced", "train", "--mode", "pair", "--dataset", "custom",
                   "--encoder", str(initial), "--train", str(train), "--validation", str(val), "--test", str(test),
                   "--output", str(output), "--epochs", "1", "--max-length", "32", "--batch-size", "3",
                   "--grad-accum", "1", "--device", "cpu", "--precision", "no", "--no-cls-feature",
                   "--no-gradient-checkpointing"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=180,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"})
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        settings = json.loads((output / "best/settings.json").read_text(encoding="utf-8"))
        self.assertFalse(settings["use_cls"])
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        args = Namespace(**config)
        args.no_cls_feature = True
        forwarded = training_command(args, self.root, self.root / "another")
        self.assertIn("--no-cls-feature", forwarded)

    def test_counterfactual_inference_for_baseline_checkpoint(self):
        from transformers import BertForSequenceClassification
        from neutral_diagnostics.metrics import LABELS
        config = BertConfig.from_dict(self.config.to_dict())
        config.num_labels = 3
        config.id2label = dict(enumerate(LABELS))
        config.label2id = {label: i for i, label in enumerate(LABELS)}
        checkpoint = self.root / "baseline"
        BertForSequenceClassification(config).save_pretrained(checkpoint)
        self.tokenizer.save_pretrained(checkpoint)
        args = Namespace(pairs=str(ROOT / "data/diagnostics/neutral_scope_pairs.csv"),
                         checkpoint=str(checkpoint), output=str(self.root / "baseline_predictions"),
                         include_unreviewed=True, batch_size=16, device="cpu", precision="no")
        with contextlib.redirect_stdout(io.StringIO()):
            predict_pairs(args)
        result = json.loads((self.root / "baseline_predictions/pair_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(result["groups"]["all"]["pairs"], 40)


if __name__ == "__main__":
    unittest.main()
