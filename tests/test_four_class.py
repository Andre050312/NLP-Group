"""Four-label regression tests with tiny local BERT fixtures; no model downloads."""
from collections import Counter
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from bert_baseline.data import LABELS, classification_metrics
from absa_enhanced import heads
from absa_enhanced.data import Aspect, Document, WindowDataset, prepare_splits
from absa_enhanced.labels import sentiment_labels
from absa_enhanced.model import JointABSA
from absa_enhanced.pair import PairABSA, PairDataset
from absa_enhanced.evaluation import evaluate
from absa_enhanced.cross_validation import aggregate, grouped_outer_folds, inner_holdout, training_command, write_documents
from absa_enhanced.cli import dataset_paths
from scripts import tune_neutral_offset as tno
from scripts.prepare_restaurant4 import export_dataset
from scripts.restaurant_format import convert_source, xml_statistics

ROOT = Path(__file__).resolve().parents[1]


def documents(count=12, suffix="train"):
    # All four labels in independent sentences keeps these tests small and deterministic.
    return [Document(f"{suffix}-{i}", f"food service price menu review {suffix} {i}",
                     (Aspect(0, 4, "negative"), Aspect(5, 12, "neutral"),
                      Aspect(13, 18, "positive"), Aspect(19, 23, "conflict"))) for i in range(count)]


class FourClassModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "food", "service", "price", "menu", "review"]
        vocab_path = self.root / "vocab.txt"
        vocab_path.write_text("\n".join(vocab), encoding="utf-8")
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
        self.config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=24, max_position_embeddings=64)

    def model(self, kind=PairABSA, labels=LABELS, **kwargs):
        return kind(BertModel(self.config), dropout=0, labels=labels, **kwargs)

    def test_pair_and_window_datasets_retain_conflict_only_when_requested(self):
        docs = documents(1)
        self.assertEqual(len(PairDataset(docs, self.tokenizer, 32)), 3)
        pair = PairDataset(docs, self.tokenizer, 32, labels=LABELS)
        self.assertEqual(len(pair), 4)
        self.assertEqual([r["targets"][0][2] for r in pair.rows], [0, 1, 2, 3])
        three = WindowDataset(docs, self.tokenizer, 32, 4)
        four = WindowDataset(docs, self.tokenizer, 32, 4, labels=LABELS)
        self.assertEqual(len(three.rows[0]["targets"]), 3)
        self.assertEqual(len(four.rows[0]["targets"]), 4)
        self.assertEqual(three.rows[0]["bio"], four.rows[0]["bio"])

    def test_four_class_softmax_dimensions_loss_and_empty_spans(self):
        for kind in (PairABSA, JointABSA):
            model = self.model(kind)
            self.assertEqual(model.sentiment[-1].out_features, 4)
            self.assertEqual(model.sentiment[2].in_features, 3 * self.config.hidden_size)
            hidden = torch.zeros(1, 8, self.config.hidden_size)
            empty = model.classify(hidden, torch.ones(1, 8, dtype=torch.bool), torch.empty(0, 3, dtype=torch.long))
            self.assertEqual(tuple(empty.shape), (0, 4))
            self.assertEqual(tuple(model.probabilities(empty).shape), (0, 4))
            loss = model.sentiment_loss(torch.randn(4, 4, requires_grad=True), torch.arange(4))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
        self.assertEqual(self.model(use_cls=False).sentiment[2].in_features, 2 * self.config.hidden_size)

    def test_four_class_rejects_every_special_head_and_reordered_labels(self):
        for head in ("gate", "ordinal", "ordered"):
            for kind in (PairABSA, JointABSA):
                with self.assertRaisesRegex(ValueError, "requires softmax"):
                    self.model(kind, sentiment_head=head)
            with self.assertRaisesRegex(ValueError, "requires softmax"):
                heads.output_dim(head, 4)
            with self.assertRaisesRegex(ValueError, "requires softmax"):
                heads.probabilities(head, torch.zeros(2, 4), num_classes=4)
            with self.assertRaisesRegex(ValueError, "requires softmax"):
                heads.loss(head, torch.zeros(2, 4), torch.tensor([0, 3]), num_classes=4)
            with self.assertRaisesRegex(ValueError, "outside"):
                heads.loss(head, torch.zeros(1, heads.output_dim(head)), torch.tensor([3]))
        with self.assertRaises(ValueError):
            sentiment_labels(("negative", "positive", "neutral", "conflict"))

    def test_neutral_offset_scales_only_index_one(self):
        logits = torch.tensor([[.1, .2, .3, .4]])
        base = heads.probabilities("softmax", logits)
        actual = heads.probabilities("softmax", logits, 3)
        expected = base * torch.tensor([1., 3., 1., 1.])
        expected /= expected.sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(LABELS[1], "neutral")

    def test_pair_and_joint_evaluation_can_choose_fourth_probability(self):
        for kind, dataset_kind in ((PairABSA, PairDataset), (JointABSA, WindowDataset)):
            with self.subTest(kind=kind.__name__):
                model = self.model(kind)
                with torch.no_grad():
                    model.sentiment[-1].weight.zero_()
                    model.sentiment[-1].bias.copy_(torch.tensor([-5., -4., -3., 5.]))
                data = dataset_kind(documents(1), self.tokenizer, 32, 4, labels=LABELS)
                metrics, rows, _ = evaluate(model, self.tokenizer, data, torch.device("cpu"))
                scores = metrics["gold_aspect_sentiment"]
                self.assertEqual(scores["label_order"], list(LABELS))
                self.assertEqual(scores["per_class"]["conflict"]["support"], 1)
                self.assertEqual(scores["per_class"]["conflict"]["recall"], 1)
                self.assertEqual(scores["examples"], 4)
                self.assertEqual(metrics["excluded_conflict_targets"], 0)
                self.assertTrue(all(r["prediction"] == "conflict" and len(r["probabilities"]) == 4 for r in rows))
                if kind is JointABSA:
                    self.assertEqual(metrics["end_to_end"]["gold"], 4)

    def test_checkpoint_axes_round_trip_and_mismatch_failure(self):
        for kind in (PairABSA, JointABSA):
            for labels in (LABELS[:3], LABELS):
                directory = self.root / f"{kind.__name__}-{len(labels)}"
                model = self.model(kind, labels)
                model.save_checkpoint(directory, self.tokenizer, {"max_length": 32, "stride": 4})
                settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
                self.assertEqual(settings["labels"], list(labels))
                restored, _, _ = JointABSA.load_checkpoint(directory)
                self.assertEqual(restored.labels, labels)
                self.assertEqual(restored.sentiment[-1].out_features, len(labels))
                settings["labels"] = list(LABELS if len(labels) == 3 else LABELS[:3])
                (directory / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "labels/output layer mismatch"):
                    JointABSA.load_checkpoint(directory)

    def test_legacy_checkpoint_without_labels_stays_three_class(self):
        directory = self.root / "legacy"
        self.model(labels=LABELS[:3]).save_checkpoint(directory, self.tokenizer, {})
        settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
        settings.pop("labels")
        (directory / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        model, _, _ = JointABSA.load_checkpoint(directory)
        self.assertEqual(model.labels, LABELS[:3])

    def run_cli(self, *args, success=True):
        result = subprocess.run([sys.executable, "-m", "absa_enhanced", *map(str, args)], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", timeout=180,
                                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def test_cli_rejects_special_heads_before_loading_any_encoder(self):
        for head in ("gate", "ordinal", "ordered"):
            result = self.run_cli("train", "--labels", "four", "--sentiment-head", head,
                                  "--encoder", self.root / "nonexistent", "--output", self.root / head, success=False)
            self.assertIn("requires softmax", result.stderr)
            self.assertFalse((self.root / head).exists())

    def test_four_class_cli_tiny_cpu_train_evaluate_predict_and_cv_plan(self):
        initial = self.root / "initial"
        BertModel(self.config).save_pretrained(initial)
        self.tokenizer.save_pretrained(initial)
        for name in ("train", "validation", "test"):
            write_documents(self.root / f"{name}.xml", documents(12 if name == "train" else 2, name))
        for mode in ("pair", "joint"):
            output = self.root / mode
            self.run_cli("train", "--labels", "four", "--mode", mode, "--dataset", "custom",
                         "--train", self.root / "train.xml", "--validation", self.root / "validation.xml",
                         "--test", self.root / "test.xml", "--output", output, "--encoder", initial,
                         "--epochs", 1, "--batch-size", 6, "--grad-accum", 1, "--max-length", 32,
                         "--stride", 4, "--device", "cpu", "--no-gradient-checkpointing")
            metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["gold_aspect_sentiment"]["per_class"]["conflict"]["support"], 2)
            self.assertEqual(json.loads((output / "best/settings.json").read_text(encoding="utf-8"))["labels"], list(LABELS))
            self.run_cli("evaluate", "--checkpoint", output / "best", "--test", self.root / "test.xml",
                         "--output", self.root / f"eval-{mode}", "--device", "cpu", "--neutral-offset", 2)
            prediction = self.run_cli("predict", "--checkpoint", output / "best", "--text", "food review",
                                      "--aspect", "food", "--device", "cpu")
            self.assertEqual(set(json.loads(prediction.stdout)["probabilities"]), set(LABELS))
        self.run_cli("cross-validate", "--labels", "four", "--mode", "pair", "--dataset", "custom",
                     "--train", self.root / "train.xml", "--encoder", self.root / "must-not-download",
                     "--output", self.root / "cv", "--folds", 3, "--validation-fraction", .25,
                     "--device", "cpu", "--plan-only")
        audit = json.loads((self.root / "cv/cv_config.json").read_text(encoding="utf-8"))["audit"]
        self.assertEqual(audit["labels"], list(LABELS))
        self.assertEqual(audit["pool_aspect_targets"], 48)
        self.assertFalse((self.root / "cv/fold_01/run").exists())
        config = json.loads((self.root / "cv/cv_config.json").read_text(encoding="utf-8"))
        command = training_command(SimpleNamespace(**config["parameters"]), self.root, self.root / "cv-run")
        self.assertEqual(command[command.index("--labels") + 1], "four")


class FourClassDataTests(unittest.TestCase):
    def test_export_preserves_original_conflict_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_documents(root / "source.xml", documents(2))
            content = (root / "source.xml").read_bytes()
            manifest = export_dataset({"train": content, "test": content}, root / "four")
            self.assertEqual(manifest["splits"]["train"]["statistics"]["label_counts"]["conflict"], 2)
            original = Counter(a.attrib["polarity"] for a in ET.fromstring(content).findall("./sentence/aspectTerms/aspectTerm"))
            exported = Counter(r["label"] for r in map(json.loads, (root / "four/train.jsonl").read_text(encoding="utf-8").splitlines()))
            self.assertEqual(original, exported)
            self.assertEqual(original["neutral"], 2)
            with self.assertRaises(ValueError):
                export_dataset({"train": content, "test": content}, root / "four")
            with self.assertRaisesRegex(ValueError, "protected"):
                export_dataset({"train": content, "test": content}, ROOT / "Database/Restaurant3/new-four")
            old, audit = convert_source(content, "semeval2014")
            self.assertEqual(audit["excluded_conflict_aspectTerm"], 2)
            self.assertNotIn("conflict", xml_statistics(old)["label_counts"])

    def test_committed_export_matches_source_manifest_and_three_class_subset(self):
        manifest = json.loads((ROOT / "Database/Restaurant4/manifest.json").read_text(encoding="utf-8"))
        catalog = json.loads((ROOT / "scripts/restaurant_sources.json").read_text(encoding="utf-8"))
        for split, count, total in (("train", 91, 3693), ("test", 14, 1134)):
            four = ET.parse(ROOT / f"Database/Restaurant4/{split}.xml")
            three = ET.parse(ROOT / f"Database/Restaurant3/{split}.xml")
            stats = xml_statistics((ROOT / f"Database/Restaurant4/{split}.xml").read_bytes(), include_conflict=True)
            self.assertEqual(stats["label_counts"]["conflict"], count)
            self.assertEqual(stats["aspect_terms"], total)
            self.assertEqual(stats, manifest["splits"][split]["statistics"])
            spec = next(s for s in catalog["files"] if s["path"] == f"Restaurant3/{split}.xml")
            self.assertEqual(manifest["splits"][split]["source_sha256"], spec["source_sha256"])
            content = (ROOT / f"Database/Restaurant4/{split}.xml").read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(content).hexdigest(), manifest["splits"][split]["sha256"])
            def targets(tree, exclude=False):
                return [(s.get("id"), a.get("term"), a.get("from"), a.get("to"), a.get("polarity"))
                        for s in tree.findall("sentence") for a in s.findall("./aspectTerms/aspectTerm")
                        if not exclude or a.get("polarity") != "conflict"]
            self.assertEqual(targets(four, True), targets(three))

    def test_four_class_splits_and_cv_include_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = documents(18)
            write_documents(root / "train.xml", docs)
            train, val, _, audit = prepare_splits(root / "train.xml", labels=LABELS)
            self.assertGreater(audit["training_sentiment_counts"]["conflict"], 0)
            self.assertFalse({d.text for d in train} & {d.text for d in val})
            pool = docs + [Document(f"bg-{i}", f"background {i}", ()) for i in range(3)]
            folds, groups = grouped_outer_folds(pool, folds=3, labels=LABELS)
            for training, held in folds:
                a, b = inner_holdout(pool, training, groups, .25, labels=LABELS)
                for indices in (a, b, held):
                    labels = {x.label for i in indices for x in pool[i].aspects}
                    self.assertEqual(labels, set(LABELS))
            plan = [{"fold": i+1, "held_out": list(range(i*6, i*6+6))} for i in range(3)]
            for fold in plan:
                run = root / f"fold_{fold['fold']:02d}/run"
                run.mkdir(parents=True)
                rows = [{"sentence_id": f"{docs[i].uid}:{j}", "text": docs[i].text,
                         "aspect": docs[i].text[a.start:a.end], "start": a.start, "end": a.end,
                         "label": a.label, "prediction": a.label}
                        for j, i in enumerate(fold["held_out"]) for a in docs[i].aspects]
                metrics = {"gold_aspect_sentiment": classification_metrics(list(range(4))*6, list(range(4))*6, LABELS),
                           "aspect_extraction": None, "end_to_end": None}
                (run / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                (run / "test_gold_predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
            summary = aggregate(root, docs, plan, labels=LABELS)
            self.assertEqual(summary["oof_gold_aspect_sentiment"]["per_class"]["conflict"]["support"], 18)
            self.assertEqual(summary["oof_gold_aspect_sentiment"]["examples"], 72)

    def test_conflict_only_sentences_participate_in_four_class_validation_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.xml"
            docs = documents(12) + [Document(f"only-{i}", f"food conflict only {i}", (Aspect(0, 4, "conflict"),))
                                    for i in range(30)]
            write_documents(path, docs)
            _, validation, _, _ = prepare_splits(path, fraction=.4, seed=42, labels=LABELS)
            self.assertEqual(len(validation), 17)
            self.assertTrue(any(d.aspects == (Aspect(0, 4, "conflict"),) for d in validation))

    def test_dataset_paths_follow_explicit_label_choice(self):
        args = SimpleNamespace(dataset="restaurant", data_root="Database", train=None, validation=None, test=None)
        self.assertEqual(dataset_paths(args)[0], Path("Database/Restaurant3/train.xml"))
        args.labels = "four"
        self.assertEqual(dataset_paths(args)[0], Path("Database/Restaurant4/train.xml"))

    def test_offset_infers_four_axes_without_changing_three_class_default(self):
        def row(labels):
            return {"label": labels[-1], "probabilities": {label: (.7 if label == labels[-1] else .1) for label in labels}}
        self.assertEqual(tno.LABELS, LABELS[:3])
        four = [row(LABELS)]
        self.assertEqual(tno.predict(four, 1), ([3], [3]))
        self.assertEqual(tno.predict(four, 10), ([3], [1]))
        self.assertEqual(tno.metrics(four, 1)["per_class"]["conflict"]["support"], 1)
        self.assertEqual(tno.labels_for_rows([row(LABELS[:3])]), LABELS[:3])
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            tno.predict(four + [row(LABELS[:3])], 1)

    def test_offset_scripts_keep_conflict_support_and_reject_axis_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv = root / "cv"
            for fold in range(2):
                run = cv / f"fold_{fold}/run"
                run.mkdir(parents=True)
                rows = [{"text": f"fold {fold} sentence {i}", "label": label,
                         "label_order": list(LABELS),
                         "probabilities": {name: .7 if name == label else .1 for name in LABELS}}
                        for i, label in enumerate(LABELS)]
                for filename in (tno.VALIDATION, tno.TEST):
                    (run / filename).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            for script, arguments, filename in (
                ("tune_neutral_offset.py", ["--runs", str(cv / "fold_0/run")], "single.json"),
                ("cv_offset.py", ["--cv-dir", str(cv)], "pooled.json")):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *arguments,
                                         "--grid", "1,2", "--bootstrap", "20", "--output", str(root / filename)],
                                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
            single = json.loads((root / "single.json").read_text(encoding="utf-8"))["run"]
            self.assertEqual(single["labels"], list(LABELS))
            self.assertEqual(single["baseline"]["test"]["per_class"]["conflict"]["support"], 1)
            summary = json.loads((root / "pooled.json").read_text(encoding="utf-8"))["cv"]
            self.assertEqual(summary["oof_argmax"]["per_class"]["conflict"]["support"], 2)
            for script, extra, output in (
                ("fixed_offset_check.py", ["--runs", str(cv / "fold_0/run"), "--cv-dir", str(cv)], "fixed.json"),
                ("offset_validation_size.py", ["--runs", str(cv / "fold_0/run"), "--sizes", "4", "--draws", "2"], "size.json")):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra,
                                         "--grid", "1,2", "--output", str(root / output)],
                                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads((root / output).read_text(encoding="utf-8"))
                self.assertEqual((data if script.startswith("fixed") else data["run"])["labels"], list(LABELS))
            path = cv / "fold_0/run" / tno.TEST
            rows = tno.read_rows(path)
            for row in rows:
                row["probabilities"].pop("conflict")
                row["label_order"] = list(LABELS[:3])
            path.write_text("".join(json.dumps(r) + "\n" for r in rows if r["label"] != "conflict"), encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/tune_neutral_offset.py"),
                                     "--runs", str(cv / "fold_0/run"), "--bootstrap", "0"],
                                    cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("axes differ", result.stderr)
