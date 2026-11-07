"""Leakage, OOF and CLI cross-validation tests using local synthetic data."""
from argparse import Namespace
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from absa_enhanced.cross_validation import (
    aggregate, build_plan, group_documents, grouped_outer_folds, inner_holdout, write_documents,
)
from absa_enhanced.data import Aspect, Document, SENTIMENTS
from bert_baseline.data import classification_metrics

ROOT = Path(__file__).resolve().parents[1]


def fixture_documents(count=30):
    aspects = (Aspect(0, 4, "positive"), Aspect(5, 12, "negative"), Aspect(13, 18, "neutral"))
    return [Document(f"r{i // 2}:s{i % 2}:{i}", f"food service price review {i}", aspects) for i in range(count)]


class CrossValidationTests(unittest.TestCase):
    def test_review_and_duplicate_text_groups_are_connected(self):
        documents = fixture_documents(4)
        documents.append(Document("other:s0:4", documents[0].text, documents[0].aspects))
        groups = group_documents(documents, by_review=True)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[4])
        self.assertNotEqual(groups[0], groups[2])

    def test_outer_and_inner_sets_do_not_share_groups(self):
        documents = fixture_documents()
        folds, groups = grouped_outer_folds(documents, folds=3, seed=42, by_review=True)
        coverage = Counter()
        for training, held in folds:
            inner_train, inner_val = inner_holdout(documents, training, groups, .2, 42)
            sets = [{groups[i] for i in indices} for indices in (inner_train, inner_val, held)]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            self.assertEqual(set(inner_train) | set(inner_val), set(training))
            coverage.update(held)
        self.assertEqual(coverage, Counter({i: 1 for i in range(len(documents))}))
        self.assertEqual(folds, grouped_outer_folds(documents, 3, 42, True)[0])

    def test_reserved_test_overlap_is_excluded_before_folding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = fixture_documents(18)
            write_documents(root / "train.xml", documents)
            write_documents(root / "test.xml", documents[:1])
            args = Namespace(group_by="sentence", dataset="custom", include_validation=False,
                             split_seed=42, seed=43, folds=3, validation_fraction=.25)
            pool, _, _, audit, _ = build_plan(args, root / "train.xml", None, root / "test.xml")
            self.assertEqual(len(pool), 17)
            self.assertNotIn(documents[0].text, {d.text for d in pool})
            self.assertEqual(audit["excluded_reserved_overlap_sentences"], 1)
            self.assertFalse(audit["official_test_evaluated"])

    def test_oof_aggregation_is_complete_and_rejects_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = fixture_documents(9)
            plan = [{"fold": i + 1, "held_out": list(range(i * 3, i * 3 + 3))} for i in range(3)]
            for fold in plan:
                run = root / f"fold_{fold['fold']:02d}" / "run"
                run.mkdir(parents=True)
                rows, labels = [], []
                for local, index in enumerate(fold["held_out"]):
                    document = documents[index]
                    for aspect in document.aspects:
                        labels.append(SENTIMENTS.index(aspect.label))
                        rows.append({"sentence_id": f"{document.uid}:{local}", "text": document.text,
                                     "aspect": document.text[aspect.start:aspect.end], "start": aspect.start,
                                     "end": aspect.end, "label": aspect.label, "prediction": aspect.label})
                metrics = {"gold_aspect_sentiment": classification_metrics(labels, labels, SENTIMENTS),
                           "aspect_extraction": None, "end_to_end": None}
                (run / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                (run / "test_gold_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = aggregate(root, documents, plan)
            self.assertEqual(result["oof_gold_aspect_sentiment"]["examples"], 27)
            self.assertEqual(result["oof_gold_aspect_sentiment"]["accuracy"], 1)
            self.assertEqual(result["fold_statistics"]["gold_aspect_sentiment"]["accuracy"]["sample_std"], 0)
            prediction_file = root / "fold_01/run/test_gold_predictions.jsonl"
            with prediction_file.open("a", encoding="utf-8") as stream:
                stream.write(prediction_file.read_text(encoding="utf-8").splitlines()[0] + "\n")
            with self.assertRaises(ValueError):
                aggregate(root, documents, plan)

    def test_cli_plan_train_and_resume_without_pretrained_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial"
            initial.mkdir()
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "food", "service", "price", "review"]
            (initial / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(initial / "vocab.txt"), do_lower_case=True)
            tokenizer.save_pretrained(initial)
            model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1,
                                          num_attention_heads=2, intermediate_size=32, max_position_embeddings=64))
            model.save_pretrained(initial)
            write_documents(root / "train.xml", fixture_documents(12))
            output = root / "cv"
            base = [sys.executable, "-m", "absa_enhanced", "cross-validate", "--mode", "pair",
                    "--dataset", "custom", "--train", str(root / "train.xml"), "--encoder", str(initial),
                    "--output", str(output), "--folds", "3", "--epochs", "1", "--batch-size", "6",
                    "--grad-accum", "1", "--max-length", "32", "--stride", "4", "--seed", "43",
                    "--split-seed", "42", "--validation-fraction", ".25", "--device", "cpu",
                    "--no-gradient-checkpointing"]
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"}

            def run(extra):
                result = subprocess.run(base + extra, cwd=ROOT, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=300, env=environment)
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
                return result.stdout

            run(["--plan-only"])
            self.assertFalse((output / "fold_01/run").exists())
            run(["--resume"])
            summary = json.loads((output / "cv_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["folds"], 3)
            self.assertFalse(summary["official_test_evaluated"])
            self.assertEqual(summary["oof_gold_aspect_sentiment"]["examples"], 36)
            self.assertEqual(run(["--resume"]).count("Skipping completed fold"), 3)


if __name__ == "__main__":
    unittest.main()
