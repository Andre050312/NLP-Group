"""Train, evaluate, or predict with the BERT ABSA baseline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from .data import LABELS, Example, read_xml, split_train_validation, ensure_disjoint, classification_metrics
from .model import AspectDataset, resolve_device


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def make_loader(examples, tokenizer, labels, args, shuffle=False):
    return DataLoader(AspectDataset(examples, tokenizer, labels, args.max_length),
                      batch_size=args.batch_size, shuffle=shuffle,
                      collate_fn=DataCollatorWithPadding(tokenizer))


def evaluate(model, loader, device, labels):
    model.eval()
    gold, predicted, probabilities = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            count = len(batch["labels"])
            total_loss += output.loss.item()*count
            gold.extend(batch["labels"].cpu().tolist())
            predicted.extend(output.logits.argmax(-1).cpu().tolist())
            probabilities.extend(output.logits.softmax(-1).cpu().tolist())
    metrics = classification_metrics(gold, predicted, labels)
    metrics["loss"] = total_loss/len(gold)
    return metrics, predicted, probabilities


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new directory")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    labels = LABELS if args.include_conflict else LABELS[:3]
    train_examples, train_stats = read_xml(args.train, args.include_conflict)
    if args.validation:
        validation_examples, validation_stats = read_xml(args.validation, args.include_conflict)
    else:
        train_examples, validation_examples = split_train_validation(train_examples, args.validation_fraction, args.seed)
        validation_stats = {"examples": len(validation_examples)}
    ensure_disjoint(train=train_examples, validation=validation_examples)
    # Audit the held-out split for leakage, but never select models on its scores.
    test_examples = None
    overlap_removed = {"train_examples": 0, "validation_examples": 0}
    if args.test:
        test_examples, _ = read_xml(args.test, args.include_conflict)
        # Official SemEval files contain duplicate texts across train/test. Preserve
        # the complete test set and explicitly exclude matching training/validation rows.
        test_texts = {x.text for x in test_examples}
        clean_train = [x for x in train_examples if x.text not in test_texts]
        clean_validation = [x for x in validation_examples if x.text not in test_texts]
        overlap_removed = {"train_examples": len(train_examples)-len(clean_train),
                           "validation_examples": len(validation_examples)-len(clean_validation)}
        train_examples, validation_examples = clean_train, clean_validation
        if any(overlap_removed.values()):
            print("Excluded exact test-text duplicates:", json.dumps(overlap_removed), flush=True)
        ensure_disjoint(train=train_examples, validation=validation_examples, test=test_examples)
    if not train_examples or not validation_examples:
        raise ValueError("Training/validation split is empty after overlap removal")
    missing = set(labels)-{x.label for x in train_examples}
    if missing:
        raise ValueError(f"Training split is missing labels: {sorted(missing)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={label:i for i,label in enumerate(labels)})
    device = resolve_device(args.device)
    model.to(device)
    if args.freeze_encoder:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    train_loader = make_loader(train_examples, tokenizer, labels, args, shuffle=True)
    validation_loader = make_loader(validation_examples, tokenizer, labels, args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    config = {**vars(args), "labels": list(labels), "resolved_device": str(device),
              "train_source_statistics": train_stats, "validation_statistics": validation_stats,
              "excluded_test_overlap": overlap_removed,
              "train_examples": len(train_examples), "validation_examples": len(validation_examples),
              "train_label_counts": dict(Counter(x.label for x in train_examples)),
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(output/"run_config.json", config)
    write_json(output/"split_manifest.json", {
        "train": sorted({x.sentence_id for x in train_examples}),
        "validation": sorted({x.sentence_id for x in validation_examples})})
    best_f1, history = -1.0, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.base_model.eval()  # fixed features also disable encoder dropout
        total_loss = 0.0
        for batch in train_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += result.loss.item()*len(batch["labels"])
        metrics, _, _ = evaluate(model, validation_loader, device, labels)
        record = {"epoch": epoch+1, "train_loss": total_loss/len(train_examples), "validation": metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            model.save_pretrained(output/"best")
            tokenizer.save_pretrained(output/"best")
        write_json(output/"history.json", history)
    write_json(output/"training_summary.json", {"seconds": time.monotonic()-started,
                                               "best_validation_macro_f1": best_f1})
    if test_examples is not None:
        model = AutoModelForSequenceClassification.from_pretrained(output/"best").to(device)
        metrics, predicted, probabilities = evaluate(model, make_loader(test_examples, tokenizer, labels, args), device, labels)
        save_results(output, test_examples, labels, metrics, predicted, probabilities)
        print("Held-out test:", json.dumps(metrics, ensure_ascii=False), flush=True)


def save_results(directory, examples, labels, metrics, predicted, probabilities):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"test_metrics.json", metrics)
    with (directory/"predictions.jsonl").open("w", encoding="utf-8") as stream:
        for example, prediction, scores in zip(examples, predicted, probabilities):
            stream.write(json.dumps({**example.to_dict(), "prediction": labels[prediction],
                                     "probabilities": dict(zip(labels, scores))}, ensure_ascii=False)+"\n")


def load_checkpoint(args):
    directory = Path(args.checkpoint)
    # Require a locally trained checkpoint, avoiding accidental evaluation of a random head.
    if not directory.is_dir():
        raise ValueError("checkpoint must be a local saved model directory")
    model = AutoModelForSequenceClassification.from_pretrained(directory)
    labels = tuple(model.config.id2label[i] for i in range(model.config.num_labels))
    if not set(labels) <= set(LABELS):
        raise ValueError("Checkpoint lacks the baseline sentiment label mapping")
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True)
    device = resolve_device(args.device)
    return model.to(device), tokenizer, labels, device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--train", required=True)
    tr.add_argument("--validation")
    tr.add_argument("--test")
    tr.add_argument("--output", required=True)
    tr.add_argument("--model", default="google-bert/bert-base-uncased")
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--learning-rate", type=float, default=2e-5)
    tr.add_argument("--weight-decay", type=float, default=0.01)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--validation-fraction", type=float, default=0.15)
    tr.add_argument("--include-conflict", action="store_true")
    tr.add_argument("--freeze-encoder", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--test", required=True)
    ev.add_argument("--output", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--text", required=True)
    pr.add_argument("--aspect", required=True)
    pr.add_argument("--aspect-start", type=int, help="Required when aspect occurs more than once")
    for command in [tr, ev, pr]:
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--max-length", type=int, default=None,
                             help="Training default 128; evaluation/prediction inherit training length")
        command.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    if args.command == "train":
        args.max_length = args.max_length or 128
        train(args)
        return
    if args.max_length is None:
        config_path = Path(args.checkpoint).parent/"run_config.json"
        args.max_length = json.loads(config_path.read_text())["max_length"] if config_path.exists() else 128
    model, tokenizer, labels, device = load_checkpoint(args)
    if args.command == "evaluate":
        examples, _ = read_xml(args.test, "conflict" in labels)
        metrics, predicted, probabilities = evaluate(model, make_loader(examples, tokenizer, labels, args), device, labels)
        save_results(args.output, examples, labels, metrics, predicted, probabilities)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        if not args.aspect or args.aspect not in args.text:
            parser.error("aspect must be a nonempty exact substring of text")
        if args.aspect_start is None and args.text.count(args.aspect) != 1:
            parser.error("Repeated aspect: provide --aspect-start character offset")
        start = args.text.index(args.aspect) if args.aspect_start is None else args.aspect_start
        if start < 0 or args.text[start:start+len(args.aspect)] != args.aspect:
            parser.error("aspect-start does not match aspect")
        example = Example("user", args.text, args.aspect, start, start+len(args.aspect), labels[0])
        _, predictions, probabilities = evaluate(model, make_loader([example], tokenizer, labels, args), device, labels)
        print(json.dumps({"aspect": args.aspect, "prediction": labels[predictions[0]],
                          "probabilities": dict(zip(labels, probabilities[0]))}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
