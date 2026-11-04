"""Windows-friendly entry points: train, evaluate and predict."""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .data import Aspect, Collator, Document, SENTIMENTS, WindowDataset, prepare_splits, read_documents
from .evaluation import autocast_context, evaluate, move_batch, save_results
from .model import JointABSA, joint_loss


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install the CUDA build of PyTorch or use --device cpu")
    return torch.device(name)


def resolve_precision(name, device):
    if device.type != "cuda":
        if name not in ("auto", "no"):
            raise ValueError("Mixed precision is only enabled for CUDA in this implementation")
        return "no"
    if name == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unavailable on this GPU; use --precision fp16")
    return name


def dataset_paths(args):
    if args.dataset == "custom":
        if not args.train:
            raise ValueError("--dataset custom requires --train")
        return args.train, args.validation, args.test
    root = Path(args.data_root)
    if args.dataset == "mams":
        directory = root / "MAMS/data/MAMS-ATSA/raw"
        paths = (directory / "train.xml", directory / "val.xml", directory / "test.xml")
    else:
        directory = root / "SemEval2014"
        domain = "Restaurants" if args.dataset == "restaurant" else "Laptop"
        candidates = list(directory.rglob(f"{domain}_Train_v2.xml"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one {domain}_Train_v2.xml under {directory}; found {len(candidates)}")
        test_file = "Restaurants_Test_Gold.xml" if args.dataset == "restaurant" else "Laptops_Test_Gold.xml"
        paths = (candidates[0], None, directory / "ABSA_Gold_TestData" / test_file)
    return args.train or paths[0], args.validation or paths[1], args.test or paths[2]


def validate_training_args(args):
    if min(args.epochs, args.batch_size, args.grad_accum, args.patience) < 1:
        raise ValueError("epochs, batch-size, grad-accum and patience must be positive")
    if args.learning_rate <= 0 or args.head_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.smoothing < 1:
        raise ValueError("warmup-ratio and smoothing must be in [0, 1)")
    if args.extraction_weight < 0 or args.local_radius < 0:
        raise ValueError("extraction-weight and local-radius must be nonnegative")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0:
        raise ValueError("dropout must be in [0, 1), and weight-decay must be nonnegative")


def train(args):
    validate_training_args(args)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new directory")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    train_path, validation_path, test_path = dataset_paths(args)
    train_docs, val_docs, test_docs, audit = prepare_splits(train_path, validation_path, test_path,
                                                        args.validation_fraction, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    tokenizer.padding_side = "right"
    train_data = WindowDataset(train_docs, tokenizer, args.max_length, args.stride)
    val_data = WindowDataset(val_docs, tokenizer, args.max_length, args.stride)
    # Validate every test annotation before spending time training. No scores are used.
    test_data = WindowDataset(test_docs, tokenizer, args.max_length, args.stride) if test_docs else None
    model = JointABSA.from_pretrained(args.encoder, local_radius=args.local_radius,
                                     fusion=not args.no_fusion, dropout=args.dropout)
    if args.max_length > getattr(model.encoder.config, "max_position_embeddings", args.max_length):
        raise ValueError("max-length exceeds this encoder's supported position range")
    if not args.no_gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    counts = audit["training_sentiment_counts"]
    class_weights = None
    if args.class_weighting == "sqrt":
        class_weights = torch.tensor([1 / math.sqrt(counts[label]) for label in SENTIMENTS], device=device)
        class_weights = class_weights / class_weights.mean()
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        encoder_parameter = name.startswith("encoder.")
        decay = not (name.endswith("bias") or "norm" in name.lower())
        key = encoder_parameter, decay
        groups.setdefault(key, []).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": parameters, "lr": args.learning_rate if encoder else args.head_learning_rate,
         "weight_decay": args.weight_decay if decay else 0.0}
        for (encoder, decay), parameters in groups.items()])
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        collate_fn=Collator(tokenizer), num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    total_updates = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"max_length": args.max_length, "stride": args.stride, "encoder": args.encoder}
    config = {**vars(args), "resolved_precision": precision, "resolved_device": str(device),
              "train_path": str(train_path), "validation_path": str(validation_path), "test_path": str(test_path),
              "audit": audit, "train_sentences": len(train_docs), "validation_sentences": len(val_docs),
              "token_alignment": {"train": train_data.alignment_stats,
                                  "validation": val_data.alignment_stats,
                                  "test": test_data.alignment_stats if test_data is not None else None},
              "train_windows": len(train_data), "effective_batch_size": args.batch_size * args.grad_accum,
              "parameters": sum(p.numel() for p in model.parameters()),
              "class_weights": class_weights.tolist() if class_weights is not None else None,
              "versions": {"torch": torch.__version__}}
    import transformers
    config["versions"]["transformers"] = transformers.__version__
    write_json(output / "run_config.json", config)
    # Record text as well as IDs so the split can be audited against the baseline.
    write_json(output / "split_manifest.json", {
        name: [{"uid": d.uid, "text": d.text} for d in documents]
        for name, documents in (("train", train_docs), ("validation", val_docs), ("test", test_docs or []))})
    print(json.dumps({"device": str(device), "precision": precision, "audit": audit,
                      "non_exact_token_spans": {name: value["non_exact_token_spans"]
                                                for name, value in config["token_alignment"].items() if value is not None},
                      "train_windows": len(train_data), "effective_batch_size": config["effective_batch_size"]}), flush=True)
    history, best_score, bad_epochs = [], -1.0, 0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = Counter()
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress):
            inputs, valid, spans, labels, bio = move_batch(batch, device)
            with autocast_context(device, precision):
                bio_logits, sentiment_logits = model(inputs, valid, spans)
                loss, sentiment_loss, extraction_loss = joint_loss(
                    bio_logits, sentiment_logits, bio, labels, class_weights,
                    args.extraction_weight, args.smoothing)
                # Correctly scale the last, potentially shorter accumulation group.
                group_size = min(args.grad_accum, len(loader) - (step // args.grad_accum) * args.grad_accum)
                scaled_loss = loss / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; retry with --precision no and a smaller learning rate")
            scaler.scale(scaled_loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            totals["loss"] += loss.item()
            totals["sentiment_loss"] += sentiment_loss.item()
            totals["extraction_loss"] += extraction_loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        val_metrics, gold_rows, sentence_rows = evaluate(model, tokenizer, val_data, device, args.batch_size, precision)
        score = (val_metrics["gold_aspect_sentiment"]["macro_f1"] if args.selection_metric == "gold_macro_f1"
                 else val_metrics["end_to_end"]["f1"])
        record = {"epoch": epoch + 1, "training": {k: v / len(loader) for k, v in totals.items()},
                  "validation": val_metrics, "selection_score": score}
        history.append(record)
        write_json(output / "history.json", history)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, bad_epochs = score, 0
            model.save_checkpoint(output / "best", tokenizer, metadata)
            save_results(output, val_metrics, gold_rows, sentence_rows, "validation_best")
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without improvement", flush=True)
            break
    write_json(output / "training_summary.json", {
        "seconds": time.monotonic() - started, "epochs_completed": len(history),
        "selection_metric": args.selection_metric, "best_validation_score": best_score,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None})
    if test_data is not None:
        # Release training tensors and optimizer state before loading the best model.
        del optimizer, scheduler, loss, scaled_loss, bio_logits, sentiment_logits, sentiment_loss, extraction_loss
        del batch, inputs, valid, spans, labels, bio
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        state = torch.load(output / "best/model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        metrics, gold_rows, sentence_rows = evaluate(model, tokenizer, test_data, device, args.batch_size, precision)
        save_results(output, metrics, gold_rows, sentence_rows)
        print("Final held-out test:", json.dumps(metrics, ensure_ascii=False, indent=2))


def predict_or_evaluate(args):
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    model, tokenizer, settings = JointABSA.load_checkpoint(args.checkpoint)
    tokenizer.padding_side = "right"
    model.to(device)
    max_length, stride = settings["max_length"], settings["stride"]
    if args.command == "evaluate":
        documents = read_documents(args.test)
    else:
        if not args.text.strip():
            raise ValueError("text must not be empty")
        aspects = ()
        if args.aspect:
            if args.aspect_start is None:
                if args.text.count(args.aspect) != 1:
                    raise ValueError("aspect must occur exactly once, or provide --aspect-start")
                start = args.text.index(args.aspect)
            else:
                start = args.aspect_start
            if start < 0 or args.text[start:start + len(args.aspect)] != args.aspect:
                raise ValueError("aspect-start does not match aspect text")
            aspects = (Aspect(start, start + len(args.aspect), "neutral"),)
        elif args.aspect_start is not None:
            raise ValueError("--aspect-start requires --aspect")
        documents = [Document("user", args.text, aspects)]
    data = WindowDataset(documents, tokenizer, max_length, stride)
    metrics, gold_rows, sentence_rows = evaluate(model, tokenizer, data, device, args.batch_size,
                                                precision, compute_metrics=args.command == "evaluate")
    if args.command == "evaluate":
        save_results(args.output, metrics, gold_rows, sentence_rows)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.aspect:
        result = gold_rows[0]
        # The placeholder neutral label is only for constructing the inference target.
        result.pop("label")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"text": args.text, "predictions": sentence_rows[0]["predictions"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", choices=["restaurant", "laptop", "mams", "custom"], default="restaurant")
    tr.add_argument("--data-root", default="Database")
    tr.add_argument("--train")
    tr.add_argument("--validation")
    tr.add_argument("--test")
    tr.add_argument("--output", required=True)
    tr.add_argument("--encoder", default="microsoft/deberta-v3-base")
    tr.add_argument("--epochs", type=int, default=6)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--learning-rate", type=float, default=1e-5)
    tr.add_argument("--head-learning-rate", type=float, default=1e-4)
    tr.add_argument("--weight-decay", type=float, default=0.01)
    tr.add_argument("--warmup-ratio", type=float, default=0.1)
    tr.add_argument("--max-length", type=int, default=192)
    tr.add_argument("--stride", type=int, default=64)
    tr.add_argument("--local-radius", type=int, default=16)
    tr.add_argument("--extraction-weight", type=float, default=0.5)
    tr.add_argument("--class-weighting", choices=["sqrt", "none"], default="sqrt")
    tr.add_argument("--smoothing", type=float, default=0.05)
    tr.add_argument("--dropout", type=float, default=0.15)
    tr.add_argument("--no-fusion", action="store_true")
    tr.add_argument("--no-gradient-checkpointing", action="store_true")
    tr.add_argument("--selection-metric", choices=["gold_macro_f1", "end_to_end_f1"], default="gold_macro_f1")
    tr.add_argument("--patience", type=int, default=2)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--validation-fraction", type=float, default=0.15)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--test", required=True)
    ev.add_argument("--output", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--text", required=True)
    pr.add_argument("--aspect")
    pr.add_argument("--aspect-start", type=int)
    for command in (tr, ev, pr):
        command.add_argument("--batch-size", type=int, default=2)
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command.add_argument("--precision", choices=["auto", "no", "bf16", "fp16"], default="auto")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        if args.batch_size < 1:
            parser.error("batch-size must be positive")
        predict_or_evaluate(args)


if __name__ == "__main__":
    main()
