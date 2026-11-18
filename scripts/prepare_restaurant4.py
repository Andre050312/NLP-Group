"""Export SemEval-2014's four original polarities separately from Restaurant3.

Use --train-source/--test-source for offline original XML, or explicitly --download
to fetch only the two pinned source XML files. Existing output is never overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

if __package__:
    from .restaurant_format import convert_source, xml_statistics
else:
    from restaurant_format import convert_source, xml_statistics

PROJECT = Path(__file__).resolve().parents[1]
LABELS = ("negative", "neutral", "positive", "conflict")
MAX_BYTES = 16 * 1024 * 1024


def export_dataset(sources, output, provenance=None):
    """Validate both original byte streams before creating a new output directory."""
    output = Path(output).resolve()
    protected = (PROJECT / "Database/Restaurant3").resolve()
    if output == protected or protected in output.parents:
        raise ValueError("Restaurant3 is protected; choose a separate Restaurant4 output")
    if output.exists():
        raise ValueError(f"Output already exists; no files overwritten: {output}")
    if set(sources) != {"train", "test"}:
        raise ValueError("Provide both original train and test sources")
    prepared, splits = {}, {}
    for split, content in sources.items():
        if len(content) > MAX_BYTES:
            raise ValueError("Unexpectedly large source XML")
        converted, audit = convert_source(content, "semeval2014", include_conflict=True)
        stats = xml_statistics(converted, include_conflict=True)
        root = ET.fromstring(content)
        original_counts = Counter(a.attrib["polarity"] for a in root.findall("./sentence/aspectTerms/aspectTerm"))
        if dict(original_counts) != stats["label_counts"]:
            raise ValueError("Four-class export must preserve every original aspect label")
        prepared[split] = converted
        splits[split] = {"statistics": stats, "conversion_audit": audit,
                         "source_sha256": hashlib.sha256(content).hexdigest(),
                         "sha256": hashlib.sha256(converted).hexdigest(),
                         **(provenance or {}).get(split, {})}
    output.mkdir(parents=True)
    for split, content in prepared.items():
        (output / f"{split}.xml").write_bytes(content)
        with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for i, sentence in enumerate(ET.fromstring(content).findall("sentence")):
                text = sentence.findtext("text")
                for a in sentence.findall("./aspectTerms/aspectTerm"):
                    row = {"sentence_id": sentence.get("id", str(i)), "text": text,
                           "aspect": a.attrib["term"], "start": int(a.attrib["from"]),
                           "end": int(a.attrib["to"]), "label": a.attrib["polarity"]}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"dataset": "SemEval-2014 Restaurant", "task": "explicit_aspect_term_sentiment",
                "labels": list(LABELS), "policy": "Preserve all original polarities; separate from Restaurant3",
                "splits": splits}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path)
    parser.add_argument("--test-source", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT / "Database/Restaurant4")
    args = parser.parse_args(argv)
    if args.download and (args.train_source or args.test_source):
        parser.error("Choose --download or two local source paths")
    if not args.download and not (args.train_source and args.test_source):
        parser.error("Provide --train-source and --test-source, or explicitly --download")
    # Refuse before networking, including the protected three-class path.
    output = args.output.resolve()
    protected = (PROJECT / "Database/Restaurant3").resolve()
    if output.exists() or output == protected or protected in output.parents:
        parser.error("Output exists or is inside protected Restaurant3; choose a new directory")
    contents, provenance = {}, {}
    if args.download:
        catalog = json.loads((Path(__file__).with_name("restaurant_sources.json")).read_text(encoding="utf-8"))
        for item in catalog["files"]:
            if item["dataset"] != "restaurant":
                continue
            request = Request(item["source_url"], headers={"User-Agent": "NLP-Group-Restaurant4/1.0"})
            with urlopen(request, timeout=60) as response:
                content = response.read(MAX_BYTES + 1)
            normalized = content.replace(b"\r\n", b"\n")
            if hashlib.sha256(normalized).hexdigest() != item["source_sha256"]:
                raise ValueError("Pinned original source checksum mismatch")
            split = Path(item["path"]).stem
            contents[split] = normalized
            provenance[split] = {"source_url": item["source_url"], "source_revision": item["source_revision"]}
    else:
        contents = {"train": args.train_source.read_bytes(), "test": args.test_source.read_bytes()}
        provenance = {"train": {"source_path": str(args.train_source)}, "test": {"source_path": str(args.test_source)}}
    result = export_dataset(contents, args.output, provenance)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
