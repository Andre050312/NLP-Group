"""Export a documented three-class derivative of original Restaurant XML."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

if __package__:
    from .download_datasets import EXPECTED_SHA256, PROJECT, main as download_main, resources, validate_xml
else:
    from download_datasets import EXPECTED_SHA256, PROJECT, main as download_main, resources, validate_xml

LABELS = ("negative", "neutral", "positive")


def export_split(source, xml_path, jsonl_path):
    tree = ET.parse(source)
    terms, excluded, categories_excluded, rows = Counter(), 0, 0, []
    for index, sentence in enumerate(tree.getroot().findall("sentence")):
        text = sentence.findtext("text")
        for container_name, element_name in (("aspectTerms", "aspectTerm"), ("aspectCategories", "aspectCategory")):
            container = sentence.find(container_name)
            if container is None:
                continue
            for element in list(container.findall(element_name)):
                label = element.attrib["polarity"]
                if label == "conflict":
                    container.remove(element)
                    if element_name == "aspectTerm":
                        excluded += 1
                    else:
                        categories_excluded += 1
                    continue
                if label not in LABELS:
                    raise ValueError(f"Unknown sentiment label: {label}")
                if element_name == "aspectTerm":
                    start, end = int(element.attrib["from"]), int(element.attrib["to"])
                    if text is None or not 0 <= start < end <= len(text) or text[start:end] != element.attrib["term"]:
                        raise ValueError("Original aspect offsets do not match original text")
                    terms[label] += 1
                    rows.append({"sentence_id": sentence.attrib.get("id", str(index)),
                                 "text": text, "aspect": element.attrib["term"],
                                 "start": start, "end": end, "label": label})
    # Serialization changes XML formatting only; text nodes are never normalized.
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    with Path(jsonl_path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    validate_xml(xml_path)
    return {"sentences": len(tree.getroot().findall("sentence")), "aspect_examples": len(rows),
            "label_counts": dict(terms), "excluded_conflict_aspects": excluded,
            "excluded_conflict_categories": categories_excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "Database")
    parser.add_argument("--output", type=Path, default=PROJECT / "Database/Restaurant3")
    args = parser.parse_args()
    # The collection is now three-class-only. Restore its already-normalized
    # files rather than recreating a mixed-label SemEval directory.
    download_main(["--dataset", "restaurant", "--data-root", str(args.data_root)])
    canonical = (args.data_root / "Restaurant3").resolve()
    output = args.output.resolve()
    if canonical != output:
        if output.exists():
            raise ValueError(f"Output exists: {output}")
        shutil.copytree(canonical, output)
    stats = {}
    for relative, url, revision in resources("restaurant"):
        path = output / Path(relative).name
        stats[path.stem] = validate_xml(path, EXPECTED_SHA256[relative])
    print(json.dumps({"output": str(output), "splits": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
