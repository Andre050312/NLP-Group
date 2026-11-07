"""Normalize restaurant sources to explicit-target three-class XML."""
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

LABELS = {"negative", "neutral", "positive"}


def convert_source(content, source_format):
    root = ET.fromstring(content)
    audit = Counter()
    if source_format in ("native", "semeval2014"):
        if root.tag != "sentences":
            raise ValueError("Expected sentence-level XML")
        for sentence in root.findall("sentence"):
            for container_name, tag in (("aspectTerms", "aspectTerm"), ("aspectCategories", "aspectCategory")):
                container = sentence.find(container_name)
                if container is None:
                    continue
                for annotation in list(container):
                    label = annotation.attrib["polarity"]
                    if label == "conflict" and source_format == "semeval2014":
                        container.remove(annotation)
                        audit[f"excluded_conflict_{tag}"] += 1
                    elif label not in LABELS:
                        raise ValueError(f"Non-three-class label: {label}")
        # Native MAMS files are already three-class; preserve their exact bytes.
        return (content if source_format == "native" else ET.tostring(root, encoding="utf-8", xml_declaration=True)), dict(audit)
    if source_format != "opinions":
        raise ValueError(f"Unknown source format: {source_format}")
    result = ET.Element("sentences")
    for index, sentence in enumerate(root.iter("sentence")):
        if sentence.attrib.get("OutOfScope", sentence.attrib.get("outOfScope", "")).lower() == "true":
            audit["excluded_out_of_scope_sentences"] += 1
            continue
        text = sentence.findtext("text")
        if text is None:
            raise ValueError("Sentence is missing text")
        output = ET.SubElement(result, "sentence", {"id": sentence.attrib.get("id", str(index))})
        ET.SubElement(output, "text").text = text
        by_span = defaultdict(list)
        for opinion in sentence.findall("./Opinions/Opinion"):
            label = opinion.attrib.get("polarity")
            if label not in LABELS:
                audit["excluded_non_three_class_opinions"] += 1
                continue
            target = opinion.attrib.get("target", "NULL")
            start, end = int(opinion.attrib.get("from", 0)), int(opinion.attrib.get("to", 0))
            if target == "NULL" or start == end == 0:
                audit["excluded_implicit_targets"] += 1
                continue
            if not 0 <= start < end <= len(text) or text[start:end] != target:
                audit["excluded_invalid_offsets"] += 1
                continue
            by_span[start, end].append((target, label))
        kept = []
        for (start, end), annotations in sorted(by_span.items()):
            if len({label for _, label in annotations}) != 1:
                audit["excluded_ambiguous_spans"] += 1
                continue
            audit["collapsed_duplicate_opinions"] += len(annotations) - 1
            target, label = annotations[0]
            kept.append((start, end, target, label))
        overlapping = set()
        for i, left in enumerate(kept):
            for j in range(i + 1, len(kept)):
                if kept[j][0] >= left[1]:
                    break
                overlapping.update((i, j))
        audit["excluded_overlapping_spans"] += len(overlapping)
        container = ET.SubElement(output, "aspectTerms")
        for i, (start, end, target, label) in enumerate(kept):
            if i not in overlapping:
                ET.SubElement(container, "aspectTerm", {"term": target, "polarity": label,
                                                        "from": str(start), "to": str(end)})
    ET.indent(result, space="  ")
    return ET.tostring(result, encoding="utf-8", xml_declaration=True), dict(audit)


def xml_statistics(content):
    root = ET.fromstring(content)
    if root.tag != "sentences":
        raise ValueError("Expected sentences XML")
    counts, categories = Counter(), Counter()
    for sentence in root.findall("sentence"):
        text = sentence.findtext("text")
        if text is None:
            raise ValueError("Missing sentence text")
        for aspect in sentence.findall("./aspectTerms/aspectTerm"):
            start, end = int(aspect.attrib["from"]), int(aspect.attrib["to"])
            label = aspect.attrib["polarity"]
            if label not in LABELS or not 0 <= start < end <= len(text) or text[start:end] != aspect.attrib["term"]:
                raise ValueError("Invalid three-class aspect annotation")
            counts[label] += 1
        for aspect in sentence.findall("./aspectCategories/aspectCategory"):
            if aspect.attrib["polarity"] not in LABELS:
                raise ValueError("Invalid three-class category label")
            categories[aspect.attrib["polarity"]] += 1
    if not counts and not categories:
        raise ValueError("No aspect annotations")
    return {"sentences": len(root.findall("sentence")), "aspect_terms": sum(counts.values()),
            "aspect_categories": sum(categories.values()), "label_counts": dict(counts or categories)}
