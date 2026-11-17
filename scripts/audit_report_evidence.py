"""Verify curated arithmetic, dataset counts and gold cases without model dependencies."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bert_baseline.data import read_xml, split_train_validation
from scripts.evaluation_stats import sign_test

ROOT = Path(__file__).resolve().parents[1]


def main():
    values = json.loads((ROOT / "docs/report_review/recorded_results.json").read_text())
    test, _ = read_xml(ROOT / "Database/Restaurant3/test.xml")
    train, _ = read_xml(ROOT / "Database/Restaurant3/train.xml")
    train, val = split_train_validation(train, seed=42)
    test_text = {x.text for x in test}
    train = [x for x in train if x.text not in test_text]
    val = [x for x in val if x.text not in test_text]
    counts = {s: {label: sum(x.label == label for x in rows)
                  for label in ("negative", "neutral", "positive")}
              for s, rows in (("train", train), ("validation", val), ("test", test))}
    assert counts == {"train": {"negative": 677, "neutral": 520, "positive": 1849},
                      "validation": {"negative": 128, "neutral": 113, "positive": 314},
                      "test": {"negative": 196, "neutral": 196, "positive": 728}}
    paired = [b-a for a, b in zip(values["control_macro"], values["no_cls_macro"])]
    offsets = {name: [b-a for a, b in zip(values[name + "_macro"], values[name + "_tuned_macro"])]
               for name in ("control", "no_cls")}
    signs = sign_test(offsets["control"] + offsets["no_cls"])
    assert signs["positive"] == 8 and signs["ties"] == 2
    assert signs["p_two_sided"] == 0.0078125
    # A valid arithmetic result is not a justification for treating correlated runs as independent.
    cases = []
    for s in ET.parse(ROOT / "Database/Restaurant3/test.xml").findall("sentence"):
        if s.get("id") in ("32899414#512291#0", "11359727#487554#2"):
            cases.append({"id": s.get("id"), "text": s.findtext("text"),
                          "gold": [{"aspect": a.get("term"), "label": a.get("polarity")}
                                   for a in s.findall("./aspectTerms/aspectTerm")]})
    assert len(cases) == 2
    paths = ["docs/neutral_results.md", "Database/Restaurant3/manifest.json",
             "Database/Restaurant3/train.xml", "Database/Restaurant3/test.xml"]
    report = {"source_commit": values["source_commit"], "model_rerun": False,
              "counts_verified_from_xml": counts, "train_unique_texts": len({x.text for x in train}),
              "paired_macro_mean": statistics.mean(paired), "paired_macro_sd": statistics.stdev(paired),
              "offset_signs_arithmetic_only": signs,
              "inference_warning": "No pooled significance claim; runs share seeds/configurations and test set",
              "gold_cases_not_model_predictions": cases,
              "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}
    out = ROOT / "docs/report_review/evidence_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
