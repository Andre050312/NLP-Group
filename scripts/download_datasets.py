"""Download pinned SemEval/MAMS XML files without Git or third-party packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
SEMEVAL_REVISION = "3fd43769112ef6517c263e303fbb7dc3002c35df"
MAMS_REVISION = "cddcdb0f423b3fdb2c76b70744737abed3a00d17"


def resources(dataset):
    items = []
    if dataset in ("all", "semeval"):
        directory = "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines"
        paths = [f"{directory}/Laptop_Train_v2.xml", f"{directory}/Restaurants_Train_v2.xml",
                 "ABSA_Gold_TestData/Laptops_Test_Gold.xml", "ABSA_Gold_TestData/Restaurants_Test_Gold.xml"]
        for path in paths:
            url = f"https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/resolve/{SEMEVAL_REVISION}/{quote(path, safe='/')}"
            items.append(("SemEval2014/" + path, url, SEMEVAL_REVISION))
    if dataset in ("all", "mams"):
        for task in ("ATSA", "ACSA"):
            for split in ("train", "val", "test"):
                path = f"data/MAMS-{task}/raw/{split}.xml"
                url = f"https://raw.githubusercontent.com/siat-nlp/MAMS-for-ABSA/{MAMS_REVISION}/{path}"
                items.append(("MAMS/" + path, url, MAMS_REVISION))
    return items


def validate_xml(path):
    root = ET.parse(path).getroot()
    if root.tag != "sentences":
        raise ValueError("Expected a sentences XML root, not a webpage or Git LFS pointer")
    sentences = root.findall("sentence")
    if not sentences:
        raise ValueError("XML contains no sentences")
    terms, categories = 0, 0
    for sentence in sentences:
        text = sentence.findtext("text")
        if text is None:
            raise ValueError("Sentence is missing text")
        for aspect in sentence.findall("./aspectTerms/aspectTerm"):
            start, end = int(aspect.attrib["from"]), int(aspect.attrib["to"])
            if not 0 <= start < end <= len(text) or text[start:end] != aspect.attrib["term"]:
                raise ValueError(f"Invalid XML character span in sentence {sentence.attrib.get('id')}")
            if aspect.attrib["polarity"] not in ("negative", "neutral", "positive", "conflict"):
                raise ValueError("Unknown sentiment label")
            terms += 1
        categories += len(sentence.findall("./aspectCategories/aspectCategory"))
    if not terms and not categories:
        raise ValueError("XML has no annotated aspects")
    content = Path(path).read_bytes()
    return {"sentences": len(sentences), "aspect_terms": terms, "aspect_categories": categories,
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def fetch_xml(url, destination, timeout=45, retries=3):
    """Validate a temporary file before replacing an invalid/missing destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        temporary = None
        try:
            request = Request(url, headers={"User-Agent": "NLP-Group-Dataset-Downloader/1.0"})
            with urlopen(request, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as stream:
                    temporary = Path(stream.name)
                    shutil.copyfileobj(response, stream)
            stats = validate_xml(temporary)
            if destination.exists():
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()[:12]
                backup = destination.with_name(destination.name + f".invalid-{digest}")
                if not backup.exists():
                    shutil.copy2(destination, backup)
            os.replace(temporary, destination)
            return stats
        except (OSError, ValueError, ET.ParseError, KeyError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "semeval", "mams"], default="all")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "Database")
    parser.add_argument("--check", action="store_true", help="Validate existing files only; do not access network or write files")
    args = parser.parse_args()
    files, failures = [], []
    for relative, url, revision in resources(args.dataset):
        destination = args.data_root / relative
        try:
            try:
                stats = validate_xml(destination)
                status = "existing_valid"
            except (OSError, ValueError, ET.ParseError, KeyError) as error:
                if args.check:
                    raise ValueError(f"Missing or invalid XML: {error}") from error
                print(f"Downloading: {relative}", flush=True)
                stats = fetch_xml(url, destination)
                status = "downloaded"
            files.append({"path": relative, "status": status, "source_url": url,
                          "source_revision": revision, **stats})
            print(f"OK [{status}] {relative}: {stats['sentences']} sentences", flush=True)
        except (OSError, ValueError, RuntimeError, ET.ParseError, KeyError) as error:
            failures.append({"path": relative, "error": str(error)})
            print(f"FAILED: {relative}: {error}", flush=True)
    if not args.check:
        args.data_root.mkdir(parents=True, exist_ok=True)
        report = {"dataset": args.dataset, "files": files, "failures": failures,
                  "note": "Existing valid XML files are kept; their remote revision is not verified. Original labels and character offsets are not modified."}
        (args.data_root / "download_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(files)} files; failed {len(failures)}.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
