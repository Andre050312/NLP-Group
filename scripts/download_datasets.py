"""Download pinned SemEval/MAMS XML files without Git or third-party packages."""
import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SEMEVAL_REVISION = "3fd43769112ef6517c263e303fbb7dc3002c35df"
MAMS_REVISION = "cddcdb0f423b3fdb2c76b70744737abed3a00d17"
BUNDLE = PROJECT / "resources/absa_xml_bundle.zip"
MAX_FILE_BYTES = 16 * 1024 * 1024
TRAIN_DIRECTORY = "SemEval2014/SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines"
# From the project's original raw-file manifest. No remote mirror is trusted
# merely because its response can be parsed as XML.
EXPECTED_SHA256 = {
    f"{TRAIN_DIRECTORY}/Laptop_Train_v2.xml": "efec2ed73b4efb83b47c697114d800149c558118884afa80362a6e975ba7248c",
    f"{TRAIN_DIRECTORY}/Restaurants_Train_v2.xml": "223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5",
    "SemEval2014/ABSA_Gold_TestData/Laptops_Test_Gold.xml": "aa3a30c9dd7e0eacf9b69e898eb4118a36c1b4f391a30a08dfc41a38d8b8745b",
    "SemEval2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml": "f21509cfa37e16534cd5b2da043be487355b64ef48fe8d6aaacaeca6b49cc0fb",
    "MAMS/data/MAMS-ATSA/raw/train.xml": "26ba39142b28169ebc80408294f4e27493bd1e127c6aa2e85ab4bc750bf8017a",
    "MAMS/data/MAMS-ATSA/raw/val.xml": "5b7d225e2f9482a3814f15aed04ff8323105171a47d22d9899d871c4e0ba86ca",
    "MAMS/data/MAMS-ATSA/raw/test.xml": "f1a5f53b785335e9c57a1f82b1415026f821040b9e9662a2e467d47a54f20aa9",
    "MAMS/data/MAMS-ACSA/raw/train.xml": "24c6a1a9f769ec5a9a1fc9a63077001339cec65317e3e72cf053d8055ec5f1ac",
    "MAMS/data/MAMS-ACSA/raw/val.xml": "d4536b0ced8e77f53f88ff1835fcf561e0de02e41a509982f76eb9b8a375b2ed",
    "MAMS/data/MAMS-ACSA/raw/test.xml": "cc0dc3c5b711653daa9ac886c6517d3ebf8752954b73715692d7765fec4a57f5",
}
FILE_ERRORS = (OSError, ValueError, ET.ParseError, KeyError, HTTPException, zipfile.BadZipFile)


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


def source_urls(primary_url, hf_endpoint=None):
    urls = [primary_url]
    if primary_url.startswith("https://huggingface.co/"):
        if hf_endpoint:
            urls.insert(0, hf_endpoint.rstrip("/") + primary_url[len("https://huggingface.co"):])
        urls.append(primary_url.replace("/resolve/", "/raw/"))
    elif primary_url.startswith("https://raw.githubusercontent.com/"):
        tail = primary_url[len("https://raw.githubusercontent.com/"):]
        owner, repo, revision, path = tail.split("/", 3)
        urls.append(f"https://github.com/{owner}/{repo}/raw/{revision}/{path}")
    return list(dict.fromkeys(urls))


def validate_xml(path, expected_sha256=None):
    if Path(path).stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Unexpectedly large XML file")
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
    digest = hashlib.sha256(content).hexdigest()
    normalized = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    if expected_sha256 and expected_sha256 not in (digest, normalized):
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    return {"sentences": len(sentences), "aspect_terms": terms, "aspect_categories": categories,
            "bytes": len(content), "sha256": digest,
            "integrity": "verified" if expected_sha256 else "structure_only",
            "line_endings_normalized_for_check": bool(expected_sha256 and digest != expected_sha256)}


def install_xml(content, destination, expected_sha256=None):
    """Validate first, preserve an old file, then install atomically on Windows."""
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Unexpectedly large XML file")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        stats = validate_xml(temporary, expected_sha256)
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()[:12]
            backup = destination.with_name(destination.name + f".invalid-{digest}")
            if not backup.exists():
                shutil.copy2(destination, backup)
        os.replace(temporary, destination)
        return stats
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_xml(url, destination, timeout=30, retries=2, expected_sha256=None):
    """Validate a temporary file before replacing an invalid/missing destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "NLP-Group-Dataset-Downloader/1.0"})
            with urlopen(request, timeout=timeout) as response:
                content = response.read(MAX_FILE_BYTES + 1)
            return install_xml(content, destination, expected_sha256)
        except FILE_ERRORS as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def fetch_from_sources(urls, destination, expected_sha256, timeout=30, retries=2):
    errors = []
    for url in urls:
        try:
            stats = fetch_xml(url, destination, timeout, retries, expected_sha256)
            return stats, url
        except (RuntimeError, *FILE_ERRORS) as error:
            errors.append(str(error))
            print(f"Source failed; trying next address: {url}", flush=True)
    raise RuntimeError("All network addresses failed:\n" + "\n".join(errors))


def import_local(source, relative, destination, expected_sha256):
    """Read only matching bytes; never extract arbitrary archive paths."""
    source = Path(source)
    errors = []
    if source.is_dir():
        candidates = sorted(source.rglob(Path(relative).name))
        for candidate in candidates:
            try:
                validate_xml(candidate, expected_sha256)
                return install_xml(candidate.read_bytes(), destination, expected_sha256), str(candidate)
            except FILE_ERRORS as error:
                errors.append(str(error))
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            candidates = [i for i in archive.infolist() if not i.is_dir()
                          and Path(i.filename).name == Path(relative).name]
            for info in candidates:
                if info.file_size > MAX_FILE_BYTES:
                    continue
                try:
                    with archive.open(info) as stream:
                        content = stream.read(MAX_FILE_BYTES + 1)
                    stats = install_xml(content, destination, expected_sha256)
                    return stats, f"{source}!{info.filename}"
                except FILE_ERRORS as error:
                    errors.append(str(error))
    raise ValueError(f"No verified local copy of {relative} in {source}" +
                     (f"; last error: {errors[-1]}" if errors else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "semeval", "mams"], default="all")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "Database")
    parser.add_argument("--check", action="store_true", help="Validate existing files only; do not access network or write files")
    parser.add_argument("--offline", action="store_true", help="Restore from local XML/bundle only; never access network")
    parser.add_argument("--import-from", type=Path, help="Directory or ZIP containing original XML files")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--hf-endpoint", help="Optional HTTPS Hugging Face mirror; downloaded bytes must match pinned hashes")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 1:
        parser.error("timeout and retries must be positive")
    if args.hf_endpoint and not args.hf_endpoint.startswith("https://"):
        parser.error("hf-endpoint must start with https://")
    files, failures = [], []
    for relative, url, revision in resources(args.dataset):
        destination = args.data_root / relative
        expected = EXPECTED_SHA256[relative]
        try:
            try:
                stats = validate_xml(destination, expected)
                status = "existing_valid"
                actual_source = str(destination)
            except FILE_ERRORS as error:
                if args.check:
                    raise ValueError(f"Missing or invalid XML: {error}") from error
                # Look for a supplied archive, bundled files or accidentally nested
                # Database/Database paths before attempting a network connection.
                stats, local_errors = None, []
                sources = [args.import_from, BUNDLE, args.data_root]
                for source in sources:
                    if source is None or not source.exists():
                        continue
                    try:
                        stats, actual_source = import_local(source, relative, destination, expected)
                        status = "restored_local"
                        break
                    except FILE_ERRORS as local_error:
                        local_errors.append(str(local_error))
                if stats is None:
                    if args.offline:
                        raise ValueError("Offline restoration failed. Copy resources/absa_xml_bundle.zip "
                                         "or use --import-from <directory-or-zip>. " + "; ".join(local_errors))
                    print(f"Downloading: {relative}", flush=True)
                    stats, actual_source = fetch_from_sources(source_urls(url, args.hf_endpoint), destination,
                                                              expected, args.timeout, args.retries)
                    status = "downloaded"
            files.append({"path": relative, "status": status, "source_url": url,
                          "actual_source": actual_source, "source_revision": revision, **stats})
            print(f"OK [{status}] {relative}: {stats['sentences']} sentences", flush=True)
        except (RuntimeError, *FILE_ERRORS) as error:
            failures.append({"path": relative, "error": str(error)})
            print(f"FAILED: {relative}: {error}", flush=True)
    if not args.check:
        args.data_root.mkdir(parents=True, exist_ok=True)
        report = {"dataset": args.dataset, "files": files, "failures": failures,
                  "note": "All files must match pinned SHA-256 (CRLF/LF differences accepted). "
                          "Original labels and character offsets are never modified."}
        (args.data_root / "download_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(files)} files; failed {len(failures)}.")
    if failures:
        print("Use the bundled offline archive or --import-from to restore files without network. "
              "Details are in Database/download_manifest.json unless --check was used.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
