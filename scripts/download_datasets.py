"""Prepare only three-class restaurant datasets; original mixed-label sources stay transient."""
import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile

if __package__:
    from .restaurant_format import convert_source, xml_statistics
else:
    from restaurant_format import convert_source, xml_statistics

PROJECT = Path(__file__).resolve().parents[1]
BUNDLE = PROJECT / "resources/restaurant_datasets_3class.zip"
MAX_FILE_BYTES = 16 * 1024 * 1024
CATALOG = json.loads((Path(__file__).with_name("restaurant_sources.json")).read_text(encoding="utf-8"))
SOURCE_SPECS = {entry["path"]: entry for entry in CATALOG["files"]}
EXPECTED_SHA256 = {path: entry["sha256"] for path, entry in SOURCE_SPECS.items()}
FILE_ERRORS = (OSError, ValueError, ET.ParseError, KeyError, HTTPException, zipfile.BadZipFile)


def resources(dataset):
    allowed = {dataset}
    if dataset == "semeval":
        allowed = {"restaurant", "restaurant2015", "restaurant2016"}
    return [(entry["path"], entry["source_url"], entry["source_revision"])
            for entry in CATALOG["files"] if dataset == "all" or entry["dataset"] in allowed]


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
    content = Path(path).read_bytes()
    stats = xml_statistics(content)
    digest = hashlib.sha256(content).hexdigest()
    normalized = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    if expected_sha256 and expected_sha256 not in (digest, normalized):
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    return {**stats,
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


def fetch_xml(url, destination, timeout=30, retries=2, expected_sha256=None, source_spec=None):
    """Validate a temporary file before replacing an invalid/missing destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "NLP-Group-Dataset-Downloader/1.0"})
            with urlopen(request, timeout=timeout) as response:
                content = response.read(MAX_FILE_BYTES + 1)
            if source_spec is not None:
                digest = hashlib.sha256(content).hexdigest()
                normalized = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
                if source_spec["source_sha256"] not in (digest, normalized):
                    raise ValueError("Original source SHA-256 mismatch")
                # Raw mixed-label XML is converted in memory and never installed.
                content, _ = convert_source(content.replace(b"\r\n", b"\n"), source_spec["source_format"])
            return install_xml(content, destination, expected_sha256)
        except FILE_ERRORS as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {url}: {last_error}") from last_error


def fetch_from_sources(urls, destination, expected_sha256, timeout=30, retries=2, source_spec=None):
    errors = []
    for url in urls:
        try:
            stats = fetch_xml(url, destination, timeout, retries, expected_sha256, source_spec)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "semeval", "restaurant", "restaurant2015", "restaurant2016", "mams"], default="all")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "Database")
    parser.add_argument("--check", action="store_true", help="Validate existing files only; do not access network or write files")
    parser.add_argument("--offline", action="store_true", help="Restore from local XML/bundle only; never access network")
    parser.add_argument("--import-from", type=Path, help="Directory or ZIP containing original XML files")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--hf-endpoint", help="Optional HTTPS Hugging Face mirror; downloaded bytes must match pinned hashes")
    args = parser.parse_args(argv)
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
                        raise ValueError("Offline restoration failed. Copy resources/restaurant_datasets_3class.zip "
                                         "or use --import-from <directory-or-zip>. " + "; ".join(local_errors))
                    print(f"Downloading: {relative}", flush=True)
                    stats, actual_source = fetch_from_sources(source_urls(url, args.hf_endpoint), destination,
                                                              expected, args.timeout, args.retries, SOURCE_SPECS[relative])
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
                          "Only three-class restaurant derivatives are installed; conversion provenance is in scripts/restaurant_sources.json."}
        (args.data_root / "download_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(files)} files; failed {len(failures)}.")
    if failures:
        print("Use the bundled offline archive or --import-from to restore files without network. "
              "Details are in Database/download_manifest.json unless --check was used.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
