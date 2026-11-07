"""Remove the known retired datasets only after the replacement collection validates."""
import argparse
import json
from pathlib import Path
import shutil

if __package__:
    from .download_datasets import PROJECT, EXPECTED_SHA256, resources, validate_xml
else:
    from download_datasets import PROJECT, EXPECTED_SHA256, resources, validate_xml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete the explicitly listed retired paths")
    args = parser.parse_args()
    checked = []
    for relative, _, _ in resources("all"):
        path = PROJECT / "Database" / relative
        checked.append({"path": relative, **validate_xml(path, EXPECTED_SHA256[relative])})
    retired = [PROJECT / "Database" / name for name in ("ASAP", "SemEval2014", "IMDB")]
    retired += [PROJECT / "resources/absa_xml_bundle.zip"]
    targets = [p for p in retired if p.exists() or p.is_symlink()]
    print(json.dumps({"validated_three_class_files": len(checked), "delete": [str(p) for p in targets],
                      "apply": args.apply}, ensure_ascii=False, indent=2))
    if args.apply:
        for path in targets:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        (PROJECT / "Database/cleanup_report.json").write_text(json.dumps(
            {"deleted_paths": [str(p.relative_to(PROJECT)) for p in targets],
             "retained_three_class_files": checked}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
