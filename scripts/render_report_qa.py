"""Use the skill renderer's rasterizer with an already exported native Word PDF.

Fallback for hosts without LibreOffice. This does not generate or simulate a PDF.
Export the current DOCX in Word first, then provide that verified PDF explicitly.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--word-pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("document_skill_renderer", args.renderer)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)

    def native_word_pdf(doc_path, user_profile, convert_tmp_dir, stem, verbose):
        target = Path(convert_tmp_dir) / (stem + ".pdf")
        shutil.copy2(args.word_pdf, target)
        return str(target), "Native Microsoft Word PDF export supplied explicitly; no LibreOffice available."

    renderer.convert_to_pdf = native_word_pdf
    paths = renderer.rasterize(str(args.docx), str(args.output_dir), 140, False, False)
    print(f"Rendered {len(paths)} actual Word pages using skill rasterization")
    from pypdf import PdfReader
    reader = PdfReader(args.word_pdf)
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        print(f"Page {i}: {len(text.split())} words | {text[:95].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
