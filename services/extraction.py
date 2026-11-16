"""Page extraction worker. Executed in an isolated, time-bounded process."""
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz


def extract(path, extension, max_pages=50, language="eng", ocr_timeout=30):
    data = Path(path).read_bytes()
    if extension == ".txt":
        text = data.decode("utf-8")
        if not text.strip():
            raise ValueError("The text file is empty.")
        return [dict(page_number=1, text=text, method="native", blocks=[], warnings=[])]
    expected = {".pdf": "pdf", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}[
        extension
    ]
    valid = (
        data.startswith(b"%PDF-")
        if expected == "pdf"
        else data.startswith(b"\x89PNG\r\n\x1a\n")
        if expected == "png"
        else data.startswith(b"\xff\xd8\xff")
    )
    if not valid:
        raise ValueError("File contents do not match the file extension.")
    with fitz.open(stream=data, filetype=expected) as original:
        if original.needs_pass:
            raise ValueError("Encrypted PDF: upload an unlocked copy.")
        if not 0 < len(original) <= max_pages:
            raise ValueError(f"Document must contain 1–{max_pages} pages.")
        converted = None
        doc = original
        if (
            expected != "pdf"
            and original[0].rect.get_area() * (200 / 72) ** 2 > 25000000
        ):
            raise ValueError("Image dimensions exceed the safe processing limit.")
        if expected != "pdf":
            converted = fitz.open("pdf", original.convert_to_pdf())
            doc = converted
        pages = []
        try:
            for number, page in enumerate(doc, 1):
                if page.rect.get_area() * (200 / 72) ** 2 > 25000000:
                    raise ValueError(f"Page {number} exceeds the safe rendering limit.")
                native = page.get_text("text", sort=True)
                # Full-page OCR replaces (never appends to) insufficient native text.
                usable = (
                    sum(c.isalnum() for c in native) >= 30 and "\ufffd" not in native
                )
                # A large image plus a short native header is often a partially scanned page.
                images = page.get_image_info()
                image_area = sum(fitz.Rect(i["bbox"]).get_area() for i in images)
                if image_area > page.rect.get_area() * 0.4:
                    usable = False
                blocks, warnings = [], []
                if usable:
                    text = native
                    for block in (
                        []
                        if page.rotation
                        else page.get_text("dict", sort=True)["blocks"]
                    ):
                        for line in block.get("lines", []):
                            content = "".join(
                                span["text"] for span in line["spans"]
                            ).strip()
                            x, y, x2, y2 = line["bbox"]
                            if content:
                                blocks.append(
                                    dict(
                                        text=content,
                                        bbox=[
                                            x / page.rect.width,
                                            y / page.rect.height,
                                            x2 / page.rect.width,
                                            y2 / page.rect.height,
                                        ],
                                        confidence=None,
                                    )
                                )
                    method = "native"
                else:
                    if page.rect.width * page.rect.height * (200 / 72) ** 2 > 25000000:
                        raise ValueError(
                            f"Page {number} is too large to render safely."
                        )
                    pix = page.get_pixmap(dpi=200, alpha=False)
                    with tempfile.TemporaryDirectory() as tmp:
                        image = Path(tmp) / "page.png"
                        pix.save(str(image))
                        try:
                            result = subprocess.run(
                                [
                                    "tesseract",
                                    str(image),
                                    "stdout",
                                    "-l",
                                    language,
                                    "--psm",
                                    "3",
                                    "tsv",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=ocr_timeout,
                            )
                        except FileNotFoundError:
                            raise ValueError(
                                "OCR is unavailable. Install Tesseract and the configured language data, then retry."
                            )
                        except subprocess.TimeoutExpired:
                            raise ValueError(
                                f"OCR timed out on page {number}. Try a smaller or clearer document."
                            )
                        if result.returncode:
                            raise ValueError(
                                "OCR failed. Check that Tesseract language data is installed and the image is readable."
                            )
                    lines = {}
                    for row in csv.DictReader(
                        io.StringIO(result.stdout), delimiter="\t"
                    ):
                        word = row.get("text", "").strip()
                        if row.get("level") != "5" or not word:
                            continue
                        x, y, w, h = (
                            int(row[k]) for k in ("left", "top", "width", "height")
                        )
                        confidence = float(row["conf"])
                        blocks.append(
                            dict(
                                text=word,
                                bbox=[
                                    x / pix.width,
                                    y / pix.height,
                                    (x + w) / pix.width,
                                    (y + h) / pix.height,
                                ],
                                confidence=confidence if confidence >= 0 else None,
                            )
                        )
                        key = tuple(
                            row[k] for k in ("block_num", "par_num", "line_num")
                        )
                        lines.setdefault(key, []).append(word)
                    text = "\n".join(" ".join(words) for words in lines.values())
                    method = "ocr"
                    warnings.append(
                        "OCR may contain recognition errors. Confidence is a Tesseract score, not a probability."
                    )
                if not text.strip():
                    warnings.append("No readable text found on this page.")
                pages.append(
                    dict(
                        page_number=number,
                        text=text,
                        method=method,
                        blocks=blocks,
                        warnings=warnings,
                    )
                )
        finally:
            if converted:
                converted.close()
    if not any(p["text"].strip() for p in pages):
        raise ValueError(
            "No readable text found. Upload a clearer scan or a text document."
        )
    return pages


if __name__ == "__main__":
    try:
        result = extract(
            sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
        )
        Path(sys.argv[6]).write_text(json.dumps(result))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
