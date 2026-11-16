"""Synthetic fixtures authored for this project; no real financial records."""
import json
from pathlib import Path
import fitz

ROOT = Path(__file__).parent / "fixtures"


def build():
    manifest = []
    # Development and held-out sets have different vendor/label/row layouts.
    for split, count in [("development", 2), ("held_out", 3)]:
        for index in range(count):
            held = split == "held_out"
            vendor = (
                f'Synthetic {"Research Supply" if held else "Design Studio"} {index}'
            )
            number = f'{"TEST" if held else "DEV"}-{index+1:03}'
            text = "\n".join(
                [
                    f'{"Supplier Name" if held else "Vendor"}: {vendor}',
                    f'{"Invoice No." if held else "Invoice Number"}: {number}',
                    "Invoice Date: 2026-03-14",
                    "Due Date: 2026-04-14",
                    "Currency: USD",
                    "Description    Quantity    Unit Price    Amount"
                    if held
                    else "Description | Quantity | Unit Price | Amount",
                    "Paper packs    3    12.25    36.75"
                    if held
                    else "Design | 3 | 12.25 | 36.75",
                    "Service fee    1    10.00    10.00"
                    if held
                    else "Review | 1 | 10.00 | 10.00",
                    "Subtotal: 46.75",
                    "Tax: 4.68",
                    "Grand Total: 51.43",
                ]
            )
            expected = dict(
                vendor_name=vendor,
                invoice_number=number,
                invoice_date="2026-03-14",
                due_date="2026-04-14",
                currency="USD",
                subtotal="46.75",
                tax="4.68",
                grand_total="51.43",
            )
            expected_items = [
                dict(
                    description="Paper packs" if held else "Design",
                    quantity="3",
                    unit_price="12.25",
                    amount="36.75",
                ),
                dict(
                    description="Service fee" if held else "Review",
                    quantity="1",
                    unit_price="10.00",
                    amount="10.00",
                ),
            ]
            folder = ROOT / split
            folder.mkdir(parents=True, exist_ok=True)
            source = fitz.open()
            page = source.new_page()
            # Different held-out placement and type size; still clean, English, single-column.
            page.insert_text(
                (72 if held else 40, 95 if held else 60),
                text,
                fontsize=13 if held else 14,
            )
            raster = page.get_pixmap(dpi=170).tobytes("png")
            for kind in ("native", "scanned", "mixed", "png", "jpeg"):
                path = (
                    folder
                    / f'{index}_{kind}{".pdf" if kind in {"native","scanned","mixed"} else ".png" if kind=="png" else ".jpg"}'
                )
                if kind == "native":
                    source.save(path, deflate=True)
                elif kind in ("png", "jpeg"):
                    image = fitz.Pixmap(raster)
                    image.save(str(path))
                else:
                    doc = fitz.open()
                    if kind == "mixed":
                        cover = doc.new_page()
                        cover.insert_text(
                            (60, 80),
                            "Synthetic invoice attachment. Invoice details are on the following page.",
                            fontsize=14,
                        )
                    p = doc.new_page()
                    p.insert_image(p.rect, stream=raster)
                    doc.save(path, deflate=True)
                    doc.close()
                manifest.append(
                    dict(
                        path=str(path.relative_to(Path(__file__).parent)),
                        split=split,
                        kind=kind,
                        reference_pages=[
                            "Synthetic invoice attachment. Invoice details are on the following page.",
                            text,
                        ]
                        if kind == "mixed"
                        else [text],
                        fields=expected,
                        line_items=expected_items,
                    )
                )
            source.close()
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    build()
