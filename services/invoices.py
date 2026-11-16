"""Conservative, deterministic invoice extraction; no inferred values."""
import csv
import io
import re
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, localcontext
from models.invoice import Invoice, Value, Evidence, LineItem, NAMES, MONEY

LABELS = {
    "vendor_name": r"(?:vendor|supplier)(?: name)?",
    "invoice_number": r"invoice (?:number|no\.?|#)",
    "invoice_date": r"invoice date",
    "due_date": r"due date",
    "currency": r"currency",
    "subtotal": r"subtotal",
    "tax": r"(?:tax|vat)",
    "grand_total": r"(?:grand total|total|amount due)",
}


def value(raw, page, text, name):
    normalized = raw.strip()
    if not normalized:
        return Value(
            raw=None, status="missing", evidence=[Evidence(page_number=page, text=text)]
        )
    status = "present"
    if name in MONEY:
        # Only unambiguous dot decimals, optionally with comma thousands groups.
        if not re.fullmatch(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", normalized):
            status, normalized = "ambiguous", None
        else:
            normalized = str(Decimal(normalized.replace(",", "")))
    elif name in {"invoice_date", "due_date"}:
        try:
            normalized = date.fromisoformat(normalized).isoformat()
        except ValueError:
            status, normalized = "ambiguous", None
    elif name == "currency":
        if not re.fullmatch("[A-Z]{3}", normalized):
            status, normalized = "ambiguous", None
    return Value(
        raw=raw,
        normalized=normalized,
        status=status,
        evidence=[Evidence(page_number=page, text=text)],
    )


def extract_invoice(pages):
    fields = {name: Value() for name in NAMES}
    items = []
    all_text = "\n".join(p["text"] for p in pages)
    for page in pages:
        for line in page["text"].splitlines():
            line = line.strip()
            for name, label in LABELS.items():
                match = re.fullmatch(label + r"\s*:\s*(.+)", line, flags=re.I)
                if match:
                    found = value(match[1], page["page_number"], line, name)
                    old = fields[name]
                    if old.raw is not None and old.raw != found.raw:
                        fields[name] = Value(
                            raw=old.raw + " | " + found.raw,
                            status="ambiguous",
                            evidence=old.evidence + found.evidence,
                        )
                    elif old.raw is None:
                        fields[name] = found
            # Explicit table rows, or aligned plain-text rows ending in three numbers.
            parts = [x.strip() for x in line.split("|")]
            if len(parts) != 4:
                match = re.fullmatch(
                    r"(.+?)\s+(\d+(?:\.\d+)?)\s+(-?[\d,]+(?:\.\d+)?)\s+(-?[\d,]+(?:\.\d+)?)",
                    line,
                )
                parts = list(match.groups()) if match else []
            if len(parts) == 4 and any(
                re.fullmatch(r"-?[\d,]+(?:\.\d+)?", part) for part in parts[1:]
            ):
                items.append(
                    LineItem(
                        **{
                            key: value(raw, page["page_number"], line, key)
                            for key, raw in zip(
                                ("description", "quantity", "unit_price", "amount"),
                                parts,
                            )
                        }
                    )
                )
    return Invoice(
        fields=fields,
        line_items=items,
        unsupported_adjustments=bool(
            re.search(
                r"\b(discount|shipping|freight|credit|withholding|rounding)\b",
                all_text,
                re.I,
            )
        ),
    )


def validate_invoice(invoice):
    warnings = []
    for name in NAMES:
        if name != "due_date" and invoice.fields[name].status != "present":
            warnings.append(f"{name}: missing or ambiguous")

    def number(v):
        return (
            Decimal(v.normalized)
            if v.status == "present" and v.normalized is not None
            else None
        )

    currency = invoice.fields["currency"].normalized
    precision = {"JPY": 0, "KRW": 0, "KWD": 3, "BHD": 3, "OMR": 3, "JOD": 3, "TND": 3}
    for code in (
        "USD",
        "EUR",
        "GBP",
        "NPR",
        "INR",
        "CAD",
        "AUD",
        "CHF",
        "CNY",
        "NZD",
        "SGD",
        "HKD",
    ):
        precision[code] = 2
    if currency not in precision:
        warnings.append(
            "Currency precision is unknown; arithmetic cannot be validated reliably."
        )
        return warnings
    tolerance = Decimal(1).scaleb(-precision[currency])
    for index, line in enumerate(invoice.line_items, 1):
        q, u, a = (
            number(getattr(line, k)) for k in ("quantity", "unit_price", "amount")
        )
        if None in (q, u, a):
            warnings.append(f"Line {index}: incomplete arithmetic evidence")
        else:
            with localcontext() as context:
                context.prec = max(
                    64, len(q.as_tuple().digits) + len(u.as_tuple().digits) + 10
                )
                mismatch = (
                    abs((q * u).quantize(tolerance, rounding=ROUND_HALF_UP) - a)
                    > tolerance
                )
            if not mismatch:
                continue
            warnings.append(
                f"Line {index}: quantity × unit price differs from amount; review adjustments"
            )
    if invoice.unsupported_adjustments:
        warnings.append(
            "Discounts, shipping, credits or adjustments detected. Summary arithmetic cannot be validated reliably."
        )
    else:
        sub, tax, total = (
            number(invoice.fields[k]) for k in ("subtotal", "tax", "grand_total")
        )
        if None not in (sub, tax, total) and abs(sub + tax - total) > tolerance:
            warnings.append("Subtotal + tax differs from grand total")
        amounts = [number(line.amount) for line in invoice.line_items]
        if (
            amounts
            and None not in amounts
            and sub is not None
            and abs(sum(amounts) - sub) > tolerance
        ):
            warnings.append(
                "Line amounts differ from subtotal; table may be incomplete"
            )
    if not invoice.line_items:
        warnings.append(
            "No supported line-item table found; line-item completeness cannot be validated."
        )
    return warnings


def business_key(invoice):
    values = [invoice.fields[k] for k in ("vendor_name", "invoice_number")]
    return (
        tuple(re.sub(r"\s+", " ", v.normalized.strip().casefold()) for v in values)
        if all(v.status == "present" for v in values)
        else None
    )


def safe_cell(value):
    text = "" if value is None else str(value)
    return (
        "'" + text
        if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n"))
        else text
    )


def export_csv(record):
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(
        [
            "document_id",
            "review_status",
            "row_type",
            "field",
            "raw",
            "normalized",
            "status",
            "evidence",
        ]
    )
    invoice = Invoice.model_validate(record["reviewed"])
    rows = [("field", k, v) for k, v in invoice.fields.items()]
    rows += [
        (f"line_{i}", k, v)
        for i, line in enumerate(invoice.line_items, 1)
        for k, v in line
    ]
    for kind, key, v in rows:
        writer.writerow(
            [
                safe_cell(x)
                for x in [
                    record["document_id"],
                    record["review_status"],
                    kind,
                    key,
                    v.raw,
                    v.normalized,
                    v.status,
                    "; ".join(f"p{e.page_number}: {e.text}" for e in v.evidence),
                ]
            ]
        )
    return out.getvalue()
