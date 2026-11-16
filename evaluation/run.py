"""Offline extraction evaluation. No embeddings, hosted APIs, or LLM calls."""
import argparse
import json
import re
import shutil
import statistics
import time
from pathlib import Path
from decimal import Decimal
import fitz
from evaluation.generate import build
from services.extraction import extract
from services.invoices import extract_invoice


def normalize(s):
    return re.sub(r"\s+", " ", s).strip()


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def baseline(path):
    # Reproduce original native-only, whole-document flattening. Images unsupported.
    if path.suffix != ".pdf":
        raise ValueError("Original pipeline did not support images")
    with fitz.open(path) as doc:
        text = "\n".join(p.get_text() for p in doc)
    if not text.strip():
        raise ValueError("No native text")
    return [dict(page_number=None, text=text)]


def same(a, b):
    if a is None:
        return b is None
    try:
        return Decimal(a) == Decimal(b)
    except Exception:
        return normalize(a) == normalize(b)


def run(split="held_out"):
    manifest = build()
    results = []
    for item in [r for r in manifest if r["split"] == split]:
        path = Path(__file__).parent / item["path"]
        for pipeline in ("baseline", "improved"):
            start = time.perf_counter()
            error = None
            try:
                pages = (
                    baseline(path)
                    if pipeline == "baseline"
                    else extract(path, path.suffix)
                )
            except Exception as exc:
                pages = []
                error = str(exc)
            latency = (time.perf_counter() - start) * 1000
            invoice = extract_invoice(
                [{**p, "page_number": p["page_number"] or 1} for p in pages]
            )
            reference = normalize(" ".join(item["reference_pages"]))
            actual = normalize(" ".join(p["text"] for p in pages))
            matches = sum(
                same(invoice.fields[k].normalized, v) for k, v in item["fields"].items()
            )
            pred = [{k: v.normalized for k, v in line} for line in invoice.line_items]
            gold = item["line_items"]
            remaining = gold.copy()
            tp = 0
            for line in pred:
                for index, expected in enumerate(remaining):
                    if all(same(line[k], v) for k, v in expected.items()):
                        tp += 1
                        remaining.pop(index)
                        break
            citation_total = len(item["reference_pages"])
            citation_correct = sum(
                p.get("page_number") == index
                and len(
                    set(normalize(p["text"]).split())
                    & set(normalize(item["reference_pages"][index - 1]).split())
                )
                >= 3
                for index, p in enumerate(pages, 1)
                if index <= len(item["reference_pages"])
            )
            results.append(
                dict(
                    path=item["path"],
                    kind=item["kind"],
                    pipeline=pipeline,
                    failure=error,
                    latency_ms=round(latency, 2),
                    character_edits=distance(reference, actual),
                    reference_characters=len(reference),
                    word_edits=distance(reference.split(), actual.split()),
                    reference_words=len(reference.split()),
                    field_matches=matches,
                    field_count=len(item["fields"]),
                    line_tp=tp,
                    line_pred=len(pred),
                    line_gold=len(gold),
                    citation_correct=citation_correct,
                    citation_total=citation_total,
                )
            )
    aggregates = []
    for kind in ("native", "scanned", "mixed", "png", "jpeg"):
        for pipeline in ("baseline", "improved"):
            rows = [
                r for r in results if r["kind"] == kind and r["pipeline"] == pipeline
            ]

            def total(key):
                return sum(r[key] for r in rows)

            aggregates.append(
                dict(
                    kind=kind,
                    pipeline=pipeline,
                    documents=len(rows),
                    failure_rate=sum(r["failure"] is not None for r in rows)
                    / len(rows),
                    mean_extraction_latency_ms=round(
                        statistics.mean(r["latency_ms"] for r in rows), 2
                    ),
                    cer=total("character_edits") / total("reference_characters"),
                    wer=total("word_edits") / total("reference_words"),
                    field_accuracy=total("field_matches") / total("field_count"),
                    line_item_exact_f1=2
                    * total("line_tp")
                    / (total("line_pred") + total("line_gold"))
                    if total("line_pred") + total("line_gold")
                    else 0,
                    page_provenance_accuracy=total("citation_correct")
                    / total("citation_total"),
                    field_correction_proxy=1
                    - total("field_matches") / total("field_count"),
                )
            )
    return dict(
        split=split,
        documents=len(results) // 2,
        development_documents=sum(r["split"] == "development" for r in manifest),
        tesseract_available=bool(shutil.which("tesseract")),
        scope="Offline extraction only. Citation metric verifies page provenance, not LLM answer support. Latency excludes queue, embeddings and indexing. Field correction proxy is reference mismatch, not observed human correction.",
        observed_human_correction_rate=None,
        results=aggregates,
        documents_detail=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["development", "held_out"], default="held_out"
    )
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()
    result = run(args.split)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "documents_detail"}, indent=2
        )
    )
