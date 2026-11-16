import asyncio
import shutil
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import fitz
import pytest
from fastapi import FastAPI
import httpx

from services.extraction import extract
from services.invoices import (
    extract_invoice,
    validate_invoice,
    export_csv,
    business_key,
)
from models.invoice import Invoice
from services.processing import page_chunks

TEXT = """Vendor: Example Studio
Invoice Number: SYN-001
Invoice Date: 2026-01-05
Due Date: 2026-02-05
Currency: USD
Consulting | 2 | 10.25 | 20.50
Subtotal: 20.50
Tax: 2.05
Grand Total: 22.55"""


def make_pdf(path, kinds=("native",)):
    doc = fitz.open()
    for kind in kinds:
        p = doc.new_page()
        if kind == "native":
            p.insert_text((45, 65), TEXT, fontsize=14)
        elif kind == "scan":
            temp = fitz.open()
            t = temp.new_page()
            t.insert_text((45, 65), TEXT, fontsize=14)
            p.insert_image(p.rect, stream=t.get_pixmap(dpi=170).tobytes("png"))
            temp.close()
    doc.save(path)
    doc.close()
    return path


def test_native_provenance_and_invoice(tmp_path):
    pages = extract(make_pdf(tmp_path / "native.pdf"), ".pdf")
    assert pages[0]["method"] == "native" and pages[0]["page_number"] == 1
    assert pages[0]["blocks"][0]["confidence"] is None
    chunks, metadata = page_chunks(pages, "fixed")
    assert chunks and all(m["page_number"] == 1 for m in metadata)
    invoice = extract_invoice(pages)
    assert invoice.fields["grand_total"].normalized == "22.55"
    assert invoice.fields["grand_total"].raw == "22.55"
    assert invoice.fields["grand_total"].evidence[0].page_number == 1
    assert len(invoice.line_items) == 1 and not validate_invoice(invoice)


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract not installed")
@pytest.mark.parametrize("kinds", [("scan",), ("native", "scan")])
def test_scanned_and_mixed(tmp_path, kinds):
    pages = extract(make_pdf(tmp_path / "input.pdf", kinds), ".pdf")
    assert [p["method"] for p in pages] == [
        "native" if k == "native" else "ocr" for k in kinds
    ]
    assert [p["page_number"] for p in pages] == list(range(1, len(kinds) + 1))
    assert pages[-1]["text"].count("SYN-001") == 1
    assert all(
        b["confidence"] is None or isinstance(b["confidence"], float)
        for b in pages[-1]["blocks"]
    )


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract not installed")
@pytest.mark.parametrize("extension", [".png", ".jpg"])
def test_images(tmp_path, extension):
    doc = fitz.open(make_pdf(tmp_path / "source.pdf"))
    path = tmp_path / ("image" + extension)
    pix = doc[0].get_pixmap(dpi=170)
    pix.save(str(path))
    doc.close()
    assert extract(path, extension)[0]["method"] == "ocr"


@pytest.mark.parametrize(
    "content,ext",
    [
        (b"", ".txt"),
        (b"garbage", ".pdf"),
        (b"%PDF-invalid", ".pdf"),
        (b"png", ".png"),
        (b"bad", ".exe"),
    ],
)
def test_invalid_input(tmp_path, content, ext):
    path = tmp_path / ("bad" + ext)
    path.write_bytes(content)
    with pytest.raises((ValueError, KeyError, fitz.FileDataError)):
        extract(path, ext)


def test_page_limit_and_encryption(tmp_path):
    path = make_pdf(tmp_path / "long.pdf", ("native", "native"))
    with pytest.raises(ValueError, match="1–1"):
        extract(path, ".pdf", max_pages=1)
    doc = fitz.open(path)
    locked = tmp_path / "locked.pdf"
    doc.save(
        locked,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="secret",
        user_pw="password",
    )
    doc.close()
    with pytest.raises(ValueError, match="Encrypted"):
        extract(locked, ".pdf")


def invoice(text=TEXT):
    return extract_invoice([dict(page_number=1, text=text)])


def test_missing_ambiguous_and_adjustments():
    inv = invoice(
        "Invoice Date: 01/02/2026\nCurrency: $\nTax: unknown\nVendor: A\nVendor: B\nDiscount: 2"
    )
    assert inv.fields["due_date"].status == "missing"
    assert inv.fields["invoice_date"].status == "ambiguous"
    assert inv.fields["currency"].normalized is None
    assert inv.fields["vendor_name"].status == "ambiguous"
    assert any("cannot be validated reliably" in w for w in validate_invoice(inv))


def test_decimal_rounding_and_schema():
    inv = invoice(TEXT.replace("2 | 10.25 | 20.50", "1 | 20.505 | 20.50"))
    assert not any(
        "quantity" in w for w in validate_invoice(inv)
    )  # half-up 20.51 differs by exactly tolerance
    inv.line_items[0].amount.normalized = "20.48"
    assert any("quantity" in w for w in validate_invoice(inv))
    inv.fields["tax"].normalized = "NaN"
    with pytest.raises(ValueError):
        Invoice.model_validate(inv.model_dump())
    inv = invoice()
    inv.fields["grand_total"].normalized = "22.55"
    assert Decimal(inv.fields["subtotal"].normalized) + Decimal(
        inv.fields["tax"].normalized
    ) == Decimal("22.55")


def test_csv_formula_and_business_duplicate():
    inv = invoice()
    inv.fields["vendor_name"].normalized = '=HYPERLINK("bad")'
    record = dict(
        document_id="doc_test", review_status="needs_review", reviewed=inv.model_dump()
    )
    assert "'=HYPERLINK" in export_csv(record)
    a = invoice()
    b = invoice(TEXT.replace("Example Studio", " example   studio "))
    assert business_key(a) == business_key(b)


@pytest.mark.asyncio
async def test_review_duplicate_export_and_conflict(database, tmp_path, monkeypatch):
    from routers import ingest
    from db.sql_db import save_intelligence
    from services.processing import now

    app = FastAPI()
    app.include_router(ingest.router, prefix="/api")
    record = dict(
        document_id="doc_1",
        filename="invoice.pdf",
        stored_name="doc_1.pdf",
        extension=".pdf",
        file_size_bytes=100,
        content_hash="same",
        strategy="fixed",
        status="ready",
        message="Ready",
        created_at=now(),
        review_status="needs_review",
        version=0,
        audit=[],
        pages=[dict(page_number=1, text=TEXT, method="native", blocks=[], warnings=[])],
        original=invoice().model_dump(mode="json"),
        reviewed=invoice().model_dump(mode="json"),
    )
    await save_intelligence(record)
    await save_intelligence(dict(record, document_id="doc_2"))
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        detail = (await client.get("/api/documents/doc_1")).json()
        assert "Exact file duplicate" in detail["duplicate_warnings"][0]
        changed = invoice()
        changed.fields["vendor_name"].normalized = "Corrected Studio"
        body = dict(
            invoice=changed.model_dump(mode="json"),
            action="approve",
            version=0,
            acknowledgement="Compared original; intentional duplicate",
        )
        approved = await client.put("/api/documents/doc_1/review", json=body)
        assert approved.status_code == 200, approved.text
        data = approved.json()
        assert data["review_status"] == "approved" and data["audit"][0]["timestamp"]
        assert (
            data["original"]["fields"]["vendor_name"]["normalized"] == "Example Studio"
        )
        assert (
            data["reviewed"]["fields"]["vendor_name"]["normalized"]
            == "Corrected Studio"
        )
        assert (
            await client.put("/api/documents/doc_1/review", json=body)
        ).status_code == 409
        assert (await client.get("/api/documents/doc_1/export")).json()[
            "review_status"
        ] == "approved"
        assert (
            "approved"
            in (await client.get("/api/documents/doc_1/export?format=csv")).text
        )
        totals = (await client.get("/api/invoices/summary")).json()
        assert totals["currencies"]["USD"]["total"] == "22.55"
        body["version"] = 1
        body["action"] = "save"
        assert (await client.put("/api/documents/doc_1/review", json=body)).json()[
            "review_status"
        ] == "needs_review"
        body["version"] = 2
        body["invoice"]["fields"]["vendor_name"]["evidence"][0]["text"] = "fabricated"
        assert (
            await client.put("/api/documents/doc_1/review", json=body)
        ).status_code == 422
        assert (await client.get("/api/documents/missing/export")).status_code == 404


@pytest.mark.asyncio
async def test_upload_limits_status_and_business_duplicates(
    database, tmp_path, monkeypatch
):
    from routers import ingest
    from services import processing

    app = FastAPI()
    app.include_router(ingest.router, prefix="/api")
    monkeypatch.setattr(ingest, "STORAGE", tmp_path)
    monkeypatch.setattr(processing, "STORAGE", tmp_path)
    monkeypatch.setattr(ingest, "queue", asyncio.Queue(20))
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        assert (
            await client.post("/api/ingest", files={"file": ("bad.exe", b"bad")})
        ).status_code == 415
        assert (
            await client.post("/api/ingest", files={"file": ("empty.pdf", b"")})
        ).status_code == 400
        monkeypatch.setattr(ingest.settings, "MAX_FILE_SIZE_MB", 1)
        assert (
            await client.post(
                "/api/ingest", files={"file": ("big.pdf", b"x" * (1024 * 1024 + 1))}
            )
        ).status_code == 413
        response = await client.post(
            "/api/ingest", files={"file": ("sample.txt", TEXT.encode())}
        )
        assert response.status_code == 202
        doc_id = response.json()["document_id"]

        async def embedding(texts):
            return [[0.1] * 768 for _ in texts]

        captured = {}

        async def upsert(*args):
            captured["metadata"] = args[3]

        monkeypatch.setattr(processing, "generate_embeddings", embedding)
        monkeypatch.setattr(processing, "upsert_embeddings", upsert)
        await processing.process_document(doc_id)
        ready = (await client.get(f"/api/documents/{doc_id}")).json()
        assert ready["status"] == "ready" and ready["pages"][0]["page_number"] == 1
        assert captured["metadata"][0]["page_number"] == 1
        assert (tmp_path / (doc_id + ".txt")).read_text() == TEXT
        response2 = await client.post(
            "/api/ingest", files={"file": ("other.txt", (TEXT + "\n").encode())}
        )
        await processing.process_document(response2.json()["document_id"])
        assert any(
            "Possible vendor/invoice duplicate" in w
            for w in (await client.get(f"/api/documents/{doc_id}")).json()[
                "duplicate_warnings"
            ]
        )


@pytest.mark.asyncio
async def test_rag_citations_untrusted_and_empty_selection(database, monkeypatch):
    from services import rag

    pipeline = rag.CustomRAGPipeline()
    chunk = {
        "score": 0.9,
        "metadata": dict(
            document_id="doc_test",
            page_number=2,
            text="The invoice total is 22.55.",
            source="Invoice.pdf",
            chunk_index=0,
        ),
    }
    monkeypatch.setattr(
        rag,
        "intelligence_records",
        AsyncMock(return_value=[dict(document_id="doc_test", status="ready")]),
    )
    monkeypatch.setattr(rag, "generate_embedding", AsyncMock(return_value=[0.1] * 768))
    query = AsyncMock(return_value=[chunk])
    monkeypatch.setattr(rag, "query_vectors", query)
    assert await pipeline.retrieve_relevant_chunks("total", []) == []
    query.assert_not_called()
    found = await pipeline.retrieve_relevant_chunks("total", ["doc_test", "not_ready"])
    assert found == [chunk] and query.call_args.kwargs["document_ids"] == ["doc_test"]
    context = pipeline.build_context(found)
    messages = await pipeline.build_messages("total", context, "test")
    assert (
        "22.55" not in messages[0]["content"] and "untrusted" in messages[1]["content"]
    )
    monkeypatch.setattr(
        pipeline,
        "generate_response",
        AsyncMock(return_value="The total is 22.55. [Source 1]"),
    )
    monkeypatch.setattr(rag, "add_to_history", AsyncMock())
    result = await pipeline.process_query(
        "What is the invoice total?", "test", ["doc_test"]
    )
    assert result["sources"][0]["page_number"] == 2
    assert result["sources"][0]["page_url"] == "/api/documents/doc_test/pages/2"
    assert result["confidence_score"] is None
    pipeline.generate_response.return_value = "Guess [Source 99]"
    assert (await pipeline.process_query("total?", "test", ["doc_test"]))[
        "sources"
    ] == []
    assert (
        "Approved invoice totals"
        in (await pipeline.process_query("sum invoices", "test"))["response"]
    )


@pytest.mark.asyncio
async def test_restart_marks_interrupted(database, tmp_path, monkeypatch):
    from services import processing
    from db.sql_db import save_intelligence, intelligence_record

    monkeypatch.setattr(processing, "STORAGE", tmp_path)
    await save_intelligence(dict(document_id="interrupted", status="extracting"))
    await processing.start_worker()
    try:
        assert (await intelligence_record("interrupted"))["status"] == "failed"
    finally:
        await processing.stop_worker()


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract not installed")
def test_blank_pdf_has_actionable_error(tmp_path):
    path = make_pdf(tmp_path / "blank.pdf", ("blank",))
    with pytest.raises(ValueError, match="No readable text"):
        extract(path, ".pdf")


@pytest.mark.asyncio
async def test_extraction_timeout_kills_worker(tmp_path, monkeypatch):
    from services import processing

    path = make_pdf(tmp_path / "native.pdf")
    monkeypatch.setattr(processing.settings, "PROCESSING_TIMEOUT_SECONDS", 0.001)
    with pytest.raises(asyncio.TimeoutError):
        await processing.run_extraction(path, ".pdf")
    assert not Path(str(path) + ".pages.json").exists()


@pytest.mark.asyncio
async def test_corrupt_background_job_becomes_failed(database, tmp_path, monkeypatch):
    from services import processing
    from db.sql_db import save_intelligence, intelligence_record

    monkeypatch.setattr(processing, "STORAGE", tmp_path)
    monkeypatch.setattr(processing, "queue", asyncio.Queue(20))
    monkeypatch.setattr(processing, "delete_document_vectors", AsyncMock())
    (tmp_path / "corrupt.pdf").write_bytes(b"corrupt")
    await save_intelligence(
        dict(
            document_id="corrupt",
            stored_name="corrupt.pdf",
            extension=".pdf",
            status="queued",
        )
    )
    worker = asyncio.create_task(processing.worker())
    try:
        await processing.queue.put("corrupt")
        await asyncio.wait_for(processing.queue.join(), 10)
        record = await intelligence_record("corrupt")
        assert record["status"] == "failed" and "extension" in record["message"]
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker


def test_short_chunk_tail_is_preserved():
    from services.chunking import TextChunker

    chunker = TextChunker()
    chunker.min_size = 10
    chunker.max_size = 15
    chunker.chunk_overlap = 0
    chunks = chunker.semantic_chunking(
        "A sentence with enough words to fill most of a chunk. Last detail."
    )
    assert "Last detail." in " ".join(chunks)


def test_partial_line_item_and_large_decimal_product():
    inv = invoice(
        TEXT.replace("Consulting | 2 | 10.25 | 20.50", "Consulting | | 10.25 | 20.50")
    )
    assert inv.line_items[0].quantity.status == "missing"
    assert inv.line_items[0].amount.normalized == "20.50"
    assert any("incomplete arithmetic" in w for w in validate_invoice(inv))
    inv = invoice()
    inv.line_items[0].quantity.normalized = "1000000000000000"
    inv.line_items[0].unit_price.normalized = "1000000000000000"
    assert any("quantity" in w for w in validate_invoice(inv))


@pytest.mark.asyncio
async def test_no_implicit_index_provisioning(monkeypatch):
    from db import vector_db as module

    class FakePinecone:
        def __init__(self, **kwargs):
            pass

        def list_indexes(self):
            return []

        def create_index(self, **kwargs):
            raise AssertionError("Must not provision resources")

    monkeypatch.setattr(module, "Pinecone", FakePinecone)
    with pytest.raises(RuntimeError, match="automatic provisioning is disabled"):
        await module.VectorDatabase().initialize()


@pytest.mark.asyncio
async def test_service_failure_is_not_insufficient_evidence(database, monkeypatch):
    from services import rag

    pipeline = rag.CustomRAGPipeline()
    monkeypatch.setattr(
        rag, "intelligence_records", AsyncMock(side_effect=ConnectionError("offline"))
    )
    with pytest.raises(RuntimeError, match="search is unavailable"):
        await pipeline.retrieve_relevant_chunks("question")
    monkeypatch.setattr(
        pipeline,
        "_generate_with_ollama",
        AsyncMock(side_effect=ConnectionError("offline")),
    )
    with pytest.raises(RuntimeError, match="generation is unavailable"):
        await pipeline.generate_response([])
