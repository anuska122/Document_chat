"""Document lifecycle and separate structured invoice review API."""
import asyncio
import hashlib
import uuid
from pathlib import Path
from decimal import Decimal

import fitz
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, Response
from config import settings
from models.schemas import ChunkingStrategy
from models.invoice import ReviewRequest, Invoice
from db.sql_db import (
    save_intelligence,
    intelligence_records,
    intelligence_record,
    update_intelligence,
)
from services.processing import STORAGE, queue, now
from services.invoices import business_key, validate_invoice, export_csv

router = APIRouter()
upload_lock = asyncio.Lock()


async def record_or_404(document_id):
    record = await intelligence_record(document_id)
    if not record:
        raise HTTPException(404, "Document not found")
    return record


async def duplicates(record):
    key = (
        business_key(Invoice.model_validate(record["reviewed"]))
        if record.get("reviewed")
        else None
    )
    warnings = []
    for other in await intelligence_records():
        if other["document_id"] == record["document_id"]:
            continue
        if other["content_hash"] == record["content_hash"]:
            warnings.append(f"Exact file duplicate: {other['document_id']}")
        elif (
            key
            and other.get("reviewed")
            and key == business_key(Invoice.model_validate(other["reviewed"]))
        ):
            warnings.append(
                f"Possible vendor/invoice duplicate: {other['document_id']}"
            )
    return warnings


@router.post("/ingest", status_code=202)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: ChunkingStrategy = Query(ChunkingStrategy.SEMANTIC),
):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".pdf", ".png", ".jpg", ".jpeg", ".txt"}:
        raise HTTPException(415, "Upload PDF, PNG, JPEG, or UTF-8 TXT.")
    data = bytearray()
    while block := await file.read(65536):
        data.extend(block)
        if len(data) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {settings.MAX_FILE_SIZE_MB} MB.")
    if not data:
        raise HTTPException(400, "File is empty.")
    document_id = "doc_" + uuid.uuid4().hex
    record = dict(
        document_id=document_id,
        filename=Path(file.filename).name,
        extension=extension,
        stored_name=document_id + extension,
        file_size_bytes=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
        strategy=strategy.value,
        status="queued",
        message="Waiting to process",
        created_at=now(),
        review_status="needs_review",
        version=0,
        audit=[],
        pages=[],
    )
    async with upload_lock:
        if queue.full():
            raise HTTPException(503, "Processing queue is full. Try again shortly.")
        STORAGE.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread((STORAGE / record["stored_name"]).write_bytes, data)
        try:
            await save_intelligence(record)
        except Exception:
            (STORAGE / record["stored_name"]).unlink(missing_ok=True)
            raise
        queue.put_nowait(document_id)
    return {
        "document_id": document_id,
        "status": "queued",
        "filename": record["filename"],
        "status_url": f"/api/documents/{document_id}",
    }


@router.get("/documents")
async def list_documents():
    records = await intelligence_records()
    return {
        "total_documents": len(records),
        "documents": [
            {
                k: r[k]
                for k in (
                    "document_id",
                    "filename",
                    "status",
                    "message",
                    "review_status",
                    "created_at",
                )
            }
            for r in records
        ],
    }


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    record = await record_or_404(document_id)
    public = {
        k: v
        for k, v in record.items()
        if k not in {"stored_name", "search_index", "search_model"}
    }
    public["duplicate_warnings"] = await duplicates(record)
    return public


@router.get("/documents/{document_id}/original")
async def original(document_id: str):
    record = await record_or_404(document_id)
    media = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".txt": "text/plain",
    }
    return FileResponse(
        STORAGE / record["stored_name"],
        media_type=media[record["extension"]],
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get("/documents/{document_id}/pages/{page_number}")
async def preview(document_id: str, page_number: int):
    record = await record_or_404(document_id)
    if not 1 <= page_number <= len(record["pages"]):
        raise HTTPException(404, "Page not available")
    if record["extension"] == ".txt":
        return Response(record["pages"][0]["text"], media_type="text/plain")

    def render():
        with fitz.open(STORAGE / record["stored_name"]) as src:
            converted = (
                fitz.open("pdf", src.convert_to_pdf()) if not src.is_pdf else None
            )
            try:
                return (
                    (converted or src)[page_number - 1]
                    .get_pixmap(dpi=100)
                    .tobytes("png")
                )
            finally:
                if converted:
                    converted.close()

    return Response(await asyncio.to_thread(render), media_type="image/png")


@router.put("/documents/{document_id}/review")
async def review(document_id: str, request: ReviewRequest):
    record = await record_or_404(document_id)
    if record["status"] != "ready":
        raise HTTPException(409, "Wait for document processing to finish.")
    # Evidence must refer to actual uploaded text; human values may be corrected without evidence.
    for value in list(request.invoice.fields.values()) + [
        v for item in request.invoice.line_items for _, v in item
    ]:
        for evidence in value.evidence:
            if (
                not 1 <= evidence.page_number <= len(record["pages"])
                or evidence.text
                not in record["pages"][evidence.page_number - 1]["text"]
            ):
                raise HTTPException(
                    422, "Source evidence must match the original page text."
                )
    warnings = validate_invoice(request.invoice)
    candidate = dict(record, reviewed=request.invoice.model_dump(mode="json"))
    duplicate_warnings = await duplicates(candidate)
    if (
        request.action == "approve"
        and (warnings or duplicate_warnings)
        and not request.acknowledgement.strip()
    ):
        raise HTTPException(
            422, "Explain your review of the warnings before approving."
        )

    def change(r):
        r["reviewed"] = request.invoice.model_dump(mode="json")
        r["validation_warnings"] = warnings
        r["review_status"] = (
            "approved" if request.action == "approve" else "needs_review"
        )
        r["version"] += 1
        r["audit"] = r["audit"] + [
            dict(
                timestamp=now(),
                actor="local-single-user",
                action=request.action,
                invoice=r["reviewed"],
                acknowledgement=request.acknowledgement,
                duplicate_warnings=duplicate_warnings,
            )
        ]

    try:
        return await update_intelligence(document_id, change, request.version)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/documents/{document_id}/export")
async def export(document_id: str, format: str = Query("json", pattern="^(json|csv)$")):
    record = await record_or_404(document_id)
    if not record.get("reviewed"):
        raise HTTPException(409, "Invoice extraction is not available yet.")
    if format == "csv":
        return Response(
            export_csv(record),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{document_id}.csv"'
            },
        )
    return {
        k: record.get(k)
        for k in (
            "document_id",
            "filename",
            "content_hash",
            "review_status",
            "original",
            "reviewed",
            "validation_warnings",
            "audit",
        )
    } | {"duplicate_warnings": await duplicates(record)}


@router.get("/invoices/summary")
async def summary():
    groups = {}
    for r in await intelligence_records():
        if r["status"] != "ready" or r["review_status"] != "approved":
            continue
        invoice = Invoice.model_validate(r["reviewed"])
        currency, total = (invoice.fields[k] for k in ("currency", "grand_total"))
        if currency.status != "present" or total.status != "present":
            continue
        group = groups.setdefault(
            currency.normalized, dict(total=Decimal(0), documents=[])
        )
        group["total"] += Decimal(total.normalized)
        group["documents"].append(r["document_id"])
    return {
        "basis": "Approved invoice grand totals; currencies kept separate; duplicates included until reviewed by user",
        "currencies": {
            k: dict(total=str(v["total"]), documents=v["documents"])
            for k, v in groups.items()
        },
    }
