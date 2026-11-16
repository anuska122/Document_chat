import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from db.sql_db import (
    intelligence_record,
    intelligence_records,
    update_intelligence,
    create_document,
)
from services.chunking import chunk_document, count_tokens
from services.embeddings import generate_embeddings
from db.vector_db import upsert_embeddings, delete_document_vectors
from services.invoices import extract_invoice, validate_invoice

logger = logging.getLogger(__name__)
STORAGE = Path(settings.STORAGE_DIR).resolve()
queue = asyncio.Queue(maxsize=20)
worker_task = None


def now():
    return datetime.now(timezone.utc).isoformat()


def page_chunks(pages, strategy):
    chunks, metadata = [], []
    for page in pages:
        for text in chunk_document(page["text"], strategy):
            chunks.append(text)
            metadata.append(
                dict(page_number=page["page_number"], extraction_method=page["method"])
            )
    return chunks, metadata


async def run_extraction(path, extension):
    output = str(path) + ".pages.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "services.extraction",
        str(path),
        extension,
        str(settings.MAX_DOCUMENT_PAGES),
        settings.OCR_LANGUAGE,
        str(settings.OCR_PAGE_TIMEOUT_SECONDS),
        output,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, err = await asyncio.wait_for(
            process.communicate(), settings.PROCESSING_TIMEOUT_SECONDS
        )
        if process.returncode:
            message = err.decode(errors="replace")[-1500:]
            raise ValueError(
                message or "Document could not be read. Upload a valid PDF or image."
            )
        return json.loads(Path(output).read_text())
    except (asyncio.TimeoutError, asyncio.CancelledError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        raise
    finally:
        Path(output).unlink(missing_ok=True)


async def process_document(document_id):
    record = await intelligence_record(document_id)
    start = time.monotonic()
    await update_intelligence(
        document_id,
        lambda r: r.update(status="extracting", message="Reading document pages"),
    )
    pages = await run_extraction(STORAGE / record["stored_name"], record["extension"])
    invoice = await asyncio.to_thread(extract_invoice, pages)
    chunks, metadata = await asyncio.to_thread(page_chunks, pages, record["strategy"])
    if not chunks:
        raise ValueError("No readable content to index.")
    await update_intelligence(
        document_id,
        lambda r: r.update(
            pages=pages,
            original=invoice.model_dump(mode="json"),
            reviewed=invoice.model_dump(mode="json"),
            validation_warnings=validate_invoice(invoice),
            status="indexing",
            message="Preparing document search",
        ),
    )
    embeddings = await generate_embeddings(chunks)
    for meta in metadata:
        meta["source"] = record["filename"]
    await upsert_embeddings(document_id, chunks, embeddings, metadata)
    await create_document(
        document_id,
        record["filename"],
        record["file_size_bytes"],
        record["strategy"],
        len(chunks),
    )
    await update_intelligence(
        document_id,
        lambda r: r.update(
            status="ready",
            message="Ready for questions and review",
            chunks_created=len(chunks),
            document_tokens=sum(count_tokens(p["text"]) for p in pages),
            processing_time_ms=round((time.monotonic() - start) * 1000),
        ),
    )


async def worker():
    while True:
        document_id = await queue.get()
        try:
            await asyncio.wait_for(
                process_document(document_id), settings.PROCESSING_TIMEOUT_SECONDS + 120
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Document processing failed: %s", document_id)
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else "Processing failed or timed out. Check OCR/search services and upload again."
            )
            await update_intelligence(
                document_id, lambda r: r.update(status="failed", message=message)
            )
            try:
                await asyncio.wait_for(delete_document_vectors(document_id), 5)
            except Exception:
                logger.warning(
                    "Failed vector cleanup; failed documents remain excluded from retrieval."
                )
        finally:
            queue.task_done()


async def start_worker():
    global worker_task
    STORAGE.mkdir(parents=True, exist_ok=True)
    # Single ASGI worker is required. Never silently leave stale jobs processing.
    for record in await intelligence_records():
        if record["status"] in {"queued", "extracting", "indexing"}:
            await update_intelligence(
                record["document_id"],
                lambda r: r.update(
                    status="failed",
                    message="Processing was interrupted by a restart. Upload the original again.",
                ),
            )
    worker_task = asyncio.create_task(worker())


async def stop_worker():
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
