"""Real local application: persistent SQLite/files, Ollama embeddings, configured chat.
No hosted search service and no generated test answers. Start with:
    python -m scripts.local_server
"""
import os
import asyncio
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_dir = Path(
    os.environ.get("LOCAL_DATA_DIR", root / "storage" / "local-preview")
).resolve()
data_dir.mkdir(parents=True, exist_ok=True)
# Local search uses the existing all-minilm Ollama embedding model (384 dimensions).
# Chat settings remain those configured in .env; models are never downloaded here.
os.environ.update(
    DATABASE_URL=f"sqlite+aiosqlite:///{data_dir}/documents.sqlite3",
    STORAGE_DIR=str(data_dir / "files"),
    EMBEDDING_PROVIDER="ollama",
    OLLAMA_EMBEDDING_MODEL="all-minilm",
    EMBEDDING_DIMENSION="384",
    PINECONE_DIMENSION="384",
)
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from db.sql_db import init_db, close_db, intelligence_records, update_intelligence
from routers import ingest, chat
from services import processing, rag, local_vectors
from services.embeddings import generate_embeddings

# Both adapters execute real vector operations; chat generation is never replaced.
processing.upsert_embeddings = local_vectors.upsert_embeddings
processing.delete_document_vectors = local_vectors.delete_document_vectors
rag.query_vectors = local_vectors.query_vectors


async def rebuild_existing_index():
    """Index preserved page text without replacing extraction, corrections or audit."""
    for record in await intelligence_records():
        if record["status"] != "ready" or (
            record.get("search_index")
            and record.get("search_model") == local_vectors.model_key()
        ):
            continue
        chunks, metadata = await asyncio.to_thread(
            processing.page_chunks, record["pages"], record["strategy"]
        )
        for item in metadata:
            item["source"] = record["filename"]
        try:
            embeddings = await generate_embeddings(chunks)
            await local_vectors.upsert_embeddings(
                record["document_id"], chunks, embeddings, metadata
            )
            await update_intelligence(
                record["document_id"], lambda r: r.pop("search_error", None)
            )
        except Exception:
            await update_intelligence(
                record["document_id"],
                lambda r: r.update(
                    search_error="Document search is unavailable. Start the local embedding service and restart the app."
                ),
            )


@asynccontextmanager
async def lifespan(app):
    await init_db()
    await rebuild_existing_index()
    await processing.start_worker()
    yield
    await processing.stop_worker()
    await close_db()


app = FastAPI(lifespan=lifespan)
app.include_router(ingest.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.mount("/assets", StaticFiles(directory=root / "web"), name="assets")


@app.get("/")
async def index():
    return FileResponse(root / "web/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8011)
