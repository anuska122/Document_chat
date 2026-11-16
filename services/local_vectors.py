"""Persistent local cosine search using the application's real embedding provider.

For the single-user local server. Normal main.py continues to use Pinecone.
"""
import math
from config import settings
from db.sql_db import intelligence_records, update_intelligence


def model_key():
    return f"{settings.EMBEDDING_PROVIDER}:{settings.OLLAMA_EMBEDDING_MODEL}:{settings.EMBEDDING_DIMENSION}"


async def upsert_embeddings(document_id, chunks, embeddings, metadata_list):
    if len(chunks) != len(embeddings) or len(chunks) != len(metadata_list):
        raise ValueError("Search indexing returned an incomplete embedding batch.")
    index = []
    for i, (text, embedding, metadata) in enumerate(
        zip(chunks, embeddings, metadata_list)
    ):
        if len(embedding) != settings.EMBEDDING_DIMENSION or not all(
            math.isfinite(x) for x in embedding
        ):
            raise ValueError(
                "Document search returned invalid embedding dimensions or values."
            )
        index.append(
            dict(
                embedding=embedding,
                metadata=dict(
                    metadata, document_id=document_id, chunk_index=i, text=text
                ),
            )
        )
    await update_intelligence(
        document_id, lambda r: r.update(search_index=index, search_model=model_key())
    )


async def query_vectors(embedding, top_k=10, document_ids=None):
    if not document_ids:
        return []
    norm = math.sqrt(sum(x * x for x in embedding))
    if not norm:
        return []
    results = []
    for record in await intelligence_records():
        if record["document_id"] not in document_ids or record["status"] != "ready":
            continue
        if record.get("search_model") != model_key() or not record.get("search_index"):
            raise RuntimeError("This document needs its local search index rebuilt.")
        for item in record["search_index"]:
            vector = item["embedding"]
            if len(vector) != len(embedding):
                raise RuntimeError(
                    "The local search index uses a different embedding model."
                )
            denominator = norm * math.sqrt(sum(x * x for x in vector))
            if not denominator:
                continue
            score = sum(a * b for a, b in zip(embedding, vector)) / denominator
            results.append(dict(score=score, metadata=item["metadata"]))
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


async def delete_document_vectors(document_id):
    await update_intelligence(document_id, lambda r: r.update(search_index=[]))
