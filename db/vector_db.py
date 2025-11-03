import logging
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
import asyncio
import time
from config import settings

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages Pinecone vector database operations"""

    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = settings.PINECONE_DIMENSION

    async def initialize(self):
        """Initializing Pinecone connection and index"""
        try:
            logger.info("Initializing Pinecone...")

            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            logger.info(f"Existing indexes: {existing_indexes}")

            if self.index_name not in existing_indexes:
                logger.warning(f"⚠️ Index '{self.index_name}' not found — creating new one...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )

                logger.info("Waiting for index creation...")
                for _ in range(30):
                    await asyncio.sleep(2)
                    idx_names = [idx.name for idx in self.pc.list_indexes()]
                    if self.index_name in idx_names:
                        logger.info("Index ready!")
                        break

            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            stats = await asyncio.to_thread(self.index.describe_index_stats)

            logger.info(f"Connected to '{self.index_name}' — "
                        f"Vectors: {stats.get('total_vector_count', 0)}, "
                        f"Dim: {stats.get('dimension', 0)}")

        except Exception as e:
            logger.error(f"Pinecone init failed: {e}", exc_info=True)
            raise

    async def upsert_vectors(self, vectors: List[tuple], namespace: str = "") -> dict:
        """Insert or update vectors"""
        try:
            if not vectors:
                logger.warning("No vectors to upsert.")
                return {}

            logger.info(f"Upserting {len(vectors)} vectors (namespace='{namespace or 'default'}')")

            response = await asyncio.to_thread(
                self.index.upsert,
                vectors=vectors,
                namespace=namespace
            )

            logger.info(f"Upserted {response.get('upserted_count', len(vectors))} vectors")

            stats = await asyncio.to_thread(self.index.describe_index_stats)
            logger.info(f"Total vectors after upsert: {stats.get('total_vector_count', 0)}")

            return response

        except Exception as e:
            logger.error(f"Upsert error: {e}", exc_info=True)
            raise

    async def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Query similar vectors"""
        try:
            logger.info(f"🔍 Querying Pinecone (Top K={top_k}, Namespace='{namespace or 'default'}')")
            logger.info(f"   Filter: {filter_dict or 'None'} | Embedding dim: {len(embedding)}")

            query_params = {
                "vector": embedding,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True,
                "include_values": False,
            }

            if filter_dict:
                query_params["filter"] = filter_dict

            results = await asyncio.to_thread(self.index.query, **query_params)

            if not hasattr(results, "matches") or not results.matches:
                logger.warning("⚠️ No matches found.")
                stats = await asyncio.to_thread(self.index.describe_index_stats)
                logger.warning(f"Index Stats: {stats}")
                return []

            matches = [
                {"id": m.id, "score": m.score, "metadata": getattr(m, "metadata", {})}
                for m in results.matches
            ]

            logger.info(f"Retrieved {len(matches)} matches.")
            for i, m in enumerate(matches[:3]):
                src = m['metadata'].get('document_id', 'unknown')
                snippet = m['metadata'].get('text', '')[:80]
                logger.info(f"   [{i+1}] Score={m['score']:.4f} | Doc={src} | Text='{snippet}...'")

            return matches

        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            raise

    async def delete_by_document(self, document_id: str, namespace: str = ""):
        """Delete all vectors belonging to a document"""
        try:
            await asyncio.to_thread(
                self.index.delete,
                filter={"document_id": document_id},
                namespace=namespace
            )
            logger.info(f"🗑️ Deleted vectors for document: {document_id}")
        except Exception as e:
            logger.error(f" Deletion error: {e}", exc_info=True)
            raise

    async def get_stats(self) -> dict:
        """Retrieve Pinecone index stats"""
        try:
            return await asyncio.to_thread(self.index.describe_index_stats)
        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {}


vector_db = VectorDatabase()

async def init_vector_db():
    await vector_db.initialize()

async def upsert_embeddings(
    document_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata_list: Optional[List[Dict]] = None
) -> dict:
    """Upsert document chunks + embeddings"""
    if not chunks or not embeddings:
        logger.warning("⚠️ No chunks or embeddings provided for upsert.")
        return {}

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        meta = {
            "document_id": document_id,
            "chunk_index": i,
            "text": chunk[:1000],
        }
        if metadata_list and i < len(metadata_list):
            meta.update(metadata_list[i])

        vectors.append((f"{document_id}_chunk_{i}", emb, meta))

    return await vector_db.upsert_vectors(vectors, namespace="")

async def query_vectors(
    embedding: List[float],
    top_k: int = 5,
    document_ids: Optional[List[str]] = None
) -> List[Dict]:
    """Query similar vectors from Pinecone"""
    
    filter_dict = {"document_id": {"$in": document_ids}} if document_ids else None

    return await vector_db.query(
        embedding=embedding,
        top_k=top_k,
        namespace="",
        filter_dict=filter_dict
    )

async def delete_document_vectors(document_id: str):
    await vector_db.delete_by_document(document_id, namespace="")
