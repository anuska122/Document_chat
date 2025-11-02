import logging
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
import asyncio

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
            # Initialized Pinecone
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Checking if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                # Waiting for index to be ready
                await asyncio.sleep(5)
            
            # Connecting to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✓ Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            raise
    
    async def upsert_vectors(
        self,
        vectors: List[tuple],
        namespace: str = ""
    ) -> dict:
        """Insert or update vectors in Pinecone"""
    
        try:
            response = await asyncio.to_thread(
                self.index.upsert,
                vectors=vectors,
                namespace=namespace
            )
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
            return response
        except Exception as e:
            logger.error(f"Vector upsert error: {str(e)}")
            raise
    
    async def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Query similar vectors from Pinecone"""
        try:
            query_params = {
                "vector": embedding,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": True
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            results = await asyncio.to_thread(
                self.index.query,
                **query_params
            )
            
            #results format
            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            logger.info(f"Found {len(matches)} matching vectors")
            return matches
            
        except Exception as e:
            logger.error(f"Vector query error: {str(e)}")
            raise
    
    async def delete_by_document(
        self,
        document_id: str,
        namespace: str = ""
    ):
        """Deleting all vectors associated with a document"""
        try:
            await asyncio.to_thread(
                self.index.delete,
                filter={"document_id": document_id},
                namespace=namespace
            )
            logger.info(f"Deleted vectors for document: {document_id}")
        except Exception as e:
            logger.error(f"Vector deletion error: {str(e)}")
            raise
    
    async def get_stats(self) -> dict:
        """index statistics"""
        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            return stats
        except Exception as e:
            logger.error(f"Stats retrieval error: {str(e)}")
            return {}


# Global vector database instance
vector_db = VectorDatabase()


async def init_vector_db():
    """Initializing vector database connection"""
    await vector_db.initialize()


async def upsert_embeddings(
    document_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata_list: Optional[List[Dict]] = None
) -> dict:
    """Upsert document chunks with embeddings"""
    vectors = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        #metadata
        metadata = {
            "document_id": document_id,
            "chunk_index": i,
            "text": chunk[:1000], 
            "source": document_id  
        }
        
        # Adding custom metadata if provided
        if metadata_list and i < len(metadata_list):
            metadata.update(metadata_list[i])
        
        #vector tuple (id, embedding, metadata)
        vector_id = f"{document_id}_chunk_{i}"
        vectors.append((vector_id, embedding, metadata))
    
    return await vector_db.upsert_vectors(vectors)


async def query_vectors(
    embedding: List[float],
    top_k: int = 5,
    document_ids: Optional[List[str]] = None
) -> List[Dict]:
    """Query similar vectors"""
    filter_dict = None
    if document_ids:
        filter_dict = {"document_id": {"$in": document_ids}}
    
    return await vector_db.query(
        embedding=embedding,
        top_k=top_k,
        filter_dict=filter_dict
    )

async def delete_document_vectors(document_id: str):
    """Deleting all vectors for a document"""
    await vector_db.delete_by_document(document_id)