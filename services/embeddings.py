from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer
import asyncio
from config import settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        logger.info(f"embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIMENSION
        logger.info(f"embedding model loaded (dimension:{self.dimension})")
    
    async def generate_embedding(self,text:str)->List[float]:
        """generation of embedding for single text"""
        try:
            text = text.replace("\n", " ").strip()
            if not text:
                logger.warning("Empty text provider for embedding")
                return [0.0] * self.dimension
            embedding = await asyncio.to_thread(
                self.model.encode,
                text,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            embedding_list = embedding.tolist()
            logger.debug(f"Generated embedding of dimension {len(embedding_list)}")
            return embedding_list
        except Exception as e:
            logger.error(f"Error generated: {str(e)}")
            raise

    async def generate_embeddings_batch(self,texts:List[str],batch_size:int=32)->List[List[float]]:
        """generating embedding for multiple text in batches"""
        try:
            cleaned_texts = [
                text.replace("\n", " ").strip()
                for text in texts
            ]
            
            # embeddings in batches (run in thread)
            all_embeddings = await asyncio.to_thread(
                self.model.encode,
                cleaned_texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=True
            )
            
            # Converting to list of lists
            embeddings_list = [emb.tolist() for emb in all_embeddings]
            
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generated {str(e)}")
            raise
embedding_generator = EmbeddingGenerator()

async def generate_embedding(text:str) -> List[float]:
    """for single embedding"""
    return await embedding_generator.generate_embedding(text)
async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """for multiple embedding"""
    return await embedding_generator.generate_embeddings_batch(texts)
