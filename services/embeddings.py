from typing import List
import logging
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self):
        logger.info(f"Initializing OpenAI embedding model: {settings.EMBEDDING_MODEL}")
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        logger.info(f"OpenAI embedding model loaded (dimension: {self.dimension})")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generation of embedding for single text"""
        try:
            text = text.replace("\n", " ").strip()
            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension
            
            #parameters
            params = {
                "input": text,
                "model": self.model
            }
            
            if "text-embedding-3" in self.model and self.dimension != 1536:
                params["dimensions"] = self.dimension
            
            response = await self.client.embeddings.create(**params)
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise

    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generates embeddings for multiple texts in batches"""
        try:
            cleaned_texts = [
                text.replace("\n", " ").strip()
                for text in texts
            ]
            
            
            all_embeddings = []
            
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(cleaned_texts)-1)//batch_size + 1}")
                
                response = await self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    dimensions=self.dimension if self.dimension < 1536 else None
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise


embedding_generator = EmbeddingGenerator()


async def generate_embedding(text: str) -> List[float]:
    """Wrapper for single embedding generation"""
    return await embedding_generator.generate_embedding(text)


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Wrapper for batch embedding generation"""
    return await embedding_generator.generate_embeddings_batch(texts)