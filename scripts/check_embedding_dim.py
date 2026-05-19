import asyncio
import os
import logging

from services.embeddings import embedding_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("check_embedding")

async def main():
    logger.info(f"Provider: {embedding_generator.provider}, Model: {embedding_generator.model}")
    sample = "This is a short test sentence to check embedding dimensionality."
    emb = await embedding_generator.generate_embedding(sample)
    logger.info(f"Generated embedding length: {len(emb)}")

if __name__ == '__main__':
    if not os.getenv('OPENAI_API_KEY') and embedding_generator.provider == 'openai':
        logger.error('OPENAI_API_KEY is not set in the environment. Set it before running this script.')
    else:
        asyncio.run(main())
