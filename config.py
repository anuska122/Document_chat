# config.py - Simplified Configuration (NO MORE ISSUES!)

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Environment configuration for RAG Backend"""

    # App info
    APP_NAME: str = "RAG BACKEND"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Hugging Face (REQUIRED in .env)
    HUGGINGFACE_API_KEY: str
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384 

    # LLM
    LLM_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 512

    # Pinecone (REQUIRED in .env)
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "rag-documents"
    PINECONE_DIMENSION: int = 384

    # Database - Will be loaded from .env
    DATABASE_URL: str = "postgresql://raguser:ragpassword@localhost:5432/ragdb"

    # Redis - Will be loaded from .env
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SESSION_EXPIRE: int = 3600

    # File config
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: list = [".pdf", ".txt"]

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    SEMANTIC_CHUNK_MIN: int = 300
    SEMANTIC_CHUNK_MAX: int = 600

    # RAG
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_CONTEXT_LENGTH: int = 3000

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance (NO CACHE - NO PROBLEMS!)
settings = Settings()