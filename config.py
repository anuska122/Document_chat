from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Environment configuration for RAG Backend"""

    # App info
    APP_NAME: str = "RAG BACKEND"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    EMBEDDING_DIMENSION: int = 1536 

    # LLM
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 512

    # Pinecone 
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "rag-documents"
    PINECONE_DIMENSION: int = 1536

    # Database 
    DATABASE_URL: str = "postgresql://raguser:ragpassword@localhost:5432/ragdb"

    # Redis
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
    TOP_K_RESULTS: int = 10
    SIMILARITY_THRESHOLD: float = 0.0
    MAX_CONTEXT_LENGTH: int = 3000

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()