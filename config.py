from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment configuration for RAG Backend"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App info
    APP_NAME: str = "RAG Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # API keys
    GROQ_API_KEY: str
    OPENAI_API_KEY: Optional[str] = None

    # Embeddings
    EMBEDDING_PROVIDER: str = "hf"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "all-minilm"
    OLLAMA_EMBEDDING_MAX_TOKENS: int = 256
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

    # LLM
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "llama3.1"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "rag-documents-768"
    PINECONE_DIMENSION: int = 768

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
    SIMILARITY_THRESHOLD: float = -1.0
    MAX_CONTEXT_LENGTH: int = 3000

    # Logging
    LOG_LEVEL: str = "INFO"

    @field_validator("EMBEDDING_PROVIDER", mode="before")
    @classmethod
    def normalize_embedding_provider(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def normalize_llm_provider(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("PINECONE_DIMENSION")
    @classmethod
    def match_embedding_dimension(cls, value, info):
        embedding_dimension = info.data.get("EMBEDDING_DIMENSION")
        if embedding_dimension is not None and value != embedding_dimension:
            raise ValueError(
                "PINECONE_DIMENSION must match EMBEDDING_DIMENSION "
                f"({value} != {embedding_dimension})"
            )
        return value

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "release",
                "prod",
                "production",
                "false",
                "0",
                "no",
                "off",
            }:
                return False
            if normalized in {
                "debug",
                "dev",
                "development",
                "true",
                "1",
                "yes",
                "on",
            }:
                return True
        return value


# Global settings instance
settings = Settings()
