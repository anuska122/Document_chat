# Project Scan Findings

## Overview

This is a **Conversational RAG Backend** built with FastAPI. It provides document ingestion, vector-based retrieval, and AI-powered chat capabilities with memory.

---

## Project Structure

```
rag_backend/
├── main.py                 # FastAPI app entry point
├── config.py              # Configuration & settings
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml    # Docker Compose setup
│
├── routers/               # API route handlers
│   ├── ingest.py         # Document ingestion endpoints
│   └── chat.py          # Chat endpoints
│
├── services/             # Core business logic
│   ├── rag.py            # RAG pipeline
│   ├── embeddings.py    # Embedding generation
│   ├── chunking.py      # Text chunking
│   └── redis_memory.py # Session memory management
│
├── db/                   # Database layer
│   ├── sql_db.py         # PostgreSQL operations
│   └── vector_db.py      # Pinecone operations
│
├── models/               # Data models
│   ├── schemas.py       # Pydantic schemas
│   └── db_models.py     # SQLAlchemy models
│
└── utils/                # Utilities
    └── file_handler.py  # File processing (PDF/TXT)
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI 0.104.1 |
| Server | Uvicorn 0.24.0 |
| SQL Database | PostgreSQL (async) |
| ORM | SQLAlchemy 2.0.23 |
| Vector DB | Pinecone 6.0.2 |
| Chat Memory | Redis 5.0.1 |
| Embeddings | OpenAI / Hugging Face |
| LLM | Groq (via Ollama CLI) |
| PDF Processing | PyMuPDF 1.23.8 |
| Tokenization | tiktoken 0.7.0 |

---

## API Endpoints

### Document Ingestion (`/api/ingest`)
- `POST /api/ingest` - Upload and process documents (PDF/TXT)
- `GET /api/documents` - List all uploaded documents

### Chat (`/api/chat`)
- `POST /api/chat` - Chat with documents using RAG
- `GET /api/chat/history/{session_id}` - Get session history
- `DELETE /api/chat/history/{session_id}` - Clear session history

### System
- `GET /health` - Health check
- `GET /` - Root endpoint
- `GET /api/stats` - System statistics

---

## Key Features

### 1. Document Ingestion
- Supports PDF and TXT files (max 10MB)
- Two chunking strategies: `fixed` and `semantic`
- Returns document, embedding, and per-chunk token usage
- Generates embeddings using OpenAI or Hugging Face
- Stores in Pinecone vector database
- Tracks metadata in PostgreSQL

### 2. RAG Pipeline
- Retrieves top-k similar chunks from Pinecone
- Builds context from retrieved chunks
- Generates responses using Ollama CLI
- Maintains conversation history in Redis
- Returns sources with relevance scores

### 3. Memory Management
- Session-based chat history in Redis
- Configurable session expiration (default: 1 hour)
- Last 6 messages used for context
- Max 20 messages per session

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_PROVIDER` | "openai" | Embedding provider |
| `EMBEDDING_DIMENSION` | 1536 | Embedding vector size |
| `CHUNK_SIZE` | 500 | Tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K_RESULTS` | 10 | Retrieved chunks |
| `LLM_MODEL` | "gpt-4o-mini" | Language model |
| `PINECONE_INDEX_NAME` | "rag-documents" | Vector index name |

---

## Database Schema

### PostgreSQL Tables

**documents**
- id (PK)
- filename
- file_size_bytes
- upload_timestamp
- chunk_count
- chunking_strategy

### Pinecone Index

- dimension: 1536
- metric: cosine
- metadata: document_id, chunk_index, text, source, token_count, chunking_strategy

---

## Security Notes

- CORS allows all origins (`*`)
- No authentication implemented
- API keys stored in `.env` file
- No rate limiting

---

## Missing Components

- No tests directory visible
- No authentication/authorization
- No API key validation
- No pagination for listing endpoints
- No document deletion endpoint

---

## Dependencies Summary

**Core**: fastapi, uvicorn, pydantic, pydantic-settings

**Database**: sqlalchemy, psycopg2-binary, redis, pinecone

**AI/ML**: openai, groq, tiktoken, sentence-transformers

**Processing**: PyMuPDF, pdfminer.six, pypdf, nltk

**Utilities**: python-dotenv, python-jose, passlib
