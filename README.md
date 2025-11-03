### RAG Backend – Document Ingestion & Conversational API
- A Retrieval-Augmented Generation (RAG) backend built with FastAPI, OpenAI embeddings, and a vector database (Pinecone/Qdrant/Weaviate/Milvus). Supports document ingestion, semantic search, multi-turn conversation with Redis memory, and interview booking.

### Features

**Document Ingestion API**
- Upload .pdf or .txt files
- Chunk text using semantic or fixed strategies
- Generate embeddings with OpenAI
- Store vectors in vector DB
- Save metadata in SQL/NoSQL DB

**Conversational RAG API**
- Multi-turn queries with Redis chat memory
- Custom retrieval (no RetrievalQAChain)
- Interview booking: name, email, date, time

