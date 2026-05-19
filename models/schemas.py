from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    FIXED = "fixed"
    SEMANTIC = "semantic"

# Document Ingestion 

class IngestionResponse(BaseModel):
    """Response from document ingestion"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    document_tokens: int = Field(..., ge=0, description="Tokens in the extracted document text")
    embedding_tokens: int = Field(..., ge=0, description="Total chunk tokens sent for embedding")
    chunk_token_counts: List[int] = Field(..., description="Token count for each chunk")
    status: str = Field(..., description="Processing status")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123",
                "filename": "report.pdf",
                "chunks_created": 42,
                "document_tokens": 12840,
                "embedding_tokens": 14120,
                "chunk_token_counts": [350, 420, 390],
                "status": "success",
                "processing_time_ms": 1250
            }
        }


#Conversational RAG

class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    session_id: str = Field(..., min_length=1, max_length=100, description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    document_ids: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123",
                "message": "What is Operating System?",
                "document_ids": None
            }
        }


class SourceChunk(BaseModel):
    """Source chunk information"""
    document: str = Field(..., description="Source document name")
    chunk_index: int = Field(..., ge=0, description="Chunk index in document")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score")
    text_preview: Optional[str] = Field(None, max_length=200, description="Text preview")


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    response: str = Field(..., description="AI-generated response")
    sources: List[SourceChunk] = Field(..., description="Source chunks used")
    session_id: str = Field(..., description="Session identifier")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Response confidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "The main findings indicate significant improvements...",
                "sources": [
                    {
                        "document": "report.pdf",
                        "chunk_index": 5,
                        "relevance_score": 0.89,
                        "text_preview": "Our analysis shows..."
                    }
                ],
                "session_id": "user_123",
                "confidence_score": 0.85
            }
        }



#Error Response

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[dict] = Field(None, description="Additional error details")
