from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime, time as dtime
from datetime import datetime, date as ddate
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
    status: str = Field(..., description="Processing status")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_abc123",
                "filename": "report.pdf",
                "chunks_created": 42,
                "status": "success",
                "processing_time_ms": 1250
            }
        }


#Conversational RAG

class ChatRequest(BaseModel):
    """Request for chat endpoint"""
    session_id: str = Field(..., min_length=1, max_length=100, description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    document_ids: Optional[List[str]] = Field(
        None,
        description="Specific documents to search (empty = search all)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123",
                "message": "What is Operating System?",
                "document_ids": ["doc_abc123"]
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


#Interview Booking

class InterviewBookingRequest(BaseModel):
    """Request to book an interview"""
    name: str = Field(..., min_length=2, max_length=100, description="Full name")
    email: EmailStr = Field(..., description="Email address")
    date: ddate = Field(..., description="Interview date (YYYY-MM-DD)")
    time: dtime = Field(..., description="Interview time (HH:MM)")
    notes: Optional[str] = Field(None, max_length=500, description="Additional notes")
    
    @field_validator('date')
    @classmethod
    def date_must_be_future(cls, v):
        """Validate that date is in the future"""
        if v < datetime.now().date():
            raise ValueError('Interview date must be in the future')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Anushka Hadkhale",
                "email": "anushka@example.com",
                "date": "2025-11-15",
                "time": "14:00",
                "notes": "Preferred platform: Zoom"
            }
        }


class InterviewBookingResponse(BaseModel):
    """Response from booking interview"""
    booking_id: str = Field(..., description="Unique booking identifier")
    status: str = Field(..., description="Booking status")
    details: dict = Field(..., description="Booking details")
    confirmation_sent: bool = Field(..., description="Whether confirmation email was sent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "booking_id": "booking_xyz789",
                "status": "confirmed",
                "details": {
                    "name": "Anushka Hadkhale",
                    "email": "anushka@example.com",
                    "scheduled_at": "2025-11-15T14:00:00"
                },
                "confirmation_sent": True
            }
        }


#Error Response

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[dict] = Field(None, description="Additional error details")