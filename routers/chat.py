import logging
import uuid
from fastapi import APIRouter, HTTPException
from models.schemas import (
    ChatRequest,
    ChatResponse,
    InterviewBookingRequest,
    InterviewBookingResponse,
    SourceChunk,
)
from services.rag import run_rag
from services.redis_memory import get_chat_history, clear_session
from db.sql_db import create_booking, get_all_bookings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Handles chat queries with optional document context."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        logger.info(f"Chat request received (session={request.session_id})")

        result = await run_rag(
            query=request.message,
            session_id=request.session_id,
            document_ids=request.document_ids,
        )

        sources_data = result.get("sources", [])
        sources = [
            SourceChunk(
                document=src.get("document", "Unknown"),
                chunk_index=src.get("chunk_index", 0),
                relevance_score=src.get("relevance_score", 0.0),
                text_preview=src.get("text_preview", "")[:200],
            )
            for src in sources_data
        ]

        logger.info(f"Response generated successfully ({len(sources)} sources).")

        return ChatResponse(
            response=result.get("response", "No response generated."),
            sources=sources,
            session_id=result.get("session_id", request.session_id),
            confidence_score=result.get("confidence_score", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing chat.")


@router.get("/chat/history/{session_id}")
async def get_session_history(session_id: str):
    """Fetch the entire chat history for a given session."""
    try:
        history = await get_chat_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history,
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history.")


@router.delete("/chat/history/{session_id}")
async def clear_session_history(session_id: str):
    """clear the chat history for a specific session."""
    try:
        await clear_session(session_id)
        logger.info(f"Cleared history for session {session_id}")
        return {"status": "success", "message": f"History cleared for session {session_id}."}
    except Exception as e:
        logger.error(f"Error clearing chat history for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear session history.")

@router.post("/book-interview", response_model=InterviewBookingResponse)
async def book_interview(request: InterviewBookingRequest):
    """Books an interview schedule and validates request and saves booking data to the database."""
    try:
        booking_id = f"booking_{uuid.uuid4().hex[:12]}"
        logger.info(f"Booking request from {request.email} (ID={booking_id})")

        await create_booking(
            booking_id=booking_id,
            name=request.name,
            email=request.email,
            interview_date=str(request.date),
            interview_time=str(request.time),
            notes=request.notes,
        )

        details = {
            "name": request.name,
            "email": request.email,
            "scheduled_at": f"{request.date}T{request.time}:00",
            "notes": request.notes,
        }

        logger.info(f"Booking created successfully (ID={booking_id})")

        return InterviewBookingResponse(
            booking_id=booking_id,
            status="confirmed",
            details=details,
            confirmation_sent=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Booking error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create interview booking.")


@router.get("/bookings")
async def list_bookings():
    """Retrieve a list of all interview bookings."""
    try:
        bookings = await get_all_bookings()
        return {
            "total_bookings": len(bookings),
            "bookings": bookings,
        }
    except Exception as e:
        logger.error(f"Error listing bookings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve bookings.")

@router.get("/stats")
async def get_system_stats():
    """
    Provides system-level statistics including:
    - Total documents
    - Total chunks
    - Total active conversations
    - Total bookings
    """
    try:
        from db.sql_db import get_stats
        from services.redis_memory import memory_manager

        stats = await get_stats()
        session_count = await memory_manager.get_session_count()

        return {
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "total_conversations": session_count,
            "total_bookings": stats.get("total_bookings", 0),
        }
    except Exception as e:
        logger.error(f"Error retrieving system stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics.")
