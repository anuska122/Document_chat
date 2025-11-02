import logging
import uuid
from fastapi import APIRouter,HTTPException
from models.schemas import(
    ChatRequest,
    ChatResponse,
    InterviewBookingRequest,
    InterviewBookingResponse,
    SourceChunk
)
from services.rag import run_rag
from services.redis_memory import get_chat_history,clear_session
from db.sql_db import create_booking,get_all_bookings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Here questions can be asked from uploaded documents"""
    try:
        logger.info(f"Chat request from session: {request.session_id}")
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        #custom rag pipeline
        result = await run_rag(
            query=request.message,
            session_id=request.session_id,
            document_ids=request.document_ids
        )
        #response formating
        sources_data = result.get("sources", [])
        sources = [
            SourceChunk(
                document=src["document"],
                chunk_index=src["chunk_index"],
                relevance_score=src["relevance_score"],
                text_preview=src.get("text_preview", "")[:200]

            )
            for src in sources_data
        ]
        logger.info(f"response generated with {len(sources)} sources")
        return ChatResponse(
            response=result["response"],
            sources=sources,
            session_id=result["session_id"],
            confidence_score=result.get("confidence_score")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history/{session_id}")
async def get_session_history(session_id: str):
    try:
        history = await get_chat_history(session_id)
        return{
            "session_id":session_id,
            "message_count":len(history),
            "message":history
        }
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/history/{session_id}")
async def clear_session_history(session_id:str):
    try:
        await clear_session(session_id)
        return{
            "status":"success",
            "message":f"Cleared history for session: {session_id}"
        }
    except Exception as e:
        logger.error(f"History clear error:{session_id}")
        raise HTTPException(status_code=500,detail=str(e))

@router.post("/book-interview",response_model=InterviewBookingResponse)
async def book_interview(request:InterviewBookingRequest):
    """interview schedule with validation"""
    try:
        logger.info(f"booking request from: {request.email}")
        booking_id = f"booking_{uuid.uuid4().hex[:12]}"
        await create_booking(booking_id=booking_id,name=request.name,email=request.email,interview_date=str(request.date),interview_time=str(request.time),notes=request.notes)
        logger.info(f"Booking created: {booking_id}")
        details = {
            "name":request.name,
            "email":request.email,
            "scheduled_at":f"{request.date}T{request.time}:00",
            "notes":request.notes
        }
        return InterviewBookingResponse(
            booking_id=booking_id,
            status="confirmed",
            details=details,
            confirmation_sent=True
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bookings")
async def list_bookings():
    """Listing all bookings"""
    try:
        bookings = await get_all_bookings()
        
        return {
            "total_bookings": len(bookings),
            "bookings": bookings
        }
    except Exception as e:
        logger.error(f"Error listing bookings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_system_stats():
    """system statistics"""
    try:
        from db.sql_db import get_stats
        from services.redis_memory import memory_manager
        
        stats = await get_stats()
        session_count = await memory_manager.get_session_count()
        
        return {
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "total_conversations": session_count,
            "total_bookings": stats["total_bookings"]
        }
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))