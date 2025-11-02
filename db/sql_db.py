import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

# Creating async engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
    future=True
)

# async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

#Database Models

class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    chunk_count = Column(Integer, default=0)
    chunking_strategy = Column(String, nullable=False)
    
    def to_dict(self):
        return {
            "document_id": self.id,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "chunk_count": self.chunk_count,
            "chunking_strategy": self.chunking_strategy
        }


class InterviewBooking(Base):
    """Interview booking table"""
    __tablename__ = "interview_bookings"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    interview_date = Column(String, nullable=False)
    interview_time = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    status = Column(String, default="confirmed")
    created_at = Column(DateTime, default=datetime.utcnow)
    confirmation_sent = Column(Boolean, default=False)
    
    def to_dict(self):
        return {
            "booking_id": self.id,
            "name": self.name,
            "email": self.email,
            "interview_date": self.interview_date,
            "interview_time": self.interview_time,
            "notes": self.notes,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "confirmation_sent": self.confirmation_sent
        }
    
#Database Functions

async def init_db():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✓ Database tables initialized")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise


async def get_db():
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


#Document CRUD

async def create_document(
    document_id: str,
    filename: str,
    file_size: int,
    chunking_strategy: str,
    chunk_count: int = 0
):
    """Create document record"""
    async with AsyncSessionLocal() as session:
        document = Document(
            id=document_id,
            filename=filename,
            file_size_bytes=file_size,
            chunking_strategy=chunking_strategy,
            chunk_count=chunk_count
        )
        session.add(document)
        await session.commit()
        logger.info(f"Created document record: {document_id}")
        return document.to_dict()


async def get_document(document_id: str):
    """Getting document by ID"""
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        return document.to_dict() if document else None


async def get_all_documents():
    """Getting all documents"""
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(select(Document))
        documents = result.scalars().all()
        return [doc.to_dict() for doc in documents]


#Interview Booking CRUD 

async def create_booking(
    booking_id: str,
    name: str,
    email: str,
    interview_date: str,
    interview_time: str,
    notes: str = None
):
    """Creating interview booking"""
    async with AsyncSessionLocal() as session:
        booking = InterviewBooking(
            id=booking_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
            notes=notes,
            confirmation_sent=True
        )
        session.add(booking)
        await session.commit()
        logger.info(f"Created booking: {booking_id}")
        return booking.to_dict()


async def get_all_bookings():
    """Getting all bookings"""
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(select(InterviewBooking))
        bookings = result.scalars().all()
        return [booking.to_dict() for booking in bookings]


async def get_stats():
    """Get database statistics"""
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select, func
        
        # Counting documents
        doc_count = await session.execute(select(func.count(Document.id)))
        total_docs = doc_count.scalar()
        
        # Counting bookings
        booking_count = await session.execute(select(func.count(InterviewBooking.id)))
        total_bookings = booking_count.scalar()
        
        # chunks sum
        chunk_sum = await session.execute(select(func.sum(Document.chunk_count)))
        total_chunks = chunk_sum.scalar() or 0
        
        return {
            "total_documents": total_docs,
            "total_chunks": int(total_chunks),
            "total_bookings": total_bookings
        }