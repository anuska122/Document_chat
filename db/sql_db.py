import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, DateTime
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


async def close_db():
    """Close database connections."""
    try:
        await engine.dispose()
        logger.info("Database connection disposed")
    except Exception as e:
        logger.error(f"Error closing database: {str(e)}")

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


async def get_stats():
    """Get database statistics"""
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select, func
        
        # Counting documents
        doc_count = await session.execute(select(func.count(Document.id)))
        total_docs = doc_count.scalar()
        
        # chunks sum
        chunk_sum = await session.execute(select(func.sum(Document.chunk_count)))
        total_chunks = chunk_sum.scalar() or 0
        
        return {
            "total_documents": total_docs,
            "total_chunks": int(total_chunks),
        }
