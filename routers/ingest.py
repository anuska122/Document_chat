#document ingestion endpoints
import logging
import uuid
import time
from fastapi import APIRouter,UploadFile,File,HTTPException,Query
from models.schemas import IngestionResponse, ChunkingStrategy
from utils.file_handler import process_uploaded_file
from services.chunking import chunk_document
from services.embeddings import generate_embeddings
from db.vector_db import upsert_embeddings
from db.sql_db import create_document

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest",response_model=IngestionResponse)
async def ingest_documnet(
    file:UploadFile=File(...),
    strategy: ChunkingStrategy=Query(ChunkingStrategy.SEMANTIC)):
    """Document ingestion endpoint"""
    start_time = time.time()
    try:
        logger.info(f"Starting ingestion for file: {file.filename}")
        #text extract
        text,file_size = await process_uploaded_file(file) #-> its converts uploaded file(pdf,docx) into plain text
        logger.info(f"Extracted {len(text)} characters")
        #text chunk
        chunks = chunk_document(text,strategy=strategy.value)
        if not chunks:
            raise HTTPException(status_code=400,detail="No chunks created from document")
        logger.info(f"Chunks {len(chunks)} created")
        #document id generation
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        #embedding generation
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = await generate_embeddings(chunks)
        logger.info(f"Embedding {len(embeddings)} generated..")
        #store in vector db
        metadata_list = [
            {
                "source":file.filename,
                "chunk_index":i,
                "chunking_strategy":strategy.value
            }
            for i in range(len(chunks))
        ]
        await upsert_embeddings(document_id=document_id,chunks=chunks,embeddings=embeddings,metadata_list=metadata_list)
        await create_document(document_id=document_id,filename=file.filename,file_size=file_size,chunking_strategy=strategy.value,chunk_count=len(chunks))

        processing_time = int((time.time()- start_time) * 1000)
        logger.info(f"Ingestion completed in {processing_time}ms")
        return IngestionResponse(document_id=document_id,filename=file.filename,chunks_created=len(chunks),status="success",processing_time_ms=processing_time)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,detail=f"Internal error: {str(e)}"
        )

@router.get("/documents")
async def list_documents():
    try:
        from db.sql_db import get_all_documents
        documents = await get_all_documents()
        return {
            "total_documents":len(documents),
            "documents":documents
        }
    except Exception as e:
        logger.error(f"Error while listing documents: {str(e)}")
        raise HTTPException(status_code=500,detail=str(e))
    

    