import logging
from pathlib import Path
from typing import Tuple
import fitz  # PyMuPDF
from fastapi import UploadFile, HTTPException

from config import settings

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file upload and text extraction"""
    
    def __init__(self):
        self.max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        self.allowed_types = settings.ALLOWED_FILE_TYPES
    
    def validate_file(self, file: UploadFile) -> Tuple[bool, str]:
        """
        Validate uploaded file """
        if not file or not file.filename:
            return False, "No file provided"
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_types:
            return False, f"File type {file_ext} not allowed. Allowed: {', '.join(self.allowed_types)}"
        
        return True, ""
    
    async def extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Extracts text from PDF using PyMuPDF"""
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            text_content = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text = page.get_text()
                text_content.append(text)
            
            pdf_document.close()
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )
    
    async def extract_text_from_txt(self, file_content: bytes) -> str:
        """Extracts text from TXT file"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text = file_content.decode(encoding)
                    logger.info(f"Decoded TXT with {encoding}: {len(text)} characters")
                    return text
                except UnicodeDecodeError:
                    continue
            
            text = file_content.decode('utf-8', errors='ignore')
            logger.warning("Decoded TXT with utf-8 (ignored errors)")
            return text
            
        except Exception as e:
            logger.error(f"TXT extraction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from TXT: {str(e)}"
            )
    
    async def process_file(self, file: UploadFile) -> Tuple[str, int]:
        """Process uploaded file and extract text"""
        # Validate file
        is_valid, error_msg = self.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Check file size
        if file_size > self.max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum ({settings.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Check if file is empty
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Extract text based on file type
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext == ".pdf":
            text = await self.extract_text_from_pdf(file_content)
        elif file_ext == ".txt":
            text = await self.extract_text_from_txt(file_content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        # Validate extracted text
        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file"
            )
        
        logger.info(f"Successfully processed {file.filename}: {len(text)} characters")
        return text, file_size


# Global file handler instance
file_handler = FileHandler()


async def process_uploaded_file(file: UploadFile) -> Tuple[str, int]:
    """Convenience function to process file"""
    return await file_handler.process_file(file)