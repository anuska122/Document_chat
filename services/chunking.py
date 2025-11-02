import re
from typing import List
import logging
from transformers import AutoTokenizer
from config import settings
logger = logging.getLogger("__name__")

class TextChunker:
    """for handling text with multiple strategies"""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL)
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.semantic_min = settings.SEMANTIC_CHUNK_MIN
        self.semantic_max = settings.SEMANTIC_CHUNK_MAX
    def count_tokens(self,text:str)->int:
        tokens = self.tokenizer.encode(text,add_special_tokens=False)
        return len(tokens)
    
    def fixed_size_chunking(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text.strip())
            start += self.chunk_size - self.chunk_overlap
        
        logger.info(f"Fixed chunking: created {len(chunks)} chunks")
        return chunks
    
    def semantic_chunking(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # spliting long sentence
            if sentence_tokens > self.semantic_max:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word)
                    if temp_tokens + word_tokens > self.semantic_max:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                continue
            
            # adding sentence to chunk
            if current_tokens + sentence_tokens > self.semantic_max:
                if current_tokens >= self.semantic_min:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.info(f"Semantic chunking: created {len(chunks)} chunks")
        return chunks
    
    def chunk_text(self, text: str, strategy: str = "semantic") -> List[str]:
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        if strategy == "fixed":
            return self.fixed_size_chunking(text)
        elif strategy == "semantic":
            return self.semantic_chunking(text)
        else:
            raise ValueError(f"Invalid chunking strategy: {strategy}")


# Global chunker instance
chunker = TextChunker()


def chunk_document(text: str, strategy: str = "semantic") -> List[str]:
    """Convenience function to chunk text"""
    return chunker.chunk_text(text, strategy)

