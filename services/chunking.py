import re
import logging
from typing import List
import tiktoken
from config import settings

logger = logging.getLogger("text_chunker")


class TextChunker:
    """Robust text chunking with fixed-size or semantic strategies using tiktoken."""

    def __init__(self):
        try:
            self.tokenizer = tiktoken.encoding_for_model(settings.LLM_MODEL)
            logger.info(f"Tokenizer loaded for model: {settings.LLM_MODEL}")
        except Exception as e:
            # Fallback
            logger.warning(f"Could not load tokenizer for {settings.LLM_MODEL}, using cl100k_base: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.min_size = settings.SEMANTIC_CHUNK_MIN
        self.max_size = settings.SEMANTIC_CHUNK_MAX
        self.max_context = settings.MAX_CONTEXT_LENGTH

        logger.info(
            f"Chunker init | size={self.chunk_size} | overlap={self.chunk_overlap} | "
            f"min={self.min_size} | max={self.max_size}"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens safely using tiktoken."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace, remove control chars."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def fixed_size_chunking(self, text: str) -> List[str]:
        """Token-based fixed-size chunking with overlap."""
        text = self._clean_text(text)
        if not text:
            return []

        tokens = self.tokenizer.encode(text)
        chunks = []
        i = 0

        while i < len(tokens):
            end = i + self.chunk_size
            chunk_tokens = tokens[i:end]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)
            i += self.chunk_size - self.chunk_overlap
            if i > 0 and i >= len(tokens):
                break

        logger.info(f"Fixed chunking → {len(chunks)} chunks")
        return chunks

    def semantic_chunking(self, text: str) -> List[str]:
        """Sentence-aware chunking with token limits."""
        text = self._clean_text(text)
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return [text[:self.max_size]]

        chunks = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            sent_tokens = self.count_tokens(sentence)

            # Handle oversized sentence
            if sent_tokens > self.max_size:
                if current:
                    chunk = " ".join(current)
                    if self.count_tokens(chunk) >= self.min_size:
                        chunks.append(chunk)
                    current = []
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                temp = []
                temp_tokens = 0
                for word in words:
                    w_tokens = self.count_tokens(word + " ")
                    if temp_tokens + w_tokens > self.max_size:
                        if temp:
                            chunks.append(" ".join(temp).strip())
                        temp = [word]
                        temp_tokens = w_tokens
                    else:
                        temp.append(word)
                        temp_tokens += w_tokens
                if temp:
                    chunks.append(" ".join(temp).strip())
                continue

            # Add to current chunk
            if current_tokens + sent_tokens > self.max_size:
                if current_tokens >= self.min_size:
                    chunks.append(" ".join(current).strip())
                    overlap_sents = []
                    overlap_tokens = 0
                    for s in reversed(current):
                        s_tok = self.count_tokens(s)
                        if overlap_tokens + s_tok <= self.chunk_overlap:
                            overlap_sents.append(s)
                            overlap_tokens += s_tok
                        else:
                            break
                    current = list(reversed(overlap_sents)) + [sentence]
                    current_tokens = overlap_tokens + sent_tokens
                else:
                    chunks.append(" ".join(current).strip())
                    current = [sentence]
                    current_tokens = sent_tokens
            else:
                current.append(sentence)
                current_tokens += sent_tokens

        if current:
            final = " ".join(current).strip()
            if self.count_tokens(final) >= self.min_size or not chunks:
                chunks.append(final)

        logger.info(f"Semantic chunking → {len(chunks)} chunks")
        return chunks

    def chunk_text(self, text: str, strategy: str = "semantic") -> List[str]:
        """Public API for chunking."""
        if not text or not text.strip():
            logger.warning("Empty text passed to chunk_text")
            return []

        if strategy == "fixed":
            return self.fixed_size_chunking(text)
        elif strategy == "semantic":
            return self.semantic_chunking(text)
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Use 'fixed' or 'semantic'")


chunker = TextChunker()


def chunk_document(text: str, strategy: str = "semantic") -> List[str]:
    """Convenience wrapper for chunking."""
    return chunker.chunk_text(text, strategy)