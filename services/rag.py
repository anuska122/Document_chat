from typing import List, Dict, Optional
import asyncio
import json
import logging
import urllib.error
import urllib.request

from groq import AsyncGroq

from services.embeddings import generate_embedding
from services.redis_memory import get_chat_history, add_to_history
from db.vector_db import query_vectors
from config import settings

logger = logging.getLogger(__name__)


class CustomRAGPipeline:
    def __init__(self):
        # Use the configured provider for LLM inference.
        self.provider = settings.LLM_PROVIDER.lower()
        self.model = settings.LLM_MODEL
        self.top_k = settings.TOP_K_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.client = None
        self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")

        if self.provider == "groq":
            self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        elif self.provider != "ollama":
            raise ValueError(
                f"Unsupported LLM_PROVIDER={settings.LLM_PROVIDER!r}. "
                "Use 'ollama' or 'groq'."
            )

        logger.info(
            "RAG pipeline initialized with %s model: %s",
            self.provider,
            self.model,
        )

    @staticmethod
    def _api_score(score: float) -> float:
        """Clamp vector similarity into the response schema's 0..1 range."""
        return max(0.0, min(1.0, score))

    async def retrieve_relevant_chunks(
        self, 
        query: str, 
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        try:
            query_embedding = await generate_embedding(query)
            logger.info(f"Query: '{query}' → embedding generated")

            results = await query_vectors(
                embedding=query_embedding,
                top_k=self.top_k,
                document_ids=document_ids
            )

            logger.info(f"Pinecone returned {len(results)} results")
            for i, r in enumerate(results):
                score = r.get("score", 0.0)
                text = r.get("metadata", {}).get("text", "NO TEXT")[:120]
                source = r.get("metadata", {}).get("source", "unknown")
                logger.info(f"  [{i+1}] Score: {score:.4f} | Source: {source} | Text: {text}...")

            filtered = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
            logger.info(f"Threshold {self.similarity_threshold} → {len(filtered)} chunks kept")

            return filtered
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            return []

    def build_context(self, chunks: List[Dict]) -> str:
        """Building context string from retrieved chunks"""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('metadata', {}).get('text', '')
            source = chunk.get('metadata', {}).get('source', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    async def build_messages(
        self, 
        query: str, 
        context: str, 
        session_id: str
    ) -> List[Dict[str, str]]:
        
        history = await get_chat_history(session_id)
        
        # System message with context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Answer questions based on the provided context. "
                    "If the context doesn't contain relevant information, say so clearly. "
                    "Be concise and accurate.\n\n"
                    f"Context:\n{context}"
                )
            }
        ]
        
        # Adding conversation history
        if history:
            for msg in history[-6:]:  # Last 6 messages
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role in ['user', 'assistant']:
                    messages.append({"role": role, "content": content})
        
        # Adding current query
        messages.append({"role": "user", "content": query})
        
        return messages

    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generates response using the configured chat provider."""
        try:
            if self.provider == "ollama":
                answer = await self._generate_with_ollama(messages)
                logger.info(f"Generated response (Ollama): {answer[:100]}...")
                return answer

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content or ""
            logger.info(f"Generated response (Groq): {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            return (
                "I apologize, but I encountered an error generating a response. "
                "Please try again or rephrase your question."
            )

    async def _generate_with_ollama(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        response = await asyncio.to_thread(
            self._post_ollama_chat,
            messages,
        )

        message = response.get("message", {})
        content = message.get("content")
        if content:
            return content.strip()

        raise RuntimeError("Ollama chat response did not include content")

    def _post_ollama_chat(self, messages: List[Dict[str, str]]) -> dict:
        request = urllib.request.Request(
            f"{self.ollama_base_url}/api/chat",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": settings.LLM_TEMPERATURE,
                        "num_predict": settings.LLM_MAX_TOKENS,
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = body.strip() or exc.reason
            raise RuntimeError(
                f"Ollama chat request failed with HTTP {exc.code}: {message}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running."
            ) from exc

    async def process_query(
        self, 
        query: str, 
        session_id: str, 
        document_ids: Optional[List[str]] = None
    ) -> Dict:
        """Complete RAG pipeline: Retrieve → Build Messages → Generate → Store"""
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Retrieve relevant chunks
            chunks = await self.retrieve_relevant_chunks(query, document_ids)
            
            if not chunks:
                return {
                    "response": (
                        "I couldn't find relevant information in the uploaded documents "
                        "to answer your question. Please try rephrasing or uploading "
                        "more relevant documents."
                    ),
                    "sources": [],
                    "session_id": session_id,
                    "confidence_score": 0.0
                }
            
            # Building context and messages
            context = self.build_context(chunks)
            messages = await self.build_messages(query, context, session_id)
            
            #response generation
            response = await self.generate_response(messages)
            
            # Storing in history
            await add_to_history(
                session_id=session_id,
                user_message=query,
                assistant_message=response
            )
            
            # sources preparation
            sources = [
                {
                    "document": chunk.get('metadata', {}).get('source', 'Unknown'),
                    "chunk_index": chunk.get('metadata', {}).get('chunk_index', 0),
                    "relevance_score": self._api_score(chunk.get('score', 0.0)),
                    "text_preview": chunk.get('metadata', {}).get('text', '')[:200]
                }
                for chunk in chunks
            ]
            
            # Calculating confidence
            avg_score = sum(c.get('score', 0) for c in chunks) / len(chunks)
            confidence_score = self._api_score(avg_score)
            
            return {
                "response": response,
                "sources": sources,
                "session_id": session_id,
                "confidence_score": round(confidence_score, 2)
            }
            
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}", exc_info=True)
            raise


rag_pipeline = CustomRAGPipeline()


async def run_rag(
    query: str,
    session_id: str,
    document_ids: Optional[List[str]] = None
) -> Dict:
    """Convenience wrapper for RAG pipeline"""
    return await rag_pipeline.process_query(query, session_id, document_ids)
