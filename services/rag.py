from typing import List, Dict, Optional
import logging
from openai import AsyncOpenAI
from services.embeddings import generate_embedding
from services.redis_memory import get_chat_history, add_to_history
from db.vector_db import query_vectors
from config import settings

logger = logging.getLogger(__name__)


class CustomRAGPipeline:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL
        self.top_k = settings.TOP_K_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        logger.info(f"RAG pipeline initialized with OpenAI model: {settings.LLM_MODEL}")

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
        """Generates response using OpenAI Chat Completion API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                top_p=0.95
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return (
                "I apologize, but I encountered an error generating a response. "
                "Please try again or rephrase your question."
            )

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
                    "relevance_score": chunk.get('score', 0.0),
                    "text_preview": chunk.get('metadata', {}).get('text', '')[:200]
                }
                for chunk in chunks
            ]
            
            # Calculating confidence
            avg_score = sum(c.get('score', 0) for c in chunks) / len(chunks)
            
            return {
                "response": response,
                "sources": sources,
                "session_id": session_id,
                "confidence_score": round(avg_score, 2)
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