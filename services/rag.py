from typing import List,Dict,Optional
import logging
from huggingface_hub import InferenceClient
import asyncio
from services.embeddings import generate_embedding
from services.redis_memory import get_chat_history, add_to_history
from db.vector_db import query_vectors
from config import settings

logger = logging.getLogger(__name__)

class CustomRAGPipeline:
    def __init__(self):
        self.client = InferenceClient(
            model=settings.LLM_MODEL,
            token = settings.HUGGINGFACE_API_KEY
        )
        self.top_k = settings.TOP_K_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        logger.info(f"RAG pipeline initialized: {settings.LLM_MODEL}")

    async def retrieve_relevant_chunks(self,query:str,document_ids:Optional[List[str]]=None) -> List[Dict]:
        try:
            query_embedding = await generate_embedding(query)
            #query vector db
            results = await query_vectors(embedding=query_embedding,top_k=self.top_k,document_ids=document_ids)

            #filtered by similarity threshold
            self.similarity_threshold = 0
            filtered_results = results
            logger.info(f"Retrieved {len(filtered_results)} chunks above threshold")
            return filtered_results
        except Exception as e:
            logger.error(f"Retrieval error:{str(e)}")
            raise

    def build_context(self,chunks:List[Dict]) -> str:
        """building context string from retrieved chunks"""
        if not chunks:
            return "No relevant context found."
        context_parts = []
        for i, chunks in enumerate(chunks, 1):
            text = chunks.get('metadata', {}).get('text','')
            source = chunks.get('metadata',{}).get('source','Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{text}\n")
        return "\n".join(context_parts)
    
    async def build_prompt(self,query:str,context:str,session_id:str)->str:
        """building complete prompt with context and history"""
        history = await get_chat_history(session_id)
        prompt = f"""<s>[INST] You are a helpful AI assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so clearly. Be concise and accurate.
        Context:
       {context}"""
        
        if history:
            prompt +="\n\nPrevious conversation:\n"
            for msg in history[-6:]:
                role = msg.get('role','')
                content = msg.get('content','')
                if role == 'user':
                    prompt += f"user: {content}\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n"
        prompt += f"\nQuestion: {query}\n\nAnswer: [/INST]"
        
        return prompt
    async def generate_response(self,prompt:str) -> str:
        try:
            # Calling HuggingFace Inference API in thread to avoid blocking
            response = await asyncio.to_thread(
                self.client.text_generation,
                prompt,
                max_new_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.15
            )
            # Cleaning up response
            answer = response.strip()
            
            # Removing any instruction tokens that might leak through
            answer = answer.replace("[INST]", "").replace("[/INST]", "").replace("</s>", "").strip()
            
            logger.info(f"Generated response: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            # Fallback response
            return "I apologize, but I encountered an error generating a response. Please try again or rephrase your question."
    async def process_query(self,query:str,session_id:str,document_ids:Optional[List[str]]=None)->Dict:
        """Complete RAG pipeline: Retrieve → Build Prompt → Generate → Store"""
        try:
            logger.info(f"Query is being processed: {query[:100]}...")
            chunks = await self.retrieve_relevant_chunks(query,document_ids)
            if not chunks:
                return{
                    "response":"I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing or uploading more relevant documents.",
                    "source":[],
                    "session_id":session_id,
                    "confidence_score":0.0
                }
            #building context
            context = self.build_context(chunks)
            #prompt with history
            prompt = await self.build_prompt(query,context,session_id)
            #generate response
            response = await self.generate_response(prompt)
            #storing
            await add_to_history(session_id=session_id,user_message=query,assistant_message=response)
            #sources 
            sources = [
                {
                    "document": chunk.get('metadata', {}).get('source', 'Unknown'),
                    "chunk_index": chunk.get('metadata', {}).get('chunk_index', 0),
                    "relevance_score": chunk.get('score', 0.0),
                    "text_preview": chunk.get('metadata', {}).get('text', '')[:200]
                }
                for chunk in chunks
            ]
            #confidence calculation
            avg_score = sum(c.get('score',0) for c in chunks) / len(chunks)
            return{
                "response":response,
                "sources":sources,
                "session_id":session_id,
                "confidence_score":round(avg_score, 2)
            }
        except Exception as e:
            logger.error(f"Error occured: {str(e)}", exc_info=True)
            raise
rag_pipeline = CustomRAGPipeline()
async def run_rag(
    query: str,
    session_id: str,
    document_ids: Optional[List[str]] = None
) -> Dict:
    return await rag_pipeline.process_query(query, session_id, document_ids)


    



