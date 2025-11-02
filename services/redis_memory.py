import json
import logging
from typing import List, Dict
import redis.asyncio as aioredis

from config import settings

logger = logging.getLogger(__name__)

class RedisMemoryManager:
    def __init__(self):
        self.redis_client = None
        self.session_expire = settings.REDIS_SESSION_EXPIRE
    
    async def connect(self):
        try:
            self.redis_client = await aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                password = settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                encoding='utf=8',
                decode_response=True                    
            )
            logger.info("Connected sucessfully")
        except Exception as e:
            logger.error(f"connection error: {str(e)}")
            raise
    
    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    def _get_key(self,session_id:str)->str:
        return f"chat:session:{session_id}"
    
    async def get_history(self,session_id:str)->List[Dict[str,str]]:
        try:
            key=self._get_key(session_id)
            history_json = await self.redis_client.get(key)
            if history_json:
                history = json.loads(history_json)
                logger.debug(f"retrived {len(history)} message for session {session_id}")
                return history
            else:
                logger.debug(f"No history found for session {session_id}")
                return []
        except Exception as e:
            logger.error(f"Error generated {str(e)}")
            return []
    
    async def add_message(self,session_id:str,role:str,content:str):
        try:
            key = self._get_key(session_id)
            #getting existing history
            history = await self.get_history(session_id)
            history.append({
                "role":role,
                "content":content
            })
            if len(history)>20:
                history=history[-20:]
            await self.redis_client.setex(key,self.session_expire,json.dumps(history))
            logger.debug(f"Added {role} message to session {session_id}")
        except Exception as e:
            logger.error(f"Error ocurred: {str(e)}")
            raise
    async def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str
    ):
        await self.add_message(session_id, "user", user_message)
        await self.add_message(session_id, "assistant", assistant_message)

    async def clear_history(self, session_id: str):
        try:
            key = self._get_key(session_id)
            await self.redis_client.delete(key)
            logger.info(f"Cleared history for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            raise

    async def get_session_count(self) -> int:
        """total number of active sessions"""
        try:
            keys = await self.redis_client.keys("chat:session:*")
            return len(keys)
        except Exception as e:
            logger.error(f"Error counting sessions: {str(e)}")
            return 0

memory_manager = RedisMemoryManager()
async def init_redis():
    """Initialize Redis connection"""
    await memory_manager.connect()


async def close_redis():
    """Close Redis connection"""
    await memory_manager.disconnect()


async def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """chat history for session"""
    return await memory_manager.get_history(session_id)


async def add_to_history(
    session_id: str,
    user_message: str,
    assistant_message: str
):
    """Adding Q&A exchange to history"""
    await memory_manager.add_exchange(session_id, user_message, assistant_message)
    
async def clear_session(session_id: str):
    """Clear session history"""
    await memory_manager.clear_history(session_id)