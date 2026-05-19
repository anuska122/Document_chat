import json
import logging
from typing import List, Dict
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from config import settings

logger = logging.getLogger(__name__)

class RedisMemoryManager:
    def __init__(self):
        self.redis_client = None
        self.session_expire = settings.REDIS_SESSION_EXPIRE
    
    async def connect(self):
        try:
            self.redis_client = aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                password = settings.REDIS_PASSWORD,
                db=settings.REDIS_DB,
                encoding='utf-8',
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except RedisError as e:
            self.redis_client = None
            logger.warning(
                "Redis unavailable; chat history will be disabled: %s",
                e,
            )
    
    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    def _get_key(self,session_id:str)->str:
        return f"chat:session:{session_id}"
    
    async def get_history(self,session_id:str)->List[Dict[str,str]]:
        if not self.redis_client:
            return []

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
        except RedisError as e:
            logger.warning(f"Could not read Redis history: {str(e)}")
            return []
    
    async def add_message(self,session_id:str,role:str,content:str):
        if not self.redis_client:
            logger.debug("Skipping chat history write because Redis is unavailable")
            return

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
        except RedisError as e:
            logger.warning(f"Could not write Redis history: {str(e)}")

    async def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str
    ):
        await self.add_message(session_id, "user", user_message)
        await self.add_message(session_id, "assistant", assistant_message)

    async def clear_history(self, session_id: str):
        if not self.redis_client:
            return

        try:
            key = self._get_key(session_id)
            await self.redis_client.delete(key)
            logger.info(f"Cleared history for session {session_id}")
        except RedisError as e:
            logger.warning(f"Could not clear Redis history: {str(e)}")

    async def get_session_count(self) -> int:
        """total number of active sessions"""
        if not self.redis_client:
            return 0

        try:
            keys = await self.redis_client.keys("chat:session:*")
            return len(keys)
        except RedisError as e:
            logger.warning(f"Could not count Redis sessions: {str(e)}")
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
