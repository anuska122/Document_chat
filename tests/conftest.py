import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Tests never read credentials or contact configured infrastructure.
os.environ.update(
    DATABASE_URL="sqlite+aiosqlite:///:memory:",
    GROQ_API_KEY="test",
    PINECONE_API_KEY="test",
    LLM_PROVIDER="ollama",
    LLM_MODEL="llama3.1",
    EMBEDDING_PROVIDER="hf",
)
import pytest_asyncio
from db.sql_db import init_db, engine, Base


@pytest_asyncio.fixture
async def database():
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await init_db()
    yield
