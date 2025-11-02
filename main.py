from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from routers import ingest, chat
from db.sql_db import init_db
from db.vector_db import init_vector_db
from config import settings

#logging configure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):
    """ Logic for startup and shutdown """
    logger.info("RAG Backend is starting")

    #DB initialize
    await init_db()
    await init_vector_db()

    #Redis initailize
    from services.redis_memory import init_redis, close_redis
    await init_redis()

    logger.info("All services initailized")
    yield
    logger.info("RAG Backend is shutting down")
    await close_redis()

#FastAPT application

app = FastAPI(
    title="Conversational RAG Backend",
    description="AI-powered document chat with custom RAG and memory",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url = "/redoc"
)

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#routers
app.include_router(ingest.router, prefix="/api",tags=['Document Ingestion'])
app.include_router(chat.router, prefix="/api",tags=["Conversational RAG"])

@app.get("/health")
async def health_check():
    return{
        "status":"healthy",
        "version":"1.0.0",
        "services":{
            "vector_db": "connected",
            "sql_db":"connected",
            "redis":"connected"
        }
    }
@app.get("/")
async def root():
    return{
        "message":"Welcome to conversational RAG Backend",
        "docs":"/docs",
        "health":"/health"
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

